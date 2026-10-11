#!/usr/bin/env python3
"""Daily briefings: RSS feeds -> Claude summary -> WhatsApp (CallMeBot).

Usage:
    python briefing.py                 # run every topic
    python briefing.py security ai     # run only these topics
    python briefing.py --dry-run ai    # print instead of sending

Environment variables:
    ANTHROPIC_API_KEY   your Claude API key
    CALLMEBOT_PHONE     your WhatsApp number with country code, e.g. +123456789
    CALLMEBOT_APIKEY    the key CallMeBot sent you
    CLAUDE_MODEL        optional, defaults to claude-haiku-5-5
"""
import calendar
import html
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
import feedparser
import requests
from anthropic import Anthropic

load_dotenv(Path(__file__).resolve().with_name(".env"))

MODEL = os.environ.get("CLAUDE_MODEL", "claude-haiku-5-5")
LOOKBACK_HOURS = 54       # 2 days + buffer; only consider stories newer than this
MAX_PER_FEED = 25         # newest N entries read from each feed
MAX_ITEMS_TO_CLAUDE = 60  # cap on stories sent to Claude per topic
MAX_MSG_CHARS = 1400      # conservative WhatsApp chunk size
USER_AGENT = "Mozilla/5.0 (compatible; PersonalBriefingBot/1.0)"

# ---------------------------------------------------------------------------
# EDIT THIS: topics, feeds, and what you care about.
# Feed URLs change over time. The script reports any feed that fails, so just
# replace or remove those.
# ---------------------------------------------------------------------------
TOPICS = {
    "security": {
        "title": "Cybersecurity",
        "focus": "Actively exploited vulnerabilities, major breaches, ransomware, "
                 "and patches affecting widely used software. Flag anything "
                 "that needs action ('patch now').",
        "feeds": [
            "https://krebsonsecurity.com/feed/",
            "https://www.bleepingcomputer.com/feed/",
            "https://feeds.feedburner.com/TheHackersNews",
            "https://www.cisa.gov/cybersecurity-advisories/all.xml",
        ],
    },
    "ai": {
        "title": "AI",
        "focus": "New model releases, research breakthroughs, major product "
                 "launches, policy and regulation, and notable funding.",
        "feeds": [
            "https://techcrunch.com/category/artificial-intelligence/feed/",
            "https://openai.com/news/rss.xml",
            "https://deepmind.google/blog/rss.xml",
            "https://hnrss.org/newest?q=AI+OR+LLM&points=150",
        ],
    },
    "tech": {
        "title": "Software & Hardware",
        "focus": "Significant software releases, developer tools, chips, devices, "
                 "and industry shifts. Skip minor rumors and routine deals.",
        "feeds": [
            "https://hnrss.org/frontpage?points=200",
            "https://www.theverge.com/rss/index.xml",
            "https://feeds.arstechnica.com/arstechnica/technology-lab",
            "https://www.tomshardware.com/feeds/all",
        ],
    },
    "markets": {
        "title": "Markets",
        "focus": "Index moves, major earnings, central bank news, and big company "
                 "stories. Mention companies in the news but do NOT give buy/sell "
                 "advice.",
        "feeds": [
            "https://www.cnbc.com/id/100003114/device/rss/rss.html",
            "https://feeds.marketwatch.com/marketwatch/topstories/",
        ],
    },
}
# ---------------------------------------------------------------------------


def clean(text: str, limit: int = 300) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = html.unescape(re.sub(r"\s+", " ", text)).strip()
    return text[:limit]


def fetch_items(feeds):
    cutoff = datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)
    items, seen = [], set()
    for url in feeds:
        try:
            parsed = feedparser.parse(url, agent=USER_AGENT)
        except Exception as exc:  # network or parse failure
            print(f"  [feed failed] {url}: {exc}")
            continue
        if not parsed.entries:
            print(f"  [feed empty/failed] {url}")
            continue
        for entry in parsed.entries[:MAX_PER_FEED]:
            ts = entry.get("published_parsed") or entry.get("updated_parsed")
            if ts:
                published = datetime.fromtimestamp(calendar.timegm(ts), tz=timezone.utc)
                if published < cutoff:
                    continue
            title = clean(entry.get("title", ""), 200)
            link = entry.get("link", "")
            if not title or title.lower() in seen:
                continue
            seen.add(title.lower())
            items.append({
                "source": clean(parsed.feed.get("title", url), 40),
                "title": title,
                "summary": clean(entry.get("summary", "")),
                "link": link,
            })
    return items[:MAX_ITEMS_TO_CLAUDE]


def summarize(client, topic, items):
    stories = "\n".join(
        f"- [{i['source']}] {i['title']} :: {i['summary']} ({i['link']})"
        for i in items
    )
    prompt = f"""You are writing a morning briefing on {topic['title']} for one reader.
Reader cares about: {topic['focus']}

From the stories below, pick the 5 most important (fewer if little matters).
Merge duplicate stories. Rank by importance.

Format for WhatsApp, plain text only:
*{topic['title']} briefing*
Then for each story:
1. *Headline in your own words*
   One sentence on why it matters. <link>

Rules: be concise, never invent details not in the stories, no preamble, no closing remarks.

Stories:
{stories}"""
    response = client.messages.create(
        model=MODEL,
        max_tokens=900,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in response.content if b.type == "text").strip()


def chunk(text, limit=MAX_MSG_CHARS):
    parts, current = [], ""
    for line in text.splitlines(keepends=True):
        if len(current) + len(line) > limit and current:
            parts.append(current.strip())
            current = ""
        current += line
    if current.strip():
        parts.append(current.strip())
    return parts


def send_whatsapp(text):
    phone = os.environ["CALLMEBOT_PHONE"]
    apikey = os.environ["CALLMEBOT_APIKEY"]
    for part in chunk(text):
        resp = requests.get(
            "https://api.callmebot.com/whatsapp.php",
            params={"phone": phone, "text": part, "apikey": apikey},
            timeout=30,
        )
        print(f"  [callmebot] HTTP {resp.status_code}")
        resp.raise_for_status()
        time.sleep(3)  # be gentle with the free service


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry_run = "--dry-run" in sys.argv
    selected = args or list(TOPICS)
    unknown = [t for t in selected if t not in TOPICS]
    if unknown:
        sys.exit(f"Unknown topic(s): {unknown}. Available: {list(TOPICS)}")

    client = Anthropic()  # reads ANTHROPIC_API_KEY
    for key in selected:
        topic = TOPICS[key]
        print(f"== {topic['title']} ==")
        items = fetch_items(topic["feeds"])
        print(f"  {len(items)} recent stories")
        if not items:
            print("  nothing to report, skipping")
            continue
        brief = summarize(client, topic, items)
        if dry_run:
            print(brief, "\n")
        else:
            send_whatsapp(brief)
            time.sleep(5)


if __name__ == "__main__":
    main()
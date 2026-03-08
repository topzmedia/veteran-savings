#!/usr/bin/env python3
"""
Scrape hooks from transitionalhooks.com/social-media-video-hook-library/
and save them to ai_ad_agency/data/inputs/transitional_hooks.json

Run from the aiagency project root:
    python scripts/scrape_hooks.py
"""
import json
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://transitionalhooks.com/social-media-video-hook-library/"
PAGES = list(range(1, 21))  # pages 1–20
OUTPUT = Path("ai_ad_agency/data/inputs/transitional_hooks.json")
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    )
}


def fetch_page(page: int) -> list[dict]:
    url = BASE_URL if page == 1 else f"{BASE_URL}page/{page}/"
    print(f"  Fetching page {page}: {url}")
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    hooks = []

    # Try common WordPress / hook-library patterns
    # Pattern 1: article or div with a title + body
    for article in soup.select("article, .hook-item, .entry, .post"):
        title_el = article.select_one("h1, h2, h3, h4, .hook-title, .entry-title")
        body_el = article.select_one("p, .hook-text, .hook-body, .entry-content p")

        title = title_el.get_text(strip=True) if title_el else ""
        body = body_el.get_text(strip=True) if body_el else ""

        if body and len(body) > 5:
            hooks.append({"title": title, "hook": body})

    # Pattern 2: fallback — every <p> that looks like a hook (short sentence)
    if not hooks:
        for p in soup.select("p"):
            text = p.get_text(strip=True)
            if 10 < len(text) < 300 and not text.startswith("©"):
                hooks.append({"title": "", "hook": text})

    print(f"    → found {len(hooks)} hooks")
    return hooks


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    all_hooks = []

    for page in PAGES:
        try:
            page_hooks = fetch_page(page)
            all_hooks.extend(page_hooks)
        except Exception as exc:
            print(f"  ERROR on page {page}: {exc}")
        time.sleep(0.8)  # be polite

    # Deduplicate by hook text
    seen = set()
    unique = []
    for item in all_hooks:
        key = item["hook"].lower().strip()
        if key not in seen and len(key) > 5:
            seen.add(key)
            unique.append(item)

    OUTPUT.write_text(json.dumps(unique, indent=2, ensure_ascii=False))
    print(f"\nDone! Saved {len(unique)} unique hooks → {OUTPUT}")
    print("Now commit and push this file:")
    print("  git add ai_ad_agency/data/inputs/transitional_hooks.json")
    print("  git commit -m 'Add TransitionalHooks.com hook bank (scraped)'")
    print("  git push -u origin claude/ai-ad-agency-platform-ZH9fb")


if __name__ == "__main__":
    main()

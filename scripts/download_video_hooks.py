#!/usr/bin/env python3
"""
Download all video hook MP4s from transitionalhooks.com via the WP REST API.

Run from the aiagency project root:
    python scripts/download_video_hooks.py

Videos are saved to:
    ai_ad_agency/data/inputs/video_hooks/<slug>.mp4
A manifest is written to:
    ai_ad_agency/data/inputs/video_hooks/manifest.json
"""
import json
import time
from pathlib import Path

import requests

API_BASE = "https://transitionalhooks.com/wp-json/wp/v2"
OUT_DIR = Path("ai_ad_agency/data/inputs/video_hooks")
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    )
}
PER_PAGE = 100


def fetch_all_posts() -> list[dict]:
    posts, page = [], 1
    while True:
        resp = requests.get(
            f"{API_BASE}/video-hook",
            params={"per_page": PER_PAGE, "page": page},
            headers=HEADERS,
            timeout=30,
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        posts.extend(batch)
        print(f"  Page {page}: fetched {len(batch)} posts (total so far: {len(posts)})")
        if len(batch) < PER_PAGE:
            break
        page += 1
        time.sleep(0.5)
    return posts


def fetch_video_url(post_id: int) -> str | None:
    resp = requests.get(
        f"{API_BASE}/media",
        params={"parent": post_id},
        headers=HEADERS,
        timeout=30,
    )
    resp.raise_for_status()
    media = resp.json()
    for item in media:
        if item.get("mime_type", "").startswith("video/"):
            return item.get("source_url")
    return None


def download_video(url: str, dest: Path) -> bool:
    if dest.exists():
        print(f"    [skip] already downloaded: {dest.name}")
        return False
    resp = requests.get(url, headers=HEADERS, stream=True, timeout=60)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 64):
            f.write(chunk)
    return True


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_DIR / "manifest.json"

    # Load existing manifest to allow resuming interrupted runs
    manifest: dict[str, dict] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())

    print("Fetching all video-hook posts...")
    posts = fetch_all_posts()
    print(f"Total posts: {len(posts)}\n")

    downloaded = skipped = errors = 0

    for i, post in enumerate(posts, 1):
        slug = post["slug"]
        post_id = post["id"]
        title = post["title"]["rendered"]
        print(f"[{i}/{len(posts)}] {title} (id={post_id})")

        if slug in manifest and manifest[slug].get("local_file"):
            local = Path(manifest[slug]["local_file"])
            if local.exists():
                print(f"    [skip] manifest entry exists: {local.name}")
                skipped += 1
                continue

        try:
            video_url = fetch_video_url(post_id)
            time.sleep(0.3)
        except Exception as exc:
            print(f"    [error] media fetch failed: {exc}")
            errors += 1
            continue

        if not video_url:
            print(f"    [warn] no video attachment found")
            manifest[slug] = {"title": title, "post_id": post_id, "source_url": None, "local_file": None}
            continue

        dest = OUT_DIR / f"{slug}.mp4"
        try:
            was_new = download_video(video_url, dest)
            if was_new:
                print(f"    [ok] downloaded → {dest.name}")
                downloaded += 1
            else:
                skipped += 1
        except Exception as exc:
            print(f"    [error] download failed: {exc}")
            errors += 1
            continue

        manifest[slug] = {
            "title": title,
            "post_id": post_id,
            "source_url": video_url,
            "local_file": str(dest),
        }

        # Save manifest after every video so progress survives interruptions
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        time.sleep(0.4)

    print(f"\nDone. Downloaded: {downloaded}  Skipped: {skipped}  Errors: {errors}")
    print(f"Videos in: {OUT_DIR}/")
    print(f"Manifest:  {manifest_path}")


if __name__ == "__main__":
    main()

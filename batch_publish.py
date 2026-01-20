#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, json, time, argparse, subprocess, re
from typing import List, Dict, Any, Optional
from yt_dlp import YoutubeDL
import requests

PROGRESS_FILE = "batch_progress.json"

def _yt_base_opts(*, skip_download: bool = False, extract_flat: bool = False) -> dict:
    """
    Base yt-dlp options everywhere in this script.
    Automatically injects cookies if YT_COOKIES_FILE is present.
    """
    opts: Dict[str, Any] = {
        "quiet": True,
        "noprogress": True,
    }
    if skip_download:
        opts["skip_download"] = True
    if extract_flat:
        # True is fine for channel/playlist listing
        opts["extract_flat"] = True
    cf = os.getenv("YT_COOKIES_FILE")
    if cf and os.path.exists(cf):
        opts["cookiefile"] = cf
    return opts

def simple_slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"['’]", "", text)
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "post"

def list_channel_videos(channel_url: str, max_results: Optional[int] = None) -> List[str]:
    """
    Returns a list of canonical video URLs from a channel or playlist page.
    Works with:
      - https://www.youtube.com/@handle/videos
      - https://www.youtube.com/channel/UCxxxx/videos
      - Uploads playlists, normal playlists, etc.
    """
    with YoutubeDL(_yt_base_opts(skip_download=True, extract_flat=True)) as ydl:
        info = ydl.extract_info(channel_url, download=False)
    entries = info.get("entries") or []
    urls: List[str] = []
    for e in entries:
        u = e.get("webpage_url") or e.get("url") or e.get("id")
        if not u:
            continue
        if not u.startswith("http"):
            u = f"https://www.youtube.com/watch?v={u}"
        urls.append(u)
        if max_results and len(urls) >= max_results:
            break
    return urls

def load_progress() -> Dict[str, Any]:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {"done": {}, "failed": {}}

def save_progress(data: Dict[str, Any]) -> None:
    tmp = PROGRESS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, PROGRESS_FILE)

def wp_has_post_with_slug(wp_url: str, wp_user: str, wp_pass: str, slug: str) -> bool:
    api = wp_url.rstrip("/") + "/wp-json/wp/v2/posts"
    try:
        r = requests.get(api, params={"slug": slug, "per_page": 1}, auth=(wp_user, wp_pass), timeout=20)
        if r.status_code == 200 and isinstance(r.json(), list) and r.json():
            return True
    except Exception:
        pass
    return False

def run_one(url: str, args, progress) -> bool:
    # Prefetch title for duplicate check
    try:
        with YoutubeDL(_yt_base_opts(skip_download=True)) as ydl:
            info = ydl.extract_info(url, download=False)
        title = (info.get("title") or "").strip()
    except Exception as e:
        title = ""
        print(f"[warn] Could not prefetch title: {e}")

    if args.skip_duplicates and title:
        slug = simple_slugify(title)
        if wp_has_post_with_slug(args.wp_url, args.wp_user, args.wp_pass, slug):
            print(f"[skip] slug exists on WP: {slug}  ({url})")
            progress["done"][url] = {"status": "skipped-duplicate", "when": time.time()}
            save_progress(progress)
            return True

    cmd = [
        sys.executable, "get_transcript.py", url,
        "--post", "--wp-status", args.wp_status,
        "--screenshots", "--screenshot-width", str(args.screenshot_width),
        "--screenshot-offset", str(args.screenshot_offset),
        "--image-quality", str(args.image_quality),
        "--featured-image",
    ]
    if args.no_local:
        cmd.append("--no-local")

    env = os.environ.copy()
    env["WP_URL"] = args.wp_url
    env["WP_USER"] = args.wp_user
    env["WP_APP_PASS"] = args.wp_pass
    # Pass cookies to child script (get_transcript.py reads YT_COOKIES_FILE)
    cf = os.getenv("YT_COOKIES_FILE")
    if cf:
        env["YT_COOKIES_FILE"] = cf

    print(f"[run] {url}")
    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=args.timeout)
        if p.returncode == 0:
            print(f"[ok]  {url}")
            progress["done"][url] = {"status": "ok", "when": time.time()}
            save_progress(progress)
            return True
        else:
            print(f"[err] {url}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}")
            progress["failed"][url] = {
                "status": f"rc={p.returncode}",
                "stdout": p.stdout[-2000:],
                "stderr": p.stderr[-2000:],
                "when": time.time()
            }
            save_progress(progress)
            return False
    except subprocess.TimeoutExpired:
        print(f"[timeout] {url}")
        progress["failed"][url] = {"status": "timeout", "when": time.time()}
        save_progress(progress)
        return False

def main():
    ap = argparse.ArgumentParser(description="Batch publish all videos from a channel to WordPress using get_transcript.py")
    ap.add_argument("channel_url", help="YouTube channel or playlist URL (e.g., https://www.youtube.com/@handle/videos)")
    ap.add_argument("--max", type=int, default=None, help="Limit number of videos (for testing)")
    ap.add_argument("--sleep", type=float, default=5.0, help="Seconds to sleep between videos (default: 5)")
    ap.add_argument("--timeout", type=int, default=3600, help="Per-video timeout in seconds (default: 3600)")
    ap.add_argument("--wp-url", default=os.getenv("WP_URL", ""), help="WP base URL")
    ap.add_argument("--wp-user", default=os.getenv("WP_USER", ""), help="WP username")
    ap.add_argument("--wp-pass", default=os.getenv("WP_APP_PASS", os.getenv("WP_PASS", "")), help="WP application password")
    ap.add_argument("--wp-status", default="draft", choices=["draft","publish","private","pending"], help="Post status (default: draft)")
    ap.add_argument("--no-local", action="store_true", help="Skip local Whisper fallback (captions-only). Good for first pass.")
    ap.add_argument("--skip-duplicates", action="store_true", help="Skip if a post with the same slug already exists on WP")
    ap.add_argument("--screenshot-width", type=int, default=1280)
    ap.add_argument("--screenshot-offset", type=float, default=0.5)
    ap.add_argument("--image-quality", type=int, default=3)
    args = ap.parse_args()

    if not (args.wp_url and args.wp_user and args.wp_pass):
        ap.error("WP_URL, WP_USER, and WP_APP_PASS must be provided (flags or env).")

    progress = load_progress()
    urls = list_channel_videos(args.channel_url, max_results=args.max)
    if not urls:
        print("No videos found. Double-check the channel/playlist URL (use the /videos page).")
        sys.exit(2)

    print(f"Found {len(urls)} videos")

    for u in urls:
        if u in progress["done"]:
            continue
        run_one(u, args, progress)
        time.sleep(args.sleep)

    print("Batch finished.")
    print(f"Success: {len([1 for v in progress['done'].values() if v['status'].startswith(('ok','skipped'))])} | "
          f"Failed: {len(progress['failed'])}")

if __name__ == "__main__":
    main()

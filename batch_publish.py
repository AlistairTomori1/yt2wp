#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, json, time, argparse, subprocess, re
from typing import List, Dict, Any, Optional
from yt_dlp import YoutubeDL
import requests

PROGRESS_FILE = "batch_progress.json"

# ---------- yt-dlp base options (cookies + stable clients) ---------
def _yt_base_opts(skip_download: bool = True, extract_flat: bool = False) -> dict:
    opts = {
        "quiet": True,
        "noprogress": True,
        "skip_download": skip_download,
        "retries": 10,
        # Enable EJS with Node + auto-fetch scripts from GitHub
        "js_runtimes": ["node"],
        "remotecomponents": ["ejs:github"],
        # DO NOT force player_client; let yt-dlp choose
        # "extractor_args": {"youtube": {"player_client": ["android,web_safari,web_embedded,default"]}},
    }
    if extract_flat:
        opts["extract_flat"] = "in_playlist"

    cookies_file = os.getenv("YT_COOKIES_FILE")
    if cookies_file and os.path.exists(cookies_file):
        opts["cookiefile"] = cookies_file
    else:
        browser = os.getenv("YT_COOKIES_BROWSER")
        if browser in ("safari", "chrome", "firefox", "edge"):
            opts["cookiesfrombrowser"] = (browser, None, None, None)

    return opts

# ---------- helpers ----------
def simple_slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"['’]", "", text)
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "post"

def list_channel_videos(channel_url: str, max_results: Optional[int] = None) -> List[str]:
    """
    Returns a list of canonical video URLs from a channel or playlist page.
      - https://www.youtube.com/@handle/videos
      - https://www.youtube.com/channel/UCxxxx/videos
      - Uploads playlists, normal playlists, etc.
    """
    opts = _yt_base_opts(skip_download=True, extract_flat=True)
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)
    if not isinstance(info, dict):
        return []

    entries = info.get("entries") or []
    urls: List[str] = []
    for e in entries:
        # Prefer 'url' (flat), then 'webpage_url', then 'id'
        u = (e.get("url") or e.get("webpage_url") or e.get("id") or "").strip()
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
    """
    Checks if a post with the exact slug already exists (published or draft).
    """
    api = wp_url.rstrip("/") + "/wp-json/wp/v2/posts"
    try:
        r = requests.get(api, params={"slug": slug, "per_page": 1}, auth=(wp_user, wp_pass), timeout=20)
        if r.status_code == 200 and isinstance(r.json(), list) and r.json():
            return True
    except Exception:
        pass
    return False

def run_one(url: str, args, progress) -> bool:
    """
    Calls get_transcript.py with your desired flags.
    Returns True on success. Records status in progress dict.
    """
    # Quick title fetch (with cookies) for duplicate detection
    title = ""
    try:
        with YoutubeDL(_yt_base_opts(skip_download=True)) as ydl:
            info = ydl.extract_info(url, download=False)
        title = (info.get("title") or "").strip()
    except Exception as e:
        print(f"[warn] Could not prefetch title: {e}")

    # Duplicate skip (optional but recommended)
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

    # Environment for WP creds + YT cookies
    env = os.environ.copy()
    env["WP_URL"] = args.wp_url
    env["WP_USER"] = args.wp_user
    env["WP_APP_PASS"] = args.wp_pass
    # Forward cookies to get_transcript.py if present
    if os.getenv("YT_COOKIES_FILE"):
        env["YT_COOKIES_FILE"] = os.getenv("YT_COOKIES_FILE")

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

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(
        description="Batch publish all videos from a channel to WordPress using get_transcript.py"
    )
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

    # Sanity
    if not (args.wp_url and args.wp_user and args.wp_pass):
        ap.error("WP_URL, WP_USER, and WP_APP_PASS must be provided (flags or env).")

    progress = load_progress()
    urls = list_channel_videos(args.channel_url, max_results=args.max)
    if not urls:
        print("No videos found. Double-check the channel/playlist URL (use the /videos page).")
        sys.exit(2)

    print(f"Found {len(urls)} videos")

    # Resume: skip URLs already marked 'ok' or 'skipped-duplicate'
    for u in urls:
        if u in progress["done"]:
            continue
        run_one(u, args, progress)
        time.sleep(args.sleep)

    print("Batch finished.")
    print(
        f"Success: {len([1 for v in progress['done'].values() if str(v['status']).startswith(('ok','skipped'))])} | "
        f"Failed: {len(progress['failed'])}"
    )

if __name__ == "__main__":
    main()

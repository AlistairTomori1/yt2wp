#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import sys
import json
import tempfile
import subprocess
from typing import List, Tuple, Optional

import requests
from yt_dlp import YoutubeDL
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    CouldNotRetrieveTranscript,
)

# ---------- small helpers ----------

def _yt_base_opts(skip_download=True):
    """
    Build a sane default options dict for yt-dlp with cookies support.
    - If cookies file is provided, avoid 'android' client (it doesn't support cookies).
    - Otherwise, include android in the rotation to dodge SABR/PO-token experiments.
    """
    opts = {
        "quiet": True,
        "noprogress": True,
        "skip_download": skip_download,
        "retries": 10,
    }

    cookies_file = os.getenv("YT_COOKIES_FILE")
    browser = os.getenv("YT_COOKIES_BROWSER")

    if cookies_file and os.path.exists(cookies_file):
        # Using exported cookies.txt
        opts["cookiefile"] = cookies_file
        clients = ["web", "web_embedded", "web_safari", "default"]  # no 'android' when cookies in use
    else:
        # Try cookies from local browser (only works on your Mac, not in GitHub Actions)
        if browser in ("safari", "chrome", "firefox", "edge"):
            opts["cookiesfrombrowser"] = (browser, None, None, None)
        # Broader mix when no cookies: include android
        clients = ["android", "web", "web_embedded", "web_safari", "default"]

    opts.setdefault("extractor_args", {})
    opts["extractor_args"]["youtube"] = {"player_client": clients}
    return opts

def hhmmss(seconds: float) -> str:
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def parse_ts(ts: str) -> float:
    """Parse 'HH:MM:SS.mmm' or 'MM:SS.mmm' to seconds."""
    ts = ts.strip().replace(",", ".")
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
    elif len(parts) == 2:
        h, m, s = 0, int(parts[0]), float(parts[1])
    else:
        raise ValueError(f"Bad timestamp: {ts}")
    return h * 3600 + m * 60 + s

def parse_vtt(vtt_text: str) -> List[dict]:
    """Very small WebVTT parser -> segments [{'start','duration','text'}]."""
    lines = vtt_text.splitlines()
    i, n = 0, len(lines)
    segments = []
    time_pat = re.compile(r"^(\d{1,2}:\d{2}(?::\d{2})?\.\d{3})\s-->\s(\d{1,2}:\d{2}(?::\d{2})?\.\d{3})")

  
    while i < n and (not lines[i].strip() or lines[i].startswith("WEBVTT")):
        i += 1

    while i < n:

        if lines[i].strip() and not time_pat.match(lines[i]):
            i += 1
            continue

        if i >= n:
            break

        m = time_pat.match(lines[i])
        if not m:
            i += 1
            continue

        start_s = parse_ts(m.group(1))
        end_s = parse_ts(m.group(2))
        i += 1

        text_lines = []
        while i < n and lines[i].strip():
            if not lines[i].startswith("NOTE"):
                text_lines.append(lines[i])
            i += 1

        while i < n and not lines[i].strip():
            i += 1

        text = " ".join(t.strip() for t in text_lines).strip()
        if text:
            segments.append({"start": start_s, "duration": end_s - start_s, "text": text})

    return segments

# ---------- YouTube info + caption paths ----------

def fetch_video_id_and_title(url: str) -> Tuple[str, str, str]:
    opts = _yt_base_opts(skip_download=True)
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return info["id"], info.get("title", ""), info.get("description", "") or ""

def try_official_transcript(video_id: str, preferred_langs: List[str]) -> Optional[List[dict]]:
    """
    Use youtube-transcript-api. Be tolerant of version differences:
    return a list of segments or None. Never raise here.
    """
    list_fn = getattr(YouTubeTranscriptApi, "list_transcripts", None)

    try:
        if list_fn is None:
            try:
                return YouTubeTranscriptApi.get_transcript(video_id, languages=preferred_langs)
            except Exception:
                return YouTubeTranscriptApi.get_transcript(video_id)
        else:
            tl = YouTubeTranscriptApi.list_transcripts(video_id)

            for lang in preferred_langs:
                try:
                    t = tl.find_manually_created_transcript([lang])
                    return t.fetch()
                except Exception:
                    pass

            for t in tl:
                if not t.is_generated:
                    return t.fetch()

            for lang in preferred_langs:
                try:
                    t = tl.find_generated_transcript([lang])
                    return t.fetch()
                except Exception:
                    pass

            for t in tl:
                if getattr(t, "is_generated", False):
                    return t.fetch()

            return None
    except (TranscriptsDisabled, CouldNotRetrieveTranscript, NoTranscriptFound, Exception):
        return None

def _choose_caption_track(tracks: dict, preferred_langs: List[str]) -> Optional[dict]:
    """
    tracks is info['subtitles'] or info['automatic_captions']: {lang: [ {ext, url, ...}, ... ]}
    Prefer preferred_langs, then any. Prefer ext order: json3/srv3, vtt, ttml/srt/sbv.
    """
    if not tracks:
        return None
    ext_order = ["json3", "srv3", "vtt", "srt", "ttml", "sbv"]
    def pick_for_lang(lang: str) -> Optional[dict]:
        items = tracks.get(lang)
        if not items:
            return None
        by_ext = {it.get("ext"): it for it in items if it.get("url")}
        for ext in ext_order:
            if ext in by_ext:
                return by_ext[ext]
        return items[0] if items else None

    for lang in preferred_langs:
        tr = pick_for_lang(lang)
        if tr:
            tr["lang"] = lang
            return tr

    for lang in tracks.keys():
        tr = pick_for_lang(lang)
        if tr:
            tr["lang"] = lang
            return tr
    return None

def _parse_json3(text: str) -> Optional[List[dict]]:
    stripped = text.lstrip()
    if stripped.startswith(")]}'"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else ""
    try:
        data = json.loads(stripped)
    except Exception:
        return None
    events = data.get("events") or []
    segs: List[dict] = []
    for ev in events:
        start = float(ev.get("tStartMs", 0.0)) / 1000.0
        dur = float(ev.get("dDurationMs", 0.0)) / 1000.0
        parts = ev.get("segs") or []
        text = "".join(p.get("utf8", "") for p in parts).replace("\n", " ").strip()
        if text:
            segs.append({"start": start, "duration": dur, "text": text})
    return segs

def try_ytdlp_captions(url: str, preferred_langs: List[str]) -> Optional[List[dict]]:
    """
    Pull caption file URLs via yt-dlp (no media download), fetch the text, parse to segments.
    Returns segments or None.
    """
    opts = _yt_base_opts(skip_download=True)
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)

    track = _choose_caption_track(info.get("subtitles") or {}, preferred_langs)
    if not track:
        track = _choose_caption_track(info.get("automatic_captions") or {}, preferred_langs)
    if not track:
        return None

    try:
        resp = requests.get(track["url"], timeout=15)
        resp.raise_for_status()
    except Exception:
        return None

    ext = (track.get("ext") or "").lower()
    text = resp.text

    if ext in ("json3", "srv3"):
        segs = _parse_json3(text)
        return segs
    elif ext in ("vtt", "srt", "ttml", "sbv"):
        if "WEBVTT" in text[:16]:
            return parse_vtt(text)
        return None
    else:
        return None

# ---------- helpers for chapters from description ----------

def _clean_chapter_title(title: str) -> str:
    title = re.sub(r'https?://\S+', '', title)
    title = re.sub(r'\s{2,}', ' ', title).strip()
    return title.strip(" -–—|:·•\t")

def parse_description_timestamps(description: str) -> List[dict]:
    """
    Parse a YouTube description to extract chapter timestamps and titles.
    Returns: [{'start': seconds, 'title': 'Intro'}, ...], sorted by start.
    Handles:
      '0:00 - Intro'
      '0:31 Package Contents'
      'Intro - 0:00'
    """
    if not description:
        return []
    chapters: List[dict] = []
    for raw in description.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.search(r'(?<!\d)(\d{1,2}:)?\d{1,2}:\d{2}(?:\.\d{1,3})?', line)
        if not m:
            continue
        ts = m.group(0)
        after = line[m.end():].strip()
        after = re.sub(r'^(?:[-–—:|]\s*)+', '', after)
        title = _clean_chapter_title(after)
        if not title:
            before = line[:m.start()].strip()
            before = re.sub(r'^(?:[-–—:|]\s*)+|(?:\s*[-–—:|])+$', '', before).strip()
            title = _clean_chapter_title(before)
        try:
            start_s = parse_ts(ts)
        except Exception:
            continue
        if title:
            chapters.append({"start": float(start_s), "title": title})

    chapters.sort(key=lambda c: c["start"])
    uniq: List[dict] = []
    seen = set()
    for ch in chapters:
        key = int(ch["start"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(ch)
    return uniq

def group_segments_by_chapters(segments: List[dict], chapters: List[dict]) -> List[tuple]:
    if not chapters:
        return []
    groups: List[tuple] = []
    for i, ch in enumerate(chapters):
        start = ch["start"]
        end = chapters[i + 1]["start"] if i + 1 < len(chapters) else float("inf")
        bucket = [seg for seg in segments if start <= float(seg["start"]) < end]
        groups.append((ch, bucket))
    return groups

def render_transcript_with_chapters(title: str, source: str, groups: List[tuple], with_timestamps: bool = True) -> str:
    out_lines: List[str] = []
    for ch, segs in groups:
        out_lines.append(f"## {ch['title']} [{hhmmss(ch['start'])}]")
        if segs:
            for seg in segs:
                text = seg["text"].replace("\n", " ").strip()
                if not text:
                    continue
                if with_timestamps:
                    out_lines.append(f"[{hhmmss(seg['start'])}] {text}")
                else:
                    out_lines.append(text)
        else:
            out_lines.append("_No speech in this section._")
        out_lines.append("")
    return "\n".join(out_lines)

def segments_to_paragraphs(segs: List[dict], target_len: int = 800) -> List[str]:
    """
    Join segment texts and split into readable paragraphs (approx target_len chars).
    Removes any timestamps; expects raw segment texts.
    """
    texts = [s["text"].strip() for s in segs if s.get("text") and s["text"].strip()]
    if not texts:
        return []
    blob = " ".join(texts)
    blob = re.sub(r"\s+", " ", blob).strip()

    sentences = re.split(r"(?<=[\.!?])\s+", blob)
    paras: List[str] = []
    cur = ""
    for s in sentences:
        if not s:
            continue
        if not cur:
            cur = s
        elif len(cur) + 1 + len(s) <= target_len:
            cur += " " + s
        else:
            paras.append(cur)
            cur = s
    if cur:
        paras.append(cur)
    return paras

# ---------- markdown -> html + wordpress publishing ----------

def md_to_html(md_text: str) -> str:
    """Convert Markdown to HTML. Uses 'markdown' if available, else a simple fallback."""
    try:
        import markdown as mdlib
        return mdlib.markdown(md_text, extensions=["extra", "sane_lists"])
    except Exception:
        html_lines = []
        for line in md_text.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("<"):
       
                html_lines.append(line)
                continue
            if line.startswith("## "):
                html_lines.append(f"<h2>{line[3:].strip()}</h2>")
            elif line.startswith("# "):
                html_lines.append(f"<h1>{line[2:].strip()}</h1>")
            else:
                if line.strip():
                    html_lines.append(f"<p>{line.strip()}</p>")
                else:
                    html_lines.append("")
        return "\n".join(html_lines)

def simple_slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"['’]", "", text)
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "post"

def make_excerpt_from_md(md_text: str, max_len: int = 220) -> str:
    t = re.sub(r"^#+\s*", "", md_text, flags=re.MULTILINE)
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)  # links -> text
    t = re.sub(r"[`*_>#]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return (t[:max_len]).strip()

def publish_to_wordpress(
    wp_url: str,
    wp_user: str,
    wp_pass: str,
    title: str,
    html_content: str,
    status: str = "draft",
    slug: Optional[str] = None,
    excerpt_text: Optional[str] = None,
    featured_media: Optional[int] = None,
) -> str:
    """
    Publish a post to WordPress via REST API using Application Passwords.
    Returns the public link on success; raises on failure.
    """
    api = wp_url.rstrip("/") + "/wp-json/wp/v2/posts"
    payload = {"title": title, "content": html_content, "status": status}
    if slug:
        payload["slug"] = slug
    if excerpt_text:
        payload["excerpt"] = excerpt_text
    if featured_media is not None:
        payload["featured_media"] = int(featured_media)

    try:
        resp = requests.post(api, json=payload, auth=(wp_user, wp_pass), timeout=30)
    except Exception as e:
        raise RuntimeError(f"Failed to contact WordPress: {e}")

    if resp.status_code not in (200, 201):
        raise RuntimeError(f"WordPress API error {resp.status_code}: {resp.text}")

    data = resp.json()
    return data.get("link") or (wp_url.rstrip("/") + f"/?p={data.get('id')}")
# ---------- video download + frame extraction + media upload ----------

def seconds_to_ffmpeg_ts(seconds: float) -> str:
    ms = int(round((seconds - int(seconds)) * 1000.0))
    secs = int(seconds)
    h = secs // 3600
    m = (secs % 3600) // 60
    s = secs % 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def build_yt_timestamp_url(video_url: str, seconds: float) -> str:
    sec_i = int(round(seconds))
    joiner = "&" if "?" in video_url else "?"
    return f"{video_url}{joiner}t={sec_i}s"

def ensure_ffmpeg_available() -> None:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception as e:
        raise RuntimeError("ffmpeg is required for --screenshots but was not found on PATH") from e

def download_video_mp4_720(url: str, outdir: str) -> str:
    """
    Download a ~720p MP4 when possible. If YouTube blocks direct download (SABR/PO token),
    fall back to an HLS (m3u8) URL that ffmpeg can read frames from remotely.
    Returns a local filepath OR a remote URL suitable for ffmpeg -i.
    """
    outtmpl = os.path.join(outdir, "%(id)s.%(ext)s")

    opts = {
        "quiet": True,
        "noprogress": True,
        "format": (
            "bv*[height<=720][vcodec~='^(avc1|h264)']+ba[acodec~='^(mp4a|m4a|aac)']/"
            "b[height<=720]"
        ),
        "merge_output_format": "mp4",
        "outtmpl": outtmpl,
        "extractor_args": {"youtube": {"player_client": ["android", "web_safari", "web_embedded", "default"]}},
        "retries": 10,
        "fragment_retries": 10,
        "concurrent_fragment_downloads": 1,
    }

    # Cookies (file in CI, or browser locally)
    cookies_file = os.getenv("YT_COOKIES_FILE")
    if cookies_file and os.path.exists(cookies_file):
        opts["cookiefile"] = cookies_file
    else:
        browser = os.getenv("YT_COOKIES_BROWSER")
        if browser in ("safari", "chrome", "firefox", "edge"):
            opts["cookiesfrombrowser"] = (browser, None, None, None)

    # Try to download a local MP4 first
    try:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filepath = ydl.prepare_filename(info)
        base, ext = os.path.splitext(filepath)
        if ext.lower() != ".mp4":
            mp4_path = base + ".mp4"
            return mp4_path if os.path.exists(mp4_path) else filepath
        return filepath
    except Exception:
        # Fallback: fetch an HLS stream URL (<=720p) and let ffmpeg grab frames remotely
        try:
            opts2 = dict(opts)
            opts2["skip_download"] = True
            with YoutubeDL(opts2) as ydl:
                info = ydl.extract_info(url, download=False)
            fmts = info.get("formats") or []
            hls = [
                f for f in fmts
                if (f.get("protocol") or "").startswith("m3u8") and (f.get("height") or 0) <= 720
            ]
            # Prefer AVC streams if present
            hls.sort(key=lambda f: (-(f.get("height") or 0), 'avc1' not in (f.get("vcodec") or "")))
            if hls and hls[0].get("url"):
                return hls[0]["url"]
        except Exception:
            pass
        # Bubble up to caller so screenshots can be skipped gracefully
        raise
        
def extract_frame_to_jpg(video_path: str, ts_seconds: float, out_path: str, width: int = 1280, qscale: int = 3) -> bool:
    """
    Use ffmpeg to extract a single frame at ts_seconds to out_path (JPEG).
    Returns True on success, False otherwise.
    """
    ts = seconds_to_ffmpeg_ts(ts_seconds)
    vf = f"scale={int(width)}:-1"
    cmd = [
        "ffmpeg",
        "-ss", ts,
        "-i", video_path,
        "-frames:v", "1",
        "-vf", vf,
        "-q:v", str(int(qscale)),
        "-y",
        out_path,
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return os.path.exists(out_path)
    except Exception:
        return False

def upload_media(wp_url: str, wp_user: str, wp_pass: str, file_path: str, file_name: Optional[str] = None, mime_type: str = "image/jpeg") -> Tuple[int, str]:
    """
    Upload a file to WP Media Library. Returns (attachment_id, source_url).
    """
    media_api = wp_url.rstrip("/") + "/wp-json/wp/v2/media"
    name = file_name or os.path.basename(file_path)
    headers = {
        "Content-Disposition": f'attachment; filename="{name}"',
        "Content-Type": mime_type,
    }
    with open(file_path, "rb") as f:
        resp = requests.post(media_api, headers=headers, data=f, auth=(wp_user, wp_pass), timeout=60)
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Media upload failed {resp.status_code}: {resp.text}")
    data = resp.json()
    return int(data["id"]), data.get("source_url")

def update_media_metadata(wp_url: str, wp_user: str, wp_pass: str, media_id: int, title: Optional[str] = None, alt_text: Optional[str] = None, caption_html: Optional[str] = None) -> None:
    """
    Update title/alt/caption for an existing media item.
    """
    api = wp_url.rstrip("/") + f"/wp-json/wp/v2/media/{int(media_id)}"
    payload = {}
    if title:
        payload["title"] = title
    if alt_text:
        payload["alt_text"] = alt_text
    if caption_html:
        payload["caption"] = caption_html
    if not payload:
        return
    resp = requests.post(api, json=payload, auth=(wp_user, wp_pass), timeout=30)
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Update media failed {resp.status_code}: {resp.text}")

def build_chapter_image_filename(video_slug: str, start_seconds: float, chapter_title: str) -> str:
    ts_label = hhmmss(start_seconds).replace(":", "-")
    ch_slug = simple_slugify(chapter_title)
    return f"{video_slug}_{ts_label}_{ch_slug}.jpg"

def build_figure_html(image_url: str, chapter_title: str, ts_seconds: float, video_url: str, show_ts: bool = False) -> str:
    """
    Build a <figure> block. If show_ts is False, do not display timestamps in caption/alt,
    but still link to the timestamped YouTube URL.
    """
    ts_label = hhmmss(ts_seconds)
    ts_url = build_yt_timestamp_url(video_url, ts_seconds)
    if show_ts:
        alt = f"Screenshot at {ts_label} – {chapter_title}"
        caption = f'{chapter_title} [{ts_label}] — <a href="{ts_url}">Watch at {ts_label}</a>'
    else:
        alt = f"Screenshot — {chapter_title}"
        caption = f'{chapter_title} — <a href="{ts_url}">Watch this section</a>'
    return (
        f'<figure class="yt-chapter">'
        f'<img src="{image_url}" alt="{alt}" />'
        f'<figcaption>{caption}</figcaption>'
        f"</figure>"
    )


# Responsive YouTube embed block (raw HTML for Gutenberg)

def build_youtube_embed_html(video_id: str, title: str, video_url: str) -> str:
    """
    Build a responsive 16:9 <iframe> embed for the YouTube video and a fallback link.
    """
    embed_url = f"https://www.youtube.com/embed/{video_id}?rel=0"
    safe_title = re.sub(r'["<>]', '', title) or "YouTube video"
    return (
        '<div class="video-embed" style="position:relative;padding-bottom:56.25%;height:0;overflow:hidden;max-width:100%;">'
        f'<iframe src="{embed_url}" title="{safe_title}" loading="lazy" '
        'allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" '
        'allowfullscreen style="position:absolute;top:0;left:0;width:100%;height:100%;border:0;"></iframe>'
        '</div>\n'
        f'<p><a href="{video_url}">Watch on YouTube</a></p>'
    )

# --- THUMBNAIL HELPERS ---
def _url_ok(url: str) -> bool:
    try:
        r = requests.head(url, allow_redirects=True, timeout=10)
        if 200 <= r.status_code < 400:
            return True
        # Some CDNs dislike HEAD; try a lightweight GET
        r = requests.get(url, stream=True, timeout=10)
        return 200 <= r.status_code < 400
    except Exception:
        return False

def fetch_best_thumbnail_url(video_page_url: str, video_id: str) -> Optional[str]:
    """
    Use yt-dlp metadata to choose the highest-quality thumbnail URL.
    Fallback to i.ytimg.com maxres/hqdefault heuristics if needed.
    """
    try:
        with YoutubeDL({"quiet": True, "skip_download": True, "noprogress": True}) as ydl:
            info = ydl.extract_info(video_page_url, download=False)
    except Exception:
        info = {}

    # Prefer the explicit 'thumbnails' list with dimensions/preference
    thumbs = info.get("thumbnails") or []
    best_url = None
    if thumbs:
        def thumb_key(t):
            w = t.get("width") or 0
            h = t.get("height") or 0
            pref = t.get("preference") or 0
            return (pref, w * h)
        thumbs_sorted = sorted(thumbs, key=thumb_key, reverse=True)
        for t in thumbs_sorted:
            url = t.get("url")
            if url:
                best_url = url
                break

    # Fallback to simple 'thumbnail' field
    if not best_url:
        best_url = info.get("thumbnail")

    # Heuristic fallback: i.ytimg.com patterns
    if not best_url:
        candidates = [
            f"https://i.ytimg.com/vi/{video_id}/maxresdefault.jpg",
            f"https://i.ytimg.com/vi/{video_id}/sddefault.jpg",
            f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
            f"https://i.ytimg.com/vi/{video_id}/mqdefault.jpg",
        ]
        for cu in candidates:
            if _url_ok(cu):
                best_url = cu
                break

    return best_url

def upload_featured_thumb_from_url(wp_url: str, wp_user: str, wp_pass: str, thumb_url: str, title: str) -> Optional[Tuple[int, str]]:
    """
    Download the remote thumbnail to a temp file, upload to WP media, set sensible title/alt,
    and return the attachment ID.
    """
    if not thumb_url:
        return None
    ext = ".jpg"
    lower = thumb_url.lower()
    if lower.endswith(".webp"):
        ext = ".webp"
        mime = "image/webp"
    elif lower.endswith(".png"):
        ext = ".png"
        mime = "image/png"
    else:
        mime = "image/jpeg"

    tmpdir = tempfile.TemporaryDirectory()
    try:
        fname = f"{simple_slugify(title)}-thumb{ext}"
        fpath = os.path.join(tmpdir.name, fname)
        try:
            with requests.get(thumb_url, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(fpath, "wb") as out:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            out.write(chunk)
        except Exception as e:
            # Could not download thumbnail
            return None

        attach_id, src_url = upload_media(wp_url, wp_user, wp_pass, fpath, file_name=fname, mime_type=mime)
        # Set media title/alt
        try:
            update_media_metadata(
                wp_url, wp_user, wp_pass, attach_id,
                title=f"{title} — Video thumbnail",
                alt_text=f"Video thumbnail — {title}",
                caption_html=None
            )
        except Exception:
            pass
        return int(attach_id), src_url
    finally:
        try:
            tmpdir.cleanup()
        except Exception:
            pass


AFFILIATE_NOTICE = ("Please note that some of the links in my video descriptions are affiliate links where I earn from qualifying purchases. As an Amazon Associate I earn from qualifying purchases.")

# Exclude donation/referral links helper
def _is_excluded_referral(line: str, url: str) -> bool:
    """
    Return True if the candidate referral should be excluded from CTA buttons.
    Currently excludes BuyMeACoffee/donation links.
    """
    l = (line or "").lower()
    u = (url or "").lower()
    if "buymeacoffee.com" in u:
        return True
    if "buymeacoffee" in l or "buy me a coffee" in l:
        return True
    return False

_URL_RE = re.compile(r'(https?://\S+|(?:www\.)?[\w.-]+\.[A-Za-z]{2,}\S*)', re.IGNORECASE)

def _normalize_url(u: str) -> str:
    u = u.strip().rstrip(').,;\'"')
    if u.startswith("http://") or u.startswith("https://"):
        return u
    # Add https scheme for bare domains/paths
    return "https://" + u.lstrip("/")

def extract_buy_referrals(description: str) -> List[Tuple[str, str]]:
    """
    Find all lines before 'TIMESTAMPS' that include the word BUY and a URL (or bare domain),
    and return a list of (label, url) pairs. Label is derived from the text between 'BUY' and the URL.
    """
    results: List[Tuple[str, str]] = []
    if not description:
        return results
    for raw in description.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.upper().startswith("TIMESTAMPS"):
            break
        if "BUY" not in line.upper():
            continue
        m = _URL_RE.search(line)
        if not m:
            continue
        url = _normalize_url(m.group(0))
        # Build label from the text after BUY and before URL
        upper = line.upper()
        idx_buy = upper.find("BUY")
        label = line[idx_buy + 3:m.start()].strip(" :-–—|·•\t")
        # Cleanup emojis and repeated spaces
        for sym in ("👉", "➡️", "✅", "☑️", "•"):
            label = label.replace(sym, "")
        label = re.sub(r'\s{2,}', ' ', label).strip()
        # Exclude donation links (e.g., BuyMeACoffee)
        if _is_excluded_referral(line, url):
            continue
        # Normalize to "Buy ..." phrasing
        if not label:
            # If nothing usable, fall back to domain
            try:
                host = re.sub(r'^https?://', '', url).split("/")[0]
            except Exception:
                host = url
            label = f"Buy on {host}"
        else:
            # Ensure it starts with 'Buy '
            if not label.lower().startswith("buy "):
                label = "Buy " + label
        results.append((label, url))
    return results

def extract_top_referral(description: str) -> Optional[Tuple[str, str]]:
    """
    Return (button_label, url) for the first referral line found near the top of the description (before TIMESTAMPS).
    The label is taken from the text adjacent to the URL on that line (e.g., 'Buy SoundPeats Air5 Pro on Amazon').
    Emojis and trailing separators/colons are stripped.
    """
    if not description:
        return None
    for raw in description.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.upper().startswith("TIMESTAMPS"):
            break
        m = re.search(r'https?://\S+', line)
        if not m:
            continue
        url = line[m.start():m.end()].rstrip(').,;\'"')
        # Prefer text immediately BEFORE the URL (common: "Buy XYZ: https://...")
        label = line[:m.start()].strip()
        # If empty, try text immediately AFTER
        if not label:
            label = line[m.end():].strip()
        # Remove common emojis and leading symbols
        for sym in ("👉", "➡️", "✅"):
            label = label.replace(sym, "")
        label = re.sub(r'\s{2,}', ' ', label)
        # Remove common leading/trailing separators and any trailing colon
        label = label.strip(" -–—|·•\t")
        label = label.rstrip(":").strip()
        # Exclude donation links (e.g., BuyMeACoffee)
        if _is_excluded_referral(line, url):
            continue
        # If label still empty, fall back to the URL itself
        if not label:
            label = url
        return (label, url)
    return None

def build_affiliate_block(description: str) -> str:
    """
    HTML block under the video: a vertical stack of CTA buttons for every 'BUY ... <url>' line
    found before the TIMESTAMPS section, followed by a single small affiliate notice.
    If no BUY lines are found, fall back to a single top referral (first URL near the top).
    """
    buttons_html: List[str] = []
    buy_refs = extract_buy_referrals(description)

    def _button(label: str, url: str) -> str:
        return (
            f'<p style="margin:12px 0 0;">'
            f'<a href="{url}" target="_blank" rel="nofollow sponsored noopener" '
            f'style="display:inline-block;background:#2b49f0;color:#fff;text-decoration:none;'
            f'padding:12px 16px;border-radius:3px;font-weight:700;line-height:1.2;">'
            f'{label}'
            f'</a>'
            f'</p>'
        )

    if buy_refs:
        for label, url in buy_refs:
            buttons_html.append(_button(label, url))
    else:
        # Fallback to the first URL above TIMESTAMPS
        # Reuse previous single-referral parsing for label + url
        fallback = extract_top_referral(description)
        if fallback:
            label, url = fallback
            buttons_html.append(_button(label, url))

    parts = []
    if buttons_html:
        parts.append("\n".join(buttons_html))
    # Disclaimer appears ONCE per stack
    parts.append(f'<p style="margin:8px 0 16px;"><em>{AFFILIATE_NOTICE}</em></p>')
    return "\n".join(parts)

def assemble_markdown_with_figures(title: str, source: str, groups: List[tuple], image_map: dict, video_url: str, with_timestamps: bool = True, paragraphs: bool = False) -> str:
    """
    image_map: { int(start_seconds): {"id": Optional[int], "url": str} }
    If paragraphs=True, render each chapter as H2 (no timestamps) + figure + paragraph(s) with no timestamps.
    """
    out_lines: List[str] = []
    for ch, segs in groups:
        # H2 without timestamp if paragraph mode, else include label
        if paragraphs:
            out_lines.append(f"## {ch['title']}")
        else:
            label = hhmmss(ch["start"])
            out_lines.append(f"## {ch['title']} [{label}]")
        key = int(ch["start"])
        # Figure: never show timestamps in caption if paragraph mode
        if key in image_map and image_map[key].get("url"):
            out_lines.append(build_figure_html(image_map[key]["url"], ch["title"], ch["start"], video_url, show_ts=not paragraphs))
        # Body
        if segs:
            if paragraphs:
                paras = segments_to_paragraphs(segs)
                for p in paras:
                    out_lines.append(p)
                    out_lines.append("")  # blank line between paragraphs
            else:
                for seg in segs:
                    text = seg["text"].replace("\n", " ").strip()
                    if not text:
                        continue
                    if with_timestamps:
                        out_lines.append(f"[{hhmmss(seg['start'])}] {text}")
                    else:
                        out_lines.append(text)
        else:
            out_lines.append("_No speech in this section._")
        out_lines.append("")  # blank line between chapters
    return "\n".join(out_lines)

# ---------- local transcription (fallback) ----------

def download_audio(url: str, outdir: str) -> str:
    outtmpl = os.path.join(outdir, "%(id)s.%(ext)s")
    opts = {
        "quiet": True,
        "noprogress": True,
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "m4a", "preferredquality": "192"}
        ],
    }
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        audio_path = ydl.prepare_filename(info)
        base = os.path.splitext(audio_path)[0]
        m4a_path = base + ".m4a"
        return m4a_path if os.path.exists(m4a_path) else audio_path

def transcribe_locally(audio_path: str, model_size: str = "tiny") -> List[dict]:
    from faster_whisper import WhisperModel
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments_iter, _info = model.transcribe(
        audio_path, beam_size=1, vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500)
    )
    segs = []
    for seg in segments_iter:
        start = float(seg.start); end = float(seg.end)
        text = seg.text.strip()
        if text:
            segs.append({"start": start, "duration": end - start, "text": text})
    return segs

# ---------- rendering (plain) ----------

def render_segments_text(segments: List[dict], with_timestamps: bool = True) -> str:
    lines = []
    for seg in segments:
        text = seg["text"].replace("\n", " ").strip()
        if not text:
            continue
        if with_timestamps:
            lines.append(f"[{hhmmss(seg['start'])}] {text}")
        else:
            lines.append(text)
    return "\n".join(lines)

# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(
        description="Get a transcript for a YouTube video.\n"
                    "Order: official captions -> yt-dlp captions -> local Whisper (unless --no-local)."
    )
    parser.add_argument("url", help="YouTube URL or ID")
    parser.add_argument("--lang", default="en", help="Preferred language code (default: en)")
    parser.add_argument("--no-local", action="store_true", help="Skip local Whisper fallback")
    parser.add_argument("--timestamps", action="store_true", help="Prepend timestamps to each line")
    parser.add_argument("--model", default="tiny", help="Whisper model: tiny/base/small/medium/large-v3")
    # WordPress options
    parser.add_argument("--post", action="store_true", help="Publish the result to WordPress using credentials")
    parser.add_argument("--wp-url", default=None, help="WordPress base URL (e.g., https://example.com). Falls back to env WP_URL")
    parser.add_argument("--wp-user", default=None, help="WordPress username (Application Password owner). Falls back to env WP_USER")
    parser.add_argument("--wp-pass", default=None, help="WordPress Application Password (or WP_APP_PASS env). Falls back to env WP_APP_PASS/WP_PASS")
    parser.add_argument("--wp-status", default="draft", choices=["draft","publish","private","pending"], help="Post status (default: draft)")
    parser.add_argument("--screenshots", action="store_true", help="Extract and insert one screenshot per chapter under each H2")
    parser.add_argument("--screenshot-width", type=int, default=1280, help="Width of screenshots (px). Height scales to maintain aspect ratio (default: 1280)")
    parser.add_argument("--screenshot-offset", type=float, default=0.5, help="Seconds to add to each chapter timestamp for frame grabbing (default: 0.5)")
    parser.add_argument("--image-quality", type=int, default=3, help="JPEG qscale for ffmpeg (lower is better, typical 2-5; default: 3)")
    parser.add_argument("--featured-image", action="store_true", help="Use the first chapter screenshot as the post featured image")
    # Paragraph mode defaults to ON; use --no-paragraphs to disable
    parser.add_argument("--paragraphs", dest="paragraphs", action="store_true", help="Render transcript as paragraphs with no timestamps (H2s and body)")
    parser.add_argument("--no-paragraphs", dest="paragraphs", action="store_false", help="Disable paragraph mode (use raw timestamped lines if --timestamps is set)")
    # Video embed defaults to ON; use --no-embed to disable
    parser.add_argument("--embed", dest="embed", action="store_true", help="Include a YouTube video embed at the very top of the post")
    parser.add_argument("--no-embed", dest="embed", action="store_false", help="Do not include the video embed")
    parser.set_defaults(paragraphs=True, embed=True)

    args = parser.parse_args()

    try:
        video_id, title, description = fetch_video_id_and_title(args.url)
    except Exception as e:
        print(f"Error: could not resolve video info - {e}", file=sys.stderr)
        sys.exit(1)

    featured_media_id: Optional[int] = None
    uploaded_thumb: Optional[Tuple[int, str]] = None
    preferred = [args.lang]
    source = None
    segments: Optional[List[dict]] = None

    # 1) youtube-transcript-api
    segments = try_official_transcript(video_id, preferred)
    if segments:
        source = "official"

    # 2) yt-dlp captions (no media download)
    if segments is None:
        segments = try_ytdlp_captions(args.url, preferred)
        if segments:
            source = "youtube"

    # 3) local transcription
    if segments is None and not args.no_local:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                audio_path = download_audio(args.url, tmp)
            except Exception as e:
                print(f"Error: could not download audio for local transcription - {e}", file=sys.stderr)
                sys.exit(2)
            try:
                segments = transcribe_locally(audio_path, model_size=args.model)
                source = "local"
            except Exception as e:
                print(f"Error: local transcription failed - {e}", file=sys.stderr)
                sys.exit(3)

    if not segments:
        print("No captions/transcript available.", file=sys.stderr)
        sys.exit(4)

    # Build Markdown output (chapters -> H2)
    chapters = parse_description_timestamps(description)
    if chapters:
        groups = group_segments_by_chapters(segments, chapters)

        image_map = {}
        featured_media_id = None

        if args.screenshots:
            ensure_ffmpeg_available()
            video_slug = simple_slugify(title)
            temp_video_dir = tempfile.TemporaryDirectory()
            try:
                video_path = download_video_mp4_720(args.url, temp_video_dir.name)
                # Optionally pre-upload the YouTube thumbnail to use as the Intro image and/or featured image
                if args.post:
                    wp_url = (args.wp_url or os.getenv("WP_URL") or "").strip()
                    wp_user = (args.wp_user or os.getenv("WP_USER") or "").strip()
                    wp_pass = (args.wp_pass or os.getenv("WP_APP_PASS") or os.getenv("WP_PASS") or "").strip()
                    if wp_url and wp_user and wp_pass:
                        try:
                            yt_thumb_url = fetch_best_thumbnail_url(args.url, video_id)
                            if yt_thumb_url:
                                uploaded_thumb = upload_featured_thumb_from_url(wp_url, wp_user, wp_pass, yt_thumb_url, title)
                        except Exception:
                            uploaded_thumb = None
                # Extract and (optionally) upload one image per chapter
                uploaded_ids = []
                for idx, ch in enumerate(chapters):
                    grab_ts = max(0.0, float(ch["start"]) + float(args.screenshot_offset))
                    fname = build_chapter_image_filename(video_slug, ch["start"], ch["title"])
                    # For the first chapter, prefer the uploaded YouTube thumbnail as the chapter image
                    if idx == 0 and uploaded_thumb:
                        thumb_id, thumb_url = uploaded_thumb
                        if args.post:
                            image_map[int(ch["start"])] = {"id": int(thumb_id), "url": thumb_url}
                        else:
                            image_map[int(ch["start"])] = {"id": None, "url": thumb_url}
                        # Skip extracting a frame for the first chapter since we used the thumbnail
                        continue
                    local_path = os.path.join(temp_video_dir.name, fname)
                    ok = extract_frame_to_jpg(
                        video_path,
                        grab_ts,
                        local_path,
                        width=args.screenshot_width,
                        qscale=args.image_quality,
                    )
                    if not ok:
                        continue

                    if args.post:
                        # Ensure WP creds exist before upload
                        wp_url = (args.wp_url or os.getenv("WP_URL") or "").strip()
                        wp_user = (args.wp_user or os.getenv("WP_USER") or "").strip()
                        wp_pass = (args.wp_pass or os.getenv("WP_APP_PASS") or os.getenv("WP_PASS") or "").strip()
                        if not (wp_url and wp_user and wp_pass):
                            print("Screenshots requested but WP credentials are missing; cannot upload images.", file=sys.stderr)
                            img_url = f"file://{local_path}"
                            image_map[int(ch['start'])] = {"id": None, "url": img_url}
                        else:
                            try:
                                attach_id, src_url = upload_media(wp_url, wp_user, wp_pass, local_path, file_name=fname, mime_type="image/jpeg")
                                if args.paragraphs:
                                    caption_html = f'<a href="{build_yt_timestamp_url(args.url, ch["start"])}">Watch this section</a>'
                                    media_title = f"{ch['title']}"
                                    alt_text = f"Screenshot — {ch['title']}"
                                else:
                                    caption_html = f'<a href="{build_yt_timestamp_url(args.url, ch["start"])}">Back to {hhmmss(ch["start"])}</a>'
                                    media_title = f"{ch['title']} ({hhmmss(ch['start'])})"
                                    alt_text = f"Screenshot at {hhmmss(ch['start'])} – {ch['title']}"
                                update_media_metadata(
                                    wp_url, wp_user, wp_pass, attach_id,
                                    title=media_title,
                                    alt_text=alt_text,
                                    caption_html=caption_html
                                )
                                image_map[int(ch["start"])] = {"id": attach_id, "url": src_url}
                                uploaded_ids.append(attach_id)
                            except Exception as e:
                                print(f"Warning: upload failed for {fname}: {e}", file=sys.stderr)
                                if args.post:
                                    continue
                                else:
                                    img_url = f"file://{local_path}"
                                    image_map[int(ch['start'])] = {"id": None, "url": img_url}
                    else:
                        img_url = f"file://{local_path}"
                        image_map[int(ch["start"])] = {"id": None, "url": img_url}

                if args.featured_image and args.post and image_map:
                    first_key = int(chapters[0]["start"])
                    first = image_map.get(first_key)
                    if first and first.get("id"):
                        featured_media_id = int(first["id"])
                else:
                    featured_media_id = None

            finally:
                # Cleanup temp video + images; uploads are now in WP
                try:
                    temp_video_dir.cleanup()
                except Exception:
                    pass
        else:
            image_map = {}
            featured_media_id = None

        # Build Markdown including figures (uses raw HTML for figure/img)
        output_md = assemble_markdown_with_figures(title, source, groups, image_map, args.url, with_timestamps=args.timestamps, paragraphs=args.paragraphs)
    else:
        if args.paragraphs:
            paras = segments_to_paragraphs(segments)
            body = "\n\n".join(paras)
        else:
            body = render_segments_text(segments, with_timestamps=args.timestamps)
        output_md = body

    try:
        affiliate_html = build_affiliate_block(description)
    except Exception:
        affiliate_html = ""

    if args.embed:
        try:
            embed_html = build_youtube_embed_html(video_id, title, args.url)
            preface = embed_html
        except Exception:
            preface = ""
    else:
        preface = ""

    pre_section = "\n\n".join([s for s in [preface, affiliate_html] if s])
    if pre_section:
        output_md = pre_section + "\n\n" + output_md

    # Append the same affiliate CTA + notice at the very bottom of the post
    if affiliate_html:
        output_md = output_md + "\n\n" + affiliate_html

    # Always print to stdout
    print(output_md)

    # Prefer the YouTube thumbnail as the featured image when requested
    if args.post and args.featured_image:
        wp_url = (args.wp_url or os.getenv("WP_URL") or "").strip()
        wp_user = (args.wp_user or os.getenv("WP_USER") or "").strip()
        wp_pass = (args.wp_pass or os.getenv("WP_APP_PASS") or os.getenv("WP_PASS") or "").strip()
        if wp_url and wp_user and wp_pass:
            try:
                if uploaded_thumb:
                    featured_media_id = int(uploaded_thumb[0])
                else:
                    thumb_url = fetch_best_thumbnail_url(args.url, video_id)
                    thumb_pair = upload_featured_thumb_from_url(wp_url, wp_user, wp_pass, thumb_url, title)
                    if thumb_pair:
                        featured_media_id = int(thumb_pair[0])
            except Exception:
                # Non-fatal: keep any previous featured_media_id (e.g., first screenshot) if thumbnail upload fails
                pass

    # Optionally publish to WordPress
    if args.post:
        wp_url = (args.wp_url or os.getenv("WP_URL") or "").strip()
        wp_user = (args.wp_user or os.getenv("WP_USER") or "").strip()
        wp_pass = (args.wp_pass or os.getenv("WP_APP_PASS") or os.getenv("WP_PASS") or "").strip()
        if not (wp_url and wp_user and wp_pass):
            print("WordPress posting requested but WP_URL, WP_USER, and WP_APP_PASS/WP_PASS are not fully provided.", file=sys.stderr)
            sys.exit(6)
        html_content = md_to_html(output_md)
        slug = simple_slugify(title)
        excerpt = make_excerpt_from_md(output_md, 220)
        # If screenshots ran, featured_media_id may have been set above; otherwise None.
        featured_media_id_local = locals().get("featured_media_id", None)
        try:
            link = publish_to_wordpress(
                wp_url, wp_user, wp_pass, title, html_content,
                status=args.wp_status, slug=slug, excerpt_text=excerpt,
                featured_media=featured_media_id_local
            )
            print(f"\nPosted to WordPress: {link}")
        except Exception as e:
            print(f"Error publishing to WordPress: {e}", file=sys.stderr)
            sys.exit(7)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import sys
import json
import tempfile
import subprocess
from typing import List, Tuple, Optional, Dict

import requests
from yt_dlp import YoutubeDL
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    CouldNotRetrieveTranscript,
)

# -------------------------------
# yt-dlp options (with cookies)
# -------------------------------

# add this helper (right after imports)
def yt_opts(**extra):
    """
    Base YoutubeDL options with optional cookies file and sane defaults.
    Pass extra opts as kwargs to override/extend.
    """
    o = {"quiet": True, "noprogress": True}
    cookies = os.getenv("YT_COOKIES_FILE")
    if cookies and os.path.exists(cookies):
        o["cookiefile"] = cookies
    # Prefer clients that currently play well with screenshots; yt-dlp will still pick best
    # (Keep your extractor_args if you already had them)
    if "extractor_args" not in extra:
        extra["extractor_args"] = {"youtube": {"player_client": ["web_embedded,web_safari,default"]}}
    o.update(extra)
    return o

def _yt_base_opts(skip_download=True):
    opts = {
        "quiet": True,
        "noprogress": True,
        "skip_download": skip_download,
        "retries": 10,
    }
    # Prefer stable clients but pass them correctly as a list

    cookies_file = os.getenv("YT_COOKIES_FILE")
    if cookies_file and os.path.exists(cookies_file):
        opts["cookiefile"] = cookies_file
    else:
        browser = os.getenv("YT_COOKIES_BROWSER")
        if browser in ("safari", "chrome", "firefox", "edge"):
            opts["cookiesfrombrowser"] = (browser, None, None, None)
    return opts

# -------------------------------
# Small helpers
# -------------------------------

def hhmmss(seconds: float) -> str:
    s = int(round(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"

def parse_ts(ts: str) -> float:
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
    lines = vtt_text.splitlines()
    i, n = 0, len(lines)
    segs = []
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
            segs.append({"start": start_s, "duration": end_s - start_s, "text": text})

    return segs

# -------------------------------
# YouTube info + captions
# -------------------------------

# fetch_video_id_and_title
def fetch_video_id_and_title(url: str) -> Tuple[str, str, str]:
    with YoutubeDL(yt_opts(skip_download=True)) as ydl:
        info = ydl.extract_info(url, download=False)
    return info["id"], info.get("title", ""), info.get("description", "") or ""

def try_official_transcript(video_id: str, preferred_langs: List[str]) -> Optional[List[dict]]:
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
                if not getattr(t, "is_generated", False):
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
    if not tracks:
        return None
    ext_order = ["json3", "srv3", "vtt", "srt", "ttml", "sbv"]

    def pick_for_lang(lang: str) -> Optional[dict]:
        items = tracks.get(lang) or []
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
        return _parse_json3(text)
    if ext in ("vtt", "srt", "ttml", "sbv"):
        if "WEBVTT" in text[:16]:
            return parse_vtt(text)
        return None
    return None

# -------------------------------
# Chapters from description
# -------------------------------

def _clean_chapter_title(title: str) -> str:
    title = re.sub(r'https?://\S+', '', title)
    title = re.sub(r'\s{2,}', ' ', title).strip()
    return title.strip(" -–—|:·•\t")

def parse_description_timestamps(description: str) -> List[dict]:
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
        after = re.sub(r'^(?:[-–—:|]\s*)+', '', line[m.end():].strip())
        title = _clean_chapter_title(after) or _clean_chapter_title(
            re.sub(r'^(?:[-–—:|]\s*)+|(?:\s*[-–—:|])+$', '', line[:m.start()].strip())
        )
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

# -------------------------------
# Rendering
# -------------------------------

def segments_to_paragraphs(segs: List[dict], target_len: int = 800) -> List[str]:
    texts = [s["text"].strip() for s in segs if s.get("text") and s["text"].strip()]
    if not texts:
        return []
    blob = re.sub(r"\s+", " ", " ".join(texts)).strip()
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

def build_figure_html(image_url: str, chapter_title: str, ts_seconds: float, video_url: str, show_ts: bool = False) -> str:
    ts_label = hhmmss(ts_seconds)
    ts_url = f"{video_url}{'&' if '?' in video_url else '?'}t={int(round(ts_seconds))}s"
    if show_ts:
        alt = f"Screenshot at {ts_label} - {chapter_title}"
        caption = f'{chapter_title} [{ts_label}] — <a href="{ts_url}">Watch at {ts_label}</a>'
    else:
        alt = f"Screenshot — {chapter_title}"
        caption = f'{chapter_title} — <a href="{ts_url}">Watch this section</a>'
    return (
        f'<figure class="yt-chapter">'
        f'<img src="{image_url}" alt="{alt}" />'
        f'<figcaption>{caption}</figcaption>'
        f'</figure>'
    )

def assemble_markdown_with_figures(title: str, groups: List[tuple], image_map: dict, video_url: str, with_timestamps: bool, paragraphs: bool) -> str:
    out_lines: List[str] = []
    for ch, segs in groups:
        out_lines.append(f"## {ch['title']}" if paragraphs else f"## {ch['title']} [{hhmmss(ch['start'])}]")
        key = int(ch["start"])
        if key in image_map and image_map[key].get("url"):
            out_lines.append(build_figure_html(image_map[key]["url"], ch["title"], ch["start"], video_url, show_ts=not paragraphs))
        if segs:
            if paragraphs:
                for p in segments_to_paragraphs(segs):
                    out_lines.append(p)
                    out_lines.append("")
            else:
                for seg in segs:
                    text = seg["text"].replace("\n", " ").strip()
                    if not text:
                        continue
                    out_lines.append(f"[{hhmmss(seg['start'])}] {text}" if with_timestamps else text)
        else:
            out_lines.append("_No speech in this section._")
        out_lines.append("")
    return "\n".join(out_lines)

def md_to_html(md_text: str) -> str:
    try:
        import markdown as mdlib
        return mdlib.markdown(md_text, extensions=["extra", "sane_lists"])
    except Exception:
        html_lines = []
        for line in md_text.splitlines():
            if line.lstrip().startswith("<"):
                html_lines.append(line)
                continue
            if line.startswith("## "):
                html_lines.append(f"<h2>{line[3:].strip()}</h2>")
            elif line.startswith("# "):
                html_lines.append(f"<h1>{line[2:].strip()}</h1>")
            else:
                html_lines.append(f"<p>{line.strip()}</p>" if line.strip() else "")
        return "\n".join(html_lines)

def simple_slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"['’]", "", text)
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "post"

def make_excerpt_from_md(md_text: str, max_len: int = 220) -> str:
    t = re.sub(r"^#+\s*", "", md_text, flags=re.MULTILINE)
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)
    t = re.sub(r"[`*_>#]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return (t[:max_len]).strip()

# -------------------------------
# WordPress API
# -------------------------------

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
    api = wp_url.rstrip("/") + "/wp-json/wp/v2/posts"
    payload = {"title": title, "content": html_content, "status": status}
    if slug:
        payload["slug"] = slug
    if excerpt_text:
        payload["excerpt"] = excerpt_text
    if featured_media is not None:
        payload["featured_media"] = int(featured_media)

    try:
        resp = requests.post(api, json=payload, auth=(wp_user, wp_pass), timeout=60)
    except Exception as e:
        raise RuntimeError(f"Failed to contact WordPress: {e}")

    if resp.status_code not in (200, 201):
        raise RuntimeError(f"WordPress API error {resp.status_code}: {resp.text}")

    data = resp.json()
    return data.get("link") or (wp_url.rstrip("/") + f"/?p={data.get('id')}")

def upload_media(wp_url: str, wp_user: str, wp_pass: str, file_path: str, file_name: Optional[str] = None, mime_type: str = "image/jpeg") -> Tuple[int, str]:
    media_api = wp_url.rstrip("/") + "/wp-json/wp/v2/media"
    name = file_name or os.path.basename(file_path)
    headers = {
        "Content-Disposition": f'attachment; filename="{name}"',
        "Content-Type": mime_type,
    }
    with open(file_path, "rb") as f:
        resp = requests.post(media_api, headers=headers, data=f, auth=(wp_user, wp_pass), timeout=120)
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Media upload failed {resp.status_code}: {resp.text}")
    data = resp.json()
    return int(data["id"]), data.get("source_url")

def update_media_metadata(wp_url: str, wp_user: str, wp_pass: str, media_id: int, title: Optional[str] = None, alt_text: Optional[str] = None, caption_html: Optional[str] = None) -> None:
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
    resp = requests.post(api, json=payload, auth=(wp_user, wp_pass), timeout=60)
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Update media failed {resp.status_code}: {resp.text}")

# -------------------------------
# Video download + frames
# -------------------------------

def seconds_to_ffmpeg_ts(seconds: float) -> str:
    ms = int(round((seconds - int(seconds)) * 1000.0))
    secs = int(seconds)
    h = secs // 3600
    m = (secs % 3600) // 60
    s = secs % 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def ensure_ffmpeg_available() -> None:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception as e:
        raise RuntimeError("ffmpeg is required for --screenshots but was not found on PATH") from e

def download_video_mp4_720(url: str, outdir: str) -> str:
    """
    Try to download a 720p MP4. If YouTube blocks direct download, return an HLS URL
    for ffmpeg to read frames remotely.
    """
    outtmpl = os.path.join(outdir, "%(id)s.%(ext)s")
    opts = _yt_base_opts(skip_download=False)
    opts = {
        "quiet": True,
        "noprogress": True,
        # prefer <=720p, then any best fallback
        "format": "bv*[height<=720]+ba/b[height<=720]/best",
        "merge_output_format": "mp4",
        "outtmpl": outtmpl,
        # do NOT force player_client; let yt-dlp pick
        "retries": 10,
        "fragment_retries": 10,
        "concurrent_fragment_downloads": 1,
    }
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
        # Fallback: get an HLS stream URL (<=720p)
        try:
            opts2 = _yt_base_opts(skip_download=True)
            with YoutubeDL(opts2) as ydl:
                info = ydl.extract_info(url, download=False)
            fmts = info.get("formats") or []
            hls = [f for f in fmts if (f.get("protocol") or "").startswith("m3u8") and (f.get("height") or 0) <= 720]
            hls.sort(key=lambda f: (-(f.get("height") or 0), 'avc1' not in (f.get("vcodec") or "")))
            if hls and hls[0].get("url"):
                return hls[0]["url"]
        except Exception:
            pass
        raise

def extract_frame_to_jpg(video_path: str, ts_seconds: float, out_path: str, width: int = 1280, qscale: int = 3) -> bool:
    ts = seconds_to_ffmpeg_ts(ts_seconds)
    vf = f"scale={int(width)}:-1"
    cmd = ["ffmpeg", "-ss", ts, "-i", video_path, "-frames:v", "1", "-vf", vf, "-q:v", str(int(qscale)), "-y", out_path]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return os.path.exists(out_path)
    except Exception:
        return False

# -------------------------------
# Thumbnails (featured + intro)
# -------------------------------

def _url_ok(url: str) -> bool:
    try:
        r = requests.head(url, allow_redirects=True, timeout=10)
        if 200 <= r.status_code < 400:
            return True
        r = requests.get(url, stream=True, timeout=10)
        return 200 <= r.status_code < 400
    except Exception:
        return False

def fetch_best_thumbnail_url(video_page_url: str, video_id: str) -> Optional[str]:
    try:
        with YoutubeDL(_yt_base_opts(skip_download=True)) as ydl:
            info = ydl.extract_info(video_page_url, download=False)
    except Exception:
        info = {}

    thumbs = info.get("thumbnails") or []
    best_url = None
    if thumbs:
        def thumb_key(t):
            w = t.get("width") or 0
            h = t.get("height") or 0
            pref = t.get("preference") or 0
            return (pref, w * h)
        for t in sorted(thumbs, key=thumb_key, reverse=True):
            url = t.get("url")
            if url:
                best_url = url
                break

    if not best_url:
        best_url = info.get("thumbnail")

    if not best_url:
        for cu in [
            f"https://i.ytimg.com/vi/{video_id}/maxresdefault.jpg",
            f"https://i.ytimg.com/vi/{video_id}/sddefault.jpg",
            f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
            f"https://i.ytimg.com/vi/{video_id}/mqdefault.jpg",
        ]:
            if _url_ok(cu):
                best_url = cu
                break
    return best_url

def upload_featured_thumb_from_url(wp_url: str, wp_user: str, wp_pass: str, thumb_url: str, title: str) -> Optional[Tuple[int, str]]:
    if not thumb_url:
        return None
    ext = ".jpg"
    lower = thumb_url.lower()
    if lower.endswith(".webp"):
        ext, mime = ".webp", "image/webp"
    elif lower.endswith(".png"):
        ext, mime = ".png", "image/png"
    else:
        mime = "image/jpeg"

    tmpdir = tempfile.TemporaryDirectory()
    try:
        fname = f"{simple_slugify(title)}-thumb{ext}"
        fpath = os.path.join(tmpdir.name, fname)
        with requests.get(thumb_url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(fpath, "wb") as out:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        out.write(chunk)

        attach_id, src_url = upload_media(wp_url, wp_user, wp_pass, fpath, file_name=fname, mime_type=mime)
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
    except Exception:
        return None
    finally:
        try:
            tmpdir.cleanup()
        except Exception:
            pass

# -------------------------------
# Affiliate buttons
# -------------------------------

AFFILIATE_NOTICE = ("Please note that some of the links in my video descriptions are affiliate links where I earn from qualifying purchases. As an Amazon Associate I earn from qualifying purchases.")

def _is_excluded_referral(line: str, url: str) -> bool:
    l = (line or "").lower()
    u = (url or "").lower()
    if "buymeacoffee.com" in u or "buy me a coffee" in l or "buymeacoffee" in l:
        return True
    return False

_URL_RE = re.compile(r'(https?://\S+|(?:www\.)?[\w.-]+\.[A-Za-z]{2,}\S*)', re.IGNORECASE)

def _normalize_url(u: str) -> str:
    u = u.strip().rstrip(').,;\'"')
    if u.startswith("http://") or u.startswith("https://"):
        return u
    return "https://" + u.lstrip("/")

def extract_buy_referrals(description: str) -> List[Tuple[str, str]]:
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
        upper = line.upper()
        idx_buy = upper.find("BUY")
        label = line[idx_buy + 3:m.start()].strip(" :-–—|·•\t")
        for sym in ("👉", "➡️", "✅", "☑️", "•"):
            label = label.replace(sym, "")
        label = re.sub(r'\s{2,}', ' ', label).strip()
        if _is_excluded_referral(line, url):
            continue
        if not label:
            try:
                host = re.sub(r'^https?://', '', url).split("/")[0]
            except Exception:
                host = url
            label = f"Buy on {host}"
        else:
            if not label.lower().startswith("buy "):
                label = "Buy " + label
        results.append((label, url))
    return results

def extract_top_referral(description: str) -> Optional[Tuple[str, str]]:
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
        label = line[:m.start()].strip() or line[m.end():].strip()
        for sym in ("👉", "➡️", "✅"):
            label = label.replace(sym, "")
        label = re.sub(r'\s{2,}', ' ', label).strip(" -–—|·•\t").rstrip(":").strip()
        if _is_excluded_referral(line, url):
            continue
        if not label:
            label = url
        return (label, url)
    return None

def build_affiliate_block(description: str) -> str:
    buttons_html: List[str] = []
    buy_refs = extract_buy_referrals(description)

    def _button(label: str, url: str) -> str:
        return (
            f'<p style="margin:12px 0 0;">'
            f'<a href="{url}" target="_blank" rel="nofollow sponsored noopener" '
            f'style="display:inline-block;background:#2b49f0;color:#fff;text-decoration:none;'
            f'padding:12px 16px;border-radius:3px;font-weight:700;line-height:1.2;">{label}</a>'
            f'</p>'
        )

    if buy_refs:
        for label, url in buy_refs:
            buttons_html.append(_button(label, url))
    else:
        fallback = extract_top_referral(description)
        if fallback:
            label, url = fallback
            buttons_html.append(_button(label, url))

    parts = []
    if buttons_html:
        parts.append("\n".join(buttons_html))
    parts.append(f'<p style="margin:8px 0 16px;"><em>{AFFILIATE_NOTICE}</em></p>')
    return "\n".join(parts)

# -------------------------------
# Embed
# -------------------------------

def build_youtube_embed_html(video_id: str, title: str, video_url: str) -> str:
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

# -------------------------------
# Local transcription (fallback)
# -------------------------------

def download_audio(url: str, outdir: str) -> str:
    outtmpl = os.path.join(outdir, "%(id)s.%(ext)s")
    opts = {
        "quiet": True,
        "noprogress": True,
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "m4a", "preferredquality": "192"}],
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
    segments_iter, _info = model.transcribe(audio_path, beam_size=1, vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500))
    segs = []
    for seg in segments_iter:
        start = float(seg.start); end = float(seg.end)
        text = seg.text.strip()
        if text:
            segs.append({"start": start, "duration": end - start, "text": text})
    return segs

# -------------------------------
# Main
# -------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Get a transcript for a YouTube video and optionally post to WordPress."
    )
    parser.add_argument("url", help="YouTube URL or ID")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--no-local", action="store_true", help="Skip local Whisper fallback")
    parser.add_argument("--timestamps", action="store_true", help="Include timestamps in lines (non-paragraph mode)")

    # WP
    parser.add_argument("--post", action="store_true", help="Publish to WordPress")
    parser.add_argument("--wp-url", default=None)
    parser.add_argument("--wp-user", default=None)
    parser.add_argument("--wp-pass", default=None)
    parser.add_argument("--wp-status", default="draft", choices=["draft", "publish", "private", "pending"])

    # Screenshots
    parser.add_argument("--screenshots", action="store_true")
    parser.add_argument("--screenshot-width", type=int, default=1280)
    parser.add_argument("--screenshot-offset", type=float, default=0.5)
    parser.add_argument("--image-quality", type=int, default=3)
    parser.add_argument("--featured-image", action="store_true", help="Use intro image (YouTube thumbnail) as featured image")

    # Mode defaults
    parser.add_argument("--paragraphs", dest="paragraphs", action="store_true")
    parser.add_argument("--no-paragraphs", dest="paragraphs", action="store_false")
    parser.add_argument("--embed", dest="embed", action="store_true")
    parser.add_argument("--no-embed", dest="embed", action="store_false")
    parser.set_defaults(paragraphs=True, embed=True)

    args = parser.parse_args()

    try:
        video_id, title, description = fetch_video_id_and_title(args.url)
    except Exception as e:
        print(f"Error: could not resolve video info - {e}", file=sys.stderr)
        sys.exit(1)

    preferred = [args.lang]
    source = None

    # 1) Official
    segments = try_official_transcript(video_id, preferred)
    if segments:
        source = "official"

    # 2) yt-dlp captions
    if segments is None:
        segments = try_ytdlp_captions(args.url, preferred)
        if segments:
            source = "youtube"

    # 3) Local Whisper
    if segments is None and not args.no_local:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                audio_path = download_audio(args.url, tmp)
            except Exception as e:
                print(f"Error: could not download audio for local transcription - {e}", file=sys.stderr)
                sys.exit(2)
            try:
                segments = transcribe_locally(audio_path, model_size="tiny")
                source = "local"
            except Exception as e:
                print(f"Error: local transcription failed - {e}", file=sys.stderr)
                sys.exit(3)

    if not segments:
        print("No captions/transcript available.", file=sys.stderr)
        sys.exit(4)

    chapters = parse_description_timestamps(description)
    if chapters:
        groups = group_segments_by_chapters(segments, chapters)
    else:
        groups = []

    # Build images map (and optionally upload)
    image_map: Dict[int, Dict[str, Optional[str]]] = {}
    uploaded_thumb: Optional[Tuple[int, str]] = None
    featured_media_id: Optional[int] = None

    if args.screenshots and chapters:
        ensure_ffmpeg_available()
        video_slug = simple_slugify(title)
        temp_video_dir = tempfile.TemporaryDirectory()
        try:
            video_path = download_video_mp4_720(args.url, temp_video_dir.name)

            # If posting, try to upload thumbnail up front (used as intro image and featured if requested)
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

            # Per-chapter images - first chapter uses the YouTube thumbnail if available
            for idx, ch in enumerate(chapters):
                grab_ts = max(0.0, float(ch["start"]) + float(args.screenshot_offset))
                key = int(ch["start"])
                fname = f"{video_slug}_{hhmmss(ch['start']).replace(':','-')}_{simple_slugify(ch['title'])}.jpg"

                if idx == 0 and uploaded_thumb:
                    thumb_id, thumb_url = uploaded_thumb
                    image_map[key] = {"id": int(thumb_id), "url": thumb_url}
                    continue

                local_path = os.path.join(temp_video_dir.name, fname)
                ok = extract_frame_to_jpg(video_path, grab_ts, local_path, width=args.screenshot_width, qscale=args.image_quality)
                if not ok:
                    continue

                if args.post:
                    wp_url = (args.wp_url or os.getenv("WP_URL") or "").strip()
                    wp_user = (args.wp_user or os.getenv("WP_USER") or "").strip()
                    wp_pass = (args.wp_pass or os.getenv("WP_APP_PASS") or os.getenv("WP_PASS") or "").strip()
                    if not (wp_url and wp_user and wp_pass):
                        image_map[key] = {"id": None, "url": f"file://{local_path}"}
                    else:
                        try:
                            attach_id, src_url = upload_media(wp_url, wp_user, wp_pass, local_path, file_name=fname, mime_type="image/jpeg")
                            caption_html = f'<a href="{args.url}{("&" if "?" in args.url else "?")}t={int(ch["start"])}s">Watch this section</a>' if args.paragraphs else f'<a href="{args.url}{("&" if "?" in args.url else "?")}t={int(ch["start"])}s">Back to {hhmmss(ch["start"])}</a>'
                            media_title = f"{ch['title']}"
                            alt_text = f"Screenshot — {ch['title']}"
                            update_media_metadata(wp_url, wp_user, wp_pass, attach_id, title=media_title, alt_text=alt_text, caption_html=caption_html)
                            image_map[key] = {"id": attach_id, "url": src_url}
                        except Exception as e:
                            print(f"Warning: upload failed for {fname}: {e}", file=sys.stderr)
                            image_map[key] = {"id": None, "url": f"file://{local_path}"}
                else:
                    image_map[key] = {"id": None, "url": f"file://{local_path}"}

            if args.featured_image and args.post:
                if uploaded_thumb:
                    featured_media_id = int(uploaded_thumb[0])
                else:
                    featured_media_id = None
        finally:
            try:
                temp_video_dir.cleanup()
            except Exception:
                pass

    # Build body
    if groups:
        output_md = assemble_markdown_with_figures(title, groups, image_map, args.url, with_timestamps=args.timestamps, paragraphs=args.paragraphs)
    else:
        if args.paragraphs:
            output_md = "\n\n".join(segments_to_paragraphs(segments))
        else:
            lines = []
            for seg in segments:
                t = seg["text"].replace("\n", " ").strip()
                if not t:
                    continue
                lines.append(f"[{hhmmss(seg['start'])}] {t}" if args.timestamps else t)
            output_md = "\n".join(lines)

    # Top embed + affiliate CTA
    preface = build_youtube_embed_html(video_id, title, args.url) if args.embed else ""
    try:
        affiliate_html = build_affiliate_block(description)
    except Exception:
        affiliate_html = ""
    pre_section = "\n\n".join([s for s in [preface, affiliate_html] if s])
    if pre_section:
        output_md = pre_section + "\n\n" + output_md
    if affiliate_html:
        output_md = output_md + "\n\n" + affiliate_html

    # Print body (for logs)
    print(output_md)

    # If posting, ensure featured image is set (prefer thumbnail)
    if args.post and args.featured_image:
        wp_url = (args.wp_url or os.getenv("WP_URL") or "").strip()
        wp_user = (args.wp_user or os.getenv("WP_USER") or "").strip()
        wp_pass = (args.wp_pass or os.getenv("WP_APP_PASS") or os.getenv("WP_PASS") or "").strip()
        if wp_url and wp_user and wp_pass:
            try:
                if not featured_media_id:
                    thumb_url = fetch_best_thumbnail_url(args.url, video_id)
                    pair = upload_featured_thumb_from_url(wp_url, wp_user, wp_pass, thumb_url, title)
                    if pair:
                        featured_media_id = int(pair[0])
            except Exception:
                pass

    # Publish
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
        try:
            link = publish_to_wordpress(
                wp_url, wp_user, wp_pass, title, html_content,
                status=args.wp_status, slug=slug, excerpt_text=excerpt,
                featured_media=featured_media_id
            )
            print(f"\nPosted to WordPress: {link}")
        except Exception as e:
            print(f"Error publishing to WordPress: {e}", file=sys.stderr)
            sys.exit(7)

if __name__ == "__main__":
    main()

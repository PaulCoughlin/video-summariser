"""Summarise a YouTube video via OpenRouter (VPS deployment).

This is the OpenRouter version of the local-Claude script. It calls the
OpenRouter Chat Completions endpoint instead of shelling out to `claude -p`,
so it runs in any headless environment with just an API key.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

import httpx
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    AgeRestricted,
    IpBlocked,
    NoTranscriptFound,
    PoTokenRequired,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeDataUnparsable,
    YouTubeRequestFailed,
    YouTubeTranscriptApiException,
)
from youtube_transcript_api.proxies import GenericProxyConfig, WebshareProxyConfig

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_TIMEOUT_SECONDS = 180


class SummariseError(Exception):
    """User-visible failure — render this message directly."""


@dataclass
class SummaryResult:
    video_id: str
    watch_url: str
    thumbnail_url: str
    body_markdown: str
    segment_count: int
    prompt_tokens: int | None  # from OpenRouter; None if absent
    completion_tokens: int | None
    total_cost_usd: float | None  # 0 for free models, None if absent
    model_id: str


# ---------- shared with the local version (URL parsing, transcripts, prompt) ----------


def extract_video_id(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower().lstrip("www.")
    if host == "youtu.be":
        vid = parsed.path.lstrip("/").split("/")[0]
    elif host.endswith("youtube.com"):
        if parsed.path == "/watch":
            vid = parse_qs(parsed.query).get("v", [""])[0]
        else:
            parts = [p for p in parsed.path.split("/") if p]
            vid = parts[1] if len(parts) >= 2 else ""
    else:
        vid = ""
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", vid):
        raise ValueError(f"Could not extract a YouTube video ID from: {url}")
    return vid


def _build_transcript_api() -> YouTubeTranscriptApi:
    """Build a YouTubeTranscriptApi, optionally with a proxy for YouTube transcript fetches.

    YouTube blocks most datacenter IPs, so on a VPS you'll usually need a
    proxy. Two ways to configure one (in priority order):

    1. PROXY_URL — a full proxy URL like `http://user:pass@1.2.3.4:8080`.
       Most flexible: works with any provider, and using a literal IP avoids
       any DNS-resolution issues inside the container.
    2. WEBSHARE_PROXY_USERNAME + WEBSHARE_PROXY_PASSWORD — uses Webshare's
       backbone gateway (`p.webshare.io`). Requires a paid Webshare plan;
       free Webshare gives you specific IPs that you'd plug into PROXY_URL
       instead.
    """
    proxy_url = os.environ.get("PROXY_URL", "").strip()
    if proxy_url:
        return YouTubeTranscriptApi(
            proxy_config=GenericProxyConfig(http_url=proxy_url, https_url=proxy_url)
        )

    user = os.environ.get("WEBSHARE_PROXY_USERNAME", "").strip()
    password = os.environ.get("WEBSHARE_PROXY_PASSWORD", "").strip()
    if user and password:
        return YouTubeTranscriptApi(proxy_config=WebshareProxyConfig(user, password))

    return YouTubeTranscriptApi()


def fetch_transcript(video_id: str) -> list[tuple[int, str]]:
    fetched = _build_transcript_api().fetch(
        video_id, languages=("en", "en-US", "en-GB")
    )
    return [(int(s.start), s.text.strip()) for s in fetched if s.text and s.text.strip()]


def format_timestamp(seconds: int) -> str:
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def canonical_watch_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def thumbnail_url(video_id: str) -> str:
    return f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"


def format_transcript(segments: list[tuple[int, str]]) -> str:
    return "\n".join(f"[{format_timestamp(t)}|{t}s] {text}" for t, text in segments)


PROMPT_TEMPLATE = """You are summarising a YouTube video for a friend who hasn't watched it.

The video URL is: {url}

The transcript below is annotated as `[M:SS|Ns]` or `[H:MM:SS|Ns]`, where
`M:SS`/`H:MM:SS` is the display time and `Ns` is the same moment in raw seconds.
Use these to build clickable deep-links — append `&t=Ns` to the video URL.

Output **pure GitHub-flavored markdown** and nothing else. No preamble, no
"Here is your summary", no closing remark. Use exactly this structure:

## TL;DR
<2-3 sentences capturing the core thesis or arc of the video>

## Key Takeaways
- <bullet>
- <bullet>
(between 5 and 8 bullets total; each bullet is one sentence, concrete, no fluff)

## Watch These Moments
- [**M:SS**]({url}&t=Ns) — <one-line description of what happens here>
- [**M:SS**]({url}&t=Ns) — <one-line description>
- [**M:SS**]({url}&t=Ns) — <one-line description>

(at least 3, up to 6 entries; pick the most rewarding moments — a key
argument, a vivid example, a turning point. The `M:SS` and `Ns` values
MUST come from a line that appears verbatim in the transcript below —
never invent a timestamp.)

Transcript:
---
{transcript}
---
"""


def build_prompt(url: str, transcript: str) -> str:
    return PROMPT_TEMPLATE.format(url=url, transcript=transcript)


# ---------- OpenRouter call ----------


def _api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise SummariseError(
            "OPENROUTER_API_KEY is not set on the server. "
            "Set it in the environment (or .env file) and restart."
        )
    return key


def _openrouter_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
        # OpenRouter recommends these for analytics; harmless if absent.
        "HTTP-Referer": os.environ.get("OPENROUTER_REFERER", "https://github.com/PaulCoughlin/video-summariser"),
        "X-Title": os.environ.get("OPENROUTER_APP_TITLE", "Video Summariser"),
    }


def call_openrouter(prompt: str, model_id: str) -> tuple[str, dict]:
    """Send the prompt to OpenRouter, return (body_markdown, usage_dict)."""
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
    }
    try:
        r = httpx.post(
            f"{OPENROUTER_BASE}/chat/completions",
            headers=_openrouter_headers(),
            json=payload,
            timeout=OPENROUTER_TIMEOUT_SECONDS,
        )
    except httpx.TimeoutException:
        raise SummariseError(
            f"OpenRouter didn't respond within {OPENROUTER_TIMEOUT_SECONDS}s. "
            "The model may be overloaded — try again, or pick a different model."
        )
    except httpx.HTTPError as e:
        raise SummariseError(f"OpenRouter network error: {e}")

    if r.status_code == 401:
        raise SummariseError("OpenRouter rejected the API key. Check OPENROUTER_API_KEY.")
    if r.status_code == 429:
        raise SummariseError(
            "OpenRouter rate limit hit for this model. Free models are limited to "
            "~50 requests/day without credits — try a different model or wait."
        )
    if r.status_code >= 400:
        raise SummariseError(
            f"OpenRouter returned {r.status_code}: {r.text[:300]}"
        )

    data = r.json()
    try:
        content = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        raise SummariseError(f"OpenRouter returned an unexpected response: {data}")

    return content, data.get("usage", {}) or {}


def check_openrouter_auth() -> tuple[bool, str]:
    """Probe the API key by hitting GET /api/v1/auth/key."""
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        return False, "OPENROUTER_API_KEY is not set."
    try:
        r = httpx.get(
            f"{OPENROUTER_BASE}/auth/key",
            headers={"Authorization": f"Bearer {key}"},
            timeout=15,
        )
    except httpx.HTTPError as e:
        return False, f"Couldn't reach OpenRouter: {e}"
    if r.status_code == 401:
        return False, "OpenRouter rejected the API key."
    if r.status_code != 200:
        return False, f"OpenRouter returned {r.status_code}: {r.text[:200]}"
    info = r.json().get("data", {})
    label = info.get("label") or "(no label)"
    limit = info.get("limit")
    usage = info.get("usage", 0)
    if limit is not None:
        return True, f"Authenticated as `{label}` — used ${usage:.4f} of ${limit:.2f} credit."
    return True, f"Authenticated as `{label}` — pay-as-you-go."


def summarise_url(url: str, model_id: str) -> SummaryResult:
    try:
        video_id = extract_video_id(url)
    except ValueError as e:
        raise SummariseError(str(e)) from e

    try:
        segments = fetch_transcript(video_id)
    except (TranscriptsDisabled, NoTranscriptFound):
        raise SummariseError("No English captions available for this video.")
    except VideoUnavailable:
        raise SummariseError("Video unavailable (private, removed, or region-locked).")
    except (RequestBlocked, IpBlocked):
        raise SummariseError(
            "YouTube blocked the transcript request from this server's IP. "
            "This is normal on datacenter IPs. Fix: sign up at https://www.webshare.io "
            "for a residential proxy (~$3/month) and set WEBSHARE_PROXY_USERNAME and "
            "WEBSHARE_PROXY_PASSWORD in the server env, then redeploy."
        )
    except AgeRestricted:
        raise SummariseError(
            "Video is age-restricted; YouTube won't return its transcript without a signed-in cookie."
        )
    except PoTokenRequired:
        raise SummariseError(
            "YouTube wants a Proof-of-Origin token for this video. Currently unsupported."
        )
    except (YouTubeRequestFailed, YouTubeDataUnparsable) as e:
        raise SummariseError(f"YouTube returned an unexpected response: {e}")
    except YouTubeTranscriptApiException as e:
        raise SummariseError(f"Transcript fetch failed: {type(e).__name__}: {e}")

    if not segments:
        raise SummariseError("Transcript was empty.")

    watch_url = canonical_watch_url(video_id)
    prompt = build_prompt(watch_url, format_transcript(segments))
    body, usage = call_openrouter(prompt, model_id)

    return SummaryResult(
        video_id=video_id,
        watch_url=watch_url,
        thumbnail_url=thumbnail_url(video_id),
        body_markdown=body,
        segment_count=len(segments),
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        total_cost_usd=usage.get("total_cost"),
        model_id=model_id,
    )

"""Summarise a YouTube video into TL;DR + takeaways + key moments.

Pipes a structured prompt to `claude -p` (uses your Claude Code subscription;
no Anthropic API key needed).
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)


class SummariseError(Exception):
    """Anything we want the caller (CLI or web) to render as a clean message."""


@dataclass
class SummaryResult:
    video_id: str
    watch_url: str
    thumbnail_url: str
    body_markdown: str  # the LLM-generated body (TL;DR, takeaways, moments)
    segment_count: int
    approx_tokens: int


def extract_video_id(url: str) -> str:
    """Pull the 11-char video ID out of any common YouTube URL shape."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower().lstrip("www.")

    if host == "youtu.be":
        vid = parsed.path.lstrip("/").split("/")[0]
    elif host.endswith("youtube.com"):
        if parsed.path == "/watch":
            vid = parse_qs(parsed.query).get("v", [""])[0]
        else:
            # /shorts/<id>, /live/<id>, /embed/<id>, /v/<id>
            parts = [p for p in parsed.path.split("/") if p]
            vid = parts[1] if len(parts) >= 2 else ""
    else:
        vid = ""

    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", vid):
        raise ValueError(f"Could not extract a YouTube video ID from: {url}")
    return vid


def fetch_transcript(video_id: str) -> list[tuple[int, str]]:
    """Return [(start_seconds, text), ...] for the best available English track."""
    fetched = YouTubeTranscriptApi().fetch(video_id, languages=("en", "en-US", "en-GB"))
    return [(int(s.start), s.text.strip()) for s in fetched if s.text and s.text.strip()]


def format_timestamp(seconds: int) -> str:
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def canonical_watch_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def thumbnail_url(video_id: str) -> str:
    # maxresdefault is missing on some videos; hqdefault is universally present.
    # We use maxresdefault and let the renderer fall back visually if absent —
    # better quality on the common case.
    return f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"


def format_transcript(segments: list[tuple[int, str]]) -> str:
    # Each line carries both the human-readable timestamp and the raw seconds,
    # so the model can write a deep-link URL without doing arithmetic.
    return "\n".join(
        f"[{format_timestamp(t)}|{t}s] {text}" for t, text in segments
    )


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


CLAUDE_TIMEOUT_SECONDS = 180

# Substrings (lowercased) that, if seen in claude's stderr, mean "you're not signed in".
# Matched loosely because the exact wording can change between CLI versions.
_AUTH_ERROR_HINTS = (
    "log in", "login", "sign in", "signin", "authenticate", "authentication",
    "unauthorised", "unauthorized", "not authorised", "not authorized",
    "credentials", "api key",
)

AUTH_INSTRUCTIONS = (
    "Claude Code isn't signed in on this machine. "
    "Stop the app (Ctrl+C in its terminal), run `claude /login` and complete "
    "the sign-in, then start the app again."
)


def _looks_like_auth_error(stderr: str) -> bool:
    s = stderr.lower()
    return any(h in s for h in _AUTH_ERROR_HINTS)


def run_claude(prompt: str) -> str:
    if shutil.which("claude") is None:
        raise SummariseError(
            "Claude Code CLI not found on PATH. Install it from "
            "https://docs.anthropic.com/claude/docs/claude-code first."
        )
    try:
        result = subprocess.run(
            ["claude", "-p"],
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=CLAUDE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        raise SummariseError(
            f"Claude didn't respond within {CLAUDE_TIMEOUT_SECONDS}s. "
            "Try again, or try a shorter video."
        )
    if result.returncode != 0:
        if _looks_like_auth_error(result.stderr):
            raise SummariseError(AUTH_INSTRUCTIONS)
        raise SummariseError(
            f"Claude failed (exit {result.returncode}): {result.stderr.strip() or '(no stderr)'}"
        )
    return result.stdout.strip()


def check_claude_auth() -> tuple[bool, str]:
    """Probe `claude -p` with a tiny prompt to confirm it's installed and authenticated.

    Returns (ok, message). Cheap-ish — takes a few seconds. Use at startup or in a
    health endpoint, not per-request.
    """
    if shutil.which("claude") is None:
        return False, "Claude Code CLI not found on PATH."
    try:
        result = subprocess.run(
            ["claude", "-p"],
            input="Reply with the single word: ok",
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return False, "Auth probe timed out — claude may be hanging on input."
    if result.returncode != 0:
        if _looks_like_auth_error(result.stderr):
            return False, AUTH_INSTRUCTIONS
        return False, f"claude exited {result.returncode}: {result.stderr.strip() or '(no stderr)'}"
    return True, "Authenticated."


def summarise_url(url: str) -> SummaryResult:
    """End-to-end: URL in, structured summary out. Raises SummariseError on user-visible failure."""
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

    if not segments:
        raise SummariseError("Transcript was empty.")

    watch_url = canonical_watch_url(video_id)
    prompt = build_prompt(watch_url, format_transcript(segments))
    body = run_claude(prompt)

    return SummaryResult(
        video_id=video_id,
        watch_url=watch_url,
        thumbnail_url=thumbnail_url(video_id),
        body_markdown=body,
        segment_count=len(segments),
        approx_tokens=len(prompt) // 4,
    )


def main() -> None:
    # Windows defaults stdout to cp1252; force UTF-8 so em-dashes survive piping.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Summarise a YouTube video.")
    parser.add_argument("url", nargs="?", help="YouTube video URL")
    parser.add_argument(
        "--check", action="store_true",
        help="Just verify Claude Code is installed and authenticated, then exit.",
    )
    args = parser.parse_args()

    if args.check:
        ok, msg = check_claude_auth()
        print(msg)
        sys.exit(0 if ok else 1)

    if not args.url:
        parser.error("url is required (or use --check)")

    try:
        result = summarise_url(args.url)
    except SummariseError as e:
        sys.exit(f"error: {e}")

    print(f"[summarise] transcript: {result.segment_count} segments, "
          f"~{result.approx_tokens} tokens.", file=sys.stderr)
    print(f"[![Video thumbnail]({result.thumbnail_url})]({result.watch_url})\n")
    print(f"**[Watch on YouTube]({result.watch_url})**\n")
    print(result.body_markdown)


if __name__ == "__main__":
    main()

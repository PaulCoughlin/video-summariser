"""Summarise a YouTube video into TL;DR + takeaways + key timestamped moments.

This module is both a **CLI** (``py summarise.py <url>``) and a **library**
imported by the FastAPI app in ``app.py``. The high-level flow is:

    1. Parse a YouTube URL and pull the 11-character video ID.
    2. Fetch the English caption track via youtube-transcript-api.
    3. Annotate every transcript line with ``[M:SS|Ns]`` markers so the LLM
       can build clickable ``&t=Ns`` deep-links without doing arithmetic.
    4. Pipe a structured prompt to ``claude -p`` — uses your Claude Code
       subscription, so no Anthropic API key is needed — and return the
       generated markdown body.

**Public surface for importers:**
    - ``summarise_url(url, on_progress=None, model=None) -> SummaryResult`` —
      end-to-end entry point. Pass ``on_progress(msg)`` to receive a status
      string at each pipeline step (used by the web UI for live logs). Pass
      ``model="Opus"`` (or ``"Sonnet"``/``"Haiku"``) to override the model;
      ``None`` or ``"Default"`` uses your ``claude`` CLI's global setting.
      Matched case-insensitively, so lowercase still works.
    - ``check_claude_auth() -> (ok, message)`` — cheap startup probe.
    - ``SUPPORTED_MODELS`` — dict of model-name → ``--model`` flag value.
    - ``SummariseError`` — raised for any user-facing failure (no captions,
      auth missing, timeout, etc.). Anything else is a bug; let it surface.

**Run as a script:**
    py summarise.py <url>                # writes <video-id>.md in CWD
    py summarise.py --check              # verify Claude Code is signed in
    py summarise.py <url> -o -           # write markdown to stdout instead
    py summarise.py <url> --model Opus   # override the model
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)


# ---------------------------------------------------------------------------
# Errors & result types
# ---------------------------------------------------------------------------


class SummariseError(Exception):
    """A user-visible failure with a message safe to render directly.

    Raised in place of leaking lower-level exceptions (subprocess errors,
    transcript-API internals, etc.) so the CLI and web layer can pass the
    message through unmodified — they never need to interpret the cause.
    """


@dataclass
class SummaryResult:
    """Everything a caller needs to render a finished summary.

    ``body_markdown`` is just the LLM-generated body (TL;DR, key takeaways,
    watch-these-moments). The CLI and web layer wrap it with their own
    header (thumbnail link + 'Watch on YouTube') so the analytic body
    stays composable.
    """

    video_id: str          # 11-char YouTube ID, e.g. "P60LqQg1RH8"
    watch_url: str         # canonical https://www.youtube.com/watch?v=<id>
    thumbnail_url: str     # max-resolution preview JPG (renderer falls back if missing)
    body_markdown: str     # the LLM-generated TL;DR + takeaways + timestamped moments
    segment_count: int     # number of transcript lines fetched (diagnostic)
    approx_tokens: int     # rough prompt size (chars / 4) for diagnostic display


# ---------------------------------------------------------------------------
# URL parsing & YouTube helpers
# ---------------------------------------------------------------------------


def extract_video_id(url: str) -> str:
    """Pull the 11-char video ID out of any common YouTube URL shape.

    Supports the watch URL (``youtube.com/watch?v=…``), short URL
    (``youtu.be/…``), shorts (``/shorts/…``), live (``/live/…``), embed
    (``/embed/…``), and the legacy ``/v/…`` form. Trailing query
    parameters like ``&t=42s`` are ignored. Raises ``ValueError`` if
    the URL doesn't yield a valid 11-character ID.
    """
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower().lstrip("www.")

    if host == "youtu.be":
        # https://youtu.be/<id>[?…]
        vid = parsed.path.lstrip("/").split("/")[0]
    elif host.endswith("youtube.com"):
        if parsed.path == "/watch":
            # https://www.youtube.com/watch?v=<id>&…
            vid = parse_qs(parsed.query).get("v", [""])[0]
        else:
            # /shorts/<id>, /live/<id>, /embed/<id>, /v/<id>
            parts = [p for p in parsed.path.split("/") if p]
            vid = parts[1] if len(parts) >= 2 else ""
    else:
        vid = ""

    # YouTube IDs are exactly 11 chars from the URL-safe-base64 alphabet.
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", vid):
        raise ValueError(f"Could not extract a YouTube video ID from: {url}")
    return vid


def fetch_transcript(video_id: str) -> list[tuple[int, str]]:
    """Fetch the best English caption track as ``[(start_seconds, text), …]``.

    Tries ``en`` → ``en-US`` → ``en-GB`` in priority order. Drops segments
    whose text is blank after stripping. Lets the library's own exception
    types (``TranscriptsDisabled``, ``NoTranscriptFound``, ``VideoUnavailable``)
    propagate — the caller maps them to user-friendly ``SummariseError``s.
    """
    fetched = YouTubeTranscriptApi().fetch(video_id, languages=("en", "en-US", "en-GB"))
    return [(int(s.start), s.text.strip()) for s in fetched if s.text and s.text.strip()]


def format_timestamp(seconds: int) -> str:
    """Render seconds as ``M:SS`` (under an hour) or ``H:MM:SS``.

    Matches the way YouTube itself displays timecodes, which is also the
    shape the LLM is asked to cite in its output.
    """
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def canonical_watch_url(video_id: str) -> str:
    """Build the canonical watch URL the deep-links append ``&t=Ns`` to."""
    return f"https://www.youtube.com/watch?v={video_id}"


def thumbnail_url(video_id: str) -> str:
    """Best-quality thumbnail URL.

    ``maxresdefault`` is missing on some videos; ``hqdefault`` is universally
    present. We optimistically use ``maxresdefault`` and let the renderer
    fall back visually if absent (the web template handles this with an
    ``onerror`` handler) — better quality on the common case.
    """
    return f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"


def format_transcript(segments: list[tuple[int, str]]) -> str:
    """Annotate the transcript so the LLM can build deep-links cheaply.

    Each line is rendered as ``[M:SS|Ns] <text>`` — both the human-readable
    timestamp the model will cite back to the user *and* the same moment
    in raw seconds. The model just appends ``&t=Ns`` to the watch URL; no
    arithmetic, no chance of off-by-one drift.
    """
    return "\n".join(
        f"[{format_timestamp(t)}|{t}s] {text}" for t, text in segments
    )


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

# The prompt deliberately:
#   - bans preamble / closing remarks so the body is paste-ready Markdown
#   - fixes the section names and ordering so we can post-process if needed
#   - constrains takeaway count to a tight 5-8 (avoids fluffy long lists)
#   - tells the model to cite *verbatim* timestamps from the transcript,
#     not invented ones — combined with the [M:SS|Ns] annotation this gives
#     the closest thing to a hard guarantee the deep-links will be valid.
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
    """Inject the watch URL and annotated transcript into ``PROMPT_TEMPLATE``."""
    return PROMPT_TEMPLATE.format(url=url, transcript=transcript)


# ---------------------------------------------------------------------------
# Claude Code invocation
# ---------------------------------------------------------------------------

# Hard ceiling on a single `claude -p` call. Long videos near the context
# limit can occasionally get close to this; bump if you hit timeouts.
CLAUDE_TIMEOUT_SECONDS = 180

# Substrings (lowercased) that, if seen in claude's stderr, mean "you're not
# signed in". Matched loosely because the exact wording can change between
# CLI versions — better to occasionally false-positive than to surface a raw
# auth error as "Claude failed (exit 1): …".
_AUTH_ERROR_HINTS = (
    "log in", "login", "sign in", "signin", "authenticate", "authentication",
    "unauthorised", "unauthorized", "not authorised", "not authorized",
    "credentials", "api key",
)

# Single source of truth for the auth-failure message, so the CLI and web
# UI tell users exactly the same thing to do.
AUTH_INSTRUCTIONS = (
    "Claude Code isn't signed in on this machine. "
    "Stop the app (Ctrl+C in its terminal), run `claude /login` and complete "
    "the sign-in, then start the app again."
)


def _looks_like_auth_error(stderr: str) -> bool:
    """True if claude's stderr smells like an auth failure (loose match)."""
    s = stderr.lower()
    return any(h in s for h in _AUTH_ERROR_HINTS)


# Models exposed to the UI / CLI. Map of "user-visible name" → flag value
# passed to `claude -p --model …`. ``None`` means "don't pass --model at all,
# use whatever model the user's claude install is globally configured to use".
#
# Editing this dict is the single point of customisation: the CLI's
# ``--model`` choices, ``run_claude``'s validation, and the web-UI dropdown
# all read from it. Add an entry to expose a new option in all three places.
#
# The value can be either:
#   - a Claude Code alias  ("sonnet", "opus", "haiku"), which Anthropic
#     bumps to the current generation as new versions ship; or
#   - a fully-qualified model ID (e.g. "claude-opus-4-7-20250930") to pin
#     to a specific version. Useful for the 1M-context Opus variant or
#     when you want reproducibility across alias bumps. Look up exact IDs
#     at https://docs.anthropic.com/en/docs/about-claude/models/overview .
SUPPORTED_MODELS: dict[str, str | None] = {
    "Default": None,
    "Sonnet": "sonnet",
    "Opus": "opus",
    "Haiku": "haiku",
}


def _resolve_model(model: str | None) -> tuple[str, str | None]:
    """Look up a model key in ``SUPPORTED_MODELS`` case-insensitively.

    Returns ``(canonical_key, flag)`` where ``canonical_key`` is the key as
    written in the dict (used for display/log) and ``flag`` is the value
    passed to ``claude --model`` (or ``None`` to skip the flag).

    Lets callers type any case (``"opus"``, ``"Opus"``, ``"OPUS"``) while
    the dict keeps a single canonical Title-Case spelling for the UI.
    Raises ``SummariseError`` for an unknown name.
    """
    target = (model or "Default").strip().lower()
    for key, flag in SUPPORTED_MODELS.items():
        if key.lower() == target:
            return key, flag
    raise SummariseError(
        f"Unknown model {model!r}. Use one of: "
        + ", ".join(SUPPORTED_MODELS.keys())
    )


def run_claude(prompt: str, model: str | None = None) -> str:
    """Pipe ``prompt`` to ``claude -p`` and return its stdout, stripped.

    ``model`` is matched case-insensitively against the keys of
    ``SUPPORTED_MODELS`` (``"Sonnet"``, ``"Opus"``, ``"Haiku"``); ``None``
    or ``"Default"`` leaves the choice to ``claude``'s global setting
    (whatever ``~/.claude/settings.json`` / ``ANTHROPIC_MODEL`` resolves
    to).

    Maps three failure modes to clean ``SummariseError`` messages:
      - ``claude`` not on PATH (hint at install URL)
      - subprocess timed out (suggest a shorter video)
      - non-zero exit (auth-shaped stderr → AUTH_INSTRUCTIONS, else generic)

    Anything else propagates — we'd rather see a real traceback than swallow
    a bug.
    """
    if shutil.which("claude") is None:
        raise SummariseError(
            "Claude Code CLI not found on PATH. Install it from "
            "https://docs.anthropic.com/claude/docs/claude-code first."
        )

    cmd: list[str] = ["claude", "-p"]
    _key, flag = _resolve_model(model)
    if flag is not None:
        cmd += ["--model", flag]

    try:
        result = subprocess.run(
            cmd,
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
    """Probe ``claude -p`` with a tiny prompt to confirm install + auth.

    Returns ``(ok, message)``. Cheap-ish — takes a few seconds because it
    actually round-trips a prompt. Use at startup or in a health endpoint,
    not per request. The ``message`` is suitable to display verbatim in
    the UI on failure (e.g., on the auth banner).
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


# ---------------------------------------------------------------------------
# Library entry point
# ---------------------------------------------------------------------------


def summarise_url(
    url: str,
    on_progress: Callable[[str], None] | None = None,
    model: str | None = None,
) -> SummaryResult:
    """End-to-end: URL in, structured ``SummaryResult`` out.

    Wires together URL parsing → transcript fetch → prompt build → claude
    call. All user-facing failure modes are translated to ``SummariseError``
    with messages safe to display unmodified.

    ``on_progress`` is an optional callback invoked at each step with a
    short human-readable status string. The web UI uses it to stream a
    live log to the browser via Server-Sent Events; the CLI doesn't pass
    one (it has its own progress prints) so the default is a no-op.

    ``model`` is an optional override (``"sonnet"`` / ``"opus"`` / ``"haiku"``);
    when ``None`` or ``"default"`` Claude uses whatever model your install is
    configured for. Useful for running a denser video through Opus.

    Used by both the CLI ``main()`` (for piping markdown to a file) and
    the FastAPI app (for rendering the web UI).
    """
    progress = on_progress or (lambda _msg: None)

    progress("parsing video URL")
    try:
        video_id = extract_video_id(url)
    except ValueError as e:
        raise SummariseError(str(e)) from e

    progress(f"fetching transcript for {video_id}")
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
    approx_tokens = len(prompt) // 4
    # Use the canonical Title-Case key from the dict, regardless of how
    # the caller spelled it.
    model_label, _ = _resolve_model(model)
    progress(
        f"got {len(segments)} segments (~{approx_tokens} tokens) — "
        f"calling Claude (model: {model_label})"
    )
    body = run_claude(prompt, model=model)
    progress("summary received")

    return SummaryResult(
        video_id=video_id,
        watch_url=watch_url,
        thumbnail_url=thumbnail_url(video_id),
        body_markdown=body,
        segment_count=len(segments),
        approx_tokens=approx_tokens,  # rough char→token heuristic
    )


# ---------------------------------------------------------------------------
# CLI — terminal spinner & main()
# ---------------------------------------------------------------------------

# Braille frames for a smooth in-place spinner. UTF-8 only; the fallback
# below covers cases where stderr isn't a real terminal anyway.
_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


@contextmanager
def _cli_spinner(label: str):
    """Animate a spinner + elapsed counter on stderr until the block exits.

    Used to keep the user informed during the slow ``claude -p`` call. Falls
    back to a single printed line when stderr isn't a terminal (e.g. when
    the script is run from a CI job or piped to a log file) so logs don't
    fill with carriage-return junk.
    """
    if not sys.stderr.isatty():
        # Non-TTY: emit one static line and skip the animation entirely.
        print(f"      {label}...", file=sys.stderr, flush=True)
        yield
        return

    stop = threading.Event()

    def loop() -> None:
        start = time.monotonic()
        i = 0
        while not stop.is_set():
            frame = _SPINNER_FRAMES[i % len(_SPINNER_FRAMES)]
            elapsed = time.monotonic() - start
            sys.stderr.write(f"\r      {label} {frame} ({elapsed:.0f}s)   ")
            sys.stderr.flush()
            stop.wait(0.1)
            i += 1
        # Wipe the spinner line on exit so the next print starts clean.
        sys.stderr.write("\r" + " " * (len(label) + 30) + "\r")
        sys.stderr.flush()

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join()


def main() -> None:
    """CLI entry point.

    Parses args, runs the three-step pipeline (fetch → summarise → save),
    and prints progress to stderr. Markdown output goes to either
    ``<video-id>.md`` in CWD (default), a path passed via ``-o``, or
    stdout when ``-o -`` is used. Non-zero exit on any failure with a
    one-line ``error: …`` message.
    """
    # Windows defaults stdout to cp1252; force UTF-8 so em-dashes and other
    # non-Latin-1 characters survive piping into files.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Summarise a YouTube video.")
    parser.add_argument("url", nargs="?", help="YouTube video URL")
    parser.add_argument(
        "--check", action="store_true",
        help="Just verify Claude Code is installed and authenticated, then exit.",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path. Defaults to <video-id>.md in the current directory. "
             "Use '-' to write to stdout instead.",
    )
    def _model_choice(s: str) -> str:
        # Case-insensitive: accept "opus" or "Opus" → return canonical "Opus".
        try:
            key, _ = _resolve_model(s)
        except SummariseError as e:
            raise argparse.ArgumentTypeError(str(e))
        return key

    parser.add_argument(
        "-m", "--model",
        type=_model_choice,
        default="Default",
        metavar="{" + ",".join(SUPPORTED_MODELS.keys()) + "}",
        help="Which Claude model to use. 'Default' (the default) leaves the "
             "choice to your `claude` CLI's global setting. Pass 'Opus' for "
             "particularly dense videos. Case-insensitive.",
    )
    args = parser.parse_args()

    # --check is a special mode: probe auth, print the result, exit 0/1.
    if args.check:
        ok, msg = check_claude_auth()
        print(msg)
        sys.exit(0 if ok else 1)

    if not args.url:
        parser.error("url is required (or use --check)")

    # Step 1 — parse URL into a video ID.
    try:
        video_id = extract_video_id(args.url)
    except ValueError as e:
        sys.exit(f"error: {e}")

    # Step 2 — fetch the transcript (network call to YouTube).
    print(f"[1/3] fetching transcript for {video_id}...", file=sys.stderr)
    try:
        segments = fetch_transcript(video_id)
    except (TranscriptsDisabled, NoTranscriptFound):
        sys.exit("error: no English captions available for this video.")
    except VideoUnavailable:
        sys.exit("error: video unavailable (private, removed, or region-locked).")

    if not segments:
        sys.exit("error: transcript was empty.")

    # Build the prompt up front so we can show the token estimate before
    # the (slow) Claude call kicks off.
    watch_url = canonical_watch_url(video_id)
    prompt = build_prompt(watch_url, format_transcript(segments))
    approx_tokens = len(prompt) // 4
    print(f"      → {len(segments)} segments, ~{approx_tokens} tokens", file=sys.stderr)

    # Step 3 — call Claude. This is the long one (typically 30-90s).
    print(
        f"[2/3] summarising with Claude (model: {args.model}, "
        f"typically 30-90s for long videos)...",
        file=sys.stderr,
    )
    try:
        with _cli_spinner("Claude is roboting.."):
            body = run_claude(prompt, model=args.model)
    except SummariseError as e:
        sys.exit(f"error: {e}")

    # Compose the final markdown: thumbnail + watch link header, then the
    # LLM body. Same shape the web UI renders.
    full_md = (
        f"[![Video thumbnail]({thumbnail_url(video_id)})]({watch_url})\n\n"
        f"**[Watch on YouTube]({watch_url})**\n\n"
        f"{body}\n"
    )

    # Step 4 — write output. `-o -` means stdout; otherwise a real file.
    if args.output == "-":
        print("[3/3] writing to stdout", file=sys.stderr)
        print(full_md)
    else:
        output_path = Path(args.output) if args.output else Path(f"{video_id}.md")
        output_path.write_text(full_md, encoding="utf-8")
        print(f"[3/3] saved {output_path.resolve()}", file=sys.stderr)


if __name__ == "__main__":
    main()

"""FastAPI web UI wrapping ``summarise.py``.

A single-page app that takes a YouTube URL, calls into ``summarise_url``,
and renders the result as styled HTML. The whole stack is intentionally
small: FastAPI for routing, Jinja2 for templates, htmx on the page for
the in-place form swap, and Tailwind via CDN for styling. No database,
no users, no rate limiting — this is meant to run on your own machine
and be exposed (if at all) through a Cloudflare Tunnel with Access in
front of it.

**Run locally:**
    py -m uvicorn app:app --reload --port 8000

Then open http://127.0.0.1:8000.

**Routes:**
    GET  /            — the form
    POST /summarise   — runs the pipeline, returns the summary fragment
                        (or an error fragment, also as 200, so htmx swaps it)
    GET  /healthz     — JSON health/auth status, for tunnels and uptime

The page is gated on Claude Code being signed in: a banner appears when
auth is missing and the form is disabled until the user runs
``claude /login`` in another terminal and restarts the app.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import markdown as md
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from summarise import (
    AUTH_INSTRUCTIONS,
    SummariseError,
    check_claude_auth,
    summarise_url,
)

BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=BASE_DIR / "templates")

app = FastAPI(title="Video Summariser")


# ---------------------------------------------------------------------------
# Auth state
# ---------------------------------------------------------------------------

# Cached auth state for the running server. Probed once at startup and
# refreshed whenever a real summarise call surfaces an auth error — that way
# a user who runs `claude /login` after the server started will see the
# banner re-appear after one failed attempt, rather than silently 500-ing.
# It's a plain dict on purpose: this app has a single uvicorn worker, no
# sharing concerns, no need for anything more.
_auth_state: dict[str, str | bool] = {"ok": False, "message": "Not yet checked."}


@app.on_event("startup")
async def probe_auth_on_startup() -> None:
    """Run ``check_claude_auth`` once on boot so the homepage renders the
    correct banner state on the very first request, without making the
    user wait for the auth probe to finish.

    The probe is sync (``subprocess.run``), so we offload it to a thread
    to avoid blocking the event loop during startup.
    """
    ok, message = await asyncio.to_thread(check_claude_auth)
    _auth_state["ok"] = ok
    _auth_state["message"] = message
    print(f"[startup] claude auth: {'OK' if ok else 'NOT OK'} — {message}")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def _render_index(request: Request) -> HTMLResponse:
    """Render the home page with the current auth state.

    Used both for the initial GET and for any handler that wants to bounce
    the user back to the form. Templates only get the auth flags they need —
    no leaking of internal state.
    """
    return templates.TemplateResponse(
        request,
        "index.html",
        {"auth_ok": _auth_state["ok"], "auth_message": _auth_state["message"]},
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Home page: the URL input form, plus an auth banner when not signed in."""
    return _render_index(request)


@app.get("/healthz", response_class=JSONResponse)
async def healthz() -> JSONResponse:
    """JSON health endpoint — useful behind Cloudflare Tunnel for uptime checks.

    Returns whether Claude Code is signed in and the most recent probe
    message. This is the only place the auth status surfaces as a machine-
    readable response (the rest of the app renders HTML).
    """
    return JSONResponse(
        {"claude_authenticated": _auth_state["ok"], "message": _auth_state["message"]}
    )


# ---------------------------------------------------------------------------
# Summarise endpoint
# ---------------------------------------------------------------------------


@app.post("/summarise", response_class=HTMLResponse)
async def summarise_endpoint(request: Request, url: str = Form(...)) -> HTMLResponse:
    """Run the full summarisation pipeline and return an HTML fragment.

    The form on the index page POSTs here via htmx, which swaps the
    response into ``#result``. Both the success and the error responses
    return 200 with HTML (rather than 4xx) so htmx will swap them in —
    a non-2xx status would be silently dropped by the default htmx config.

    On a clean ``SummariseError`` whose message matches ``AUTH_INSTRUCTIONS``,
    the global auth flag is flipped back to false so the banner reappears
    on the next page load — picking up "user signed out mid-session" without
    a server restart.
    """
    try:
        # `summarise_url` is sync (it shells out to claude -p, blocking for
        # tens of seconds); offload to a thread so the event loop stays free.
        result = await asyncio.to_thread(summarise_url, url.strip())
    except SummariseError as e:
        message = str(e)
        # If this looks like an auth failure, flip the global flag so the
        # banner appears on the next page load.
        if message == AUTH_INSTRUCTIONS:
            _auth_state["ok"] = False
            _auth_state["message"] = message
        return templates.TemplateResponse(
            request, "error.html", {"message": message}
        )

    # Render the LLM-generated markdown body into HTML. The `extra` and
    # `sane_lists` extensions cover GitHub-flavoured tables, fenced code,
    # and consistent list parsing.
    body_html = md.markdown(result.body_markdown, extensions=["extra", "sane_lists"])

    # All summary links are external (YouTube deep-links) — open in a new
    # tab. Negative lookahead skips any <a> that already has target=
    # (defensive — markdown shouldn't add one but if a future extension does
    # we don't want to duplicate it).
    body_html = re.sub(
        r'<a (?![^>]*\btarget=)',
        '<a target="_blank" rel="noopener noreferrer" ',
        body_html,
    )

    return templates.TemplateResponse(
        request,
        "summary.html",
        {
            "watch_url": result.watch_url,
            "thumbnail_url": result.thumbnail_url,
            # `maxresdefault` is missing for some videos — pass an `hqdefault`
            # fallback so the template's <img onerror> can swap to a URL
            # that's universally present.
            "thumbnail_fallback": f"https://img.youtube.com/vi/{result.video_id}/hqdefault.jpg",
            "body_html": body_html,
            "segment_count": result.segment_count,
            "approx_tokens": result.approx_tokens,
        },
    )

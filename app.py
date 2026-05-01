"""FastAPI web UI wrapping ``summarise.py``.

A single-page app that takes a YouTube URL, calls into ``summarise_url``,
and streams progress + the rendered summary back to the browser. The
whole stack is intentionally small: FastAPI for routing, Jinja2 for
templates, Server-Sent Events for live progress, and Tailwind via CDN
for styling. No database, no users, no rate limiting — this is meant
to run on your own machine and be exposed (if at all) through a
Cloudflare Tunnel with Access in front of it.

**Run locally:**
    py -m uvicorn app:app --reload --port 8000

Then open http://127.0.0.1:8000.

**Routes:**
    GET  /                    — the form
    GET  /summarise/stream    — SSE: streams progress events ending in either
                                a ``result`` (rendered HTML) or ``error`` event.
                                The page's submit handler uses this.
    POST /summarise           — non-streaming fall-back: runs the pipeline
                                synchronously and returns the rendered
                                summary fragment in one response. Useful
                                for scripts/curl; not used by the page.
    GET  /healthz             — JSON health/auth status, for tunnels and uptime.

The page is gated on Claude Code being signed in: a banner appears when
auth is missing and the form is disabled until the user runs
``claude /login`` in another terminal and restarts the app.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import markdown as md
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from summarise import (
    AUTH_INSTRUCTIONS,
    SUPPORTED_MODELS,
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
    """Render the home page with the current auth state and the list of
    selectable models.

    Used both for the initial GET and for any handler that wants to bounce
    the user back to the form. Templates only get the values they need —
    no leaking of internal state.
    """
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "auth_ok": _auth_state["ok"],
            "auth_message": _auth_state["message"],
            "models": list(SUPPORTED_MODELS.keys()),  # ["default","sonnet",…]
        },
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
async def summarise_endpoint(
    request: Request,
    url: str = Form(...),
    model: str = Form("default"),
) -> HTMLResponse:
    """Non-streaming fall-back: run the pipeline and return the summary in one shot.

    The page itself uses ``GET /summarise/stream`` (Server-Sent Events) so the
    user sees live progress. This endpoint is kept around for scripted
    callers that just want the rendered HTML (e.g. ``curl -X POST -d url=…``).

    Both success and error responses return HTTP 200 with an HTML fragment —
    the caller can decide how to display it.

    On a clean ``SummariseError`` whose message matches ``AUTH_INSTRUCTIONS``,
    the global auth flag is flipped back to false so the banner reappears on
    the next page load — picking up "user signed out mid-session" without a
    server restart.
    """
    try:
        # `summarise_url` is sync (it shells out to claude -p, blocking for
        # tens of seconds); offload to a thread so the event loop stays free.
        result = await asyncio.to_thread(summarise_url, url.strip(), None, model)
    except SummariseError as e:
        message = str(e)
        if message == AUTH_INSTRUCTIONS:
            _auth_state["ok"] = False
            _auth_state["message"] = message
        return templates.TemplateResponse(
            request, "error.html", {"message": message}
        )

    return templates.TemplateResponse(request, "summary.html", _summary_context(result))


# ---------------------------------------------------------------------------
# Live-progress streaming (Server-Sent Events)
# ---------------------------------------------------------------------------


def _render_body_html(body_markdown: str) -> str:
    """Markdown → HTML, with all anchor tags forced to open in a new tab.

    The negative lookahead skips any ``<a>`` that already has ``target=``
    (defensive — the markdown extensions shouldn't add one, but if a future
    extension does, we don't want to double up).
    """
    html = md.markdown(body_markdown, extensions=["extra", "sane_lists"])
    return re.sub(
        r'<a (?![^>]*\btarget=)',
        '<a target="_blank" rel="noopener noreferrer" ',
        html,
    )


def _summary_context(result) -> dict:
    """Build the template context for the summary fragment.

    Centralised so both the regular POST endpoint and the streaming
    endpoint render identical output.
    """
    return {
        "watch_url": result.watch_url,
        "thumbnail_url": result.thumbnail_url,
        # `maxresdefault` is missing for some videos — pass an `hqdefault`
        # fallback so the template's <img onerror> can swap to a URL
        # that's universally present.
        "thumbnail_fallback": f"https://img.youtube.com/vi/{result.video_id}/hqdefault.jpg",
        "body_html": _render_body_html(result.body_markdown),
        "segment_count": result.segment_count,
        "approx_tokens": result.approx_tokens,
    }


def _sse(event: str, data: str) -> str:
    """Format a Server-Sent Events frame.

    Each ``data:`` line is one chunk; we split on newlines so multi-line
    payloads (e.g. rendered HTML) are reassembled correctly by the browser.
    """
    safe = data.replace("\r", "")
    lines = "\n".join(f"data: {line}" for line in safe.split("\n"))
    return f"event: {event}\n{lines}\n\n"


@app.get("/summarise/stream")
async def summarise_stream(
    request: Request,
    url: str,
    model: str = "default",
) -> StreamingResponse:
    """Stream the summarisation pipeline as Server-Sent Events.

    Emits one ``progress`` event per pipeline step (parsing → fetching →
    calling Claude → rendering) so the browser can show a live log while
    the slow ``claude -p`` call runs. Ends with either:
      - a ``result`` event whose data is JSON-encoded ``{"html": "..."}``
        containing the rendered summary fragment, or
      - an ``error`` event whose data is the user-facing error message.

    The summarise pipeline is sync (blocks on subprocess), so the worker
    runs in the default threadpool and bridges progress messages onto the
    asyncio event loop via ``call_soon_threadsafe``.
    """
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def on_progress(msg: str) -> None:
        # Called from the worker thread — hand the message off to the
        # event loop so the streamer can yield it.
        loop.call_soon_threadsafe(queue.put_nowait, ("progress", msg))

    def worker() -> None:
        try:
            result = summarise_url(
                url.strip(), on_progress=on_progress, model=model
            )
            loop.call_soon_threadsafe(queue.put_nowait, ("result", result))
        except SummariseError as e:
            message = str(e)
            # If this looks like an auth failure, flip the global flag so
            # the banner reappears on the next page load.
            if message == AUTH_INSTRUCTIONS:
                _auth_state["ok"] = False
                _auth_state["message"] = message
            loop.call_soon_threadsafe(queue.put_nowait, ("error", message))
        except Exception as e:
            # Catch-all so a bug doesn't leave the browser hanging on an
            # open SSE stream. Real traceback goes to the server log.
            import traceback
            traceback.print_exc()
            loop.call_soon_threadsafe(
                queue.put_nowait,
                ("error", f"Unexpected server error ({type(e).__name__}): {e}"),
            )

    asyncio.get_event_loop().run_in_executor(None, worker)

    async def event_stream():
        try:
            yield _sse("progress", "starting")
            while True:
                kind, data = await queue.get()
                if kind == "progress":
                    yield _sse("progress", data)
                elif kind == "error":
                    yield _sse("error", data)
                    return
                elif kind == "result":
                    html = templates.get_template("summary.html").render(
                        _summary_context(data)
                    )
                    yield _sse("progress", "rendering")
                    yield _sse("result", json.dumps({"html": html}))
                    return
        except asyncio.CancelledError:
            # Browser disconnected — let the worker finish silently.
            pass

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",  # tells nginx/Caddy not to buffer the stream
    }
    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)

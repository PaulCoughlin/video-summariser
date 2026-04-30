"""FastAPI web UI wrapping summarise.py.

Run:  py -m uvicorn app:app --reload --port 8000
Then: http://127.0.0.1:8000
"""

from __future__ import annotations

import asyncio
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

# Auth state for the running server. Probed at startup and refreshed whenever a
# real summarise call surfaces an auth error (so re-signing in is picked up
# without restarting the server, after one failed attempt).
_auth_state: dict[str, str | bool] = {"ok": False, "message": "Not yet checked."}


@app.on_event("startup")
async def probe_auth_on_startup() -> None:
    ok, message = await asyncio.to_thread(check_claude_auth)
    _auth_state["ok"] = ok
    _auth_state["message"] = message
    print(f"[startup] claude auth: {'OK' if ok else 'NOT OK'} — {message}")


def _render_index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {"auth_ok": _auth_state["ok"], "auth_message": _auth_state["message"]},
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return _render_index(request)


@app.get("/healthz", response_class=JSONResponse)
async def healthz() -> JSONResponse:
    """JSON health endpoint — useful behind Cloudflare Tunnel for uptime checks."""
    return JSONResponse({"claude_authenticated": _auth_state["ok"], "message": _auth_state["message"]})


@app.post("/auth/refresh", response_class=HTMLResponse)
async def refresh_auth(request: Request) -> HTMLResponse:
    """Re-probe auth state. Lets the user click 'Re-check' after signing in
    without having to restart the server."""
    ok, message = await asyncio.to_thread(check_claude_auth)
    _auth_state["ok"] = ok
    _auth_state["message"] = message
    return _render_index(request)


@app.post("/summarise", response_class=HTMLResponse)
async def summarise_endpoint(request: Request, url: str = Form(...)) -> HTMLResponse:
    try:
        result = await asyncio.to_thread(summarise_url, url.strip())
    except SummariseError as e:
        message = str(e)
        # If this looks like an auth failure, flip the global flag so the
        # banner appears on the next page load.
        if message == AUTH_INSTRUCTIONS:
            _auth_state["ok"] = False
            _auth_state["message"] = message
        return templates.TemplateResponse(
            request, "error.html", {"message": message}, status_code=400
        )

    body_html = md.markdown(result.body_markdown, extensions=["extra", "sane_lists"])
    return templates.TemplateResponse(
        request,
        "summary.html",
        {
            "watch_url": result.watch_url,
            "thumbnail_url": result.thumbnail_url,
            "thumbnail_fallback": f"https://img.youtube.com/vi/{result.video_id}/hqdefault.jpg",
            "body_html": body_html,
            "segment_count": result.segment_count,
            "approx_tokens": result.approx_tokens,
        },
    )

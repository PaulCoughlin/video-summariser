"""FastAPI web UI — public/VPS edition (OpenRouter, model picker, rate limit, cache).

Run locally:  py -m uvicorn app:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections import OrderedDict, deque
from pathlib import Path
from threading import Lock

import markdown as md
from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from models import DEFAULT_MODEL_ID, MODELS, get as get_model, is_known
from summarise import (
    SummariseError,
    check_openrouter_auth,
    summarise_url,
)

load_dotenv()

BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=BASE_DIR / "templates")

app = FastAPI(title="Video Summariser (Public)")

# ---- per-IP rate limit -------------------------------------------------------
RATE_LIMIT_PER_DAY = int(os.environ.get("RATE_LIMIT_PER_DAY", "10"))
WINDOW_SECONDS = 24 * 60 * 60
_rate_buckets: dict[str, deque[float]] = {}
_rate_lock = Lock()


def _client_ip(request: Request) -> str:
    # Trust X-Forwarded-For when present (Coolify/Caddy/Cloudflare set it).
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _check_rate_limit(ip: str) -> tuple[bool, int]:
    """Returns (allowed, remaining)."""
    now = time.time()
    with _rate_lock:
        bucket = _rate_buckets.setdefault(ip, deque())
        while bucket and now - bucket[0] > WINDOW_SECONDS:
            bucket.popleft()
        if len(bucket) >= RATE_LIMIT_PER_DAY:
            return False, 0
        bucket.append(now)
        return True, RATE_LIMIT_PER_DAY - len(bucket)


# ---- in-memory cache ---------------------------------------------------------
CACHE_MAX = 100
_cache: OrderedDict[tuple[str, str], dict] = OrderedDict()
_cache_lock = Lock()


def _cache_get(key: tuple[str, str]) -> dict | None:
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    return None


def _cache_put(key: tuple[str, str], value: dict) -> None:
    with _cache_lock:
        _cache[key] = value
        _cache.move_to_end(key)
        while len(_cache) > CACHE_MAX:
            _cache.popitem(last=False)


# ---- auth state --------------------------------------------------------------
_auth_state: dict = {"ok": False, "message": "Not yet checked."}


@app.on_event("startup")
async def probe_auth_on_startup() -> None:
    ok, message = await asyncio.to_thread(check_openrouter_auth)
    _auth_state["ok"] = ok
    _auth_state["message"] = message
    print(f"[startup] OpenRouter auth: {'OK' if ok else 'NOT OK'} — {message}")


# ---- markdown helpers --------------------------------------------------------
def _render_md(body: str) -> str:
    html = md.markdown(body, extensions=["extra", "sane_lists"])
    return re.sub(
        r'<a (?![^>]*\btarget=)',
        '<a target="_blank" rel="noopener noreferrer" ',
        html,
    )


# ---- routes ------------------------------------------------------------------
def _index_context() -> dict:
    return {
        "auth_ok": _auth_state["ok"],
        "auth_message": _auth_state["message"],
        "models": MODELS,
        "default_model_id": DEFAULT_MODEL_ID,
        "rate_limit_per_day": RATE_LIMIT_PER_DAY,
    }


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html", _index_context())


@app.get("/healthz", response_class=JSONResponse)
async def healthz() -> JSONResponse:
    return JSONResponse({
        "openrouter_authenticated": _auth_state["ok"],
        "message": _auth_state["message"],
        "cache_entries": len(_cache),
        "tracked_ips": len(_rate_buckets),
    })


def _build_summary_ctx(result, remaining: int | None, from_cache: bool) -> dict:
    model_meta = get_model(result.model_id)
    return {
        "watch_url": result.watch_url,
        "thumbnail_url": result.thumbnail_url,
        "thumbnail_fallback": f"https://img.youtube.com/vi/{result.video_id}/hqdefault.jpg",
        "body_html": _render_md(result.body_markdown),
        "segment_count": result.segment_count,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_cost_usd": result.total_cost_usd,
        "model_label": model_meta.label if model_meta else result.model_id,
        "remaining_today": remaining,
        "from_cache": from_cache,
    }


def _sse(event: str, data: str) -> str:
    """Format a Server-Sent Events frame. Each `data:` line is one chunk;
    multi-line data is split so the browser reassembles correctly."""
    safe = data.replace("\r", "")
    lines = "\n".join(f"data: {line}" for line in safe.split("\n"))
    return f"event: {event}\n{lines}\n\n"


@app.get("/summarise/stream")
async def summarise_stream(
    request: Request,
    url: str,
    model: str = DEFAULT_MODEL_ID,
) -> StreamingResponse:
    """Server-Sent Events endpoint that streams progress + final summary HTML.

    Events emitted:
      progress  — single line of human-readable status (multiple)
      result    — JSON-encoded {"html": "..."} when done
      error     — error message; closes the stream
    """
    ip = _client_ip(request)
    url = url.strip()

    async def stream_one_event(event: str, data: str):
        yield _sse(event, data)

    if not is_known(model):
        return StreamingResponse(
            stream_one_event("error", "Unknown model selected."),
            media_type="text/event-stream",
        )

    cache_key = (url, model)
    cached = _cache_get(cache_key)
    if cached:
        cached_html = templates.get_template("summary.html").render(
            {**cached, "from_cache": True}
        )
        async def cached_stream():
            yield _sse("progress", "served from cache")
            yield _sse("result", json.dumps({"html": cached_html}))
        return StreamingResponse(cached_stream(), media_type="text/event-stream")

    allowed, remaining = _check_rate_limit(ip)
    if not allowed:
        return StreamingResponse(
            stream_one_event(
                "error",
                f"Rate limit reached: {RATE_LIMIT_PER_DAY} summaries per IP per 24h.",
            ),
            media_type="text/event-stream",
        )

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def on_progress(msg: str) -> None:
        # Called from the worker thread; hand off to the event loop.
        loop.call_soon_threadsafe(queue.put_nowait, ("progress", msg))

    def worker() -> None:
        try:
            res = summarise_url(url, model, on_progress=on_progress)
            loop.call_soon_threadsafe(queue.put_nowait, ("result", res))
        except SummariseError as e:
            loop.call_soon_threadsafe(queue.put_nowait, ("error", str(e)))
        except Exception as e:
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
                    # Refund the rate-limit slot since we never returned a result.
                    with _rate_lock:
                        bucket = _rate_buckets.get(ip)
                        if bucket:
                            bucket.pop()
                    yield _sse("error", data)
                    return
                elif kind == "result":
                    ctx = _build_summary_ctx(data, remaining, from_cache=False)
                    html = templates.get_template("summary.html").render(ctx)
                    _cache_put(cache_key, ctx)
                    yield _sse("progress", "rendering")
                    yield _sse("result", json.dumps({"html": html}))
                    return
        except asyncio.CancelledError:
            # Client disconnected — let the worker finish silently.
            pass

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",  # tells nginx/Caddy not to buffer
    }
    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)


@app.get("/diag", response_class=JSONResponse)
async def diag() -> JSONResponse:
    """Container-side network diagnostic — DNS + proxy reachability.

    Useful when YouTube transcript fetches fail with cryptic errors and we
    need to know whether DNS or the proxy itself is the problem. Exposed
    publicly because it doesn't reveal anything sensitive (proxy URL is
    masked).
    """
    import socket
    from urllib.parse import urlparse

    def _resolve(host: str) -> dict:
        try:
            return {"host": host, "ok": True, "ipv4": socket.gethostbyname(host)}
        except Exception as e:
            return {"host": host, "ok": False, "error": f"{type(e).__name__}: {e}"}

    hosts = ["openrouter.ai", "www.youtube.com", "p.webshare.io",
             "proxy.webshare.io", "ipv4.webshare.io"]
    dns = [_resolve(h) for h in hosts]

    proxy_url = os.environ.get("PROXY_URL", "").strip()
    proxy_check: dict = {"configured": bool(proxy_url)}
    if proxy_url:
        try:
            parsed = urlparse(proxy_url)
            proxy_check["host"] = parsed.hostname
            proxy_check["port"] = parsed.port
            # Try resolving the proxy host
            proxy_check["proxy_host_resolves"] = _resolve(parsed.hostname or "")
            # Try a HEAD request through the proxy to a Webshare echo endpoint
            import httpx
            try:
                r = httpx.get(
                    "https://ipv4.webshare.io/",
                    proxy=proxy_url,
                    timeout=15,
                )
                proxy_check["echo_status"] = r.status_code
                proxy_check["echo_body"] = r.text.strip()[:120]
            except Exception as e:
                proxy_check["echo_error"] = f"{type(e).__name__}: {e}"
        except Exception as e:
            proxy_check["parse_error"] = f"{type(e).__name__}: {e}"

    # /etc/resolv.conf to see what DNS server the container is using
    resolv = ""
    try:
        with open("/etc/resolv.conf") as f:
            resolv = f.read()
    except Exception as e:
        resolv = f"(couldn't read: {e})"

    return JSONResponse({"dns": dns, "proxy": proxy_check, "resolv_conf": resolv})


@app.get("/diag/proxy-test", response_class=JSONResponse)
async def diag_proxy_test() -> JSONResponse:
    """Test the configured proxy against several proxy HOSTS (the hostname
    plus a couple of literal IPs) and several target URLs, using requests —
    the same library youtube-transcript-api uses internally.

    Lets us isolate whether the failure is DNS for the proxy host, the
    proxy itself, the targets (YouTube specifically), or our app's wrapping.
    """
    import requests
    from urllib.parse import urlparse

    proxy_url = os.environ.get("PROXY_URL", "").strip()
    if not proxy_url:
        return JSONResponse({"error": "PROXY_URL is not set."})

    parsed = urlparse(proxy_url)
    auth = f"{parsed.username}:{parsed.password}" if parsed.username else ""
    if not auth:
        return JSONResponse({"error": "PROXY_URL is missing user:pass."})

    proxy_hosts = [
        "p.webshare.io:80",        # the canonical hostname (DNS-dependent)
        "169.150.245.196:80",       # literal IP 1
        "37.9.62.134:80",           # literal IP 2
    ]
    targets = [
        "https://ipv4.webshare.io/",        # echoes the egress IP
        "https://paulcoughlin.com/",         # neutral target
        "https://www.google.com/",           # well-known
        "https://www.youtube.com/",          # the real test
        "https://www.youtube.com/watch?v=P60LqQg1RH8",
    ]

    def _hit(proxy: str, url: str) -> dict:
        try:
            r = requests.get(
                url,
                proxies={"http": proxy, "https": proxy},
                timeout=20,
                allow_redirects=False,
            )
            return {
                "target": url,
                "status": r.status_code,
                "bytes": len(r.content),
                "snippet": r.text[:160] if r.text else "",
            }
        except Exception as e:
            return {"target": url, "error": f"{type(e).__name__}: {str(e)[:200]}"}

    out = []
    for host in proxy_hosts:
        proxy = f"http://{auth}@{host}"
        out.append({
            "proxy_host": host,
            "results": [_hit(proxy, u) for u in targets],
        })

    return JSONResponse({"tests": out})


@app.post("/summarise", response_class=HTMLResponse)
async def summarise_endpoint(
    request: Request,
    url: str = Form(...),
    model: str = Form(DEFAULT_MODEL_ID),
) -> HTMLResponse:
    ip = _client_ip(request)

    if not is_known(model):
        return templates.TemplateResponse(
            request, "error.html", {"message": "Unknown model selected."}
        )

    # Cache check happens BEFORE the rate-limit decrement: same video+model
    # served from cache shouldn't cost the user a daily slot.
    cache_key = (url.strip(), model)
    cached = _cache_get(cache_key)
    if cached:
        return templates.TemplateResponse(
            request, "summary.html", {**cached, "from_cache": True}
        )

    allowed, remaining = _check_rate_limit(ip)
    if not allowed:
        return templates.TemplateResponse(
            request,
            "error.html",
            {"message": f"Rate limit reached: {RATE_LIMIT_PER_DAY} summaries per IP per 24h. Come back later."},
        )

    try:
        result = await asyncio.to_thread(summarise_url, url.strip(), model)
    except SummariseError as e:
        # Don't burn a rate-limit slot for failed calls — refund it.
        with _rate_lock:
            bucket = _rate_buckets.get(ip)
            if bucket:
                bucket.pop()
        return templates.TemplateResponse(
            request, "error.html", {"message": str(e)}
        )
    except Exception as e:
        # Last-line defence so the user sees a styled message rather than
        # FastAPI's plain "Internal Server Error" text. Log to stderr so the
        # actual traceback shows in Coolify's logs.
        import traceback
        traceback.print_exc()
        with _rate_lock:
            bucket = _rate_buckets.get(ip)
            if bucket:
                bucket.pop()
        return templates.TemplateResponse(
            request,
            "error.html",
            {"message": f"Unexpected server error ({type(e).__name__}): {e}"},
        )

    ctx = _build_summary_ctx(result, remaining, from_cache=False)
    _cache_put(cache_key, ctx)
    return templates.TemplateResponse(request, "summary.html", ctx)

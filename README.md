# video-summariser

Paste a YouTube URL → get a TL;DR, key takeaways, and clickable timestamped moments.

Two ways to use it:
- A CLI that prints markdown to stdout — `summarise.py`
- A small FastAPI web UI — `app.py`

Both shell out to your local **Claude Code** CLI (`claude -p`), so the LLM call uses your existing Claude subscription rather than a paid API key.

## How it works

```
URL → youtube-transcript-api → transcript with [M:SS|Ns] markers
                              ↓
                  prompt template (TL;DR + 5–8 takeaways + ≥3 deep-linked moments)
                              ↓
                       claude -p (subscription)
                              ↓
                     markdown out, rendered to HTML in the web UI
```

The transcript is sent to Claude in a single shot — chunking and map-reduce aren't needed for any video that fits in Claude's context window (90-minute podcasts are ~12–18k tokens; the limit is 200k). Each transcript line is annotated with both `M:SS` and the same time in raw seconds, so Claude can write `&t=Ns` deep-links into the watch URL without doing arithmetic.

## Requirements

- Python 3.10+
- [Claude Code](https://docs.anthropic.com/claude/docs/claude-code) installed and signed in (`claude` on your PATH; run `claude` once interactively to authenticate)
- Network access to YouTube (the transcript API uses captions, so cloud IPs are commonly blocked — run this on your own machine)

## Install

```bash
git clone https://github.com/<you>/video-summariser.git
cd video-summariser
py -m pip install -r requirements.txt
```

## CLI usage

```bash
# Verify Claude Code is installed + signed in:
py summarise.py --check

# Summarise a video — writes <video-id>.md in the current directory:
py summarise.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Custom output path:
py summarise.py "https://youtu.be/VIDEO_ID" -o my-notes.md

# Or print to stdout (for piping):
py summarise.py "https://youtu.be/VIDEO_ID" -o -
```

If you're not signed in, you'll get:

> error: Claude Code isn't signed in on this machine. Open a terminal, run `claude` once, complete the sign-in, then try again.

## Web UI

```bash
py -m uvicorn app:app --reload --port 8000
```

Open http://127.0.0.1:8000 in your browser.

The page probes Claude Code's auth status on startup. If you're not signed in, the form is disabled and a banner explains how to fix it; click **Re-check authentication** after running `claude` to clear it.

### Exposing it publicly

The app intentionally has no auth/rate-limiting and runs on your own machine using your subscription, so don't put it on the public internet without a gate. The intended pattern is:

```bash
cloudflared tunnel --url http://localhost:8000
```

…with [Cloudflare Access](https://developers.cloudflare.com/cloudflare-one/policies/access/) in front for a password gate. Don't host this on a cloud provider — Claude Code's subscription auth only works on your local machine, and YouTube blocks transcript requests from datacenter IPs.

## Health endpoint

`GET /healthz` returns JSON:

```json
{ "claude_authenticated": true, "message": "Authenticated." }
```

Useful for uptime checks behind the tunnel.

## Output shape

Every summary contains, in order:

1. A clickable thumbnail linking to the video
2. **TL;DR** — 2–3 sentences
3. **Key Takeaways** — 5–8 single-sentence bullets
4. **Watch These Moments** — at least 3 entries, each a clickable `&t=Ns` deep-link with a one-line description

Timestamps are constrained by the prompt to come from lines that appear verbatim in the transcript — in practice, hallucinated timestamps are rare-to-nonexistent.

## Limitations

- **English captions only.** The transcript fetcher tries `en`, `en-US`, `en-GB` and gives up otherwise.
- **No fallback to audio transcription.** If the video has captions disabled (common on music videos), the script errors cleanly rather than reaching for Whisper.
- **Single-shot only.** A video whose transcript exceeds the model's context window will fail. None of the test videos in this README hit that limit.
- **Subprocess timeout: 3 minutes.** Long videos near the context limit can occasionally bump up against it.

## License

MIT — see [LICENSE](LICENSE).

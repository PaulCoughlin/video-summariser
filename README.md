# Video Summariser
## Agentic Labs Challenge

Paste a YouTube URL → get a TL;DR, key takeaways, and clickable timestamped moments.

Two ways to use it:
- A CLI that prints markdown to stdout — `summarise.py`
- A small FastAPI web UI — `app.py`

Both shell out to your local **Claude Code** CLI (`claude -p`), so the LLM call uses your existing Claude subscription rather than a paid API key.

## Notes on the brief

> *"The interesting part is how you handled the 30+ minute videos, so tell us what you picked: chunking, big context, map-reduce, or something weirder."*

**Picked: single-shot, big context window. No chunking, no map-reduce.**

The temptation with long videos is to immediately reach for map-reduce — chunk the transcript, summarise each chunk, then summarise the summaries. It's the classically "right-feeling" answer. But it's the wrong one here, and the math is worth doing once.

A 90-minute podcast transcript is **~12-18k tokens** (verified empirically across the test videos in this repo: a 31-minute talk came in at ~10k, a 42-minute build at ~12k). Claude Sonnet's context window is **200k tokens**. That's a 10× headroom over the *worst case* of the brief's hard requirement. Even a three-hour conversation slots in comfortably. Chunking solves a problem this app doesn't have.

The cost of chunking-when-you-don't-need-to is real:

- **Each chunk-summary loses global context.** The model can't tell which moment is *the* turning point of a 90-minute talk if it only sees 5 minutes at a time. The "Watch These Moments" section in particular degrades to "here's the most interesting bit of each chunk" — not the same thing.
- **Tone and emphasis drift between chunks.** The TL;DR ends up reading like a bullet list of segments rather than a coherent thesis.
- **More LLM calls = more chances to hallucinate.** Each round of summarisation is another chance for invented facts or drifted timestamps.
- **More code, more bugs.** Chunk-boundary heuristics, overlap windows, summary-of-summaries plumbing — all surface area for things to go subtly wrong.

So the rule applied here: **only chunk when forced by the context window**, not because the input "feels long". For the 30-minute minimum the brief specifies, single-shot wins on every axis (quality, cost, complexity).

If a video genuinely overflowed (it'd need to be ~5+ hours of dense speech), the next move would still not be map-reduce — it'd be a model with a larger context (Gemini's 1M, etc.) or sliding-window summarisation that preserves global cross-references. Map-reduce is a last resort, not a default.

### Other design choices worth flagging

- **Subscription-priced, not API-priced.** Shelling out to `claude -p` means the LLM call costs nothing at the margin (uses your Max subscription) and there's no API key to manage. Trade-off: only works on a machine you've signed into Claude Code on, so it can't be deployed headless without swapping the back end (which I tried — see commit history — and the YouTube transcript API actively blocks datacenter IPs anyway, so "deploy publicly" stops being free for unrelated reasons).
- **Grounded timestamps.** Every transcript line is annotated as `[M:SS|Ns]` — both the human-readable timecode *and* the same moment in raw seconds. The prompt instructs the model to use both verbatim from the transcript, so it doesn't have to do arithmetic and can't quietly drift. Combined with the deep-link template (`&t=Ns`), every "Watch These Moments" entry lands within the right second. Empirically, 6/6 timestamps were verbatim across the test videos.
- **Strict output format, no preamble.** The prompt bans "Here is your summary" boilerplate and fixes the section names so the markdown is paste-ready and the body could be machine-parsed if anything ever wanted to.
- **A subprocess, not a library.** Calling `claude -p` rather than the Python SDK keeps this a 100-ish-line script. The only real downside is a single 30-90s blocking call per request — fine for a personal tool, and mitigated in the web UI by streaming pipeline progress over Server-Sent Events so the user sees a live log instead of staring at a static spinner.

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

> error: Claude Code isn't signed in on this machine. Stop the app (Ctrl+C in its terminal), run `claude /login` and complete the sign-in, then start the app again.

## Web UI

```bash
py -m uvicorn app:app --reload --port 8000
```

Open http://127.0.0.1:8000 in your browser.

While the summary is generating, a small log card streams the pipeline status in real time over Server-Sent Events:

```
[14:32:11] starting
[14:32:11] parsing video URL
[14:32:11] fetching transcript for P60LqQg1RH8
[14:32:13] got 768 segments (~10154 tokens) — calling Claude
[14:33:04] summary received
[14:33:04] rendering
```

When Claude responds, the log card is replaced with the rendered summary (thumbnail, TL;DR, takeaways, deep-linked moments). If anything fails mid-flow, an error card appears above the log so you can see exactly which step blew up.

The page probes Claude Code's auth status on startup. If you're not signed in, the form is disabled and a banner walks you through the fix: stop the app, run `claude /login` in another terminal, restart the app.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | The form |
| `GET` | `/summarise/stream?url=…` | SSE: live progress + final rendered summary. The page uses this. |
| `POST` | `/summarise` | Non-streaming fall-back: runs the pipeline and returns the rendered HTML in one response. Useful for `curl` / scripts. |
| `GET` | `/healthz` | JSON health/auth status, for tunnel uptime checks. |

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

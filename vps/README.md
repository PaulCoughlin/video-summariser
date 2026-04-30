# Video Summariser — VPS edition - this is just for my own experiment..

The same web UI as the root project, but it calls **OpenRouter** instead of the local Claude Code CLI — so it runs headless on a VPS or any container platform with just an API key.

Differences from the root version:

| Capability | Root (`../app.py`) | VPS (this folder) |
|---|---|---|
| LLM | `claude -p` subprocess (subscription) | OpenRouter Chat Completions (API key) |
| Model | Whatever `claude` defaults to | Dropdown — switch per request |
| Hosting | Your own machine + Cloudflare Tunnel | Any VPS, Docker host, or PaaS |
| Per-IP rate limit | None | 10/day (configurable) |
| Cache | None | In-memory, by `(url, model)` |

The transcript fetcher, prompt template, and overall UI are identical.

## Run locally first

```bash
cd vps
py -m pip install -r requirements.txt
cp .env.example .env
# edit .env and paste your key from https://openrouter.ai/keys
py -m uvicorn app:app --reload --port 8000
```

Then open <http://127.0.0.1:8000>. The startup log should print:

```
[startup] OpenRouter auth: OK — Authenticated as `…` — pay-as-you-go.
```

If it says `NOT OK`, the banner on the page will tell you what's wrong.

## Picking a model

The dropdown is populated from [`models.py`](models.py). Edit that file to add or reorder models — every entry is just an OpenRouter slug + display label. Slugs come from <https://openrouter.ai/models>.

Reasonable starting picks:

- **DeepSeek V3.1 (`:free`)** — best free option; rate-limited to ~50/day per key without credits, ~1000/day with $10+ in lifetime credit.
- **Claude 3.5 Haiku** — paid, ~$0.80/1M input. Excellent format adherence (rarely hallucinates timestamps).
- **Claude Sonnet 4.5** — paid, ~$3/1M input. Best overall, ~1¢ per long-video summary.

For each summary the footer shows tokens used and exact cost — useful when comparing models side-by-side.

## Hostinger VPS + Coolify deploy

Why this combo: Hostinger gives you a Linux VPS for ~$5–8/month, Coolify gives you a Heroku-style "git push to deploy" UI on top of it for free, and the GitHub repo is already Dockerised so deployment is a few clicks.

### 1. Provision the VPS

In Hostinger:

1. **Hosting → VPS → Get Plan.** A KVM 2 (2 vCPU, 8 GB RAM, ~$8/month) is more than enough; KVM 1 works if you're tight. [Click here for 20% discount (my affiliate link)](https://www.hostinger.com/cart?product=vps%3Avps_kvm_2&period=12&referral_type=cart_link&REFERRALCODE=IYWZPAUL29MC&referral_id=019dddb9-d104-7097-8e29-5e8051f32a1c)
2. **Choose the OS template: Ubuntu 24.04 with Coolify** (Hostinger has a one-click image). If you don't see it, pick plain Ubuntu 24.04 and install Coolify manually in step 2.
3. Set a root password / add your SSH key.
4. Wait ~2 minutes for provisioning. Note the public IP.

### 2. Install Coolify (skip if you used the one-click image)

SSH in:

```bash
ssh root@YOUR_VPS_IP
curl -fsSL https://cdn.coollabs.io/coolify/install.sh | sudo bash
```

The installer prints a URL like `http://YOUR_VPS_IP:8000` — open it, register the admin user. (Coolify's UI port is also 8000; that's fine, the app will be exposed on its own subdomain via Coolify's built-in proxy.)

### 3. Point a domain at the VPS

In your DNS provider, create an A record:

```
summarise.yourdomain.com → YOUR_VPS_IP
```

Wait for it to propagate (~1–10 min).

### 4. Add the app in Coolify

In the Coolify UI:

1. **Projects → New Project**, name it `video-summariser`.
2. **+ New → Application → Public Repository**.
3. **Repository URL**: `https://github.com/PaulCoughlin/video-summariser`
4. **Branch**: `main`
5. **Build Pack**: **Dockerfile**
6. **Base Directory**: `vps` *(critical — the repo contains a non-VPS version at the root)*
7. **Dockerfile Location**: `Dockerfile` (relative to the base directory, so it resolves to `vps/Dockerfile`)
8. **Port**: `8000`
9. **Domain**: `https://summarise.yourdomain.com` — Coolify will provision a Let's Encrypt cert automatically.

### 5. Set environment variables

Still in the application's settings, **Environment Variables** tab:

| Key | Value | Mark as secret? |
|---|---|---|
| `OPENROUTER_API_KEY` | sk-or-v1-... | ✅ |
| `OPENROUTER_REFERER` | `https://summarise.yourdomain.com` | no |
| `OPENROUTER_APP_TITLE` | `Video Summariser` | no |
| `RATE_LIMIT_PER_DAY` | `10` (or whatever) | no |

### 6. Deploy

Hit **Deploy**. First build takes ~3 minutes (pulling the base image, installing deps). Subsequent builds are cached and complete in ~30s.

Coolify's logs view streams build + runtime output. If anything's wrong, the most likely culprits are:
- Missing `OPENROUTER_API_KEY` → banner appears, form is disabled. Add the env var, redeploy.
- Build fails on `pip install` → check Python version in `Dockerfile` matches your dep constraints.
- 502 from Coolify proxy → port mismatch. Confirm app is on `8000` and Coolify is configured for `8000`.

### 7. Verify

```bash
curl https://summarise.yourdomain.com/healthz
# {"openrouter_authenticated":true,"message":"Authenticated...","cache_entries":0,"tracked_ips":0}
```

Then try a real video in the UI.

## YouTube IP-block reality check

YouTube's transcript endpoint blocks many datacenter IPs. Whether your VPS works is luck-of-the-draw — try a handful of videos first.

If the page shows "YouTube blocked the transcript request from this server's IP":

1. **Add a residential proxy** *(easiest)*. Sign up at [webshare.io](https://www.webshare.io/), pick the **Residential** plan (~$3/month), and grab your proxy username + password. In Coolify, add two env vars and redeploy:
   - `WEBSHARE_PROXY_USERNAME` = (your Webshare username)
   - `WEBSHARE_PROXY_PASSWORD` = (your Webshare password)

   The code auto-detects these and routes transcript requests through the residential pool. No code change needed.
2. **Try a different VPS region** in Hostinger's dashboard (free move within the first 30 days). Some IP ranges are less blocked than others — luck of the draw.
3. **Use cookies** *(free but fragile)*. Log into YouTube in a browser, export the cookies for `youtube.com` to a file, mount it into the container, and modify [`summarise.py`](summarise.py) to pass it via `YouTubeTranscriptApi(http_client=...)`. Cookies expire — expect to refresh occasionally.

## Cost notes

- **Free models on OpenRouter**: 50 requests/day per key without credits, 1000/day after $10 lifetime credits. Rate limits are model-specific and shared across all your projects.
- **Paid models**: per-summary cost shown in the footer of every result. For a typical 30-min video at ~12k input tokens, expect ~½¢ on Claude Haiku, ~3¢ on Claude Sonnet, ~0.3¢ on DeepSeek.
- **Caching means repeats are free**: same `(url, model)` pair returns instantly without burning a daily slot. Cache resets on container restart.

## Limitations

Same as the root version, plus:
- Free models follow the strict-format prompt about 70% of the time vs. Sonnet's ~99% — expect occasionally hallucinated timestamps and wonkier prose. Switch models per request from the dropdown to compare.
- No persistent storage — rate-limit buckets and cache reset on every redeploy.
- `RATE_LIMIT_PER_DAY` is per-IP from the request's first `X-Forwarded-For` hop. Coolify's reverse proxy sets this correctly; if you put another proxy in front, double-check.

## License

MIT (inherits from the parent repo).

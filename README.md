# VideoForge

Turn any topic into an animated or cartoon video using multi-scene AI generation, scene planning, subtitle rendering with Devanagari text shaping, and crossfade video composition.

---

## Quickstart (Windows)

1. Open PowerShell or Command Prompt in `videoforge`:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

2. Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.

---

## Features Implemented

### Phase 1: Pages, Multi-Language & Design System
- **11 Server-Rendered Jinja2 Pages**: Landing, Pricing, Signup, Login, Dashboard, Create Video, My Videos, Settings, Terms, Privacy, Content Policy.
- **Modern Dark UI**: Distinct dark background with high-contrast orange accent (`#FF6B35`). No generic purple-blue gradients.
- **Bilingual Interface**: English and Hindi interface toggle with instant language switching (`translations.py`).
- **Dynamic Footer**: Automatically renders current year.

### Phase 2: Accounts, Security & Isolation
- **Database**: SQLite (`videoforge.db`) with SQLAlchemy ORM (seamlessly portable to PostgreSQL).
- **Security**:
  - Passwords hashed with `bcrypt` (12 rounds).
  - Secure, HTTP-only signed session cookies via `itsdangerous`.
  - CSRF protection across all state-modifying POST requests.
  - Rate limiting on login attempts (5 attempts per minute).
- **Development Mode**: Email verification and password reset links printed to server logs and displayed on screen banners.
- **User Isolation**: Users only have access to their own projects and videos.

### Phase 3: Create-Video Flow & Composition
- **Scene Planning**: Automatically breaks topics into sequential ~5-second scenes using Claude (via Anthropic API if key is present) or rich template fallbacks.
- **Interactive Scene Review**: Edit, add, delete, and reorder prompts and captions before generating.
- **Credit Accounting**:
  - Free scene planning; credits deducted only when queued for generation.
  - Automatic refund for any failed clip.
- **Hindi / Devanagari Typography**: High-precision text shaping with `uharfbuzz` and `freetype-py` using `NotoSansDevanagari.ttf`. Ligatures and conjuncts (e.g. "क्या कोई गिरफ्तार हुआ?") join correctly.
- **Video Composition**: Concatenates clips with 0.3-second crossfades using `imageio-ffmpeg`.
- **Audio & Subtitle Exports**:
  - Dual MP4 output (with music and clean no-music version).
  - SRT subtitle export.
  - Raw clips ZIP archive.

---

## Running with Real Models (Replicate)

1. Edit `.env`:
```env
PROVIDER=replicate
REPLICATE_API_TOKEN=your_replicate_token_here
REPLICATE_MODEL=wan-video/wan-2.1-1.3b
```
*(Or `lightricks/ltx-video`)*

2. Input parameters are automatically mapped to the model's verified schema on Replicate.

---

## Running Automated Tests

```powershell
# Phase 2 test suite (auth, rate limiting, data isolation)
venv\Scripts\python test_phase2.py

# Phase 3 test suite (scene planning, crossfade stitching, Hindi captions, refunds)
venv\Scripts\python test_phase3.py
```

---

## Deploying to Production

1. In `.env` set `APP_ENV=production`, a strong `SESSION_SECRET` (the app refuses to start without one),
   `APP_BASE_URL=https://your-domain.com`, the `SMTP_*` settings, and a real `PROVIDER` (not `mock`).
2. Serve over **HTTPS** (Caddy, Nginx, or your host's proxy). Cookies are marked `Secure` in production.
3. Run **without** `--reload` and with **one worker** (the job queue and login rate-limit live in memory):

```powershell
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --proxy-headers --forwarded-allow-ips="*"
```

4. Back up `videoforge.db` and the `videos/` folder regularly. Never commit `.env`.

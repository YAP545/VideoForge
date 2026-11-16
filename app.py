"""VideoForge — Main Application.

PHASE 1: Jinja2 Templates, Design system, English/Hindi language toggle.
PHASE 2: Full user accounts with SQLAlchemy, Bcrypt password hashing,
         HTTP-only session cookies, CSRF protection, login rate-limiting,
         development-mode email verification & password reset links,
         complete user data isolation, and admin role support.
PHASE 3: Complete Create-Video flow:
         - Scene planning with Anthropic API or intelligent template fallbacks.
         - User scene editing, ordering, adding, deleting.
         - Credit cost validation and per-clip charging.
         - Automatic credit refund upon clip generation failures.
         - High precision Devanagari text shaping using HarfBuzz + FreeType + Noto Sans Devanagari.
         - Video crossfade stitching via imageio-ffmpeg.
         - Background music upload & mixing.
         - Dual output: final video with music and clean video without music.
         - SRT subtitle generation and Raw clips ZIP package.
"""
import unicode_compat
import hashlib
import hmac
import json
import logging
from html import escape as html_escape
import os
import queue
import re
import secrets
import shutil
import threading
import time
import uuid
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Union

from dotenv import load_dotenv

BASE = Path(__file__).parent
load_dotenv(BASE / ".env")

logger = logging.getLogger("videoforge")

from fastapi import (  # noqa: E402
    FastAPI,
    HTTPException,
    Request,
    Response,
    Depends,
    Form,
    File,
    UploadFile,
)
from fastapi.responses import FileResponse, PlainTextResponse, HTMLResponse, RedirectResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from fastapi.templating import Jinja2Templates  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from moderation import check_prompt  # noqa: E402
from providers import get_provider  # noqa: E402
from translations import get_strings  # noqa: E402
from models import init_db, get_db, User, Video, Scene, Payment, SessionLocal  # noqa: E402
from planner import plan_scenes  # noqa: E402
from compositor import stitch_clips, export_srt  # noqa: E402
from auth import (  # noqa: E402
    hash_password,
    verify_password,
    create_session_token,
    decode_session_token,
    generate_csrf_token,
    verify_csrf_token,
    check_rate_limit,
    clear_rate_limit,
)

VIDEO_DIR = Path(os.getenv("VIDEO_DIR", str(BASE / "videos"))).resolve()
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
MAX_ACTIVE = int(os.getenv("MAX_ACTIVE_JOBS", "5"))
_DEFAULT_SECRET = "super-secret-videoforge-key-change-in-prod-32"
SESSION_SECRET = os.getenv("SESSION_SECRET", _DEFAULT_SECRET)
IS_PROD = os.getenv("APP_ENV", "development").strip().lower() == "production"
if IS_PROD and (SESSION_SECRET == _DEFAULT_SECRET or len(SESSION_SECRET) < 32):
    raise RuntimeError(
        "APP_ENV=production requires a strong SESSION_SECRET (32+ characters) in .env. "
        "Generate one: python -c \"import secrets; print(secrets.token_hex(32))\""
    )
EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")


def public_base_url(request: Request) -> str:
    """Base URL for links in emails. Set APP_BASE_URL in production (prevents Host-header spoofing)."""
    return (os.getenv("APP_BASE_URL", "").strip() or str(request.base_url)).rstrip("/")


def safe_path(base_dir: Union[str, Path], user_path: Union[str, Path]) -> Path:
    """Resolves user_path against base_dir and raises ValueError if it leaves base_dir."""
    base = os.path.realpath(str(base_dir))
    u_str = str(user_path).strip()
    if u_str.startswith("/videos/") or u_str.startswith("\\videos\\"):
        u_str = u_str[8:]
    elif u_str.startswith("/static/") or u_str.startswith("\\static\\"):
        u_str = u_str[8:]
    if not u_str:
        raise ValueError("Empty path provided")
    if os.path.isabs(u_str):
        target = os.path.realpath(u_str)
    else:
        target = os.path.realpath(os.path.join(base, u_str))
    try:
        common = os.path.commonpath([base, target])
    except ValueError:
        raise ValueError(f"Path traversal detected: {user_path}")
    if common != base:
        raise ValueError(f"Path traversal detected: {user_path}")
    return Path(target)


def is_credits_enabled() -> bool:
    """Returns True if credit checks, deductions, and refunds are enabled."""
    val = os.getenv("CREDITS_ENABLED", "true").strip().lower()
    return val not in ("false", "0", "no", "off")


def mark_job_failed(
    video_id: str,
    user_message: str,
    technical_error: Optional[str] = None,
    db: Optional[Session] = None,
):
    """Sets status=failed, stores plain-language user message with next steps,
    logs technical error with video id, and refunds credits once only when credits are enabled."""
    close_db = False
    if db is None:
        db = SessionLocal()
        close_db = True
    try:
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            logger.warning(f"mark_job_failed: Video {video_id} not found")
            return

        video.status = "failed"
        video.error = user_message
        video.progress_text = user_message

        # Log technical error with video id
        logger.error(f"[Job {video_id}] Failure: {user_message} | Technical details: {technical_error or 'None'}")

        # Refund credits once only when credits are enabled
        if is_credits_enabled() and not getattr(video, "credits_refunded", False):
            user = db.query(User).filter(User.id == video.user_id).first()
            if user:
                scenes_count = db.query(Scene).filter(Scene.video_id == video.id).count()
                refund_amount = scenes_count if scenes_count > 0 else 5
                user.credits += refund_amount
                logger.info(f"[Job {video_id}] Refunded {refund_amount} credits to user {user.id}")
            video.credits_refunded = True

        db.commit()
    except Exception as e:
        logger.error(f"[Job {video_id}] Error in mark_job_failed: {e}")
        db.rollback()
    finally:
        if close_db:
            db.close()


def recover_stuck_jobs(timeout_minutes: int = 30):
    """Marks videos left in processing or queued for > timeout_minutes as failed through mark_job_failed."""
    db = SessionLocal()
    try:
        cutoff = datetime.utcnow() - timedelta(minutes=timeout_minutes)
        stuck_statuses = ["generating", "stitching", "running", "queued", "processing"]
        stuck_videos = db.query(Video).filter(
            Video.status.in_(stuck_statuses),
            Video.created_at < cutoff,
        ).all()
        for v in stuck_videos:
            mark_job_failed(
                v.id,
                user_message="Generation timed out. Please try creating or regenerating the video again.",
                technical_error=f"Job exceeded {timeout_minutes}m timeout (status: {v.status}, created: {v.created_at})",
                db=db,
            )
        db.commit()
    except Exception as e:
        logger.error(f"Error recovering stuck jobs: {e}")
        db.rollback()
    finally:
        db.close()

# ── Templates ───────────────────────────────────────────────────────────────
templates = Jinja2Templates(directory=str(BASE / "templates"))

EXAMPLE_PROMPTS = [
    "A curious traveler exploring a snowy forest in watercolor style",
    "A street food vendor making pav bhaji in Mumbai, cartoon animation",
    "Playful creatures exploring an enchanted jungle, 3D animated film",
    "A paper boat sailing down a rainy street, flat vector art",
    "An inventive creator learning to paint in an art studio, anime style",
    "Two friends debating who is the better explorer, claymation",
]

# ── Background Worker for Video Project Generation ──────────────────────────
project_q: "queue.Queue[str]" = queue.Queue()
_provider = None


def generate_project_pipeline(video_id: str):
    """Generates all scene clips for a video project, stitches them, and exports SRT."""
    global _provider
    db = SessionLocal()
    try:
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            return

        scenes = db.query(Scene).filter(Scene.video_id == video.id).order_by(Scene.order_idx).all()
        if not scenes:
            mark_job_failed(video_id, "No scenes found for this project. Please create scenes and try again.", "No scenes found", db=db)
            return

        video.status = "generating"
        db.commit()

        provider_name = os.getenv("PROVIDER", "storyboard").lower()

        if provider_name == "storyboard":
            from pipeline import run_storyboard_pipeline

            def update_progress(txt: str):
                db_sub = SessionLocal()
                try:
                    v = db_sub.query(Video).filter(Video.id == video_id).first()
                    if v:
                        v.progress_text = txt
                        db_sub.commit()
                finally:
                    db_sub.close()

            # Prefer rich scenes stored in scenes_json with motion instructions and character references
            pre_scenes = None
            if video.scenes_json:
                try:
                    loaded = json.loads(video.scenes_json)
                    if isinstance(loaded, list) and len(loaded) == 5:
                        pre_scenes = [
                            {
                                "image_prompt": s.get("visual_prompt", s.get("image_prompt", "")),
                                "caption": s.get("caption", ""),
                                "narration": s.get("narration", ""),
                                "motion_instructions": s.get("motion_instructions", s.get("motion_prompt", "")),
                                "character_reference": s.get("character_reference", ""),
                            }
                            for s in loaded
                        ]
                except Exception:
                    pass
            if not pre_scenes:
                pre_scenes = [{"image_prompt": s.visual_prompt, "caption": s.caption, "narration": s.narration or ""} for s in scenes]
            try:
                result = run_storyboard_pipeline(
                    job_id=video.id,
                    topic=video.topic,
                    style=video.style,
                    aspect_ratio=video.aspect_ratio,
                    caption_lang=video.caption_lang,
                    on_progress=update_progress,
                    pre_planned_scenes=pre_scenes if len(pre_scenes) == 5 else None,
                    length_seconds=video.length_seconds,
                    keep_consistency=getattr(video, "keep_consistency", False),
                )
            except Exception as sb_exc:
                import traceback
                tb_str = traceback.format_exc()
                logger.error(f"[Job {video_id}] Full traceback:\n{tb_str}")
                reason_str = str(sb_exc) if str(sb_exc) else type(sb_exc).__name__
                user_msg = f"pipeline: {reason_str}"
                try:
                    for s in scenes:
                        s.status = "failed"
                        s.error = reason_str
                except Exception:
                    pass
                mark_job_failed(video_id, user_msg, tb_str, db=db)
                return

            # run_storyboard_pipeline returns (mp4, scenes, placeholder_scenes, caption_results)
            if isinstance(result, tuple) and len(result) >= 3:
                final_mp4 = result[0]
                generated_scenes = result[1]
                placeholder_scene_nums = result[2]
            else:
                final_mp4, generated_scenes = result
                placeholder_scene_nums = []

            # Update scene rows and mark placeholders / reused
            reused_nums = []
            for i, s in enumerate(scenes):
                s.status = "done"
                s.clip_path = f"/videos/{video.id}_scene_{s.order_idx}.mp4"
                s.is_placeholder = (i + 1) in placeholder_scene_nums
                if i < len(generated_scenes) and generated_scenes[i].get("is_reused"):
                    s.is_placeholder = False
                    s.is_reused = True
                    s.reused_from = generated_scenes[i].get("reused_from")
                    s.error = f"Reused image from Scene {s.reused_from}"
                    reused_nums.append(i + 1)

            clean_mp4 = final_mp4
            srt_file = safe_path(VIDEO_DIR, f"{video.id}.srt")
            zip_file = safe_path(VIDEO_DIR, f"{video.id}_clips.zip")
            clip_files = [safe_path(VIDEO_DIR, f"{video.id}_scene_{s.order_idx}.mp4") for s in scenes]

            # Export SRT subtitles
            scene_dicts = [{"caption": s.caption} for s in scenes]
            export_srt(scene_dicts, 4.0, srt_file)

            with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zf:
                for c in clip_files:
                    if c.exists():
                        zf.write(c, arcname=c.name)
                if srt_file.exists():
                    zf.write(srt_file, arcname=srt_file.name)

            placeholder_note = ""
            if placeholder_scene_nums:
                placeholder_note = " ⚠ Scenes used placeholder cards."
            reused_note = ""
            if reused_nums:
                reused_note = f" ℹ Reused image for scene{'s' if len(reused_nums) > 1 else ''} {', '.join(str(n) for n in reused_nums)}."

            video.status = "done"
            video.progress_text = "Completed" + placeholder_note + reused_note
            video.video_path = f"/videos/{final_mp4.name}"
            video.no_music_video_path = f"/videos/{final_mp4.name}"
            video.srt_path = f"/videos/{srt_file.name}"
            video.zip_path = f"/videos/{zip_file.name}"
            db.commit()
            return

        if _provider is None:
            _provider = get_provider()

        clip_files: List[Path] = []
        captions: List[str] = []
        user = db.query(User).filter(User.id == video.user_id).first()

        for s in scenes:
            s.status = "running"
            video.progress_text = f"Scene {s.order_idx + 1} of {len(scenes)}: generating clip..."
            db.commit()

            clip_path = safe_path(VIDEO_DIR, f"{video.id}_scene_{s.order_idx}.mp4")
            try:
                _provider.generate(
                    prompt=s.visual_prompt,
                    seconds=5,
                    out_path=clip_path,
                    aspect_ratio=video.aspect_ratio,
                    caption=s.caption,
                    scene_idx=s.order_idx,
                )
                s.status = "done"
                s.clip_path = f"/videos/{clip_path.name}"
                db.commit()
                clip_files.append(clip_path)
                captions.append(s.caption)
            except Exception as clip_exc:
                s.status = "failed"
                s.error = "Clip generation failed. Please retry."
                if is_credits_enabled() and user:
                    user.credits += 1  # Automatic refund for failed clip
                db.commit()
                logger.warning(f"Clip generation failed for scene {s.id}: {clip_exc}")

        if not clip_files:
            mark_job_failed(video_id, "All scene video generations failed. Please try again.", "All clip generations failed", db=db)
            return

        # Stitch all generated clips
        video.status = "stitching"
        video.progress_text = "Stitching all video clips..."
        db.commit()

        final_mp4 = safe_path(VIDEO_DIR, f"{video.id}.mp4")
        clean_mp4 = safe_path(VIDEO_DIR, f"{video.id}_nomusic.mp4")
        srt_file = safe_path(VIDEO_DIR, f"{video.id}.srt")
        zip_file = safe_path(VIDEO_DIR, f"{video.id}_clips.zip")

        music_file_path = safe_path(VIDEO_DIR, video.music_filename) if video.music_filename else None

        stitch_clips(
            clip_paths=clip_files,
            scene_captions=captions,
            output_path=final_mp4,
            width=1080 if video.aspect_ratio in ("9:16", "1:1") else 1280,
            height=1920 if video.aspect_ratio == "9:16" else (1080 if video.aspect_ratio == "1:1" else 720),
            fps=24,
            clip_duration=5.0,
            crossfade_sec=0.3,
            music_path=music_file_path,
            no_music_output_path=clean_mp4,
        )

        # Export SRT subtitles
        scene_dicts = [{"caption": s.caption} for s in scenes]
        export_srt(scene_dicts, 5.0, srt_file)

        # Package raw clips ZIP
        with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zf:
            for c in clip_files:
                if c.exists():
                    zf.write(c, arcname=c.name)
            if srt_file.exists():
                zf.write(srt_file, arcname=srt_file.name)

        video.status = "done"
        video.progress_text = "Completed"
        video.video_path = f"/videos/{final_mp4.name}"
        video.no_music_video_path = f"/videos/{clean_mp4.name}"
        video.srt_path = f"/videos/{srt_file.name}"
        video.zip_path = f"/videos/{zip_file.name}"
        db.commit()
    except Exception as exc:
        db.rollback()
        mark_job_failed(video_id, "Video generation encountered an error. Please try generating again.", str(exc), db=db)
    finally:
        db.close()
        project_q.task_done()


def background_project_worker():
    while True:
        vid_id = project_q.get()
        generate_project_pipeline(vid_id)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()

    # Recover jobs stuck for > 30 minutes
    recover_stuck_jobs(timeout_minutes=30)

    # Recover interrupted video jobs from server restart
    db = SessionLocal()
    try:
        interrupted = db.query(Video).filter(Video.status.in_(["generating", "stitching"])).all()
        for v in interrupted:
            mark_job_failed(
                v.id,
                user_message="Generation was interrupted by server restart. Please try again.",
                technical_error="Server restarted while job was active",
                db=db,
            )
        db.commit()
    finally:
        db.close()

    threading.Thread(target=background_project_worker, daemon=True).start()
    yield


app = FastAPI(
    title="VideoForge",
    lifespan=lifespan,
    docs_url=None if IS_PROD else "/docs",
    redoc_url=None if IS_PROD else "/redoc",
    openapi_url=None if IS_PROD else "/openapi.json",
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    resp.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    if IS_PROD:
        resp.headers.setdefault("Strict-Transport-Security", "max-age=31536000")
    return resp


# ── Email: SMTP when SMTP_HOST is set, console banner in development ────────
def _smtp_send(to: str, subject: str, body: str):
    import smtplib
    from email.message import EmailMessage

    msg = EmailMessage()
    msg["From"] = os.getenv("SMTP_FROM") or os.getenv("SMTP_USER") or "no-reply@localhost"
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    try:
        with smtplib.SMTP(os.getenv("SMTP_HOST"), int(os.getenv("SMTP_PORT", "587")), timeout=15) as s:
            s.starttls()
            if os.getenv("SMTP_USER"):
                s.login(os.getenv("SMTP_USER"), os.getenv("SMTP_PASSWORD", ""))
            s.send_message(msg)
    except Exception as e:
        logger.error(f"Email to {to} failed: {e}")


def _notify(email: str, subject: str, body: str, label: str, url: str):
    if os.getenv("SMTP_HOST", "").strip():
        threading.Thread(target=_smtp_send, args=(email, subject, body), daemon=True).start()
    elif IS_PROD:
        logger.warning(f"SMTP_HOST is not set: could not email {label} to {email}.")
    else:
        print("\n" + "=" * 70)
        print(f"[DEVELOPMENT MODE EMAIL - {label}]")
        print(f"To: {email}")
        print(f"Link: {url}")
        print("=" * 70 + "\n")


def send_verification_email(email: str, verify_url: str):
    _notify(email, "Verify your VideoForge account",
            f"Welcome to VideoForge!\n\nVerify your email: {verify_url}\n", "VERIFY ACCOUNT", verify_url)


def send_password_reset_email(email: str, reset_url: str):
    _notify(email, "Reset your VideoForge password",
            f"Reset your password: {reset_url}\n\nIf you did not ask for this, ignore this email.\n",
            "PASSWORD RESET", reset_url)


# ── Authentication & Session Helpers ─────────────────────────────────────────
def get_session_id(request: Request) -> str:
    sid = request.cookies.get("vf_sid")
    return sid or "anon"


def get_current_user(request: Request, db: Session = Depends(get_db)) -> Optional[User]:
    token = request.cookies.get("vf_session")
    if not token:
        return None
    user_id = decode_session_token(token, SESSION_SECRET)
    if not user_id:
        return None
    return db.query(User).filter(User.id == user_id).first()


def require_user(request: Request, db: Session = Depends(get_db)) -> User:
    user = get_current_user(request, db)
    if not user:
        raise HTTPException(status_code=303, headers={"Location": "/login"})
    return user


def get_lang(request: Request) -> str:
    lang = request.query_params.get("lang") or request.cookies.get("vf_lang", "en")
    return lang if lang in ("en", "hi") else "en"


def tpl_ctx(
    request: Request,
    active_page: str = "",
    user: Optional[User] = None,
    flash: Optional[dict] = None,
    dev_link: Optional[dict] = None,
    is_video_page: bool = False,
):
    lang = get_lang(request)
    session_id = request.cookies.get("vf_sid")
    if not session_id:
        session_id = getattr(request.state, "vf_sid", None)
        if not session_id:
            session_id = secrets.token_hex(16)
            request.state.vf_sid = session_id

    csrf_token = generate_csrf_token(SESSION_SECRET, session_id)
    provider_name = os.getenv("PROVIDER", "mock").lower()

    return {
        "lang": lang,
        "t": get_strings(lang),
        "active_page": active_page,
        "user": user,
        "flash": flash,
        "dev_link": dev_link,
        "csrf_token": csrf_token,
        "example_prompts": EXAMPLE_PROMPTS,
        "current_year": datetime.utcnow().year,
        "is_mock_mode": provider_name == "mock",
        "is_video_page": is_video_page,
        "credits_enabled": is_credits_enabled(),
    }



def render_page(request: Request, name: str, ctx: dict, response: Optional[Response] = None):
    resp = templates.TemplateResponse(request=request, name=name, context=ctx)
    if not request.cookies.get("vf_sid"):
        sid = getattr(request.state, "vf_sid", None) or secrets.token_hex(16)
        resp.set_cookie("vf_sid", sid, httponly=True, samesite="lax", secure=IS_PROD)
    return resp


def set_user_session(response: Response, user_id: str):
    token = create_session_token(user_id, SESSION_SECRET)
    response.set_cookie(
        "vf_session",
        token,
        httponly=True,
        samesite="lax",
        secure=IS_PROD,
        max_age=14 * 86400,
    )


def clear_user_session(response: Response):
    response.delete_cookie("vf_session")


def validate_csrf(request: Request, csrf_token: str):
    session_id = get_session_id(request)
    if not verify_csrf_token(csrf_token, SESSION_SECRET, session_id):
        raise HTTPException(status_code=403, detail="Invalid or expired CSRF token")


# ════════════════════════════════════════════════════════════════════════════
#  PAGE ROUTES & VIEWS
# ════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def page_home(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    ctx = tpl_ctx(request, active_page="home", user=user)
    resp = render_page(request, "landing.html", ctx)
    if "lang" in request.query_params:
        resp.set_cookie("vf_lang", get_lang(request), max_age=365 * 86400, samesite="lax")
    return resp


@app.get("/pricing", response_class=HTMLResponse)
async def page_pricing(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    ctx = tpl_ctx(request, active_page="pricing", user=user)
    return render_page(request, "pricing.html", ctx)


@app.get("/signup", response_class=HTMLResponse)
async def page_signup(request: Request, db: Session = Depends(get_db)):
    if get_current_user(request, db):
        return RedirectResponse("/dashboard", status_code=302)
    ctx = tpl_ctx(request, active_page="signup")
    ctx["error"] = None
    ctx["form_email"] = ""
    return render_page(request, "signup.html", ctx)


@app.post("/signup", response_class=HTMLResponse)
async def page_signup_post(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    csrf_token: str = Form(...),
    db: Session = Depends(get_db),
):
    validate_csrf(request, csrf_token)
    email_clean = email.strip().lower()

    ctx = tpl_ctx(request, active_page="signup")
    ctx["form_email"] = email_clean

    if not EMAIL_RE.fullmatch(email_clean) or len(email_clean) > 255:
        ctx["error"] = "Please enter a valid email address."
        return render_page(request, "signup.html", ctx)

    if len(password) < 8:
        ctx["error"] = "Password must be at least 8 characters long."
        return render_page(request, "signup.html", ctx)

    existing = db.query(User).filter(User.email == email_clean).first()
    if existing:
        ctx["error"] = "An account with this email already exists."
        return render_page(request, "signup.html", ctx)

    is_first = db.query(User).count() == 0
    user_id = str(uuid.uuid4())
    token = secrets.token_hex(24)

    new_user = User(
        id=user_id,
        email=email_clean,
        hashed_password=hash_password(password),
        credits=20,
        is_admin=is_first,
        is_verified=False,
        verification_token=token,
    )
    db.add(new_user)
    db.commit()

    base_url = public_base_url(request)
    verify_url = f"{base_url}/verify-email?token={token}"
    send_verification_email(email_clean, verify_url)

    resp = RedirectResponse("/dashboard?verified_nudge=1", status_code=303)
    set_user_session(resp, user_id)
    return resp


@app.get("/verify-email", response_class=HTMLResponse)
async def page_verify_email(request: Request, token: str = "", db: Session = Depends(get_db)):
    user = db.query(User).filter(User.verification_token == token).first() if token else None
    current_u = get_current_user(request, db)

    if not user:
        ctx = tpl_ctx(request, user=current_u, flash={"type": "error", "msg": "Invalid or expired verification link."})
        return render_page(request, "dashboard.html", ctx)

    user.is_verified = True
    user.verification_token = None
    db.commit()

    ctx = tpl_ctx(request, user=user, flash={"type": "success", "msg": "Your email has been verified successfully!"})
    resp = render_page(request, "dashboard.html", ctx)
    set_user_session(resp, user.id)
    return resp


@app.get("/login", response_class=HTMLResponse)
async def page_login(request: Request, db: Session = Depends(get_db)):
    if get_current_user(request, db):
        return RedirectResponse("/dashboard", status_code=302)
    ctx = tpl_ctx(request, active_page="login")
    ctx["error"] = None
    ctx["form_email"] = ""
    return render_page(request, "login.html", ctx)


@app.post("/login", response_class=HTMLResponse)
async def page_login_post(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    csrf_token: str = Form(...),
    db: Session = Depends(get_db),
):
    validate_csrf(request, csrf_token)
    email_clean = email.strip().lower()

    client_ip = request.client.host if request.client else "unknown"
    rate_key = f"{client_ip}:{email_clean}"

    if not check_rate_limit(rate_key):
        ctx = tpl_ctx(request, active_page="login")
        ctx["form_email"] = email_clean
        ctx["error"] = "Too many failed login attempts. Please wait 60 seconds."
        return render_page(request, "login.html", ctx)

    user = db.query(User).filter(User.email == email_clean).first()
    if not user or not verify_password(password, user.hashed_password):
        ctx = tpl_ctx(request, active_page="login")
        ctx["form_email"] = email_clean
        ctx["error"] = "Incorrect email or password."
        return render_page(request, "login.html", ctx)

    clear_rate_limit(rate_key)
    resp = RedirectResponse("/dashboard", status_code=303)
    set_user_session(resp, user.id)
    return resp


@app.get("/logout")
async def page_logout():
    resp = RedirectResponse("/login", status_code=302)
    clear_user_session(resp)
    return resp


@app.get("/forgot-password", response_class=HTMLResponse)
async def page_forgot_password(request: Request):
    ctx = tpl_ctx(request)
    ctx["error"] = None
    ctx["form_email"] = ""
    return render_page(request, "forgot_password.html", ctx)


@app.post("/forgot-password", response_class=HTMLResponse)
async def page_forgot_password_post(
    request: Request,
    email: str = Form(...),
    csrf_token: str = Form(...),
    db: Session = Depends(get_db),
):
    validate_csrf(request, csrf_token)
    email_clean = email.strip().lower()
    user = db.query(User).filter(User.email == email_clean).first()

    ctx = tpl_ctx(request)
    dev_link = None
    if user:
        reset_tok = secrets.token_hex(24)
        user.reset_token = reset_tok
        db.commit()
        base_url = public_base_url(request)
        reset_url = f"{base_url}/reset-password/{reset_tok}"
        send_password_reset_email(email_clean, reset_url)
        if not IS_PROD:
            dev_link = {"label": "Password Reset URL", "url": reset_url}

    ctx["flash"] = {"type": "info", "msg": f"If an account exists for {email_clean}, a reset link has been issued."}
    ctx["dev_link"] = dev_link
    return render_page(request, "login.html", ctx)


@app.get("/reset-password/{token}", response_class=HTMLResponse)
async def page_reset_password(request: Request, token: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.reset_token == token).first()
    if not user:
        ctx = tpl_ctx(request, flash={"type": "error", "msg": "Invalid or expired reset token."})
        return render_page(request, "login.html", ctx)

    ctx = tpl_ctx(request)
    ctx["token"] = token
    ctx["error"] = None
    return render_page(request, "reset_password.html", ctx)


@app.post("/reset-password/{token}", response_class=HTMLResponse)
async def page_reset_password_post(
    request: Request,
    token: str,
    password: str = Form(...),
    csrf_token: str = Form(...),
    db: Session = Depends(get_db),
):
    validate_csrf(request, csrf_token)
    user = db.query(User).filter(User.reset_token == token).first()
    if not user:
        ctx = tpl_ctx(request, flash={"type": "error", "msg": "Invalid or expired reset token."})
        return render_page(request, "login.html", ctx)

    if len(password) < 8:
        ctx = tpl_ctx(request)
        ctx["token"] = token
        ctx["error"] = "Password must be at least 8 characters long."
        return render_page(request, "reset_password.html", ctx)

    user.hashed_password = hash_password(password)
    user.reset_token = None
    db.commit()

    ctx = tpl_ctx(request, flash={"type": "success", "msg": "Password updated successfully. Please log in."})
    return render_page(request, "login.html", ctx)


@app.get("/dashboard", response_class=HTMLResponse)
async def page_dashboard(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=302)

    videos = db.query(Video).filter(Video.user_id == user.id).order_by(Video.created_at.desc()).all()
    dev_link = None
    if request.query_params.get("verified_nudge") and user.verification_token and not IS_PROD:
        base_url = public_base_url(request)
        dev_link = {
            "label": "Click to verify your account email",
            "url": f"{base_url}/verify-email?token={user.verification_token}",
        }

    ctx = tpl_ctx(request, active_page="dashboard", user=user, dev_link=dev_link)
    ctx["video_count"] = len(videos)
    ctx["recent_videos"] = videos[:6]
    return render_page(request, "dashboard.html", ctx)


# ── STEP 1: Enter Topic & Settings ──────────────────────────────────────────
@app.get("/create", response_class=HTMLResponse)
async def page_create(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=302)

    ctx = tpl_ctx(request, active_page="create", user=user)
    ctx["form_topic"] = ""
    return render_page(request, "create.html", ctx)


@app.post("/create", response_class=HTMLResponse)
async def page_create_post(
    request: Request,
    topic: str = Form(...),
    length: int = Form(15),
    style: str = Form("2d_cartoon"),
    ratio: str = Form("16:9"),
    caption_lang: str = Form("none"),
    csrf_token: str = Form(...),
    db: Session = Depends(get_db),
):
    validate_csrf(request, csrf_token)
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=302)

    MAX_PROMPT_LENGTH = 20000
    if len(topic) > MAX_PROMPT_LENGTH:
        ctx = tpl_ctx(request, active_page="create", user=user)
        ctx["form_topic"] = topic[:MAX_PROMPT_LENGTH]
        ctx["flash"] = {
            "type": "error",
            "msg": f"Prompt exceeds maximum limit of {MAX_PROMPT_LENGTH:,} characters (current length: {len(topic):,} characters). Please shorten your prompt.",
        }
        return render_page(request, "create.html", ctx)

    ok, reason = check_prompt(topic)
    if not ok:
        ctx = tpl_ctx(request, active_page="create", user=user)
        ctx["form_topic"] = topic
        ctx["flash"] = {"type": "error", "msg": reason}
        return render_page(request, "create.html", ctx)

    # Detect style from main prompt if user left default "3d_animated" so main prompt is source of truth
    topic_low = topic.lower()
    if style == "3d_animated":
        if any(k in topic_low for k in ["dark cinematic", "cinematic", "science-fiction", "sci-fi", "space station"]):
            style = "scifi" if ("sci-fi" in topic_low or "science" in topic_low or "space station" in topic_low) else "cinematic"
        elif "photorealistic" in topic_low or "realistic" in topic_low:
            style = "realistic"
        elif "dark fantasy" in topic_low:
            style = "dark_fantasy"
        elif "anime" in topic_low:
            style = "anime"
        elif "watercolor" in topic_low:
            style = "watercolor"
        elif "claymation" in topic_low or "clay" in topic_low:
            style = "clay"

    # Auto-detect Hindi captions if prompt includes Devanagari text or requests captions
    has_devanagari = any("\u0900" <= ch <= "\u097f" for ch in topic)
    if caption_lang == "none" and has_devanagari:
        caption_lang = "hi"

    # Plan scenes (5 scenes for storyboard, or duration-based for other providers)
    provider_name = os.getenv("PROVIDER", "storyboard").lower()
    is_captions_off = str(caption_lang).lower().strip() in ("none", "", "off", "disabled", "false", "0")
    try:
        if provider_name == "storyboard":
            from pipeline import plan_storyboard_scenes
            planned_raw = plan_storyboard_scenes(topic, style, caption_lang, length_seconds=length, fallback_to_template=True)
            planned_scenes = [
                {
                    "visual_prompt": s.get("visual_prompt") or s.get("image_prompt") or s.get("prompt") or s.get("visual_description") or "",
                    "caption": "" if is_captions_off else (s.get("caption") or s.get("burned_in_caption") or ""),
                    "narration": s.get("narration", ""),
                    "motion_instructions": s.get("motion_instructions", s.get("motion_prompt", "")),
                    "character_reference": s.get("character_reference", ""),
                }
                for s in planned_raw
            ]
        else:
            planned_raw = plan_scenes(topic, length, style, caption_lang, fallback_to_template=True)
            planned_scenes = [
                {
                    "visual_prompt": s.get("visual_prompt") or s.get("image_prompt") or s.get("prompt") or s.get("visual_description") or "",
                    "caption": "" if is_captions_off else (s.get("caption") or s.get("burned_in_caption") or ""),
                    "narration": s.get("narration", ""),
                    "motion_instructions": s.get("motion_instructions", s.get("motion_prompt", "")),
                    "character_reference": s.get("character_reference", ""),
                }
                for s in planned_raw
            ]
    except Exception as plan_err:
        logger.error(f"Scene planning failed: {plan_err}", exc_info=True)
        # Universal generic fallback: plan_scenes_template() works for ALL topics without LLM
        try:
            from planner import plan_scenes_template
            planned_raw_fb = plan_scenes_template(topic, length, style, caption_lang)
            planned_scenes = [
                {
                    "visual_prompt": s.get("visual_prompt") or s.get("image_prompt") or "",
                    "caption": "" if is_captions_off else (s.get("caption") or ""),
                    "narration": s.get("narration", ""),
                    "motion_instructions": s.get("motion_instructions", ""),
                    "character_reference": s.get("character_reference", ""),
                }
                for s in planned_raw_fb
            ]
            logger.info(f"[app] Generic template fallback produced {len(planned_scenes)} scenes for: {topic[:60]!r}")
        except Exception as fb_err:
            logger.error(f"Template fallback also failed: {fb_err}", exc_info=True)
            ctx = tpl_ctx(request, active_page="create", user=user)
            ctx["form_topic"] = topic
            err_msg = f"planner: {str(plan_err)[:120]}" if str(plan_err) else f"planner: {type(plan_err).__name__}"
            ctx["flash"] = {"type": "error", "msg": f"{err_msg}. Please click Create Video to retry."}
            return render_page(request, "create.html", ctx)

    # STRICT VALIDATION: Ensure planned_scenes has >= 5 scenes and every visual_prompt is non-empty
    if not planned_scenes or len(planned_scenes) < 5 or any(not (s.get("visual_prompt") or "").strip() for s in planned_scenes):
        logger.warning(f"[app] Planned scenes incomplete or contain blank prompts. Triggering template fallback...")
        try:
            from planner import plan_scenes_template
            fb_scenes = plan_scenes_template(topic, length, style, caption_lang)
            if fb_scenes and len(fb_scenes) >= 5:
                planned_scenes = [
                    {
                        "visual_prompt": s.get("visual_prompt") or s.get("image_prompt") or "",
                        "caption": "" if is_captions_off else (s.get("caption") or ""),
                        "narration": s.get("narration", ""),
                        "motion_instructions": s.get("motion_instructions", ""),
                        "character_reference": s.get("character_reference", ""),
                    }
                    for s in fb_scenes
                ]
        except Exception as fb_err2:
            logger.error(f"[app] Second template fallback failed: {fb_err2}", exc_info=True)

    # GUARANTEE: Never leave a blank visual_prompt
    clean_topic_summary = " ".join(topic.strip().split()[:20])
    while len(planned_scenes) < 5:
        idx_pad = len(planned_scenes)
        planned_scenes.append({
            "visual_prompt": f"Cinematic detailed animation scene {idx_pad+1} depicting {clean_topic_summary}, vibrant colors, balanced lighting, high production quality.",
            "caption": "",
            "narration": "",
            "motion_instructions": "",
            "character_reference": "",
        })
    for idx_sc, sc_item in enumerate(planned_scenes):
        if not (sc_item.get("visual_prompt") or "").strip():
            sc_item["visual_prompt"] = f"Cinematic detailed animation scene {idx_sc+1} depicting {clean_topic_summary}, vibrant colors, balanced lighting, high production quality."

    video_id = str(uuid.uuid4())[:12]
    first_line = topic.strip().splitlines()[0] if topic.strip().splitlines() else "Untitled Video"
    new_video = Video(
        id=video_id,
        user_id=user.id,
        title=first_line[:40] + ("..." if len(first_line) > 40 else ""),
        topic=topic.strip(),
        length_seconds=length,
        style=style,
        aspect_ratio=ratio,
        caption_lang=caption_lang,
        status="planning",
        progress_text="Planning complete",
        scenes_json=json.dumps(planned_scenes, ensure_ascii=False),
    )
    db.add(new_video)

    for i, sc in enumerate(planned_scenes):
        scene = Scene(
            id=str(uuid.uuid4())[:12],
            video_id=video_id,
            order_idx=i,
            visual_prompt=sc["visual_prompt"],
            caption=sc.get("caption", ""),
            narration=sc.get("narration", ""),
            status="pending",
        )
        db.add(scene)

    db.commit()
    return RedirectResponse(f"/project/{video_id}/plan", status_code=303)


# ── STEP 2: Review & Edit Scenes ────────────────────────────────────────────
@app.get("/project/{video_id}/plan", response_class=HTMLResponse)
async def page_project_plan(video_id: str, request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=302)

    video = db.query(Video).filter(Video.id == video_id, Video.user_id == user.id).first()
    if not video:
        raise HTTPException(404, "Video project not found")

    scenes = db.query(Scene).filter(Scene.video_id == video.id).order_by(Scene.order_idx).all()
    topic_low = (video.topic or "").lower()
    is_captions_off = str(video.caption_lang).lower().strip() in ("none", "", "off", "disabled", "false", "0")

    needs_restore = not scenes or len(scenes) < 5 or any(not (s.visual_prompt or "").strip() for s in scenes)
    if needs_restore:
        restored = []
        if video.scenes_json:
            try:
                raw_json = json.loads(video.scenes_json)
                # STRICT: only accept saved scenes_json if ALL entries have non-empty visual_prompt
                if isinstance(raw_json, list) and len(raw_json) >= 5:
                    if all(bool((sc.get("visual_prompt") or sc.get("image_prompt") or sc.get("prompt") or "").strip()) for sc in raw_json):
                        restored = raw_json
                    else:
                        logger.warning(f"[app] video {video.id} scenes_json contained blank prompts; discarding and re-planning.")
            except Exception as e:
                logger.warning(f"[app] Failed to parse video {video.id} scenes_json: {e}")
                restored = []

        # If scenes_json is empty, bad, or had blank prompts, use universal template planner
        if not restored:
            try:
                from planner import plan_scenes_template
                replanned = plan_scenes_template(
                    video.topic or "animated story",
                    video.length_seconds or 25,
                    video.style or "3d_animated",
                    video.caption_lang or "none",
                )
                restored = [
                    {
                        "visual_prompt": s.get("visual_prompt") or s.get("image_prompt") or "",
                        "caption": "" if is_captions_off else (s.get("caption") or ""),
                        "narration": s.get("narration", ""),
                    }
                    for s in replanned
                ]
                logger.info(f"[app] Re-planned {len(restored)} scenes via template for video {video.id}")
            except Exception as rp_err:
                logger.error(f"[app] Template re-plan failed: {rp_err}", exc_info=True)

        clean_topic_summary = " ".join((video.topic or "animated story").strip().split()[:20])
        while len(restored) < 5:
            idx_pad = len(restored)
            restored.append({
                "visual_prompt": f"Cinematic detailed animation scene {idx_pad+1} depicting {clean_topic_summary}, vibrant colors, balanced lighting, high production quality.",
                "caption": "",
                "narration": "",
            })

        db.query(Scene).filter(Scene.video_id == video.id).delete()
        scenes = []
        for idx, sc in enumerate(restored):
            vp = (sc.get("visual_prompt") or sc.get("image_prompt") or sc.get("prompt") or sc.get("visual_description") or sc.get("description") or "").strip()
            if not vp:
                vp = f"Cinematic detailed animation scene {idx+1} depicting {clean_topic_summary}, vibrant colors, balanced lighting, high production quality."
            cap = "" if is_captions_off else (sc.get("caption") or sc.get("burned_in_caption") or "")
            nar = sc.get("narration") or cap
            sc_obj = Scene(
                id=str(uuid.uuid4())[:12],
                video_id=video.id,
                order_idx=idx,
                visual_prompt=vp,
                caption=cap,
                narration=nar,
                status="pending",
            )
            db.add(sc_obj)
            scenes.append(sc_obj)
        video.scenes_json = json.dumps([
            {
                "visual_prompt": s.visual_prompt,
                "caption": s.caption,
                "narration": s.narration,
            }
            for s in scenes
        ], ensure_ascii=False)
        db.commit()

    # Diagnostic reporting: safe information only (never secrets)
    diag_info = {
        "video_id": video.id,
        "scene_count": len(scenes),
        "scenes": [
            {
                "order_idx": s.order_idx,
                "id": s.id,
                "length": len(s.visual_prompt or ""),
                "status": "populated" if (s.visual_prompt or "").strip() else ("none" if s.visual_prompt is None else "empty"),
            }
            for s in scenes
        ],
    }
    status_summary = [f"S{s['order_idx']+1}:{s['status']}({s['length']}c)" for s in diag_info['scenes']]
    print(f"[DIAGNOSTIC] GET /project/{video.id}/plan: {len(scenes)} scenes. {status_summary}", flush=True)

    ctx = tpl_ctx(request, active_page="create", user=user)
    ctx["video"] = video
    ctx["scenes"] = scenes
    ctx["diag_info"] = diag_info

    resp = render_page(request, "plan_scenes.html", ctx)
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    resp.headers["X-Diag-Video-Id"] = video.id
    resp.headers["X-Diag-Scene-Count"] = str(len(scenes))
    resp.headers["X-Diag-Prompt-Lengths"] = ",".join(str(len(s.visual_prompt or "")) for s in scenes)
    resp.headers["X-Diag-Prompt-Status"] = ",".join(s["status"] for s in diag_info["scenes"])
    return resp


# ── STEP 3: Queue & Generate Project ────────────────────────────────────────
@app.post("/project/{video_id}/generate")
async def page_project_generate(video_id: str, request: Request, db: Session = Depends(get_db)):
    form = await request.form()
    validate_csrf(request, str(form.get("csrf_token", "")))

    user = get_current_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=302)

    video = db.query(Video).filter(Video.id == video_id, Video.user_id == user.id).first()
    if not video:
        raise HTTPException(404, "Video project not found")

    # Parse scenes from form
    enable_voiceover = bool(form.get("enable_voiceover"))
    idx = 0
    parsed_scenes = []
    blank_scenes = []

    while f"visual_prompt_{idx}" in form:
        vp = str(form.get(f"visual_prompt_{idx}", "")).strip()
        cap = str(form.get(f"caption_{idx}", "")).strip()
        orig_cap = str(form.get(f"orig_caption_{idx}", "")).strip()
        nar = str(form.get(f"narration_{idx}", "")).strip()

        if not vp or len(vp) < 10:
            blank_scenes.append(idx + 1)

        is_captions_off = str(video.caption_lang).lower().strip() in ("none", "", "off", "disabled", "false", "0")
        scene_caption = "" if is_captions_off else cap

        if not enable_voiceover:
            scene_narration = ""
        elif orig_cap and cap != orig_cap and not is_captions_off:
            scene_narration = cap
        else:
            scene_narration = nar or (cap if not is_captions_off else "")

        sc = Scene(
            id=str(uuid.uuid4())[:12],
            video_id=video.id,
            order_idx=idx,
            visual_prompt=vp,
            caption=scene_caption,
            narration=scene_narration,
            status="pending",
        )
        parsed_scenes.append(sc)
        idx += 1

    # Pre-Flight Validation 1: Scene count
    if len(parsed_scenes) < 5:
        ctx = tpl_ctx(request, active_page="create", user=user)
        ctx["video"] = video
        ctx["scenes"] = parsed_scenes or db.query(Scene).filter(Scene.video_id == video.id).order_by(Scene.order_idx).all()
        ctx["flash"] = {
            "type": "error",
            "msg": f"Validation Error: Exactly 5 scenes are required for generation (found {len(parsed_scenes)}).",
        }
        return render_page(request, "plan_scenes.html", ctx)

    # Pre-Flight Validation 2: All 5 Visual Description prompts must be populated
    if blank_scenes:
        ctx = tpl_ctx(request, active_page="create", user=user)
        ctx["video"] = video
        ctx["scenes"] = parsed_scenes
        blanks_str = ", ".join(f"Scene {n}" for n in blank_scenes)
        ctx["flash"] = {
            "type": "error",
            "msg": f"Validation Error: All five scene prompts must be populated before generation. Blank: {blanks_str}.",
        }
        return render_page(request, "plan_scenes.html", ctx)

    # Pre-Flight Validation 3: Aspect ratio validation
    req_ar = str(form.get("aspect_ratio") or form.get("ratio") or video.aspect_ratio or "16:9").strip()
    if req_ar not in ("16:9", "9:16", "1:1"):
        ctx = tpl_ctx(request, active_page="create", user=user)
        ctx["video"] = video
        ctx["scenes"] = parsed_scenes
        ctx["flash"] = {
            "type": "error",
            "msg": f"Validation Error: Selected aspect ratio '{req_ar}' is invalid. Supported: 16:9, 9:16, 1:1.",
        }
        return render_page(request, "plan_scenes.html", ctx)
    video.aspect_ratio = req_ar

    # Pre-Flight Validation 4: Provider readiness check
    img_provider = os.getenv("IMAGE_PROVIDER", "pollinations").lower().strip()
    if img_provider == "cloudflare":
        from pipeline import check_cloudflare_config
        cf_status = check_cloudflare_config()
        if not cf_status.get("configured"):
            ctx = tpl_ctx(request, active_page="create", user=user)
            ctx["video"] = video
            ctx["scenes"] = parsed_scenes
            ctx["flash"] = {
                "type": "error",
                "msg": f"Validation Error: Cloudflare Workers AI configuration incomplete: {cf_status.get('status')}.",
            }
            return render_page(request, "plan_scenes.html", ctx)

    # Check APP_ENV=production safety guard
    app_env = os.getenv("APP_ENV", "development").lower()
    provider_name = os.getenv("PROVIDER", "mock").lower()
    if app_env == "production" and provider_name == "mock":
        ctx = tpl_ctx(request, active_page="create", user=user)
        ctx["video"] = video
        ctx["scenes"] = parsed_scenes
        ctx["flash"] = {
            "type": "error",
            "msg": "Generation blocked: This server is in PRODUCTION mode (APP_ENV=production), but PROVIDER=mock is set. Please configure a live AI model provider (replicate or local) to generate real videos.",
        }
        return render_page(request, "plan_scenes.html", ctx)

    # All validations passed! Now deduct credits strictly based on verified scenes
    new_scenes = parsed_scenes
    if is_credits_enabled():
        required_credits = len(new_scenes)
        if user.credits < required_credits:
            ctx = tpl_ctx(request, active_page="create", user=user)
            ctx["video"] = video
            ctx["scenes"] = new_scenes
            ctx["flash"] = {"type": "error", "msg": f"Insufficient credits! Need {required_credits} credits, but you have {user.credits}."}
            return render_page(request, "plan_scenes.html", ctx)
        user.credits -= required_credits

    # Save validated scenes into DB
    db.query(Scene).filter(Scene.video_id == video.id).delete()
    for sc in new_scenes:
        db.add(sc)

    has_pollinations_key = bool(os.getenv("POLLINATIONS_API_KEY", "").strip())
    est_time = "1-2 minutes" if has_pollinations_key else "5-6 minutes"
    video.keep_consistency = bool(form.get("keep_consistency"))
    video.status = "generating"
    video.progress_text = f"Queued (~{est_time} estimated)..."

    # Sync video.scenes_json with edited scenes so the queued pipeline respects user prompt edits
    old_scenes_data = []
    if video.scenes_json:
        try:
            old_scenes_data = json.loads(video.scenes_json)
        except Exception:
            old_scenes_data = []

    updated_json_scenes = []
    for i, sc in enumerate(new_scenes):
        old_item = old_scenes_data[i] if i < len(old_scenes_data) and isinstance(old_scenes_data[i], dict) else {}
        updated_json_scenes.append({
            "visual_prompt": sc.visual_prompt,
            "caption": sc.caption,
            "narration": sc.narration,
            "motion_instructions": old_item.get("motion_instructions", ""),
            "character_reference": old_item.get("character_reference", ""),
        })
    video.scenes_json = json.dumps(updated_json_scenes, ensure_ascii=False)
    db.commit()

    # Queue video generation task
    project_q.put(video.id)
    return RedirectResponse(f"/video/{video.id}", status_code=303)


# ── STEP 4: Video Details, Progress & Player ────────────────────────────────
@app.get("/video/{video_id}", response_class=HTMLResponse)
async def page_video_detail(video_id: str, request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=302)

    video = db.query(Video).filter(Video.id == video_id, Video.user_id == user.id).first()
    if not video:
        raise HTTPException(404, "Video not found")

    scenes = db.query(Scene).filter(Scene.video_id == video.id).order_by(Scene.order_idx).all()
    ctx = tpl_ctx(request, active_page="library", user=user, is_video_page=True)
    ctx["video"] = video
    ctx["scenes"] = scenes
    return render_page(request, "video_detail.html", ctx)


@app.get("/api/project/{video_id}/status")
async def api_project_status(video_id: str, request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        raise HTTPException(401, "Login required")
    video = db.query(Video).filter(Video.id == video_id, Video.user_id == user.id).first()
    if not video:
        raise HTTPException(404, "Project not found")
    scenes = db.query(Scene).filter(Scene.video_id == video.id).order_by(Scene.order_idx).all()
    return {
        "id": video.id,
        "status": video.status,
        "error": video.error,
        "progress_text": video.progress_text or (
            "Processing..." if video.status in ("generating", "stitching") else
            "Completed" if video.status == "done" else "Failed"
        ),
        "video_path": video.video_path,
        "scenes": [{"id": s.id, "order_idx": s.order_idx, "status": s.status, "clip_path": s.clip_path, "error": s.error} for s in scenes],
    }


# ── Jobs API Alias for Dashboard / API compatibility ────────────────────────
@app.get("/api/jobs")
async def api_jobs_list(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        return []
    query = db.query(Video).filter(Video.user_id == user.id)
    videos = query.order_by(Video.created_at.desc()).limit(10).all()
    return [
        {
            "id": v.id,
            "prompt": v.topic,
            "status": v.status,
            "progress_text": v.progress_text or (
                "Processing..." if v.status in ("generating", "stitching") else
                "Completed" if v.status == "done" else "Failed"
            ),
            "video_url": v.video_path,
            "seconds": v.length_seconds,
            "error": v.error,
        }
        for v in videos
    ]


@app.post("/api/jobs/{video_id}/scenes/{scene_num}/regenerate")
async def api_regenerate_job_scene(
    video_id: str,
    scene_num: int,
    request: Request,
    db: Session = Depends(get_db),
):
    """Regenerates a single scene (by 1-based or 0-based scene number) and re-stitches video."""
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(404, "Job/Video not found")

    target_idx = scene_num - 1 if scene_num > 0 and scene_num <= 5 else scene_num
    scene = db.query(Scene).filter(Scene.video_id == video.id, Scene.order_idx == target_idx).first()
    if not scene:
        raise HTTPException(404, f"Scene {scene_num} not found")

    provider_name = os.getenv("PROVIDER", "storyboard").lower()
    try:
        clip_path = safe_path(VIDEO_DIR, f"{video.id}_scene_{scene.order_idx}.mp4")
    except ValueError:
        raise HTTPException(400, "Invalid file path")
    scene.status = "running"
    db.commit()

    try:
        import random
        provider = get_provider()
        provider.generate(
            prompt=scene.visual_prompt,
            seconds=4 if provider_name == "storyboard" else 5,
            out_path=clip_path,
            scene_idx=scene.order_idx,
            aspect_ratio=video.aspect_ratio,
            caption=scene.caption,
            seed=random.randint(100000, 9999999),
        )
        scene.status = "done"
        scene.clip_path = f"/videos/{clip_path.name}"
        db.commit()

        # Re-stitch
        all_scenes = db.query(Scene).filter(Scene.video_id == video.id).order_by(Scene.order_idx).all()
        clip_files = [safe_path(VIDEO_DIR, f"{video.id}_scene_{s.order_idx}.mp4") for s in all_scenes]
        final_mp4 = safe_path(VIDEO_DIR, f"{video.id}.mp4")

        if provider_name == "storyboard":
            from pipeline import stitch_storyboard_clips
            stitch_storyboard_clips(clip_files, final_mp4)
        else:
            captions = [s.caption for s in all_scenes]
            stitch_clips(
                clip_paths=clip_files,
                scene_captions=captions,
                output_path=final_mp4,
                width=1080 if video.aspect_ratio in ("9:16", "1:1") else 1280,
                height=1920 if video.aspect_ratio == "9:16" else (1080 if video.aspect_ratio == "1:1" else 720),
            )

        video.status = "done"
        video.progress_text = "Completed"
        db.commit()
        return {"ok": True, "scene_num": scene_num, "video_url": f"/videos/{final_mp4.name}"}
    except Exception as e:
        scene.status = "failed"
        scene.error = "Scene generation failed. Please try again."
        db.commit()
        logger.error(f"[Job {video.id}] Failed to regenerate scene {scene_num}: {e}")
        raise HTTPException(500, "Failed to regenerate scene. Please try again.")


# ── Background Music Upload & Re-mix ─────────────────────────────────────────
@app.post("/project/{video_id}/music")
async def upload_music_and_mix(
    video_id: str,
    request: Request,
    csrf_token: str = Form(...),
    music_file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    validate_csrf(request, csrf_token)
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=302)

    video = db.query(Video).filter(Video.id == video_id, Video.user_id == user.id).first()
    if not video:
        raise HTTPException(404, "Video not found")

    ext = Path(music_file.filename).suffix.lower()
    if ext not in (".mp3", ".wav"):
        raise HTTPException(400, "Only .mp3 and .wav audio files are supported.")

    music_filename = f"{video.id}_music{ext}"
    try:
        music_dest = safe_path(VIDEO_DIR, music_filename)
    except ValueError:
        raise HTTPException(400, "Invalid file path")

    with open(music_dest, "wb") as f:
        shutil.copyfileobj(music_file.file, f)

    video.music_filename = music_filename
    db.commit()

    # Re-stitch video with audio
    scenes = db.query(Scene).filter(Scene.video_id == video.id).order_by(Scene.order_idx).all()
    clip_files = [safe_path(VIDEO_DIR, f"{video.id}_scene_{s.order_idx}.mp4") for s in scenes]
    captions = [s.caption for s in scenes]
    final_mp4 = safe_path(VIDEO_DIR, f"{video.id}.mp4")
    clean_mp4 = safe_path(VIDEO_DIR, f"{video.id}_nomusic.mp4")

    stitch_clips(
        clip_paths=clip_files,
        scene_captions=captions,
        output_path=final_mp4,
        width=1280,
        height=720,
        music_path=music_dest,
        no_music_output_path=clean_mp4,
    )
    return RedirectResponse(f"/video/{video.id}", status_code=303)


# ── Single Scene Regeneration ───────────────────────────────────────────────
@app.post("/scene/{scene_id}/regenerate")
async def regenerate_single_scene(scene_id: str, request: Request, db: Session = Depends(get_db)):
    form = await request.form()
    validate_csrf(request, str(form.get("csrf_token", "")))

    user = get_current_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=302)

    scene = db.query(Scene).filter(Scene.id == scene_id).first()
    if not scene:
        raise HTTPException(404, "Scene not found")

    video = db.query(Video).filter(Video.id == scene.video_id, Video.user_id == user.id).first()
    if not video:
        raise HTTPException(404, "Video not found")

    if is_credits_enabled():
        if user.credits < 1:
            raise HTTPException(400, "Insufficient credits to regenerate scene.")
        user.credits -= 1

    scene.status = "running"
    db.commit()

    provider = get_provider()
    try:
        clip_path = safe_path(VIDEO_DIR, f"{video.id}_scene_{scene.order_idx}.mp4")
    except ValueError:
        raise HTTPException(400, "Invalid file path")

    try:
        provider.generate(scene.visual_prompt, 5, clip_path, aspect_ratio=video.aspect_ratio)
        scene.status = "done"
        scene.clip_path = f"/videos/{clip_path.name}"
        db.commit()

        # Re-stitch entire video with updated clip
        scenes = db.query(Scene).filter(Scene.video_id == video.id).order_by(Scene.order_idx).all()
        clip_files = [safe_path(VIDEO_DIR, f"{video.id}_scene_{s.order_idx}.mp4") for s in scenes]
        captions = [s.caption for s in scenes]
        final_mp4 = safe_path(VIDEO_DIR, f"{video.id}.mp4")
        clean_mp4 = safe_path(VIDEO_DIR, f"{video.id}_nomusic.mp4")
        music_path = safe_path(VIDEO_DIR, video.music_filename) if video.music_filename else None

        stitch_clips(
            clip_paths=clip_files,
            scene_captions=captions,
            output_path=final_mp4,
            width=1280,
            height=720,
            music_path=music_path,
            no_music_output_path=clean_mp4,
        )
    except Exception as exc:
        if is_credits_enabled() and user:
            user.credits += 1  # Refund failed single scene regeneration
        scene.status = "failed"
        scene.error = "Scene generation failed. Please try again."
        logger.error(f"[Job {video.id}] Failed to regenerate scene {scene_id}: {exc}")
        db.commit()

    return RedirectResponse(f"/video/{video.id}", status_code=303)


# ── Library: Isolated My Videos ─────────────────────────────────────────────
@app.get("/my-videos", response_class=HTMLResponse)
async def page_library(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=302)

    videos = db.query(Video).filter(Video.user_id == user.id).order_by(Video.created_at.desc()).all()
    ctx = tpl_ctx(request, active_page="library", user=user)
    ctx["videos"] = videos
    return render_page(request, "library.html", ctx)


@app.post("/video/{video_id}/delete")
async def delete_video(
    request: Request,
    video_id: str,
    csrf_token: str = Form(...),
    db: Session = Depends(get_db),
):
    validate_csrf(request, csrf_token)
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=302)

    v = db.query(Video).filter(Video.id == video_id).first()
    if v and (v.user_id == user.id or user.is_admin):
        for path_attr in [v.video_path, v.no_music_video_path, v.srt_path, v.zip_path]:
            if path_attr:
                try:
                    f = safe_path(VIDEO_DIR, path_attr)
                    if f.exists() and f.is_file():
                        f.unlink()
                except Exception:
                    pass
        for s in v.scenes:
            if s.clip_path:
                try:
                    f = safe_path(VIDEO_DIR, s.clip_path)
                    if f.exists() and f.is_file():
                        f.unlink()
                except Exception:
                    pass
        db.delete(v)
        db.commit()

    return RedirectResponse("/my-videos", status_code=303)


# ── Settings ────────────────────────────────────────────────────────────────
def settings_provider_info() -> dict:
    """Admin provider status shown on the Settings page (never exposes the token itself)."""
    provider_name = os.getenv("PROVIDER", "mock").lower()
    raw_token = os.getenv("REPLICATE_API_TOKEN", "")
    return {
        "name": provider_name,
        "has_token": bool(raw_token),
        "token_masked": "Configured" if raw_token else "Not Set",
        "model": os.getenv("REPLICATE_MODEL", "wan-video/wan-2.1-1.3b") if provider_name == "replicate" else os.getenv("LOCAL_MODEL", "Lightricks/LTX-Video"),
    }


@app.get("/settings", response_class=HTMLResponse)
async def page_settings(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=302)

    ctx = tpl_ctx(request, active_page="settings", user=user)
    ctx["profile_success"] = False
    ctx["pw_success"] = False
    ctx["pw_error"] = None

    ctx["provider_info"] = settings_provider_info()
    return render_page(request, "settings.html", ctx)


@app.get("/api/admin/provider/test")
async def api_admin_provider_test(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if not user or not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required.")

    provider_name = os.getenv("PROVIDER", "mock").lower()
    if provider_name == "mock":
        return {
            "ok": True,
            "message": "PROVIDER=mock is active. Synthesizes full-resolution test clips with prompt & watermark overlays.",
        }

    try:
        provider = get_provider()
        if hasattr(provider, "test_connection"):
            res = provider.test_connection()
            return res
        return {"ok": True, "message": f"Provider '{provider_name}' is initialized."}
    except Exception as e:
        logger.error(f"Provider test connection failed: {e}")
        return {"ok": False, "error": "Provider connection failed. Check your API token configuration."}


@app.post("/settings/password", response_class=HTMLResponse)
async def page_settings_password(
    request: Request,
    old_password: str = Form(...),
    new_password: str = Form(...),
    csrf_token: str = Form(...),
    db: Session = Depends(get_db),
):
    validate_csrf(request, csrf_token)
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=302)

    ctx = tpl_ctx(request, active_page="settings", user=user)
    ctx["provider_info"] = settings_provider_info()
    ctx["profile_success"] = False
    ctx["pw_success"] = False

    if not verify_password(old_password, user.hashed_password):
        ctx["pw_error"] = "Current password is incorrect."
        return render_page(request, "settings.html", ctx)

    if len(new_password) < 8:
        ctx["pw_error"] = "New password must be at least 8 characters long."
        return render_page(request, "settings.html", ctx)

    user.hashed_password = hash_password(new_password)
    db.commit()
    ctx["pw_success"] = True
    ctx["pw_error"] = None
    return render_page(request, "settings.html", ctx)


@app.post("/settings/profile", response_class=HTMLResponse)
async def page_settings_profile(
    request: Request,
    email: str = Form(...),
    csrf_token: str = Form(...),
    db: Session = Depends(get_db),
):
    validate_csrf(request, csrf_token)
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=302)

    new_email = email.strip().lower()
    ctx = tpl_ctx(request, active_page="settings", user=user)
    ctx.update(profile_success=False, pw_success=False, pw_error=None, provider_info=settings_provider_info())

    if not EMAIL_RE.fullmatch(new_email) or len(new_email) > 255:
        ctx["flash"] = {"type": "error", "msg": "Please enter a valid email address."}
        return render_page(request, "settings.html", ctx)

    if new_email != user.email:
        if db.query(User).filter(User.email == new_email, User.id != user.id).first():
            ctx["flash"] = {"type": "error", "msg": "That email is already in use."}
            return render_page(request, "settings.html", ctx)
        token = secrets.token_hex(24)
        user.email = new_email
        user.is_verified = False
        user.verification_token = token
        db.commit()
        send_verification_email(new_email, f"{public_base_url(request)}/verify-email?token={token}")

    ctx["profile_success"] = True
    return render_page(request, "settings.html", ctx)


@app.post("/settings/delete-account")
async def page_delete_account(
    request: Request,
    csrf_token: str = Form(...),
    db: Session = Depends(get_db),
):
    validate_csrf(request, csrf_token)
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=302)

    db.delete(user)
    db.commit()
    resp = RedirectResponse("/", status_code=303)
    clear_user_session(resp)
    return resp


# ── Payments (Razorpay) ─────────────────────────────────────────────────────
# Amounts are in paise and are decided here on the server, never by the browser.
PACKS = {
    "creator": {"name": "Creator", "credits": 100, "amount": 29900},
    "studio": {"name": "Studio", "credits": 350, "amount": 79900},
}


def fulfil_payment(db: Session, order_id: str, payment_id: str) -> Optional[int]:
    """Marks an order paid and adds its credits exactly once. Returns the user's new credit
    balance, or None if the order is unknown or was already fulfilled."""
    done = db.query(Payment).filter(Payment.id == order_id, Payment.status == "created").update(
        {"status": "paid", "payment_id": payment_id}, synchronize_session=False
    )
    if not done:
        db.rollback()
        return None
    pay = db.query(Payment).filter(Payment.id == order_id).first()
    user = db.query(User).filter(User.id == pay.user_id).first()
    user.credits += pay.credits
    db.commit()
    logger.info(f"Payment {payment_id} fulfilled: +{pay.credits} credits for user {user.id}")
    return user.credits


@app.post("/api/pay/order")
async def api_pay_order(
    request: Request,
    pack: str = Form(...),
    csrf_token: str = Form(...),
    db: Session = Depends(get_db),
):
    validate_csrf(request, csrf_token)
    user = get_current_user(request, db)
    if not user:
        return JSONResponse(status_code=401, content={"error": "Please log in to buy credits."})
    key_id = os.getenv("RAZORPAY_KEY_ID", "").strip()
    key_secret = os.getenv("RAZORPAY_KEY_SECRET", "").strip()
    if not key_id or not key_secret:
        return JSONResponse(status_code=503, content={"error": "Payments are not available yet. Please try again later."})
    item = PACKS.get(pack)
    if not item:
        return JSONResponse(status_code=400, content={"error": "Unknown credit pack."})

    import httpx
    try:
        r = httpx.post(
            "https://api.razorpay.com/v1/orders",
            auth=(key_id, key_secret),
            json={
                "amount": item["amount"],
                "currency": "INR",
                "receipt": f"vf_{uuid.uuid4().hex[:16]}",
                "notes": {"user_id": user.id, "pack": pack},
            },
            timeout=20,
        )
        r.raise_for_status()
        order = r.json()
    except Exception as e:
        logger.error(f"Razorpay order creation failed: {e}")
        return JSONResponse(status_code=502, content={"error": "Could not start the payment. Please try again."})

    db.add(Payment(id=order["id"], user_id=user.id, pack=pack, credits=item["credits"], amount=item["amount"]))
    db.commit()
    return {
        "key": key_id,
        "order_id": order["id"],
        "amount": item["amount"],
        "currency": "INR",
        "description": f"{item['name']} pack - {item['credits']} credits",
        "email": user.email,
    }


@app.post("/api/pay/verify")
async def api_pay_verify(
    request: Request,
    razorpay_order_id: str = Form(...),
    razorpay_payment_id: str = Form(...),
    razorpay_signature: str = Form(...),
    csrf_token: str = Form(...),
    db: Session = Depends(get_db),
):
    validate_csrf(request, csrf_token)
    user = get_current_user(request, db)
    if not user:
        return JSONResponse(status_code=401, content={"error": "Please log in."})
    pay = db.query(Payment).filter(Payment.id == razorpay_order_id, Payment.user_id == user.id).first()
    if not pay:
        return JSONResponse(status_code=404, content={"error": "Order not found."})

    secret = os.getenv("RAZORPAY_KEY_SECRET", "").strip()
    expected = hmac.new(secret.encode(), f"{razorpay_order_id}|{razorpay_payment_id}".encode(), hashlib.sha256).hexdigest()
    if not secret or not hmac.compare_digest(expected, razorpay_signature):
        return JSONResponse(status_code=400, content={"error": "Payment could not be verified."})

    fulfil_payment(db, razorpay_order_id, razorpay_payment_id)  # no-op if the webhook already did it
    db.refresh(user)
    return {"ok": True, "credits": user.credits}


@app.post("/api/pay/webhook")
async def api_pay_webhook(request: Request, db: Session = Depends(get_db)):
    """Razorpay calls this after payment, so credits are added even if the buyer closes the tab."""
    secret = os.getenv("RAZORPAY_WEBHOOK_SECRET", "").strip()
    if not secret:
        return JSONResponse(status_code=503, content={"error": "Webhook not configured."})
    raw = await request.body()
    expected = hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, request.headers.get("X-Razorpay-Signature", "")):
        return JSONResponse(status_code=400, content={"error": "Bad signature."})
    try:
        data = json.loads(raw)
        if data.get("event") in ("order.paid", "payment.captured"):
            ent = data["payload"]["payment"]["entity"]
            fulfil_payment(db, ent["order_id"], ent["id"])
    except Exception as e:
        logger.error(f"Webhook handling error: {e}")
    return {"ok": True}


# ── Informational Pages ─────────────────────────────────────────────────────
@app.get("/terms", response_class=HTMLResponse)
async def page_terms(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    ctx = tpl_ctx(request, active_page="terms", user=user)
    return render_page(request, "terms.html", ctx)


@app.get("/privacy", response_class=HTMLResponse)
async def page_privacy(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    ctx = tpl_ctx(request, active_page="privacy", user=user)
    return render_page(request, "privacy.html", ctx)


@app.get("/content-policy", response_class=HTMLResponse)
async def page_content_policy(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    ctx = tpl_ctx(request, active_page="content_policy", user=user)
    return render_page(request, "content_policy.html", ctx)


# ── System API Info ─────────────────────────────────────────────────────────
@app.get("/api/info")
def api_info():
    return {"provider": os.getenv("PROVIDER", "mock")}



# ── Health & SEO Endpoints ──────────────────────────────────────────────────
@app.get("/health")
def health_check():
    """Health-check endpoint for Railway, Docker, and uptime probes."""
    return {
        "status": "healthy",
        "service": "videoforge",
        "version": "1.0.0",
        "provider": os.getenv("PROVIDER", "storyboard"),
    }


@app.get("/robots.txt", response_class=PlainTextResponse)
def robots_txt(request: Request):
    """Robots exclusion standard: allow public pages, disallow private and video paths."""
    base_url = public_base_url(request)
    return (
        "User-agent: *\n"
        "Allow: /\n"
        "Allow: /pricing\n"
        "Allow: /login\n"
        "Allow: /signup\n"
        "Allow: /terms\n"
        "Allow: /privacy\n"
        "Allow: /content-policy\n"
        "Disallow: /dashboard\n"
        "Disallow: /create\n"
        "Disallow: /project/\n"
        "Disallow: /video/\n"
        "Disallow: /videos/\n"
        "Disallow: /my-videos\n"
        "Disallow: /settings\n"
        "Disallow: /api/\n"
        f"Sitemap: {base_url}/sitemap.xml\n"
    )


@app.get("/sitemap.xml")
def sitemap_xml(request: Request):
    """Dynamically generated XML sitemap with canonical public marketing and policy pages."""
    base_url = public_base_url(request)
    today = datetime.utcnow().strftime("%Y-%m-%d")
    urls = [
        {"loc": f"{base_url}/", "changefreq": "daily", "priority": "1.0"},
        {"loc": f"{base_url}/pricing", "changefreq": "weekly", "priority": "0.8"},
        {"loc": f"{base_url}/login", "changefreq": "monthly", "priority": "0.5"},
        {"loc": f"{base_url}/signup", "changefreq": "monthly", "priority": "0.6"},
        {"loc": f"{base_url}/terms", "changefreq": "monthly", "priority": "0.3"},
        {"loc": f"{base_url}/privacy", "changefreq": "monthly", "priority": "0.3"},
        {"loc": f"{base_url}/content-policy", "changefreq": "monthly", "priority": "0.3"},
    ]
    xml_items = "\n".join(
        f"  <url>\n    <loc>{u['loc']}</loc>\n    <lastmod>{today}</lastmod>\n    <changefreq>{u['changefreq']}</changefreq>\n    <priority>{u['priority']}</priority>\n  </url>"
        for u in urls
    )
    xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n{xml_items}\n</urlset>"""
    return Response(content=xml_content, media_type="application/xml")


# ── Safe Media Serving & Static Mounts ──────────────────────────────────────
@app.get("/videos/{filename:path}")
async def serve_video_file(filename: str):
    """Safely serves media files, preventing path traversal outside VIDEO_DIR."""
    try:
        fp = safe_path(VIDEO_DIR, filename)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    if not fp.exists() or not fp.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(fp)


app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail}) if request.url.path.startswith("/api/") else HTMLResponse(f"<h1>Error {exc.status_code}</h1><p>{html_escape(str(exc.detail))}</p>", status_code=exc.status_code)
    short_reason = f"{type(exc).__name__}: {str(exc)[:100]}" if str(exc) else type(exc).__name__
    if request.url.path.startswith("/api/"):
        return JSONResponse(status_code=500, content={"error": f"An error occurred: {short_reason}"})
    return HTMLResponse(f"<h1>Something went wrong</h1><p>{html_escape(short_reason)}</p>", status_code=500)



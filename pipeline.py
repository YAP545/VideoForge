"""Storyboard Video Pipeline.
Generates finished 5-scene videos with:
1. Character sheet (max 2 characters with 4 distinctive features, word for word identical at start of each prompt).
2. Limit of 1 simple prop/everyday object, no new creatures after scene 1.
3. Plain easy-to-draw poses, zero banned words, capped at ~60 words.
4. Image validation (PIL variance and edge energy) with single-scene retry.
5. Pollinations.ai image downloads with 4 retries, backoff waits, watermark crop (bottom 6%),
   and illustrated gradient placeholder fallback.
6. 0.25s crossfade with captions fading out before transition starts so no two captions ever overlap.
7. Voiceover using edge-tts with audio duration-driven pacing.
"""
import unicode_compat
import asyncio
import base64
import io
import json
import os
import random
import re
import shutil
import subprocess
import time
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

import httpx
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from animation_engine import render_realistic_animation_clip
from compositor import get_font_path, shape_and_render_multiline_prompt
from planner import (
    ALL_ANIMATION_EFFECTS,
    ANIMATION_LIBRARY,
    BANNED_PROMPT_WORDS,
    CAMERA_ANGLES,
    DEFAULT_EMOTION_ARC,
    EMOTION_STYLE,
    STYLE_AVOID_TEXT,
    STYLE_DESCRIPTIONS,
    STYLE_PRESET_PROMPTS,
    VALID_EMOTIONS,
    build_character_reference_sheets,
    build_scene_motion_instructions,
    check_scene_plan_diversity,
    clean_and_parse_json,
    extract_default_cues,
    extract_story_sheets,
    generate_character_sheet,
    parse_topic_elements,
    plan_scenes,
    plan_scenes_template,
    sanitize_caption,
    sanitize_image_prompt,
    verify_and_fix_character_sheet_features,
)

BASE = Path(__file__).parent
FONTS_DIR = BASE / "static" / "fonts"
VIDEO_DIR = Path(os.getenv("VIDEO_DIR", str(BASE / "videos"))).resolve()
VIDEO_DIR.mkdir(parents=True, exist_ok=True)


def detect_voice(topic: str, narration: str = "") -> str:
    """Detect voice based on script/language in topic or narration:
    Devanagari: checks for Marathi markers, else Hindi; otherwise English.
    """
    combined = (topic or "") + " " + (narration or "")
    if any("\u0900" <= ch <= "\u097F" for ch in combined):
        if re.search(r"\b(आहे|झाला|झाली|झाले|मध्ये|करा|आणि|नाही)\b", combined):
            return "mr-IN-AarohiNeural"
        return "hi-IN-SwaraNeural"
    low = combined.lower()
    if any(k in low for k in ["shayari", "hindi", "shairi", "kahani", "dosti", "pyar", "ishq", "mohabbat", "zindagi"]):
        return "hi-IN-MadhurNeural"
    return "en-US-JennyNeural"


def crop_bottom_and_resize(img: Image.Image, bottom_pct: float = 0.06) -> Image.Image:
    """Crops bottom 6% of an image to remove watermarks, then scales back to full size."""
    w, h = img.size
    crop_h = max(1, int(h * (1.0 - bottom_pct)))
    cropped = img.crop((0, 0, w, crop_h))
    return cropped.resize((w, h), Image.Resampling.LANCZOS)


def validate_image(img: Image.Image) -> bool:
    """Checks if an image is valid (not flat solid color, not extreme blur).
    Uses grayscale variance and edge energy statistics.
    """
    try:
        gray = img.convert("L")
        arr = np.array(gray, dtype=np.float32)
        variance = float(np.var(arr))
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edge_energy = float(np.mean(np.array(edges, dtype=np.float32)))

        if variance < 80.0 or edge_energy < 3.5:
            print(f"[validation] Image failed check: variance={variance:.1f}, edge_energy={edge_energy:.1f}")
            return False
        return True
    except Exception as e:
        print(f"[validation] Error checking image: {e}")
        return True


def get_audio_duration(audio_path: Path) -> float:
    """Returns audio duration in seconds using ffmpeg inspection."""
    if not audio_path.exists() or audio_path.stat().st_size < 200:
        return 0.0
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    try:
        cmd = [ffmpeg_exe, "-i", str(audio_path)]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", res.stderr)
        if m:
            h = float(m.group(1))
            mn = float(m.group(2))
            s = float(m.group(3))
            return h * 3600.0 + mn * 60.0 + s
    except Exception:
        pass
    return 0.0


def plan_storyboard_scenes(
    topic: str,
    style: str = "3d_animated",
    caption_lang: str = "none",
    length_seconds: int = 25,
    fallback_to_template: bool = True,
) -> List[Dict[str, Any]]:
    """Step 1: Plan 5 sequential scenes using Gemini or template planner.
    Enforces generic sheets, color/accessory lock, camera validation, and prompt formatting.
    """
    from planner import plan_scenes
    return plan_scenes(topic, length_seconds=length_seconds, style=style, caption_lang=caption_lang, fallback_to_template=fallback_to_template)


def validate_narration(narration: str, topic: str) -> Tuple[bool, str]:
    """Check if narration copies a long run of words from topic or contains style/character description words.
    Returns (ok, reason). Used to trigger a re-ask of the LLM once.
    """
    if not narration or not topic:
        return True, ""
    # If narration is Devanagari (Hindi shayari, poetry, poem), it is explicit and intentional
    if any("\u0900" <= ch <= "\u097F" for ch in narration):
        return True, ""
    style_leak = re.compile(
        r"\b(animated|emotional story|character with|wearing a|storyboard|visual style|claymation|"
        r"flat vector|watercolor|3d animated|2d cartoon|scene \d)\b",
        re.IGNORECASE,
    )
    if style_leak.search(narration):
        return False, "narration contains style/description words"
    # Check for 4+ consecutive word run shared with topic
    topic_words = [w.lower() for w in topic.split() if len(w) >= 3]
    narr_lower = narration.lower()
    for i in range(len(topic_words) - 3):
        run = " ".join(topic_words[i : i + 4])
        if run and run in narr_lower:
            return False, f"narration copies topic run: '{run}'"
    return True, ""


def is_unwanted_caption_text(text: str, prompt: str = "") -> bool:
    """Returns True if the text contains scene descriptions, visual prompts,
    motion instructions, or internal planning text that should NEVER be shown as a subtitle.
    """
    if not text or not text.strip():
        return False
    t_lower = text.strip().lower()

    prompt_indicators = [
        "stylized 3d", "animated movie character", "smooth rounded shapes",
        "big expressive eyes", "bright cheerful colors", "soft studio lighting",
        "clean simple background", "detailed background", "natural skin tone",
        "warm tan skin", "rich brown skin", "wide shot", "medium shot",
        "close-up shot", "establishing shot", "camera=", "camera:",
        "setting_detail", "must_show", "characters_present",
        "motion_instructions", "motion_prompt", "character_reference",
        "shifting posture naturally", "blinking naturally", "widens eyes in astonished",
        "holds head high with a confident", "gazes forward with an uplifted",
        "cinematic lighting", "photorealistic", "hyperrealistic", "8k", "4k",
        "unreal engine", "octane render", "vivid colors", "wearing a",
        "standing near", "standing quietly", "walks gracefully",
    ]
    if any(k in t_lower for k in prompt_indicators):
        return True

    if prompt:
        p_lower = prompt.lower().strip()
        if t_lower in p_lower and len(t_lower) > 25:
            return True

    words = text.split()
    if len(words) > 16:
        visual_words = ["wearing", "standing", "sitting", "dressed", "view", "lighting", "background", "foreground", "shot"]
        if sum(1 for w in words if w.lower() in visual_words) >= 2:
            return True

    return False


def clean_caption_language(text: str, caption_lang: str = "none") -> str:
    """Strip stray characters from other scripts.
    If caption_lang is 'none' or disabled, returns empty string.
    """
    if not text:
        return ""
    lang = str(caption_lang).lower().strip()
    if lang in ("none", "", "off", "disabled", "false", "0"):
        return ""
    if is_unwanted_caption_text(text):
        return ""
    try:
        if lang in ("hi", "mr", "hindi", "marathi"):
            is_devanagari = True
        elif lang in ("en", "english"):
            is_devanagari = False
        else:
            dev_chars = sum(1 for ch in text if "\u0900" <= ch <= "\u097F")
            lat_chars = sum(1 for ch in text if ch.isalpha() and ch.isascii())
            is_devanagari = dev_chars > lat_chars

        if is_devanagari:
            cleaned = re.sub(r"\b[A-Za-z]{3,}\b", "", text)
        else:
            cleaned = re.sub(r"[\u0900-\u097F]+", "", text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = re.sub(r"\s+([.,!?।])", r"\1", cleaned)
        return cleaned if cleaned else text
    except Exception:
        return text



def redact_secrets(text: str) -> str:
    """Never log or display keys or URLs containing them."""
    if not text:
        return ""
    text_str = str(text)
    poll_key = (os.getenv("POLLINATIONS_API_KEY") or "").strip()
    if poll_key:
        text_str = text_str.replace(poll_key, "[REDACTED]")
    gem_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if gem_key:
        text_str = text_str.replace(gem_key, "[REDACTED]")
    rep_key = (os.getenv("REPLICATE_API_TOKEN") or "").strip()
    if rep_key:
        text_str = text_str.replace(rep_key, "[REDACTED]")
    text_str = re.sub(r"([?&]key=)[^&\s'\"]+", r"\1[REDACTED]", text_str)
    return text_str



def check_pollinations_config() -> Dict[str, Any]:
    """Safe configuration check reporting whether POLLINATIONS_API_KEY is configured
    without printing or exposing the secret key in logs or responses.
    """
    key = (os.getenv("POLLINATIONS_API_KEY") or "").strip()
    configured = bool(key)
    model = (os.getenv("POLLINATIONS_IMAGE_MODEL") or "flux").strip()
    return {
        "configured": configured,
        "status": "configured" if configured else "missing",
        "model": model,
        "endpoint": "https://gen.pollinations.ai/image/" if configured else "https://image.pollinations.ai/prompt/",
    }



# ═══════════════════════════════════════════════════════════════════════════════
# CLOUDFLARE WORKERS AI IMAGE GENERATION PROVIDER
# ═══════════════════════════════════════════════════════════════════════════════

def check_cloudflare_config() -> Dict[str, Any]:
    """Safe configuration check reporting whether Cloudflare Workers AI is configured
    without exposing the account ID or secret API token in logs or UI.
    """
    account_id = (os.getenv("CLOUDFLARE_ACCOUNT_ID") or "").strip()
    api_token = (os.getenv("CLOUDFLARE_API_TOKEN") or "").strip()
    configured = bool(account_id and api_token)
    model = (os.getenv("CLOUDFLARE_IMAGE_MODEL") or "@cf/black-forest-labs/flux-1-schnell").strip()
    return {
        "configured": configured,
        "status": "configured" if configured else "missing",
        "model": model,
        "endpoint": f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}" if configured else "",
    }



# Conservative local safety cap reserving a safety buffer:
# Cloudflare gives 10,000 Neurons/day free. We cap local usage at 7,500 Neurons (25% safety buffer).
_CLOUDFLARE_SAFE_NEURONS_CAP = float(os.getenv("CLOUDFLARE_SAFE_NEURONS_CAP", "7500.0"))
_CLOUDFLARE_NEURONS_USED_TODAY = 0.0
_CLOUDFLARE_QUOTA_EXHAUSTED = False
_CLOUDFLARE_AUTH_ERROR = False
_CLOUDFLARE_LAST_RESET_UTC_DAY = None

def check_and_reset_cloudflare_daily_counter():
    """Resets local usage counters when the UTC calendar day rolls over (Cloudflare resets at 00:00 UTC)."""
    global _CLOUDFLARE_NEURONS_USED_TODAY, _CLOUDFLARE_QUOTA_EXHAUSTED, _CLOUDFLARE_LAST_RESET_UTC_DAY
    from datetime import datetime, timezone
    current_utc_day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if _CLOUDFLARE_LAST_RESET_UTC_DAY != current_utc_day:
        _CLOUDFLARE_NEURONS_USED_TODAY = 0.0
        _CLOUDFLARE_QUOTA_EXHAUSTED = False
        _CLOUDFLARE_LAST_RESET_UTC_DAY = current_utc_day

def reset_cloudflare_quota():
    """Manual reset of the in-memory Cloudflare quota exhaustion flag and tracker."""
    global _CLOUDFLARE_QUOTA_EXHAUSTED, _CLOUDFLARE_AUTH_ERROR, _CLOUDFLARE_NEURONS_USED_TODAY, _CLOUDFLARE_LAST_RESET_UTC_DAY
    from datetime import datetime, timezone
    _CLOUDFLARE_QUOTA_EXHAUSTED = False
    _CLOUDFLARE_AUTH_ERROR = False
    _CLOUDFLARE_NEURONS_USED_TODAY = 0.0
    _CLOUDFLARE_LAST_RESET_UTC_DAY = datetime.now(timezone.utc).strftime("%Y-%m-%d")

def estimate_cloudflare_neurons(width: int = 1024, height: int = 1024, steps: int = 4) -> float:
    """Calculates official Cloudflare Workers AI Neurons for FLUX.1 Schnell:
    - FLUX.1 Schnell natively generates at 1024x1024 (4 tiles of 512x512).
    - Base cost: 4 tiles * 4.80 Neurons = 19.20 Neurons.
    - Step cost: 4 tiles * 4 steps * 9.60 Neurons = 153.60 Neurons.
    - Total per image: 172.80 Neurons.
    """
    tiles = 4  # Native 1024x1024 grid (2x2 tiles of 512x512)
    base_cost = tiles * 4.80
    step_cost = tiles * steps * 9.60
    return round(base_cost + step_cost, 2)


def fit_image_to_aspect_ratio(
    im: Image.Image,
    target_width: int,
    target_height: int,
) -> Image.Image:
    """Proportionally scales an image to fit target canvas without stretching or cropping subject.
    For aspect ratio mismatches (e.g. 1024x1024 or 1280x720 on 1080x1920 9:16 portrait canvas):
    - Background: heavily blurred, darkened version of the image scaled to fill canvas.
    - Foreground: sharp original image scaled proportionally to fit inside canvas without cropping.
    - Result: clean, high-production cinematic presentation with zero distortion or black borders.
    """
    if im.width == target_width and im.height == target_height:
        return im

    src_ar = im.width / im.height
    tgt_ar = target_width / target_height

    # If aspect ratio matches within 1.5%, direct lanczos resize
    if abs(src_ar - tgt_ar) < 0.015:
        return im.resize((target_width, target_height), Image.Resampling.LANCZOS)

    # 1. Background fill: scale proportionally to fill entire target canvas, then heavy blur + dim
    scale_fill = max(target_width / im.width, target_height / im.height)
    bg_w = max(target_width, int(round(im.width * scale_fill)))
    bg_h = max(target_height, int(round(im.height * scale_fill)))
    bg = im.resize((bg_w, bg_h), Image.Resampling.LANCZOS)

    # Center crop background to exact target dimensions
    left = max(0, (bg_w - target_width) // 2)
    top = max(0, (bg_h - target_height) // 2)
    bg = bg.crop((left, top, left + target_width, top + target_height))

    # Apply cinematic blur and dimming to background
    bg = bg.filter(ImageFilter.GaussianBlur(radius=28))
    bg = Image.blend(bg, Image.new("RGB", bg.size, (15, 15, 20)), 0.35)

    # 2. Foreground: scale proportionally to fit inside target canvas without any cropping
    scale_fit = min(target_width / im.width, target_height / im.height)
    fg_w = max(1, int(round(im.width * scale_fit)))
    fg_h = max(1, int(round(im.height * scale_fit)))
    fg = im.resize((fg_w, fg_h), Image.Resampling.LANCZOS)

    # 3. Paste foreground centered onto blurred background
    paste_x = (target_width - fg_w) // 2
    paste_y = (target_height - fg_h) // 2

    canvas = bg.copy()
    canvas.paste(fg, (paste_x, paste_y))
    return canvas

def download_cloudflare_image(
    prompt: str,
    out_path: Path,
    width: int = 1280,
    height: int = 720,
    seed: int = 42,
    scene_num: int = 1,
    client: Optional[httpx.Client] = None,
    on_wait: Optional[Callable[[str], None]] = None,
    draw_placeholder_on_fail: bool = False,
    request_tracker: Optional[Dict[str, Any]] = None,
    negative_prompt: Optional[str] = None,
    **kwargs: Any,
) -> bool:
    """Generates an image via Cloudflare Workers AI using the official REST API.
    Supports @cf/black-forest-labs/flux-1-schnell and @cf/stabilityai/stable-diffusion-xl-base-1.0.
    Handles JSON base64 string or binary image/png responses cleanly.
    Never exposes credentials in logs.
    """
    import base64
    account_id = (os.getenv("CLOUDFLARE_ACCOUNT_ID") or "").strip()
    api_token = (os.getenv("CLOUDFLARE_API_TOKEN") or "").strip()

    if not account_id or not api_token:
        print(f"[pipeline:cloudflare] Missing CLOUDFLARE_ACCOUNT_ID or CLOUDFLARE_API_TOKEN for scene {scene_num}.")
        if draw_placeholder_on_fail:
            draw_placeholder_image(out_path, scene_num, prompt[:60], width=width, height=height)
        return False

    global _CLOUDFLARE_NEURONS_USED_TODAY, _CLOUDFLARE_QUOTA_EXHAUSTED, _CLOUDFLARE_AUTH_ERROR
    check_and_reset_cloudflare_daily_counter()

    if _CLOUDFLARE_AUTH_ERROR:
        err_msg = "[pipeline:cloudflare] Generation stopped: Cloudflare Workers AI authentication failed (HTTP 401: Authentication error)."
        print(err_msg)
        if request_tracker is not None:
            request_tracker["cloudflare_error"] = err_msg
        if on_wait:
            on_wait(err_msg)
        return False

    if _CLOUDFLARE_QUOTA_EXHAUSTED:
        err_msg = "[pipeline:cloudflare] Generation stopped: Cloudflare Workers AI local safety limit reached."
        print(err_msg)
        if request_tracker is not None:
            request_tracker["cloudflare_error"] = err_msg
        if on_wait:
            on_wait(err_msg)
        return False

    model = (os.getenv("CLOUDFLARE_IMAGE_MODEL") or "@cf/black-forest-labs/flux-1-schnell").strip()
    steps_count = int(os.getenv("CLOUDFLARE_INFERENCE_STEPS", "4" if "schnell" in model else "20"))
    # Estimate neurons before sending request (172.8 Neurons for native 1024x1024 @ 4 steps)
    est_neurons = estimate_cloudflare_neurons(width=width, height=height, steps=steps_count)
    if (_CLOUDFLARE_NEURONS_USED_TODAY + est_neurons) > _CLOUDFLARE_SAFE_NEURONS_CAP:
        _CLOUDFLARE_QUOTA_EXHAUSTED = True
        err_msg = (
            f"[pipeline:cloudflare] Hard Safety Stop: Request of {est_neurons} Neurons would exceed "
            f"local conservative safety cap ({_CLOUDFLARE_SAFE_NEURONS_CAP} Neurons; 25% allowance reserved). "
            "Generation halted to prevent any charges."
        )
        print(err_msg)
        if request_tracker is not None:
            request_tracker["cloudflare_error"] = err_msg
        if on_wait:
            on_wait(err_msg)
        return False

    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "prompt": prompt,
        "steps": steps_count,
    }
    # FLUX.1 Schnell strictly forbids additional properties like 'seed' (additionalProperties: false)
    if seed and "flux" not in model.lower():
        payload["seed"] = seed

    # Negative prompt support: only pass if model supports it (e.g. SDXL)
    # FLUX.1 Schnell rejects negative_prompt with HTTP 400 Bad Request
    model_lower = model.lower()
    supports_negative = any(k in model_lower for k in ("stable-diffusion", "sdxl", "sd-", "diffusers"))
    if supports_negative and negative_prompt:
        payload["negative_prompt"] = negative_prompt

    if on_wait:
        on_wait(f"Scene {scene_num}: Generating scene image via Cloudflare Workers AI ({model})...")

    try:
        if client is not None:
            resp = client.post(url, json=payload, headers=headers, timeout=60.0)
        else:
            with httpx.Client(timeout=60.0) as local_c:
                resp = local_c.post(url, json=payload, headers=headers)

        ct = resp.headers.get("content-type", "").lower()
        print(
            f"[pipeline:diagnostic] Scene {scene_num} | Provider: Cloudflare Workers AI | "
            f"Model: {model} | Status: {resp.status_code} | ContentType: {ct or 'unknown'} | "
            f"Bytes: {len(resp.content)}"
        )

        if resp.status_code == 200:
            raw_img_bytes: Optional[bytes] = None
            if "application/json" in ct:
                try:
                    data = resp.json()
                    # Standard Cloudflare Workers AI format: {"result": {"image": "<base64>"}} or {"image": "<base64>"}
                    b64_str = ""
                    if isinstance(data, dict):
                        if "result" in data and isinstance(data["result"], dict):
                            b64_str = data["result"].get("image", "")
                        else:
                            b64_str = data.get("image", "")
                    if b64_str:
                        raw_img_bytes = base64.b64decode(b64_str)
                except Exception as je:
                    print(f"[pipeline:cloudflare] Failed parsing JSON response for scene {scene_num}: {je}")
            elif "image" in ct or len(resp.content) > 500:
                raw_img_bytes = resp.content

            if raw_img_bytes and len(raw_img_bytes) > 200:
                im = Image.open(io.BytesIO(raw_img_bytes)).convert("RGB")
                # Post-generation resize: FLUX.1 Schnell outputs native 1024x1024; crop/resize to video dimensions (e.g. 1280x720)
                if width and height:
                    im = fit_image_to_aspect_ratio(im, width, height)
                if str(out_path).lower().endswith(".png"):
                    im.save(str(out_path), "PNG")
                else:
                    im.save(str(out_path), "JPEG", quality=92)
                _CLOUDFLARE_NEURONS_USED_TODAY += est_neurons
                print(
                    f"[pipeline] Scene {scene_num} image generated successfully via Cloudflare Workers AI. "
                    f"Est. Neurons: {est_neurons} (Local Daily Total: {_CLOUDFLARE_NEURONS_USED_TODAY:.1f}/{_CLOUDFLARE_SAFE_NEURONS_CAP} safe cap)."
                )
                if on_wait:
                    on_wait(f"Scene {scene_num}: image generated via Cloudflare Workers AI.")
                return True

        # Non-200 or decode failure handling
        body_snip = redact_secrets(resp.text[:250])
        print(f"[pipeline:cloudflare] Scene {scene_num} failed (HTTP {resp.status_code}): {body_snip}")

        cf_err_detail = f"HTTP {resp.status_code}"
        try:
            err_data = resp.json()
            if isinstance(err_data, dict) and "errors" in err_data and isinstance(err_data["errors"], list):
                err_msgs = [f"{e.get('message', '')} (code {e.get('code', '')})" for e in err_data["errors"] if isinstance(e, dict)]
                if err_msgs:
                    cf_err_detail += f": {', '.join(err_msgs)}"
            elif isinstance(err_data, dict) and "error" in err_data:
                cf_err_detail += f": {err_data.get('error')}"
        except Exception:
            cf_err_detail += f": {body_snip}"

        if resp.status_code == 401:
            _CLOUDFLARE_AUTH_ERROR = True
            cf_err_detail += " - Cloudflare Workers AI Authentication Error. Verify that CLOUDFLARE_API_TOKEN in .env is valid and has 'Account > Workers AI > Read' (or Edit) permissions for Account ID."
            print(f"[pipeline:cloudflare] Authentication error (HTTP 401). Halting further requests.")
        elif resp.status_code in (429, 402):
            _CLOUDFLARE_QUOTA_EXHAUSTED = True
            cf_err_detail += " - Cloudflare Workers AI quota / rate limit reached."
            print(f"[pipeline:cloudflare] Quota / rate limit encountered (HTTP {resp.status_code}). Halting further requests.")

        if request_tracker is not None:
            request_tracker["cloudflare_error"] = cf_err_detail
            if "scene_errors" not in request_tracker:
                request_tracker["scene_errors"] = {}
            request_tracker["scene_errors"][scene_num] = cf_err_detail

    except Exception as e:
        print(f"[pipeline:cloudflare] Scene {scene_num} exception: {e}")
        if request_tracker is not None:
            request_tracker["cloudflare_error"] = str(e)

    if draw_placeholder_on_fail:
        draw_placeholder_image(out_path, scene_num, prompt[:60], width=width, height=height)
    return False

def download_scene_image(
    prompt: str,
    out_path: Path,
    width: int = 1280,
    height: int = 720,
    seed: int = 42,
    scene_num: int = 1,
    caption: str = "",
    skip_network: bool = False,
    download_fn: Optional[Callable] = None,
    client: Optional[httpx.Client] = None,
    sleep_fn: Optional[Callable[[float], None]] = None,
    on_wait: Optional[Callable[[str], None]] = None,
    max_attempts: int = 3,
    request_tracker: Optional[Dict[str, Any]] = None,
    budget_cap: int = 15,
    draw_placeholder_on_fail: bool = False,
    negative_prompt: Optional[str] = None,
    **kwargs: Any,
) -> bool:
    """Unified scene image generator respecting IMAGE_PROVIDER in .env:
    - 'cloudflare': Uses Cloudflare Workers AI (@cf/black-forest-labs/flux-1-schnell).
    - 'pollinations': Uses Pollinations (paid if configured, else anonymous).
    """
    if download_fn is not None:
        ok = download_fn(prompt, out_path, scene_num)
        if ok and out_path.exists() and out_path.stat().st_size > 500:
            try:
                im = Image.open(out_path).convert("RGB")
                if not validate_image(im):
                    ok = download_fn(prompt, out_path, scene_num)
                    if ok and out_path.exists():
                        im = Image.open(out_path).convert("RGB")
                im = crop_bottom_and_resize(im, 0.06)
                im.save(str(out_path), "JPEG", quality=92)
                return True
            except Exception:
                return True
        return ok

    img_provider = os.getenv("IMAGE_PROVIDER", "pollinations").strip().lower()
    if img_provider in ("cloudflare", "cf", "auto"):
        cf_cfg = check_cloudflare_config()
        if cf_cfg["configured"]:
            return download_cloudflare_image(
                prompt=prompt,
                out_path=out_path,
                width=width,
                height=height,
                seed=seed,
                scene_num=scene_num,
                client=client,
                on_wait=on_wait,
                draw_placeholder_on_fail=draw_placeholder_on_fail,
                request_tracker=request_tracker,
                negative_prompt=negative_prompt,
                **kwargs,
            )
        elif img_provider in ("cloudflare", "cf"):
            warn_msg = (
                "[pipeline] IMAGE_PROVIDER=cloudflare is active but Cloudflare credentials "
                "(CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN) are missing. "
                "Gracefully falling back to free Pollinations image generation."
            )
            print(warn_msg)
            if on_wait:
                on_wait(f"Scene {scene_num}: Cloudflare credentials missing, using free Pollinations fallback...")
            # Fall through to download_pollinations_image below

    return download_pollinations_image(
        prompt=prompt,
        out_path=out_path,
        width=width,
        height=height,
        seed=seed,
        scene_num=scene_num,
        caption=caption,
        skip_network=skip_network,
        download_fn=download_fn,
        client=client,
        sleep_fn=sleep_fn,
        on_wait=on_wait,
        max_attempts=max_attempts,
        request_tracker=request_tracker,
        budget_cap=budget_cap,
        draw_placeholder_on_fail=draw_placeholder_on_fail,
        negative_prompt=negative_prompt,
        **kwargs,
    )


def parse_retry_after(resp: httpx.Response, default: int = 20) -> int:
    """Parses number of seconds from 'Retry after Ns' in body or Retry-After header."""
    header_val = resp.headers.get("retry-after")
    if header_val and header_val.strip().isdigit():
        return int(header_val.strip())
    try:
        text = resp.text
        m = re.search(r"Retry after\s*(\d+)\s*s", text, re.IGNORECASE)
        if m:
            return int(m.group(1))
        data = resp.json()
        if isinstance(data, dict):
            err = data.get("error")
            msg = err.get("message", "") if isinstance(err, dict) else str(err)
            m = re.search(r"Retry after\s*(\d+)\s*s", msg, re.IGNORECASE)
            if m:
                return int(m.group(1))
    except Exception:
        pass
    return default


def draw_placeholder_image(
    out_path: Path,
    scene_num: int = 1,
    caption: str = "",
    width: int = 1280,
    height: int = 720,
) -> None:
    """Step 2 fallback: Draw a plain gradient placeholder card with NO caption text and NO 'SCENE N' label."""
    top_color = np.array([20, 24, 38], dtype=float)
    bottom_color = np.array([48, 28, 48], dtype=float)
    grad_arr = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        alpha = y / float(height - 1)
        c = (1.0 - alpha) * top_color + alpha * bottom_color
        grad_arr[y, :, :] = c.astype(np.uint8)

    base = Image.fromarray(grad_arr, mode="RGB")
    base.save(str(out_path), "JPEG", quality=92)


_POLLINATIONS_402_EXHAUSTED = False   # True only when anonymous endpoint ALSO fails
_POLLINATIONS_KEY_EXHAUSTED = False   # True when paid key hits insufficient_balance

def is_pollinations_exhausted() -> bool:
    """Returns True only when BOTH paid key AND anonymous endpoint are exhausted."""
    global _POLLINATIONS_402_EXHAUSTED
    return _POLLINATIONS_402_EXHAUSTED

def reset_pollinations_exhausted():
    global _POLLINATIONS_402_EXHAUSTED, _POLLINATIONS_KEY_EXHAUSTED
    _POLLINATIONS_402_EXHAUSTED = False
    _POLLINATIONS_KEY_EXHAUSTED = False
    if hasattr(download_pollinations_image, "_key_disabled"):
        download_pollinations_image._key_disabled = False
    if hasattr(download_pollinations_image, "_disabled_keys"):
        download_pollinations_image._disabled_keys.clear()


def download_free_cloud_image(
    prompt: str,
    out_path: Path,
    width: int = 1280,
    height: int = 720,
    seed: int = 42,
    request_tracker: Optional[Dict[str, Any]] = None,
    negative_prompt: Optional[str] = None,
    **kwargs: Any,
) -> bool:
    """Free image generation fallback using Hugging Face Spaces (FLUX.1-schnell via Gradio Client).
    Never blocks or hangs; short-circuits if ZeroGPU quota is exhausted.
    """
    if request_tracker and request_tracker.get("free_cloud_image_exhausted"):
        return False

    try:
        from gradio_client import Client
        client = Client("black-forest-labs/FLUX.1-schnell")
        flux_w = 512 if width <= 720 else 768
        flux_h = 768 if height >= 1080 else 512
        res = client.predict(
            prompt=prompt[:500],
            seed=seed,
            randomize_seed=False,
            width=flux_w,
            height=flux_h,
            num_inference_steps=4,
            api_name="/infer",
        )
        if isinstance(res, (tuple, list)) and len(res) > 0:
            img_file = res[0]
            if isinstance(img_file, dict):
                img_file = img_file.get("path")
            if img_file and Path(img_file).exists():
                im = Image.open(img_file).convert("RGB")
                if validate_image(im):
                    im.save(str(out_path), "JPEG", quality=92)
                    return True
    except Exception as e:
        err_str = str(e)
        print(f"[pipeline] Free cloud image fallback notice: {err_str[:200]}")
        if any(w in err_str.lower() for w in ["quota", "exceeded", "zerogpu", "rate limit", "429"]):
            if request_tracker is not None:
                request_tracker["free_cloud_image_exhausted"] = True
                request_tracker["free_cloud_image_error"] = "ZeroGPU quota exceeded"
    return False


def download_pollinations_image(
    prompt: str,
    out_path: Path,
    width: int = 1280,
    height: int = 720,
    seed: int = 42,
    scene_num: int = 1,
    caption: str = "",
    skip_network: bool = False,
    download_fn: Optional[Callable] = None,
    client: Optional[httpx.Client] = None,
    sleep_fn: Optional[Callable[[float], None]] = None,
    on_wait: Optional[Callable[[str], None]] = None,
    max_attempts: int = 3,
    request_tracker: Optional[Dict[str, Any]] = None,
    budget_cap: int = 15,
    draw_placeholder_on_fail: bool = False,
    negative_prompt: Optional[str] = "inset picture, picture-in-picture, photo frame, split-screen, collage, extra people, duplicate faces, bad anatomy",
    **kwargs: Any,
) -> bool:
    """Download scene image from the official Pollinations API:
    https://gen.pollinations.ai/image/{prompt}
    Uses Authorization: Bearer <POLLINATIONS_API_KEY> and model from catalog (default: flux).
    Validates status code, content-type (image/*), and bytes.
    Handles HTTP 429 using Retry-After and bounded retries.
    Fails fast on 401/402 without bypassing auth or quota.
    """
    if download_fn is not None:
        ok = download_fn(prompt, out_path, scene_num)
        if ok and out_path.exists() and out_path.stat().st_size > 500:
            try:
                im = Image.open(out_path).convert("RGB")
                if not validate_image(im):
                    ok = download_fn(prompt, out_path, scene_num)
                    if ok and out_path.exists():
                        im = Image.open(out_path).convert("RGB")
                im = crop_bottom_and_resize(im, 0.06)
                im.save(str(out_path), "JPEG", quality=92)
                return True
            except Exception:
                pass
        if draw_placeholder_on_fail:
            draw_placeholder_image(out_path, scene_num, caption, width=width, height=height)
        return False

    if skip_network:
        if draw_placeholder_on_fail:
            draw_placeholder_image(out_path, scene_num, caption, width=width, height=height)
        return False

    global _POLLINATIONS_402_EXHAUSTED, _POLLINATIONS_KEY_EXHAUSTED
    if getattr(download_pollinations_image, "_key_disabled", False):
        _POLLINATIONS_KEY_EXHAUSTED = True
    if request_tracker is not None and request_tracker.get("pollinations_key_exhausted"):
        _POLLINATIONS_KEY_EXHAUSTED = True
    if request_tracker is not None and request_tracker.get("pollinations_exhausted"):
        _POLLINATIONS_402_EXHAUSTED = True

    # Short-circuit ALL Pollinations only if the anonymous endpoint is also known to be down
    if _POLLINATIONS_402_EXHAUSTED:
        print(f"[pipeline] Pollinations fully short-circuited (paid key + anonymous both exhausted). Skipping scene {scene_num}.")
        if not skip_network and download_fn is None:
            try:
                ok_fb = download_free_cloud_image(prompt, out_path, width=width, height=height, seed=seed, request_tracker=request_tracker)
                if ok_fb and out_path.exists() and out_path.stat().st_size > 500:
                    return True
            except Exception:
                pass
        if draw_placeholder_on_fail:
            draw_placeholder_image(out_path, scene_num, caption, width=width, height=height)
        return False

    poll_key = (os.getenv("POLLINATIONS_API_KEY") or "").strip()
    # If the paid key is known to be exhausted, immediately skip to anonymous endpoint
    if poll_key and _POLLINATIONS_KEY_EXHAUSTED:
        poll_key = ""
        if on_wait:
            on_wait(f"Scene {scene_num}: Paid Pollinations key has 0 balance – using free anonymous endpoint (rate-limited to 1 image/45s).")

    model = (os.getenv("POLLINATIONS_IMAGE_MODEL") or "flux").strip()
    encoded = urllib.parse.quote(prompt[:3000])

    if poll_key:
        url = f"https://gen.pollinations.ai/image/{encoded}"
        params = {
            "model": model,
            "width": width,
            "height": height,
            "seed": seed,
            "nologo": "true",
            "key": poll_key,
        }
        if negative_prompt:
            params["negative"] = negative_prompt
        headers = {
            "Authorization": f"Bearer {poll_key}",
        }
    else:
        # Anonymous endpoint: free but rate-limited to 1 image per 45 seconds per IP
        url = f"https://image.pollinations.ai/prompt/{encoded}"
        params = {
            "width": width,
            "height": height,
            "seed": seed,
            "nologo": "true",
        }
        if negative_prompt:
            params["negative"] = negative_prompt
        headers = {}

    sleep = sleep_fn if sleep_fn is not None else time.sleep
    rate_limit_retries = 0
    max_rate_limit_retries = 2
    attempts_left = max_attempts

    while attempts_left > 0:
        attempts_left -= 1
        if request_tracker is not None:
            if request_tracker.get("requests_made", 0) >= budget_cap:
                print(f"[pipeline] Image request budget cap ({budget_cap}) reached for scene {scene_num}.")
                break
            request_tracker["requests_made"] = request_tracker.get("requests_made", 0) + 1

        try:
            if client is not None:
                resp = client.get(url, params=params, headers=headers, timeout=45.0, follow_redirects=True)
            else:
                with httpx.Client(timeout=45.0, follow_redirects=True) as local_c:
                    resp = local_c.get(url, params=params, headers=headers)

            ct = resp.headers.get("content-type", "").lower()
            endpoint_type = "paid (gen.pollinations.ai)" if "gen.pollinations.ai" in url else "anonymous (image.pollinations.ai)"
            has_auth = "Authorization" in headers
            print(
                f"[pipeline:diagnostic] Scene {scene_num} | Provider: Pollinations | "
                f"Endpoint: {endpoint_type} | Auth: {has_auth} | Status: {resp.status_code} | "
                f"ContentType: {ct or 'unknown'} | Bytes: {len(resp.content)}"
            )
            if resp.status_code == 200 and ("image" in ct or not ct) and len(resp.content) > 100:
                try:
                    im = Image.open(io.BytesIO(resp.content)).convert("RGB")
                    if validate_image(im):
                        im = crop_bottom_and_resize(im, 0.06)
                        im.save(str(out_path), "JPEG", quality=92)
                        return True
                    else:
                        im.save(str(out_path), "JPEG", quality=92)
                        return True
                except Exception as proc_e:
                    print(f"[pipeline] Error processing Pollinations image for scene {scene_num}: {proc_e}")

            # Strict error handling
            if resp.status_code == 401:
                body_snippet = redact_secrets(resp.text[:300])
                err = f"Pollinations authentication failed (HTTP 401): {body_snippet}"
                print(f"[pipeline] {err}")
                _POLLINATIONS_KEY_EXHAUSTED = True
                download_pollinations_image._key_disabled = True
                if request_tracker is not None:
                    request_tracker["pollinations_key_exhausted"] = True
                    request_tracker["pollinations_402_error"] = err
                if on_wait:
                    on_wait(f"Scene {scene_num}: Pollinations key invalid (HTTP 401). Switching to anonymous endpoint.")
                break  # Fall through to anonymous fallback

            elif resp.status_code == 403:
                body_snippet = redact_secrets(resp.text[:300])
                err = f"Pollinations forbidden / bad permissions (HTTP 403): {body_snippet}"
                print(f"[pipeline] {err}")
                _POLLINATIONS_KEY_EXHAUSTED = True
                download_pollinations_image._key_disabled = True
                if request_tracker is not None:
                    request_tracker["pollinations_key_exhausted"] = True
                    request_tracker["pollinations_402_error"] = err
                if on_wait:
                    on_wait(f"Scene {scene_num}: Pollinations permissions error (HTTP 403). Switching to anonymous endpoint.")
                break  # Fall through to anonymous fallback

            elif resp.status_code == 402:
                body_snippet = redact_secrets(resp.text[:300])
                is_free_limit = "free_limit" in resp.text.lower() or "free limit" in resp.text.lower()
                if is_free_limit:
                    retry_s = parse_retry_after(resp, default=20)
                    wait_s = min(60.0, max(float(retry_s + 3), float(retry_s)))
                    if request_tracker is not None:
                        request_tracker["total_wait"] = request_tracker.get("total_wait", 0) + wait_s
                        tot_wait = request_tracker["total_wait"]
                        max_total_wait = request_tracker.get("max_total_wait", 360.0)
                        if tot_wait > max_total_wait:
                            err = f"Pollinations rate limit timeout exceeded ({max_total_wait:.0f}s) for scene {scene_num}."
                            print(f"[pipeline] {err}")
                            raise RuntimeError(err)
                    msg = f"Pollinations rate limit (402 free limit); waiting {wait_s:.1f} s (scene {scene_num} of 5)..."
                    print(f"[pipeline] {msg}")
                    if on_wait:
                        on_wait(msg)
                    if rate_limit_retries < max_rate_limit_retries:
                        rate_limit_retries += 1
                    else:
                        break
                    sleep(wait_s)
                    continue
                else:
                    err = f"Pollinations paid key has insufficient balance (HTTP 402): {body_snippet}"
                    print(f"[pipeline] {err}")
                    _POLLINATIONS_KEY_EXHAUSTED = True
                    download_pollinations_image._key_disabled = True
                    if request_tracker is not None:
                        request_tracker["pollinations_key_exhausted"] = True
                        request_tracker["pollinations_402_error"] = err
                    if on_wait:
                        on_wait(
                            f"Scene {scene_num}: Pollinations paid key has 0 balance (HTTP 402). "
                            "Switching to free anonymous endpoint (1 image/45s rate limit)."
                        )
                    # Do NOT set _POLLINATIONS_402_EXHAUSTED – fall through to anonymous endpoint below
                    break  # Exit paid-key retry loop; anonymous fallback will run next

            elif resp.status_code in (429, 503):
                retry_s = parse_retry_after(resp, default=20)
                wait_s = min(60.0, max(float(retry_s + 3), float(retry_s)))
                if request_tracker is not None:
                    request_tracker["total_wait"] = request_tracker.get("total_wait", 0) + wait_s
                    tot_wait = request_tracker["total_wait"]
                    max_total_wait = request_tracker.get("max_total_wait", 360.0)
                    if tot_wait > max_total_wait:
                        err = f"Pollinations rate limit timeout exceeded ({max_total_wait:.0f}s) for scene {scene_num}."
                        print(f"[pipeline] {err}")
                        raise RuntimeError(err)
                msg = f"Pollinations temporary limit ({resp.status_code}); waiting {wait_s:.1f} s (scene {scene_num} of 5)..."
                print(f"[pipeline] {msg}")
                if on_wait:
                    on_wait(msg)

                if rate_limit_retries < max_rate_limit_retries:
                    rate_limit_retries += 1
                else:
                    break
                sleep(wait_s)
                continue

            elif resp.status_code >= 400:
                body_snippet = redact_secrets(resp.text[:300])
                err = f"Pollinations image generation failed (HTTP {resp.status_code}): {body_snippet}"
                print(f"[pipeline] {err}")
                if attempts_left <= 0:
                    break

        except RuntimeError:
            raise
        except Exception as e:
            safe_err = redact_secrets(str(e))
            print(f"[pipeline] Pollinations download attempt failed for scene {scene_num}: {safe_err}")
            if attempts_left > 0:
                sleep(2.0)

    # If the paid key was used and failed with 402, retry once using the anonymous endpoint
    current_poll_key = (os.getenv("POLLINATIONS_API_KEY") or "").strip()
    if current_poll_key and _POLLINATIONS_KEY_EXHAUSTED and not skip_network and download_fn is None:
        print(f"[pipeline] Paid key exhausted – retrying scene {scene_num} via anonymous image.pollinations.ai")
        anon_encoded = urllib.parse.quote(prompt[:3000])
        anon_url = f"https://image.pollinations.ai/prompt/{anon_encoded}"
        anon_params = {"width": width, "height": height, "seed": seed, "nologo": "true"}
        if negative_prompt:
            anon_params["negative"] = negative_prompt
        anon_retries = 0
        max_anon_retries = 2
        while anon_retries <= max_anon_retries:
            try:
                if client is not None:
                    anon_resp = client.get(anon_url, params=anon_params, timeout=60.0, follow_redirects=True)
                else:
                    with httpx.Client(timeout=60.0, follow_redirects=True) as lc:
                        anon_resp = lc.get(anon_url, params=anon_params)
                anon_ct = anon_resp.headers.get("content-type", "").lower()
                if anon_resp.status_code == 200 and ("image" in anon_ct or not anon_ct) and len(anon_resp.content) > 100:
                    try:
                        im = Image.open(io.BytesIO(anon_resp.content)).convert("RGB")
                        im = crop_bottom_and_resize(im, 0.06)
                        im.save(str(out_path), "JPEG", quality=92)
                        print(f"[pipeline] Scene {scene_num} image generated via anonymous Pollinations (free tier).")
                        if on_wait:
                            on_wait(f"Scene {scene_num}: image generated via Pollinations free anonymous endpoint.")
                        return True
                    except Exception as ae:
                        print(f"[pipeline] Anonymous Pollinations image save error: {ae}")
                        break
                elif anon_resp.status_code == 402:
                    is_free_limit = "free_limit" in anon_resp.text.lower() or "free limit" in anon_resp.text.lower()
                    if is_free_limit:
                        retry_s = parse_retry_after(anon_resp, default=45)
                        wait_s = min(120.0, retry_s + 3)
                        msg = f"Scene {scene_num}: Anonymous Pollinations rate-limited (FREE_LIMIT); waiting {wait_s:.0f}s..."
                        print(f"[pipeline] {msg}")
                        if on_wait:
                            on_wait(msg)
                        if anon_retries < max_anon_retries:
                            sleep(wait_s)
                            anon_retries += 1
                            continue
                    else:
                        # Anonymous endpoint also hitting pay-wall – mark full exhaustion
                        _POLLINATIONS_402_EXHAUSTED = True
                        if request_tracker is not None:
                            request_tracker["pollinations_exhausted"] = True
                    break
                elif anon_resp.status_code in (429, 503):
                    retry_s = parse_retry_after(anon_resp, default=20)
                    wait_s = min(120.0, retry_s + 3)
                    if anon_retries < max_anon_retries:
                        sleep(wait_s)
                        anon_retries += 1
                        continue
                    break
                else:
                    break
            except RuntimeError:
                raise
            except Exception as ae:
                print(f"[pipeline] Anonymous Pollinations attempt failed: {ae}")
                break
            anon_retries += 1

    # Try free cloud image fallback (FLUX.1-schnell via Gradio) if all Pollinations failed
    if not skip_network and download_fn is None:
        try:
            ok_fb = download_free_cloud_image(prompt, out_path, width=width, height=height, seed=seed, request_tracker=request_tracker)
            if ok_fb and out_path.exists() and out_path.stat().st_size > 500:
                print(f"[pipeline] Scene {scene_num} image generated successfully via free cloud fallback.")
                if on_wait:
                    on_wait(f"Scene {scene_num}: generated using free cloud image model.")
                return True
        except Exception as fb_e:
            print(f"[pipeline] Free cloud image fallback failed: {fb_e}")

    if draw_placeholder_on_fail:
        draw_placeholder_image(out_path, scene_num, caption, width=width, height=height)
    return False

download_pollinations_image._key_disabled = False
download_pollinations_image._disabled_keys = set()


def split_layers(img_path: Path, out_dir: Path) -> Optional[Tuple[Path, Path]]:
    """C8: LAYERS: if a background-removal step is available, split each scene image into
    background and subject layers so subjects can move separately (parallax).
    If it is not available or fails, returns None. Never fails the video.
    """
    try:
        import rembg
        with open(img_path, "rb") as f:
            input_bytes = f.read()
        subject_bytes = rembg.remove(input_bytes)
        subject_path = out_dir / f"subject_{img_path.name}"
        with open(subject_path, "wb") as f:
            f.write(subject_bytes)
        bg_path = out_dir / f"bg_{img_path.name}"
        shutil.copy2(img_path, bg_path)
        return bg_path, subject_path
    except Exception as e:
        return None


def generate_procedural_particles(
    effect: str,
    width: int,
    height: int,
    duration: float,
    out_dir: Path,
    fps: int = 12,
    max_particles: int = 35,
) -> Optional[Path]:
    """C9: Generates procedural particle overlay in code (capped at max 35 particles).
    Returns path to an animated transparent video or None on error.
    """
    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        num_frames = max(4, int(min(duration, 2.0) * fps))
        frame_paths = []

        import random
        rng = random.Random(42)
        particles = [
            {
                "x": rng.uniform(0, width),
                "y": rng.uniform(0, height),
                "speed_x": rng.uniform(-2, 2) if effect not in ["rain"] else -4,
                "speed_y": rng.uniform(8, 16) if effect == "rain" else (rng.uniform(2, 6) if effect == "snow" else rng.uniform(-4, -1) if effect == "embers" else rng.uniform(-1, 1)),
                "size": rng.randint(2, 5) if effect != "rain" else rng.randint(10, 20),
                "alpha": rng.randint(140, 220),
            }
            for _ in range(min(max_particles, 35))
        ]

        for f_idx in range(num_frames):
            frame_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(frame_img)

            for p in particles:
                px = (p["x"] + p["speed_x"] * f_idx * 4) % width
                py = (p["y"] + p["speed_y"] * f_idx * 4) % height
                sz = p["size"]
                alpha = p["alpha"]

                if effect == "rain":
                    draw.line([(px, py), (px - 4, py + sz)], fill=(200, 220, 255, alpha), width=2)
                elif effect == "snow":
                    draw.ellipse([px - sz, py - sz, px + sz, py + sz], fill=(255, 255, 255, alpha))
                elif effect in ["sparkles", "glow"]:
                    draw.line([(px - sz, py), (px + sz, py)], fill=(255, 220, 100, alpha), width=2)
                    draw.line([(px, py - sz), (px, py + sz)], fill=(255, 220, 100, alpha), width=2)
                elif effect == "embers":
                    draw.ellipse([px - sz, py - sz, px + sz, py + sz], fill=(255, 120, 40, alpha))
                elif effect == "leaves":
                    draw.polygon([(px, py - sz), (px + sz, py), (px, py + sz), (px - sz, py)], fill=(210, 140, 50, alpha))
                else:
                    draw.ellipse([px - sz, py - sz, px + sz, py + sz], fill=(230, 240, 255, int(alpha * 0.7)))

            frame_p = out_dir / f"particle_{effect}_{f_idx:03d}.png"
            frame_img.save(str(frame_p), "PNG")
            frame_paths.append(frame_p)

        out_overlay_mov = out_dir / f"overlay_{effect}.mov"
        input_pattern = str(out_dir / f"particle_{effect}_%03d.png").replace("\\", "/")
        cmd = [
            ffmpeg_exe, "-y",
            "-framerate", str(fps),
            "-i", input_pattern,
            "-c:v", "png",
            "-pix_fmt", "rgba",
            "-t", str(duration),
            str(out_overlay_mov),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return out_overlay_mov if out_overlay_mov.exists() else None
    except Exception as e:
        print(f"[pipeline] Procedural overlay generation failed ({e}), skipping overlay.")
        return None
    finally:
        for fp in frame_paths:
            try:
                fp.unlink()
            except Exception:
                pass


def animate_scene_image(
    image_path: Path,
    out_clip_path: Path,
    motion_instructions: str = "",
    visual_prompt: str = "",
    width: int = 1080,
    height: int = 1920,
    duration: float = 4.0,
    fps: int = 24,
    scene_idx: int = 0,
    camera: str = "medium",
    emotion: str = "calm",
    cues: Optional[List[Dict[str, Any]]] = None,
    aspect_ratio: str = "16:9",
    character_info: Optional[str] = None,
    on_progress: Optional[Callable[[str], None]] = None,
) -> bool:
    """Dynamic motion generation coordinated by VideoModelAdapter.
    Executes genuine cloud/local AI video if configured and funded,
    or truthful, artifact-free FFmpeg cinematic camera motion.
    """
    from video_adapter import video_adapter
    ok, engine, meta = video_adapter.generate_motion(
        image_path=image_path,
        out_clip_path=out_clip_path,
        prompt=visual_prompt,
        motion_instructions=motion_instructions,
        width=width,
        height=height,
        duration=duration,
        fps=fps,
        scene_idx=scene_idx,
        camera=camera,
        emotion=emotion,
        cues=cues,
        aspect_ratio=aspect_ratio,
        on_progress=on_progress,
    )
    return ok


def generate_motion_clip(
    image_path: Path,
    out_clip_path: Path,
    scene_idx: int,
    width: int = 1280,
    height: int = 720,
    fps: int = 24,
    duration: float = 4.0,
    camera: str = "",
    flipped_pan: bool = False,
    emotion: str = "calm",
    cues: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Step 3: Make motion clip with emotion-tuned idle motion speed, color grade,
    animation cues, procedural overlays, and layer parallax if available.
    Fail-safe: falls back to whole-frame idle motion if any advanced step fails.
    """
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    frames = int(duration * fps)

    # Ensure source image is proportionally fitted to target canvas without stretching or cropping subject
    try:
        with Image.open(image_path) as src_im:
            if src_im.size != (width, height):
                fitted = fit_image_to_aspect_ratio(src_im.convert("RGB"), width, height)
                fmt = "PNG" if str(image_path).lower().endswith(".png") else "JPEG"
                if fmt == "PNG":
                    fitted.save(str(image_path), "PNG")
                else:
                    fitted.save(str(image_path), "JPEG", quality=95)
    except Exception as img_err:
        print(f"[pipeline] fit_image_to_aspect_ratio notice: {img_err}")

    emo = EMOTION_STYLE.get(emotion, EMOTION_STYLE["calm"])
    idle_speed = emo.get("idle_speed", 1.0)
    color_grade = emo.get("color_grade", {"tint": (255, 255, 255), "vignette": 0.2, "brightness": 1.0})

    is_close_up = "close" in str(camera).lower()
    max_z = round(min(1.18, (1.06 if is_close_up else 1.12) * (0.9 + 0.1 * idle_speed)), 3)
    step_z = round((0.0004 if is_close_up else 0.0008) * idle_speed, 5)

    mode = scene_idx % 3
    if flipped_pan:
        if mode == 0:
            zoom_vf = (
                f"zoompan=z='if(lte(zoom,1.0),{max_z},max(1.001,zoom-{step_z}))':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                f"d={frames}:s={width}x{height}:fps={fps}"
            )
        elif mode == 1:
            zoom_vf = (
                f"zoompan=z='min(zoom+{step_z},{max_z})':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                f"d={frames}:s={width}x{height}:fps={fps}"
            )
        else:
            zoom_vf = (
                f"zoompan=z={max_z}:x='(1-(on/{frames}))*(iw-iw/zoom)':y='ih/2-(ih/zoom/2)':"
                f"d={frames}:s={width}x{height}:fps={fps}"
            )
    else:
        if mode == 0:
            zoom_vf = (
                f"zoompan=z='min(zoom+{step_z},{max_z})':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                f"d={frames}:s={width}x{height}:fps={fps}"
            )
        elif mode == 1:
            zoom_vf = (
                f"zoompan=z='if(lte(zoom,1.0),{max_z},max(1.001,zoom-{step_z}))':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                f"d={frames}:s={width}x{height}:fps={fps}"
            )
        else:
            zoom_vf = (
                f"zoompan=z={max_z}:x='(on/{frames})*(iw-iw/zoom)':y='ih/2-(ih/zoom/2)':"
                f"d={frames}:s={width}x{height}:fps={fps}"
            )

    # Color grading
    tint = color_grade.get("tint", (255, 255, 255))
    rr = round(tint[0] / 255.0, 2)
    gg = round(tint[1] / 255.0, 2)
    bb = round(tint[2] / 255.0, 2)
    vig = round(color_grade.get("vignette", 0.2) * 3.14159, 2)
    bright = round(color_grade.get("brightness", 1.0) - 1.0, 2)
    pre_filter = f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}"
    vf = f"{pre_filter},{zoom_vf},colorchannelmixer=rr={rr}:gg={gg}:bb={bb},eq=brightness={bright},vignette={vig}"

    # Check for procedural overlay cues
    overlay_path = None
    overlay_cue = None
    if cues:
        for c in cues:
            eff = c.get("effect", "")
            if eff in ANIMATION_LIBRARY.get("overlay", []):
                overlay_cue = c
                break

    try:
        layers = split_layers(image_path, out_clip_path.parent)
        if layers:
            pass
    except Exception as e:
        print(f"[pipeline] Parallax layers step failed ({e}), falling back to whole-frame effects.")

    if overlay_cue:
        try:
            overlay_path = generate_procedural_particles(
                effect=overlay_cue["effect"],
                width=width,
                height=height,
                duration=duration,
                out_dir=out_clip_path.parent,
            )
        except Exception as e:
            print(f"[pipeline] Overlay cue error ({e}), skipping overlay.")

    try:
        if overlay_path and overlay_path.exists():
            c_start = max(0.0, float(overlay_cue.get("start_time", 0.0)))
            c_dur = min(2.0, max(0.5, float(overlay_cue.get("duration", 1.0))))
            c_end = min(duration, c_start + c_dur)
            cmd = [
                ffmpeg_exe, "-y",
                "-loop", "1",
                "-i", str(image_path),
                "-i", str(overlay_path),
                "-filter_complex",
                f"[0:v]{vf}[base];[base][1:v]overlay=0:0:enable='between(t,{c_start:.2f},{c_end:.2f})'[v]",
                "-map", "[v]",
                "-t", str(duration),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                str(out_clip_path),
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode == 0 and out_clip_path.exists() and out_clip_path.stat().st_size > 500:
                return
    except Exception as e:
        print(f"[pipeline] Overlay composite error ({e}), proceeding with whole-frame idle motion.")

    try:
        cmd = [
            ffmpeg_exe, "-y",
            "-loop", "1",
            "-i", str(image_path),
            "-vf", vf,
            "-t", str(duration),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(out_clip_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        print(f"[pipeline] Advanced motion render failed ({e}), using safe fallback zoompan.")
        fallback_vf = f"zoompan=z='min(zoom+0.0008,1.10)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s={width}x{height}:fps={fps},vignette=PI/5"
        cmd = [
            ffmpeg_exe, "-y",
            "-loop", "1",
            "-i", str(image_path),
            "-vf", fallback_vf,
            "-t", str(duration),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(out_clip_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    finally:
        if overlay_path and overlay_path.exists():
            try:
                overlay_path.unlink()
            except Exception:
                pass



def get_word_timestamps(
    narration: str,
    duration: float,
    tts_word_boundaries: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], bool]:
    """B4: After voiceover is generated, get word-level timestamps for each scene:
    TTS word boundaries if available, else forced alignment, else even split by character count.
    Saves as scene.words = [{word, start, end}].
    Must work for every supported caption language.
    All timestamps strictly bounded within [0.0, max(0.1, duration - 0.25)] to ensure no transition overlap.
    Returns (words, is_real).
    """
    try:
        max_t = max(0.2, duration - 0.25)
        if tts_word_boundaries and isinstance(tts_word_boundaries, list) and len(tts_word_boundaries) > 0:
            words = []
            for b in tts_word_boundaries:
                w = str(b.get("word") or b.get("text") or "").strip()
                if not w:
                    continue
                s = max(0.0, float(b.get("start", 0.0)))
                e = max(s + 0.05, float(b.get("end", s + 0.2)))
                if s < max_t:
                    words.append({
                        "word": w,
                        "start": round(s, 3),
                        "end": round(min(max_t, e), 3),
                    })
            if words:
                return words, True
    except Exception as e:
        print(f"[pipeline] get_word_timestamps error parsing TTS boundaries ({e}), falling back to even-split.")

    # Fallback: even split by character count / word count
    try:
        words_raw = narration.strip().split()
        if not words_raw:
            return [], False

        max_t = max(0.2, duration - 0.25)
        total_chars = sum(len(w) for w in words_raw)
        words = []

        if total_chars > 0:
            char_dur = max_t / total_chars
            curr_t = 0.0
            for w in words_raw:
                w_dur = max(0.1, len(w) * char_dur)
                start = curr_t
                end = min(max_t, curr_t + w_dur)
                words.append({
                    "word": w,
                    "start": round(start, 3),
                    "end": round(end, 3),
                })
                curr_t = end
        else:
            dt = max_t / len(words_raw)
            for i, w in enumerate(words_raw):
                words.append({
                    "word": w,
                    "start": round(i * dt, 3),
                    "end": round(min(max_t, (i + 1) * dt), 3),
                })

        return words, False
    except Exception as e:
        print(f"[pipeline] Even-split timestamp fallback error ({e}).")
        return [], False


def segment_words_into_phrases(
    words: List[Dict[str, Any]],
    min_words: int = 3,
    max_words: int = 6,
    target_chars: int = 34,
) -> List[List[Dict[str, Any]]]:
    """Segment word timestamp stream into natural, complete phrases.
    Respects sentence and clause boundaries (punctuation, conjunctions),
    ensures readable phrase chunks (3-6 words), and avoids orphaned single words.
    """
    if not words:
        return []

    n = len(words)
    if n <= max_words:
        return [words]

    chunks: List[List[Dict[str, Any]]] = []
    i = 0

    while i < n:
        best_split = -1
        limit = min(n, i + max_words + 1)

        # 1. Sentence-ending punctuation (. ! ? ।)
        for j in range(i + min_words - 1, limit):
            w_text = str(words[j].get("word", "")).strip()
            if any(w_text.endswith(p) for p in [".", "!", "?", "।"]):
                best_split = j + 1
                break

        # 2. Clause punctuation (, ; : — -)
        if best_split == -1:
            for j in range(i + min_words - 1, limit):
                w_text = str(words[j].get("word", "")).strip()
                if any(w_text.endswith(p) for p in [",", ";", ":", "—", "-"]):
                    best_split = j + 1
                    break

        # 3. Conjunction / preposition boundary
        if best_split == -1:
            conjunctions = {"and", "but", "or", "so", "yet", "for", "with", "when", "as", "while", "that", "where", "because"}
            for j in range(i + min_words, min(n, i + max_words + 1)):
                clean_w = re.sub(r"[^\w]", "", str(words[j].get("word", "")).lower())
                if clean_w in conjunctions:
                    best_split = j
                    break

        # 4. Length/character based split
        if best_split == -1:
            char_count = 0
            split_candidate = min(n, i + 4)
            for j in range(i, min(n, i + max_words)):
                char_count += len(str(words[j].get("word", ""))) + 1
                if char_count > target_chars and j >= i + min_words:
                    split_candidate = j
                    break
                split_candidate = j + 1
            best_split = split_candidate

        # Guard: Avoid leaving an orphaned 1-word chunk at the end
        if n - best_split == 1 and best_split < n:
            best_split = n

        # Ensure forward progress
        if best_split <= i:
            best_split = min(n, i + max_words)

        chunks.append(words[i:best_split])
        i = best_split

    if len(chunks) > 1 and len(chunks[-1]) < 2:
        last = chunks.pop()
        chunks[-1].extend(last)

    return chunks


def burn_timed_captions(
    clip_path: Path,
    words: List[Dict[str, Any]],
    caption_text: str,
    out_path: Path,
    width: int = 1280,
    height: int = 720,
    duration: float = 4.0,
    on_error: Optional[Callable[[str], None]] = None,
) -> bool:
    """B5: Captions: show natural, complete phrases synchronized with narration.
    Highlights the active word with zero empty gaps or flickering.
    Safe area: MarginV positioned above UI controls (~22% from bottom for 9:16).
    """
    # STRICT GUARD: If captions are not explicitly provided, or if unwanted prompt text, NEVER burn captions!
    if not caption_text or not caption_text.strip() or is_unwanted_caption_text(caption_text):
        shutil.copy2(clip_path, out_path)
        return True

    if not words:
        return burn_caption_overlay(clip_path, caption_text, out_path, width=width, height=height, on_error=on_error)

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    ass_path = out_path.parent / f"timed_cap_{out_path.stem}.ass"

    try:
        max_end_time = max(0.1, duration - 0.25)
        chunks = segment_words_into_phrases(words)

        def _fmt_ass_time(sec: float) -> str:
            cs = int(round(sec * 100))
            h = cs // 360000
            m = (cs % 360000) // 6000
            s = (cs % 6000) // 100
            c = cs % 100
            return f"{h}:{m:02d}:{s:02d}.{c:02d}"

        is_devanagari = any("\u0900" <= ch <= "\u097F" for ch in caption_text)
        font_name = ("Nirmala UI" if sys.platform == "win32" else "Noto Sans Devanagari") if is_devanagari else "Arial"
        if height > width:  # 9:16 vertical (e.g. 1080x1920)
            fontsize = 54
            margin_v = int(height * 0.22)  # ~422px safe area above Reels/Shorts controls
            outline = 4
            shadow = 2
        else:  # 16:9 landscape or square
            fontsize = 36
            margin_v = 50
            outline = 3
            shadow = 1

        ass_lines = [
            "[Script Info]",
            "ScriptType: v4.00+",
            f"PlayResX: {width}",
            f"PlayResY: {height}",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            f"Style: Default,{font_name},{fontsize},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{outline},{shadow},2,30,30,{margin_v},1",
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        ]

        for c_idx, chunk in enumerate(chunks):
            if not chunk:
                continue
            chunk_speech_start = chunk[0]["start"]
            chunk_speech_end = chunk[-1]["end"]

            # Display window for the phrase chunk to eliminate empty flashes between words/chunks
            if c_idx + 1 < len(chunks) and chunks[c_idx + 1]:
                next_chunk_start = chunks[c_idx + 1][0]["start"]
                chunk_display_end = min(max_end_time, max(chunk_speech_end + 0.4, next_chunk_start))
            else:
                chunk_display_end = min(max_end_time, max(chunk_speech_end + 0.8, duration - 0.25))

            if chunk_display_end <= chunk_speech_start:
                chunk_display_end = min(max_end_time, chunk_speech_start + 0.5)

            for active_idx, w_active in enumerate(chunk):
                w_start = w_active["start"]
                if active_idx + 1 < len(chunk):
                    w_end = max(w_start + 0.05, chunk[active_idx + 1]["start"])
                else:
                    w_end = max(w_start + 0.05, chunk_display_end)

                w_end = min(max_end_time, w_end)
                if w_end <= w_start:
                    w_end = min(max_end_time, w_start + 0.1)

                formatted_words = []
                for idx, w_item in enumerate(chunk):
                    if idx == active_idx:
                        # Bright golden yellow highlight with solid white for non-active words
                        formatted_words.append(r"{\c&H003BF8FF&}" + w_item["word"] + r"{\c&HFFFFFF&}")
                    else:
                        formatted_words.append(w_item["word"])
                line_str = " ".join(formatted_words)
                ass_lines.append(f"Dialogue: 0,{_fmt_ass_time(w_start)},{_fmt_ass_time(w_end)},Default,,0,0,0,,{line_str}")

        with open(ass_path, "w", encoding="utf-8") as f:
            f.write("\n".join(ass_lines))

        escaped_ass = str(ass_path).replace("\\", "/").replace(":", r"\:")
        cmd = [
            ffmpeg_exe, "-y",
            "-i", str(clip_path),
            "-vf", f"ass='{escaped_ass}'",
            "-map", "0:v",
            "-map", "0:a?",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            str(out_path),
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
        if res.returncode == 0 and out_path.exists() and out_path.stat().st_size > 500:
            return True
        else:
            raise RuntimeError(f"FFmpeg ass filter failed: {res.stderr[:300]}")
    except Exception as e:
        err_msg = f"Timed caption burn failed ({e}), falling back to HarfBuzz overlay"
        print(f"[pipeline] {err_msg}")
        if on_error:
            on_error(err_msg)
        return burn_caption_overlay(clip_path, caption_text, out_path, width=width, height=height, on_error=on_error)
    finally:
        if ass_path.exists():
            try:
                ass_path.unlink()
            except Exception:
                pass


def burn_caption_plain(
    clip_path: Path,
    caption: str,
    out_path: Path,
    width: int = 1280,
    height: int = 720,
) -> bool:
    """Step 4 fallback: Plain burn-in with PIL rounded rectangle at bottom 12% and white text."""
    if not caption or not caption.strip() or is_unwanted_caption_text(caption):
        shutil.copy2(clip_path, out_path)
        return True

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    margin_x = int(width * 0.06)
    box_h = int(height * 0.12)
    box_y = max(10, height - box_h - int(height * (0.22 if height > width else 0.04)))
    box_w = width - (2 * margin_x)

    draw.rounded_rectangle(
        [margin_x, box_y, margin_x + box_w, box_y + box_h],
        radius=10,
        fill=(12, 14, 20, 190),
        outline=(255, 179, 71, 160),
        width=2,
    )

    font_path = get_font_path()
    try:
        font = ImageFont.truetype(str(font_path), 50 if height > width else 32)
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), caption, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = margin_x + max(10, (box_w - tw) // 2)
    ty = box_y + max(5, (box_h - th) // 2)
    draw.text((tx, ty), caption, fill=(255, 255, 255, 255), font=font)

    png_tmp = out_path.parent / f"plain_cap_{out_path.stem}.png"
    img.save(str(png_tmp), "PNG")
    try:
        cmd = [
            ffmpeg_exe, "-y",
            "-i", str(clip_path),
            "-loop", "1",
            "-i", str(png_tmp),
            "-filter_complex", "[0:v][1:v]overlay=0:0:shortest=1[v]",
            "-map", "[v]",
            "-map", "0:a?",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            str(out_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except Exception as e:
        print(f"[pipeline] Plain caption burn failed: {e}")
        shutil.copy2(clip_path, out_path)
        return False
    finally:
        if png_tmp.exists():
            png_tmp.unlink()


def burn_caption_overlay(
    clip_path: Path,
    caption: str,
    out_path: Path,
    width: int = 1280,
    height: int = 720,
    on_error: Optional[Callable[[str], None]] = None,
    **kwargs,
) -> bool:
    """Step 4: Burn caption into clip at bottom 12% safe area with HarfBuzz text shaping.
    Burned into each scene's clip BEFORE crossfading.
    No fades applied to guarantee 100% visibility across every single frame.
    Falls back to burn_caption_plain if shaping fails. Never silently skips captions.
    """
    if not caption or not caption.strip() or is_unwanted_caption_text(caption):
        shutil.copy2(clip_path, out_path)
        return True

    font_path = get_font_path()
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    try:
        cap_img = shape_and_render_multiline_prompt(
            caption,
            font_path,
            font_size=52 if height > width else 34,
            width=width,
            height=height,
            margin_pct=0.06,
            text_color=(255, 255, 255, 255),
            bg_box_color=(12, 14, 20, 180),
            border_color=(255, 107, 53, 130),
        )
        png_tmp = out_path.parent / f"cap_{out_path.stem}.png"
        cap_img.save(str(png_tmp), "PNG")
        try:
            cmd = [
                ffmpeg_exe, "-y",
                "-i", str(clip_path),
                "-loop", "1",
                "-i", str(png_tmp),
                "-filter_complex", "[0:v][1:v]overlay=0:0:shortest=1[v]",
                "-map", "[v]",
                "-map", "0:a?",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-c:a", "copy",
                str(out_path),
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        finally:
            if png_tmp.exists():
                png_tmp.unlink()
    except Exception as e:
        err_msg = f"HarfBuzz caption burn failed ({e}), using plain burn-in fallback"
        print(f"[pipeline] {err_msg}")
        if on_error:
            on_error(err_msg)
        return burn_caption_plain(clip_path, caption, out_path, width=width, height=height)


def check_video_captions(
    video_path: Path,
    scenes: List[Dict[str, str]],
    durations: List[float],
    crossfade_sec: float = 0.25,
    sample_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Automated check: samples a frame from the middle of each scene in final video,
    and confirms the caption region contains bright text pixels (>=180 intensity).
    Returns list of check results per scene.
    """
    from typing import Any
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    results = []

    if sample_dir:
        sample_dir.mkdir(parents=True, exist_ok=True)

    for i, sc in enumerate(scenes):
        cap = sc.get("caption", "").strip()
        dur = durations[i] if i < len(durations) else 4.0
        t_start = sum(durations[j] - crossfade_sec for j in range(i)) if i > 0 else 0.0
        t_mid = max(0.2, t_start + (dur / 2.0))

        sample_img_path = (sample_dir / f"scene_{i+1}.png") if sample_dir else (video_path.parent / f"sample_{video_path.stem}_scene_{i+1}.png")

        try:
            cmd = [
                ffmpeg_exe, "-y",
                "-ss", f"{t_mid:.3f}",
                "-i", str(video_path),
                "-vframes", "1",
                "-f", "image2",
                str(sample_img_path),
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            img = Image.open(sample_img_path).convert("L")
            w, h = img.size

            # Caption box region: Safe area according to aspect ratio
            if h > w:  # 9:16 vertical
                y1 = int(h * 0.68)
                y2 = int(h * 0.85)
                max_gap = int(h * 0.08)
            else:  # 16:9 landscape or square
                y1 = int(h * 0.76)
                y2 = int(h * 0.98)
                max_gap = 45
            x1 = int(w * 0.06)
            x2 = int(w * 0.94)

            crop = img.crop((x1, y1, x2, y2))
            arr = np.array(crop)

            # Count bright white text pixels (intensity >= 180)
            bright_pixels = int(np.sum(arr >= 180))

            # Verify only one text box is present (no duplicate/stacked caption boxes)
            text_rows = np.where(np.sum(arr >= 180, axis=1) > 5)[0]
            if len(text_rows) > 0:
                gaps = np.diff(text_rows)
                large_gaps = int(np.sum(gaps > max_gap))
                one_text_box = (large_gaps <= 1)
            else:
                one_text_box = True

            # If caption was expected, require at least 50 bright pixels and no stacked duplicate boxes
            if cap:
                passed = (bright_pixels >= 50) and one_text_box
            else:
                passed = True

            results.append({
                "scene": i + 1,
                "caption": cap,
                "timestamp": round(t_mid, 2),
                "bright_pixels": bright_pixels,
                "one_text_box": one_text_box,
                "passed": passed,
                "sample_path": str(sample_img_path),
            })
        except Exception as e:
            results.append({
                "scene": i + 1,
                "caption": cap,
                "timestamp": round(t_mid, 2),
                "bright_pixels": 0,
                "one_text_box": True,
                "passed": False,
                "error": str(e),
            })

    return results


def generate_voiceover(
    narration: str,
    voice: str,
    out_audio_path: Path,
    emotion: str = "calm",
    skip_network: bool = False,
    word_boundaries: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[bool, str]:
    """Step 5: Generate voiceover using edge-tts with emotion voice style/rate/pitch.
    Collects word boundaries if available.
    Returns (success, warning_message). warning_message is empty string on success.
    Retries once with .save() fallback if streaming produces an empty file.
    Never silently produces a 0-byte audio file.
    """
    if skip_network or not narration.strip():
        return False, ""

    emo = EMOTION_STYLE.get(emotion, EMOTION_STYLE["calm"])
    rate = emo.get("voice", {}).get("rate", "+0%")
    pitch = emo.get("voice", {}).get("pitch", "+0Hz")
    if isinstance(pitch, str) and pitch.endswith("%"):
        pitch = pitch.replace("%", "Hz")

    async def _tts_stream():
        import edge_tts
        communicate = edge_tts.Communicate(narration, voice, rate=rate, pitch=pitch)
        with open(out_audio_path, "wb") as f:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    f.write(chunk["data"])
                elif chunk["type"] == "WordBoundary" and word_boundaries is not None:
                    start_sec = chunk["offset"] / 10_000_000.0
                    dur_sec = chunk["duration"] / 10_000_000.0
                    word_boundaries.append({
                        "word": chunk["text"],
                        "start": start_sec,
                        "end": start_sec + dur_sec,
                    })

    async def _tts_save():
        import edge_tts
        communicate = edge_tts.Communicate(narration, voice, rate=rate, pitch=pitch)
        await communicate.save(str(out_audio_path))

    # Attempt 1: streaming (collects word boundaries)
    try:
        asyncio.run(asyncio.wait_for(_tts_stream(), timeout=20.0))
        if out_audio_path.exists() and out_audio_path.stat().st_size > 500:
            return True, ""
        print(f"[pipeline] edge-tts streaming produced empty file, retrying with .save()...")
    except Exception as e:
        print(f"[pipeline] edge-tts streaming error ({e}), retrying with .save()...")

    # Attempt 2: fallback .save() (no word boundaries collected on retry)
    try:
        asyncio.run(asyncio.wait_for(_tts_save(), timeout=20.0))
        if out_audio_path.exists() and out_audio_path.stat().st_size > 500:
            print(f"[pipeline] edge-tts .save() fallback succeeded.")
            return True, ""
    except Exception as e2:
        print(f"[pipeline] edge-tts .save() fallback also failed ({e2}).")

    warn = f"voice failed: edge-tts produced no audio for '{narration[:40]}...'"
    print(f"[pipeline] {warn}")
    return False, warn


def mux_clip_audio(
    video_clip_path: Path,
    audio_path: Optional[Path],
    out_path: Path,
    duration: float = 4.0,
) -> None:
    """Pads or trims voiceover audio to exactly match clip duration and muxes with video."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    if audio_path and audio_path.exists() and audio_path.stat().st_size > 500:
        cmd = [
            ffmpeg_exe, "-y",
            "-i", str(video_clip_path),
            "-i", str(audio_path),
            "-filter_complex",
            f"[1:a]apad=whole_dur={duration},atrim=0:{duration}[a]",
            "-map", "0:v",
            "-map", "[a]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-ar", "44100",
            "-ac", "2",
            "-t", str(duration),
            str(out_path),
        ]
    else:
        cmd = [
            ffmpeg_exe, "-y",
            "-i", str(video_clip_path),
            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
            "-map", "0:v",
            "-map", "1:a",
            "-c:v", "copy",
            "-c:a", "aac",
            "-t", str(duration),
            str(out_path),
        ]

    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def stitch_storyboard_clips(
    clip_paths: List[Path],
    out_final_mp4: Path,
    durations: Optional[List[float]] = None,
    crossfade_sec: float = 0.25,
) -> None:
    """Step 6: Concatenate clips using 0.25s crossfade (xfade/acrossfade),
    with graceful fallback to concat demuxer if xfade fails.
    """
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    if len(clip_paths) <= 1:
        if clip_paths:
            shutil.copy2(clip_paths[0], out_final_mp4)
        return

    if not durations or len(durations) != len(clip_paths):
        durations = [4.0] * len(clip_paths)

    try:
        cf = crossfade_sec
        filter_parts = []
        v_prev = "[0:v]"
        a_prev = "[0:a]"
        offset = 0.0

        for i in range(1, len(clip_paths)):
            offset += durations[i - 1] - cf
            v_out = f"[v{i}]" if i < len(clip_paths) - 1 else "[v_final]"
            a_out = f"[a{i}]" if i < len(clip_paths) - 1 else "[a_final]"
            filter_parts.append(f"{v_prev}[{i}:v]xfade=transition=fade:duration={cf}:offset={offset:.3f}{v_out}")
            filter_parts.append(f"{a_prev}[{i}:a]acrossfade=d={cf}:c1=tri:c2=tri{a_out}")
            v_prev = v_out
            a_prev = a_out

        total_stitched_dur = offset + durations[-1]
        fade_start = max(0.0, total_stitched_dur - 1.0)
        filter_parts.append(f"[v_final]fade=t=out:st={fade_start:.3f}:d=1.0[v_faded]")
        filter_parts.append(f"[a_final]afade=t=out:st={fade_start:.3f}:d=1.0[a_faded]")

        cmd = [ffmpeg_exe, "-y"]
        for c in clip_paths:
            cmd.extend(["-i", str(c)])
        cmd.extend([
            "-filter_complex", ";".join(filter_parts),
            "-map", "[v_faded]",
            "-map", "[a_faded]",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-r", "24",
            "-t", f"{total_stitched_dur:.3f}",
            "-c:a", "aac",
            "-ar", "44100",
            "-ac", "2",
            str(out_final_mp4),
        ])
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return
    except Exception as e:
        print(f"[pipeline] xfade stitching encountered issue ({e}), falling back to concat demuxer...")

    concat_list = out_final_mp4.parent / f"concat_{out_final_mp4.stem}.txt"
    lines = [f"file '{c.resolve().as_posix()}'" for c in clip_paths]
    concat_list.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        cmd = [
            ffmpeg_exe, "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-ar", "44100",
            "-ac", "2",
            str(out_final_mp4),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    finally:
        if concat_list.exists():
            concat_list.unlink()


def validate_subtitle_sequence(scenes: List[Dict[str, Any]], expected_lines: List[str]) -> Tuple[bool, List[str]]:
    """Strictly validates that captions and narrations match the expected poetry/shayari sequence:
    - If 4 expected lines:
      * Scene 1 (idx 0): Line 1
      * Scene 2 (idx 1): Line 2
      * Scene 3 (idx 2): Line 3
      * Scene 4 (idx 3): Line 4
      * Scene 5 (idx 4): caption == "" and narration == ""
    Returns: (is_valid, list_of_errors)
    """
    errors = []
    if not expected_lines or len(expected_lines) != 4:
        return True, []

    if len(scenes) < 5:
        errors.append(f"Expected at least 5 scenes, got {len(scenes)}")
        return False, errors

    for idx in range(4):
        exp = expected_lines[idx].strip()
        actual_cap = str(scenes[idx].get("caption") or "").strip()
        actual_nar = str(scenes[idx].get("narration") or "").strip()
        if actual_cap != exp:
            errors.append(f"Scene {idx + 1} caption mismatch: expected '{exp}', got '{actual_cap}'")
        if actual_nar != exp:
            errors.append(f"Scene {idx + 1} narration mismatch: expected '{exp}', got '{actual_nar}'")

    s5_cap = str(scenes[4].get("caption") or "").strip()
    s5_nar = str(scenes[4].get("narration") or "").strip()
    if s5_cap:
        errors.append(f"Scene 5 must have empty caption, got '{s5_cap}'")
    if s5_nar:
        errors.append(f"Scene 5 must have empty narration, got '{s5_nar}'")

    return len(errors) == 0, errors


def check_picture_gemini(
    image_path: Path,
    must_show: List[str],
    accessories: List[Dict[str, str]],
    api_key: str,
    model: str = "gemini-3.8-flash",
    gemini_fn: Optional[Callable] = None,
    setting_detail: str = "",
    characters_present: Optional[List[str]] = None,
    expected_count: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Checks image with Gemini for setting_visible, skin_natural, characters_match,
    character count, inset pictures/split screens, duplicate clones, and anatomy.
    Never blocks or throws; returns None on failure/timeout/rate-limit.
    """
    if gemini_fn is not None:
        try:
            return gemini_fn(image_path, must_show, accessories)
        except Exception:
            return None

    if not api_key or not image_path.exists():
        return None

    import base64
    try:
        with open(image_path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")

        items_str = ", ".join(str(m) for m in must_show) if must_show else "none"
        acc_str = ", ".join(f"{a.get('color', '')} {a.get('object', '')}" for a in accessories) if accessories else "none"
        exp_count = expected_count if expected_count is not None else (len(characters_present) if characters_present is not None else len(must_show))

        question = (
            f"Inspect this image from an animated film. Expected setting: {setting_detail or 'full setting'}. "
            f"Expected characters: {items_str}. Expected character count: {exp_count}. "
            f"Required accessories with exact colors: {acc_str}. "
            "Respond in JSON ONLY with keys: "
            "\"detected_character_count\" (integer count of distinct human characters visible in the main scene), "
            "\"has_inset_picture_or_split_screen\" (boolean, true if there is an inset photo frame, picture-in-picture, split-screen, or collage containing an alternate scene), "
            "\"has_duplicate_characters_or_clones\" (boolean, true if the same person is duplicated, cloned, or appears multiple times), "
            "\"clothing_and_appearance_match\" (boolean, true if characters match their described outfits, hair, and appearance), "
            "\"setting_visible\" (boolean), "
            "\"skin_natural\" (boolean), "
            "\"characters_match\" (boolean), "
            "\"anatomy_clean\" (boolean, false if hands have extra/fused fingers or face is severely distorted)."
        )

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": question},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": b64_data,
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {"response_mime_type": "application/json"},
        }
        with httpx.Client(timeout=12.0) as client:
            resp = client.post(url, json=payload, headers=headers)
            if resp.status_code != 200:
                return None
            data = resp.json()
            raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            res = clean_and_parse_json(raw_text)
            if not isinstance(res, dict):
                return None

            raw_detected = res.get("detected_character_count")
            try:
                detected_count = int(raw_detected) if raw_detected is not None else exp_count
            except Exception:
                detected_count = exp_count

            has_inset = bool(res.get("has_inset_picture_or_split_screen", False))
            has_dups = bool(res.get("has_duplicate_characters_or_clones", False))
            if "no_duplicate_characters" in res and not res["no_duplicate_characters"]:
                has_dups = True
            clothing_match = bool(res.get("clothing_and_appearance_match", res.get("characters_match", True)))
            setting_ok = bool(res.get("setting_visible", True))
            skin_ok = bool(res.get("skin_natural", True))
            anat_ok = bool(res.get("anatomy_clean", True))
            char_ok = bool(res.get("characters_match", True)) and clothing_match

            return {
                "setting_visible": setting_ok,
                "skin_natural": skin_ok,
                "characters_match": char_ok,
                "clothing_and_appearance_match": clothing_match,
                "must_show_visible": char_ok,
                "accessory_colors_ok": skin_ok,
                "readable": setting_ok,
                "detected_character_count": detected_count,
                "expected_character_count": exp_count,
                "has_inset_picture_or_split_screen": has_inset,
                "has_duplicate_characters_or_clones": has_dups,
                "no_duplicate_characters": not has_dups,
                "anatomy_clean": anat_ok,
            }
    except Exception as e:
        print(f"[picture_check] Gemini check skipped: {e}")
        return None


def run_storyboard_pipeline(
    job_id: str,
    topic: str,
    style: str = "3d_animated",
    aspect_ratio: str = "16:9",
    caption_lang: str = "none",
    on_progress: Optional[Callable[[str], None]] = None,
    pre_planned_scenes: Optional[List[Dict[str, str]]] = None,
    skip_network: bool = False,
    delays: Tuple[float, ...] = (20.0, 30.0, 45.0),
    download_fn: Optional[Callable] = None,
    caption_samples_dir: Optional[Path] = None,
    gemini_fn: Optional[Callable] = None,
    client: Optional[httpx.Client] = None,
    sleep_fn: Optional[Callable[[float], None]] = None,
    length_seconds: int = 25,
    keep_consistency: bool = False,
) -> Tuple[Path, List[Dict[str, str]], List[int], List[Dict[str, Any]]]:
    """Executes the full 5-scene storyboard video pipeline.
    Returns: (output_mp4_path, scenes_list, placeholder_scenes_list, caption_check_results)
    """
    from typing import Any
    if aspect_ratio == "9:16":
        w, h = 1080, 1920
    elif aspect_ratio == "1:1":
        w, h = 1080, 1080
    else:
        w, h = 1280, 720
    seed = abs(hash(job_id)) % 10000000

    def report(text: str):
        if on_progress:
            on_progress(text)
        print(f"[storyboard {job_id}] {text}")

    # 1. Scenes planning
    report("Planning storyboard scenes...")
    from planner import extract_explicit_devanagari_lines
    explicit_lines = extract_explicit_devanagari_lines(topic)
    if pre_planned_scenes and len(pre_planned_scenes) == 5:
        scenes = pre_planned_scenes
        sheets = scenes[0].get("sheets") if "sheets" in scenes[0] else extract_story_sheets(topic)
        story_chars = sheets.get("characters") or [] if isinstance(sheets, dict) else []
        has_human = any(c.get("is_human", True) for c in story_chars if isinstance(c, dict)) if story_chars else True
        if style == "3d_animated":
            if has_human:
                style_text = "stylized 3D animation, expressive eyes, vibrant colors, studio lighting"
            else:
                style_text = "stylized 3D animation, expressive eyes, soft toy-like fur, vibrant colors"
        else:
            style_text = STYLE_PRESET_PROMPTS.get(style, STYLE_PRESET_PROMPTS["3d_animated"])
            if has_human:
                style_text = style_text.replace(", soft glossy toy-like fur", "").replace("toy-like fur", "")

        for sc in scenes:
            p = sc.get("image_prompt") or sc.get("visual_prompt", "")
            if p and not any(k in p.lower() for k in ["3d animation", "3d animated", "2d cartoon", "anime style", "claymation", "flat vector", "watercolor", "stylized", "cinematic", "sci-fi", "science-fiction", "photorealistic", "realistic", "dark fantasy"]):
                sc["image_prompt"] = f"{style_text}. {p}".strip()
                sc["visual_prompt"] = sc["image_prompt"]
    else:
        scenes = plan_storyboard_scenes(topic, style=style, caption_lang=caption_lang, length_seconds=length_seconds)

    # Strictly enforce 4-line Devanagari poetry/shayari sequence across all scene plans
    if explicit_lines and len(explicit_lines) == 4:
        for s_i in range(len(scenes)):
            if s_i < 4:
                scenes[s_i]["caption"] = explicit_lines[s_i]
                scenes[s_i]["narration"] = explicit_lines[s_i]
            else:
                scenes[s_i]["caption"] = ""
                scenes[s_i]["narration"] = ""

    sub_ok, sub_errs = validate_subtitle_sequence(scenes, explicit_lines)
    if not sub_ok:
        print(f"[pipeline] Subtitle sequence notice: {sub_errs}")

    for s_i, sc_item in enumerate(scenes, 1):
        chars_s = ", ".join(sc_item.get("characters_present", [])) or "none (establishing)"
        c_cam = sc_item.get("camera", "medium")
        s_det = sc_item.get("setting_detail", "")
        print(f"[plan] Scene {s_i}: camera={c_cam}, characters={chars_s}, setting_detail='{s_det}'")

    sheets = scenes[0].get("sheets") if scenes and "sheets" in scenes[0] else extract_story_sheets(topic)
    char_sheet_desc = sheets.get("character_reference_sheet") if isinstance(sheets, dict) else ""
    if not char_sheet_desc and isinstance(sheets, dict):
        char_sheet_desc = build_character_reference_sheets(sheets)
        sheets["character_reference_sheet"] = char_sheet_desc

    # Requirement 9: Stage - Preparing character references
    report("Preparing character references...")
    if char_sheet_desc:
        print(f"[character_reference] {char_sheet_desc}")

    # Requirement 9: Stage - Creating motion instructions
    report("Creating motion instructions...")
    for s_idx, sc in enumerate(scenes, 1):
        if not sc.get("motion_instructions"):
            sc["motion_instructions"] = build_scene_motion_instructions(sc, sheets, style=style)
            sc["motion_prompt"] = sc["motion_instructions"]
        print(f"[motion_instructions] Scene {s_idx}: {sc['motion_instructions'][:120]}...")

    voice = detect_voice(topic, " ".join(s.get("narration", "") for s in (pre_planned_scenes or [])))
    final_clips: List[Path] = []
    clip_durations: List[float] = []
    placeholder_scenes: List[int] = []

    tmp_dir = VIDEO_DIR / f"storyboard_tmp_{job_id}"
    tmp_dir.mkdir(exist_ok=True)

    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    gemini_model = os.getenv("GEMINI_TEXT_MODEL") or os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    total_gemini_checks = 0

    est_time = "1-2 minutes"
    report(f"Starting storyboard generation (~{est_time} estimated)...")

    # 8. Consistency logging: log first 100 characters of every scene's final prompt and seed
    for idx, sc in enumerate(scenes, 1):
        sc_seed = seed + (idx * 17)
        sc_prompt = sc.get("image_prompt", topic)
        print(f"[consistency] Scene {idx} seed={sc_seed} prompt: {sc_prompt[:100]}")

    request_tracker: Dict[str, Any] = {"requests_made": 0, "total_wait": 0, "max_total_wait": 360.0}
    budget_cap = 15
    sleep = sleep_fn if sleep_fn is not None else time.sleep

    try:
        # Reset provider short-circuit for fresh run
        reset_pollinations_exhausted()

        # Phase A: Master Reference & Scene Image Generation
        report("Generating scene images...")
        scenes_with_image: List[int] = []

        # Master Reference Image Workflow
        master_img_path = VIDEO_DIR / f"{job_id}_master.png"
        is_reference_mode = keep_consistency or (sheets and isinstance(sheets, dict) and any("hanuman" in str(c.get("name", "")).lower() for c in (sheets.get("characters") or [])))

        if is_reference_mode:
            report("Character consistency mode active. Checking master visual reference...")
            # If not yet generated, generate the master reference image once
            if not master_img_path.exists() and not skip_network and download_fn is None:
                master_prompt = scenes[0].get("image_prompt", topic)
                report("Generating master reference image for visual consistency...")
                download_scene_image(
                    prompt=master_prompt,
                    out_path=master_img_path,
                    width=w,
                    height=h,
                    seed=seed,
                    scene_num=1,
                    client=client,
                    sleep_fn=sleep,
                    on_wait=report,
                    request_tracker=request_tracker,
                    budget_cap=budget_cap,
                    negative_prompt="extra people, duplicate faces, split-screen, collage, bad anatomy",
                )

            # Honest Fallback: Reuse master reference image across all 5 scenes with subtle progressive camera navigation and environmental lighting
            if master_img_path.exists():
                report("Applying Master Visual Anchor across all 5 scenes (zero character drift)...")
                for m_idx in range(1, len(scenes) + 1):
                    scene_target = tmp_dir / f"scene_{m_idx}.jpg"
                    shutil.copy2(master_img_path, scene_target)
                    scenes_with_image.append(m_idx)
        poll_key = (os.getenv("POLLINATIONS_API_KEY") or "").strip()
        _poll_key_present = bool(poll_key)  # True if key env var is set (may still be exhausted)

        for idx, sc in enumerate(scenes, 1):
            if idx in scenes_with_image:
                continue
            if idx > 1 and not skip_network and download_fn is None:
                # Recalculate each scene: once paid key exhausted, anonymous needs 50 s spacing
                _using_paid = _poll_key_present and not _POLLINATIONS_KEY_EXHAUSTED
                inter_scene_delay = 3.0 if _using_paid else 50.0
                sleep(inter_scene_delay)
            if _CLOUDFLARE_AUTH_ERROR or _CLOUDFLARE_QUOTA_EXHAUSTED:
                report(f"Halting image generation due to provider error: {request_tracker.get('cloudflare_error')}")
                break
            report(f"Scene {idx} of 5: Generating scene image...")
            img_path = tmp_dir / f"scene_{idx}.jpg"
            is_real = download_scene_image(
                prompt=sc.get("image_prompt", topic),
                out_path=img_path,
                width=w,
                height=h,
                seed=seed + (idx * 17),
                scene_num=idx,
                caption=sc.get("caption", ""),
                skip_network=skip_network,
                download_fn=download_fn,
                client=client,
                sleep_fn=sleep,
                on_wait=report,
                request_tracker=request_tracker,
                budget_cap=budget_cap,
                max_attempts=3,
                draw_placeholder_on_fail=False,
                negative_prompt="inset picture, picture-in-picture, photo frame, split-screen, collage, extra people, duplicate faces, bad anatomy",
            )
            if is_real and img_path.exists() and img_path.stat().st_size > 500:
                scenes_with_image.append(idx)
            img_size = img_path.stat().st_size if img_path.exists() else 0
            prompt_str = sc.get("image_prompt", topic)
            print(f"[scene {idx} image] prompt: '{prompt_str}' | words: {len(prompt_str.split())} | size: {img_size} bytes")

        # Phase B: Round 2 (If scenes still have no image, retry for just those scenes)
        missing_scenes = [i for i in range(1, len(scenes) + 1) if i not in scenes_with_image]
        if missing_scenes and not skip_network and download_fn is None:
            if _CLOUDFLARE_AUTH_ERROR or _CLOUDFLARE_QUOTA_EXHAUSTED:
                print(f"[pipeline:cloudflare] Skipping Round 2 retry due to hard provider error: {request_tracker.get('cloudflare_error')}")
            else:
                report(f"Retrying image generation for {len(missing_scenes)} scene(s)...")
            _using_paid_r2 = _poll_key_present and not _POLLINATIONS_KEY_EXHAUSTED
            round2_delay = 2.0 if _using_paid_r2 else 60.0
            sleep(round2_delay)
            for m_idx in missing_scenes:
                report(f"Scene {m_idx} of 5 (Round 2): making image...")
                img_path = tmp_dir / f"scene_{m_idx}.jpg"
                sc = scenes[m_idx - 1]
                is_real = download_scene_image(
                    prompt=sc.get("image_prompt", topic),
                    out_path=img_path,
                    width=w,
                    height=h,
                    seed=seed + (m_idx * 17) + 500,
                    scene_num=m_idx,
                    caption=sc.get("caption", ""),
                    skip_network=skip_network,
                    download_fn=download_fn,
                    client=client,
                    sleep_fn=sleep,
                    on_wait=report,
                    request_tracker=request_tracker,
                    budget_cap=budget_cap,
                    max_attempts=3,
                    draw_placeholder_on_fail=False,
                    negative_prompt="inset picture, picture-in-picture, photo frame, split-screen, collage, extra people, duplicate faces, bad anatomy",
                )
                if is_real and img_path.exists() and img_path.stat().st_size > 500:
                    scenes_with_image.append(m_idx)

        # Phase C: Picture Check (Gemini) - triggers ONLY after all 5 scenes have an image, capped at 10 requests
        if len(scenes_with_image) == 5:
            can_check = (bool(gemini_key) or gemini_fn is not None) and (not skip_network or gemini_fn is not None)
            if can_check and total_gemini_checks < 12:
                for idx in range(1, len(scenes) + 1):
                    if total_gemini_checks >= 12 or request_tracker["requests_made"] >= budget_cap:
                        break
                    sc = scenes[idx - 1]
                    img_path = tmp_dir / f"scene_{idx}.jpg"
                    scene_checks = 0
                    check_status = "skipped"
                    for attempt in range(3):
                        if scene_checks >= 3 or total_gemini_checks >= 12:
                            break
                        scene_checks += 1
                        total_gemini_checks += 1

                        chars_present = sc.get("characters_present") or []
                        exp_count = len(chars_present)

                        chk = check_picture_gemini(
                            img_path,
                            sc.get("must_show", []),
                            sheets.get("accessories", []),
                            gemini_key,
                            gemini_model,
                            gemini_fn=gemini_fn,
                            setting_detail=sc.get("setting_detail", ""),
                            characters_present=chars_present,
                            expected_count=exp_count,
                        )
                        if chk is None:
                            check_status = "skipped"
                            break

                        setting_ok = chk.get("setting_visible", True)
                        skin_ok = chk.get("skin_natural", True)
                        char_ok = chk.get("characters_match", True)
                        prop_ok = chk.get("must_show_visible", True)
                        color_ok = chk.get("accessory_colors_ok", True)
                        dup_ok = not chk.get("has_duplicate_characters_or_clones", False) and chk.get("no_duplicate_characters", True)
                        inset_ok = not chk.get("has_inset_picture_or_split_screen", False)
                        clothing_ok = chk.get("clothing_and_appearance_match", True)
                        anat_ok = chk.get("anatomy_clean", True)

                        det_count = chk.get("detected_character_count")
                        count_ok = True
                        if exp_count is not None and det_count is not None and exp_count > 0:
                            count_ok = (det_count == exp_count)

                        if setting_ok and skin_ok and char_ok and prop_ok and color_ok and dup_ok and inset_ok and clothing_ok and anat_ok and count_ok:
                            check_status = "passed"
                            break
                        else:
                            missing = []
                            if not count_ok:
                                missing.append(f"character count mismatch (expected {exp_count}, found {det_count})")
                            if not inset_ok:
                                missing.append("inset picture or split-screen")
                            if not dup_ok:
                                missing.append("duplicate character or extra people")
                            if not clothing_ok or not char_ok:
                                missing.append("clothing or character mismatch")
                            if not anat_ok:
                                missing.append("hand or anatomical distortion")
                            if not setting_ok:
                                missing.append("setting unclear")
                            if not skin_ok or not color_ok:
                                missing.append("skin or accessory color discolored")

                            if attempt < 2 and request_tracker["requests_made"] < budget_cap:
                                report(f"Scene {idx}: Defect detected ({', '.join(missing)}), retrying {attempt + 1}/2...")
                                retry_seed = seed + (idx * 100) + (attempt + 1) * 37
                                anti_defect = f"clear continuous single camera shot, {', '.join(missing)}"
                                if exp_count == 2:
                                    anti_defect += ", exactly two people together in frame, Kabir wearing dark blue shirt and Meera wearing light pink traditional outfit, no photo frame"
                                elif exp_count == 1:
                                    anti_defect += ", single person only, exactly one character visible, no duplicate copies"
                                emphasized_prompt = sc.get("image_prompt", "") + f", emphasized {anti_defect}"
                                retry_img_path = tmp_dir / f"retry_{idx}_{attempt + 1}.jpg"
                                if not skip_network and download_fn is None:
                                    sleep(2.0)
                                ok_retry = download_scene_image(
                                    prompt=emphasized_prompt,
                                    out_path=retry_img_path,
                                    width=w,
                                    height=h,
                                    seed=retry_seed,
                                    scene_num=idx,
                                    caption=sc.get("caption", ""),
                                    skip_network=skip_network,
                                    download_fn=download_fn,
                                    client=client,
                                    sleep_fn=sleep,
                                    on_wait=report,
                                    request_tracker=request_tracker,
                                    budget_cap=budget_cap,
                                    draw_placeholder_on_fail=False,
                                    negative_prompt="inset picture, picture-in-picture, photo frame, split-screen, collage, extra people, duplicate faces, bad anatomy",
                                )
                                if ok_retry and retry_img_path.exists() and retry_img_path.stat().st_size > 500:
                                    shutil.copy2(retry_img_path, img_path)
                            else:
                                check_status = f"warning: {', '.join(missing)}"
                                report(f"Scene {idx}: Image validation notice: {check_status}")
                                break

                    sc["picture_check"] = check_status
                    report(f"Scene {idx} of 5: picture check {check_status}")

        # Fallback reuse for missing scenes when at least one scene has an image
        missing_scenes = [i for i in range(1, len(scenes) + 1) if i not in scenes_with_image]
        if missing_scenes and scenes_with_image:
            for m in missing_scenes:
                prev_candidates = [s for s in scenes_with_image if s < m]
                donor_s = max(prev_candidates) if prev_candidates else min(scenes_with_image)
                donor_path = tmp_dir / f"scene_{donor_s}.jpg"
                target_path = tmp_dir / f"scene_{m}.jpg"
                if donor_path.exists():
                    shutil.copy2(donor_path, target_path)
                    scenes[m - 1]["is_reused"] = True
                    scenes[m - 1]["reused_from"] = donor_s
                    scenes[m - 1]["flipped_pan"] = True
                    scenes_with_image.append(m)
                    report(f"Scene {m}: reused image from Scene {donor_s}")

        # Phase D: Image Verification
        # If any scene still has no image, fail cleanly with descriptive provider failure details.
        missing_scenes = [i for i in range(1, len(scenes) + 1) if i not in scenes_with_image]
        if missing_scenes:
            missing_str = ", ".join(f"Scene {i}" for i in missing_scenes)
            poll_err = request_tracker.get("pollinations_402_error") if request_tracker else ""
            key_exhausted = _POLLINATIONS_KEY_EXHAUSTED or bool(request_tracker and request_tracker.get("pollinations_key_exhausted"))
            if _POLLINATIONS_402_EXHAUSTED or poll_err:
                err_detail = (
                    "Pollinations insufficient balance (HTTP 402: available balance is 0.0000 pollen) "
                    "and the free anonymous endpoint (image.pollinations.ai) also failed"
                )
                if request_tracker and request_tracker.get("free_cloud_image_error"):
                    err_detail += f". Free cloud fallback (FLUX.1-schnell) also unavailable: {request_tracker.get('free_cloud_image_error')}"
                raise RuntimeError(f"Image generation unavailable for {missing_str}: {err_detail}. Generation halted.")
            elif key_exhausted:
                err_detail = (
                    "Pollinations paid key has 0 balance (HTTP 402). "
                    "Anonymous fallback (image.pollinations.ai) was used but could not complete all scenes. "
                    "Retry in a few minutes when the anonymous rate limit resets."
                )
                raise RuntimeError(f"Image generation incomplete for {missing_str}: {err_detail}")
            cf_err = request_tracker.get("cloudflare_error") if request_tracker else None
            if cf_err:
                raise RuntimeError(f"Image generation failed for {missing_str}: {cf_err}. Generation halted.")
            raise RuntimeError(f"Image generation failed for {missing_str}. Generation halted.")

        # Determine target length for duration calculation
        # length_seconds comes from plan_storyboard_scenes but we re-derive it safely
        scene_count = len(scenes)

        # Phase E: Clip rendering & compilation
        all_voice_warnings: List[str] = []
        for idx, sc in enumerate(scenes, 1):
            img_path = tmp_dir / f"scene_{idx}.jpg"
            scene_emo = sc.get("emotion") or "calm"
            scene_expr = sc.get("expression") or EMOTION_STYLE.get(scene_emo, EMOTION_STYLE["calm"])["expression"]

            # Narration: validate it does not copy topic text; re-ask once via caption regeneration
            narration = sc.get("narration") or sc.get("caption", "")
            try:
                narr_ok, narr_reason = validate_narration(narration, topic)
                if not narr_ok:
                    print(f"[scene {idx}] Narration validation failed ({narr_reason}); re-asking LLM fallback.")
                    # Simple re-generation: ask planner for a single storyteller line
                    storyteller_prompt = (
                        f"Narrate scene {idx} of a story about '{topic[:1000]}' as a storyteller in 1 sentence. "
                        f"Do NOT copy the topic text. Do NOT mention style, characters wearing things, or visual descriptions. "
                        f"Make it narrative and emotional. Return only the narration sentence."
                    )
                    gemini_k = os.getenv("GEMINI_API_KEY", "").strip()
                    if gemini_k and not skip_network:
                        try:
                            import httpx as _httpx
                            _model = os.getenv("GEMINI_TEXT_MODEL") or os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
                            _url = f"https://generativelanguage.googleapis.com/v1beta/models/{_model}:generateContent"
                            _resp = _httpx.post(
                                _url,
                                headers={"x-goog-api-key": gemini_k, "Content-Type": "application/json"},
                                json={"contents": [{"parts": [{"text": storyteller_prompt}]}]},
                                timeout=10.0,
                            )
                            if _resp.status_code == 200:
                                _raw = _resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                                if _raw and len(_raw.split()) >= 3:
                                    narration = _raw
                                    sc["narration"] = narration
                                    print(f"[scene {idx}] Narration re-asked OK.")
                        except Exception as _e:
                            print(f"[scene {idx}] Narration re-ask failed ({_e}), keeping original.")
            except Exception as _ve:
                print(f"[scene {idx}] Narration validation error ({_ve}), keeping original.")

            # Caption: only enabled if user explicitly requested a caption language (not 'none')
            if explicit_lines and len(explicit_lines) == 4 and idx == 5:
                captions_enabled = False
                caption_text = ""
                narration = ""
                sc["caption"] = ""
                sc["narration"] = ""
            else:
                captions_enabled = (str(caption_lang).lower().strip() not in ("none", "", "off", "disabled", "false", "0")) or bool(sc.get("caption") and str(sc.get("caption")).strip())
                caption_text = sc.get("caption", "") if captions_enabled else ""
            if is_unwanted_caption_text(caption_text, sc.get("image_prompt", "")):
                caption_text = ""
            if caption_text:
                try:
                    caption_text = clean_caption_language(caption_text, caption_lang)
                except Exception:
                    pass
            lang_ok = True

            # 2. Pacing: Synthesize voice first if narration is present
            audio_path = tmp_dir / f"voice_{idx}.mp3"
            tts_word_boundaries: List[Dict[str, Any]] = []
            has_voice = False
            voice_warn = ""
            if narration and narration.strip():
                report(f"Scene {idx} of 5: synthesizing voice...")
                try:
                    scene_voice = detect_voice(topic, narration)
                    tts_result = generate_voiceover(
                        narration,
                        scene_voice,
                        audio_path,
                        emotion=scene_emo,
                        skip_network=skip_network,
                        word_boundaries=tts_word_boundaries,
                    )
                    has_voice, voice_warn = tts_result if isinstance(tts_result, tuple) else (tts_result, "")
                except Exception as e:
                    print(f"[pipeline] Voice synthesis error ({e}), proceeding with fallback.")
                    voice_warn = f"voice error: {e}"

                if voice_warn:
                    all_voice_warnings.append(f"Scene {idx}: {voice_warn}")
                    report(f"Warning: {voice_warn}")

            audio_dur = get_audio_duration(audio_path) if has_voice else 0.0

            # Duration: account for crossfade overlap so total stitched duration equals exactly length_seconds
            # For N scenes with crossfade cf (0.25s), total stitched length = sum(durations) - (N - 1) * cf
            # Therefore target per-clip duration = (length_seconds + (scene_count - 1) * 0.25) / scene_count
            target_clip_dur = (float(length_seconds) + (float(scene_count) - 1.0) * 0.25) / float(scene_count)
            scene_dur = max(audio_dur + 0.5 if has_voice else 0.0, target_clip_dur)
            scene_dur = round(scene_dur, 2)
            clip_durations.append(scene_dur)

            # Word timestamps (B4)
            words = []
            is_real_ts = False
            try:
                words, is_real_ts = get_word_timestamps(
                    narration,
                    duration=scene_dur,
                    tts_word_boundaries=tts_word_boundaries,
                )
            except Exception as e:
                print(f"[pipeline] Word timing error ({e}), using empty words.")
            sc["words"] = words

            # Align animation cues with word timestamps (C6, C7)
            cues = sc.get("cues") or []
            if not isinstance(cues, list):
                cues = []
            for c in cues:
                w_idx = c.get("word_index", 0)
                if words and 0 <= w_idx < len(words):
                    c["word"] = words[w_idx]["word"]
                    c["start_time"] = words[w_idx]["start"]
                    c["duration"] = min(2.0, max(0.5, round(words[w_idx]["end"] - words[w_idx]["start"] + 0.5, 2)))
                elif words:
                    c["word"] = words[0]["word"]
                    c["start_time"] = words[0]["start"]
                    c["duration"] = 1.0
                else:
                    c["word"] = ""
                    c["start_time"] = 0.0
                    c["duration"] = 1.0

            # Requirement 11 Logging + new per-scene log (narration, voice_dur, scene_dur, caption_lang_ok)
            print(
                f"[scene {idx}] emotion={scene_emo}, expression='{scene_expr}', "
                f"cues={len(cues)}, timestamps={'real' if is_real_ts else 'fallback'}"
            )
            print(
                f"[scene {idx}] narration='{narration[:60]}...', "
                f"voice_dur={audio_dur:.2f}s, scene_dur={scene_dur:.2f}s, "
                f"caption_lang_ok={lang_ok}"
            )
            for c_i, cue in enumerate(cues):
                print(f"  [cue {c_i+1}] word='{cue.get('word')}', effect='{cue.get('effect')}', start_time={cue.get('start_time', 0.0):.2f}s")

            # 3. Motion clip: genuine image-to-video animation
            report(f"Scene {idx} of {len(scenes)}: Rendering motion & framing...")
            motion_clip = tmp_dir / f"motion_{idx}.mp4"
            animate_scene_image(
                image_path=img_path,
                out_clip_path=motion_clip,
                motion_instructions=sc.get("motion_instructions") or sc.get("motion_prompt", ""),
                visual_prompt=sc.get("image_prompt", topic),
                width=w,
                height=h,
                duration=scene_dur,
                fps=24,
                scene_idx=idx - 1,
                camera=sc.get("camera", "medium"),
                emotion=scene_emo,
                cues=cues,
                aspect_ratio=aspect_ratio,
                character_info=char_sheet_desc,
                on_progress=report,
            )

            # 4. Captions: word-timed captions with highlight (B5) - only if explicitly enabled
            caption_clip = tmp_dir / f"caption_{idx}.mp4"
            if captions_enabled and caption_text and caption_text.strip():
                report(f"Scene {idx} of 5: Combining video and captions...")
                try:
                    burn_timed_captions(
                        motion_clip,
                        words=words,
                        caption_text=caption_text,
                        out_path=caption_clip,
                        width=w,
                        height=h,
                        duration=scene_dur,
                        on_error=report,
                    )
                except Exception as e:
                    print(f"[pipeline] burn_timed_captions exception ({e}), falling back to plain caption.")
                    burn_caption_overlay(motion_clip, caption_text, caption_clip, width=w, height=h, on_error=report)
            else:
                report(f"Scene {idx} of 5: Combining video and audio (captions off)...")
                shutil.copy2(motion_clip, caption_clip)

            # 5. Mux audio & video with dynamic duration
            muxed_clip = tmp_dir / f"muxed_{idx}.mp4"
            mux_clip_audio(caption_clip, audio_path if has_voice else None, muxed_clip, duration=scene_dur)
            final_clips.append(muxed_clip)

            # Persist individual scene clip to videos folder
            scene_dest = VIDEO_DIR / f"{job_id}_scene_{idx - 1}.mp4"
            shutil.copy2(muxed_clip, scene_dest)
            img_dest = VIDEO_DIR / f"{job_id}_scene_{idx - 1}.jpg"
            if img_path.exists():
                shutil.copy2(img_path, img_dest)

        # 6. Stitching with 0.25s crossfade
        report("Rendering final video...")
        out_mp4 = VIDEO_DIR / f"{job_id}.mp4"
        stitch_storyboard_clips(final_clips, out_mp4, durations=clip_durations, crossfade_sec=0.25)

        # Log requested length vs final length
        total_dur = sum(clip_durations)
        print(f"[duration] requested={length_seconds}s, final_total={total_dur:.2f}s")

        # ffmpeg volumedetect check
        try:
            _ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            _vd = subprocess.run(
                [_ffmpeg, "-i", str(out_mp4), "-af", "volumedetect", "-f", "null", "-"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
            )
            _mean_m = re.search(r"mean_volume:\s*([-\d.]+)\s*dB", _vd.stderr)
            _mean_vol = float(_mean_m.group(1)) if _mean_m else None
            print(f"[pipeline] volumedetect mean_volume={_mean_vol} dB")
            if _mean_vol is not None and _mean_vol < -60:
                all_voice_warnings.append(f"video audio appears silent (mean={_mean_vol} dB)")
        except Exception as _ve:
            print(f"[pipeline] volumedetect skipped ({_ve})")

        # Surface voice warnings to caller via report
        if all_voice_warnings:
            report(f"VOICE WARNING: {'; '.join(all_voice_warnings)}")

        # 7. Automated caption check across all scenes in final video (only if captions enabled)
        caption_results = []
        if captions_enabled:
            report("Checking captions visibility across all scenes in final video...")
            sample_dir = caption_samples_dir or (tmp_dir / "caption_samples")
            caption_results = check_video_captions(
                video_path=out_mp4,
                scenes=scenes,
                durations=clip_durations,
                crossfade_sec=0.25,
                sample_dir=sample_dir,
            )

        failed_scenes = [r for r in caption_results if not r["passed"]]
        if failed_scenes:
            for f in failed_scenes:
                idx = f["scene"]
                report(f"Warning: Scene {idx} caption missing ({f.get('bright_pixels', 0)} pixels). Rebuilding with plain burn-in...")
                motion_clip = tmp_dir / f"motion_{idx}.mp4"
                plain_caption_clip = tmp_dir / f"plain_caption_{idx}.mp4"
                burn_caption_plain(motion_clip, scenes[idx - 1].get("caption", ""), plain_caption_clip, width=w, height=h)
                muxed_clip = tmp_dir / f"muxed_{idx}.mp4"
                audio_path = tmp_dir / f"voice_{idx}.mp3"
                has_voice = audio_path.exists() and audio_path.stat().st_size > 500
                mux_clip_audio(plain_caption_clip, audio_path if has_voice else None, muxed_clip, duration=clip_durations[idx - 1])
                final_clips[idx - 1] = muxed_clip

            report("Re-stitching final video after plain burn-in rebuild...")
            stitch_storyboard_clips(final_clips, out_mp4, durations=clip_durations, crossfade_sec=0.25)
            caption_results = check_video_captions(
                video_path=out_mp4,
                scenes=scenes,
                durations=clip_durations,
                crossfade_sec=0.25,
                sample_dir=sample_dir,
            )

        if placeholder_scenes:
            warn_scenes = ", ".join(f"Scene {i}" for i in placeholder_scenes)
            report(f"Completed (Note: {warn_scenes} used placeholder cards)")
        else:
            report("Finished video ready! (Engine: FFmpeg cinematic camera motion)")

        return out_mp4, scenes, placeholder_scenes, caption_results

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


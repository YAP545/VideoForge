"""Video generators. Pick one with PROVIDER in .env: mock, replicate or local."""
import unicode_compat
import json
import os
import shutil
import time
from pathlib import Path


class Provider:
    def generate(self, prompt: str, seconds: int, out_path: Path, **kwargs) -> None:
        raise NotImplementedError


class MockProvider(Provider):
    """Makes a moving colour test video with large, clear prompt text and scene details.
    Uses HarfBuzz and FreeType (via compositor) so Hindi prompts and conjuncts render with
    typographical accuracy. Overlays 'PLACEHOLDER' watermark and scene info.
    """

    def generate(self, prompt, seconds, out_path, scene_idx=0, **kwargs):
        import imageio.v2 as imageio
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        from compositor import get_font_path, shape_and_render_caption

        time.sleep(1.5)  # simulate brief model processing
        w, h, fps = 1280, 720, 24
        n = seconds * fps
        xs = np.linspace(0, 1, w)[None, :]
        ys = np.linspace(0, 1, h)[:, None]

        # Pre-render text overlays on a transparent RGBA canvas
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        font_path = get_font_path()

        try:
            watermark_font = ImageFont.truetype(str(font_path), 22)
            scene_font = ImageFont.truetype(str(font_path), 26)
        except Exception:
            watermark_font = ImageFont.load_default()
            scene_font = ImageFont.load_default()

        # 1. Corner watermark: PLACEHOLDER (DEMO CLIP) - Large, high-contrast and readable
        watermark_text = "PLACEHOLDER · DEMO CLIP"
        draw.rounded_rectangle([24, 24, 380, 74], radius=10, fill=(10, 12, 16, 235), outline=(255, 107, 53, 220), width=2)
        draw.text((36, 35), watermark_text, fill=(255, 107, 53, 255), font=watermark_font)

        # 2. Scene indicator pill - Larger font and clear padding
        scene_num = kwargs.get("scene_num") or (scene_idx + 1 if isinstance(scene_idx, int) else 1)
        scene_label = f"SCENE {scene_num} (MOCK)"
        draw.rounded_rectangle([24, 88, 260, 140], radius=10, fill=(255, 107, 53, 240))
        draw.text((36, 98), scene_label, fill=(255, 255, 255, 255), font=scene_font)

        # 3. Main center prompt box wrapped onto multiple lines with safe margins
        from compositor import shape_and_render_multiline_prompt
        prompt_overlay = shape_and_render_multiline_prompt(
            prompt,
            font_path,
            font_size=38,
            width=w,
            height=h,
            margin_pct=0.06,  # 6% margin (> 5% requirement) on each side
            text_color=(255, 255, 255, 255),
            bg_box_color=(12, 14, 20, 235),
            border_color=(255, 107, 53, 160),
        )

        # Composite prompt box onto main overlay
        overlay = Image.alpha_composite(overlay, prompt_overlay)
        overlay_np = np.array(overlay)
        alpha = overlay_np[..., 3:4] / 255.0
        overlay_rgb = overlay_np[..., :3]

        with imageio.get_writer(
            str(out_path), fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=16
        ) as writer:
            for i in range(n):
                t = i / n
                r = (0.5 + 0.5 * np.sin(2 * np.pi * (xs + t))) * np.ones((h, 1))
                g = (0.5 + 0.5 * np.sin(2 * np.pi * (ys + t * 1.5))) * np.ones((1, w))
                b = 0.5 + 0.5 * np.sin(2 * np.pi * (xs + ys + t * 2))
                frame = (np.stack([r, g, b], axis=-1) * 255).astype("uint8")

                # Moving pulse block in background
                cx = int((0.15 + 0.7 * t) * w)
                frame[h // 2 - 25 : h // 2 + 25, max(cx - 25, 0) : min(cx + 25, w)] = 255

                # Alpha-blend the crisp text overlays
                blended = (frame * (1.0 - alpha) + overlay_rgb * alpha).astype("uint8")
                writer.append_data(blended)


class ReplicateProvider(Provider):
    """Runs open-source video models hosted on replicate.com (e.g. Wan 2.1, LTX-Video).
    Automatically maps real input field names confirmed from replicate.com documentation:
      - ltx-video: 'prompt', 'aspect_ratio', 'cfg', 'steps', 'length' (8n+1)
      - wan-2.1:   'prompt', 'aspect_ratio', 'frame_num' (81 ~ 5s), 'resolution' ('480p')
    """

    def __init__(self):
        if not os.getenv("REPLICATE_API_TOKEN"):
            raise RuntimeError("Set REPLICATE_API_TOKEN in the .env file.")
        self.model = os.getenv("REPLICATE_MODEL", "wan-video/wan-2.1-1.3b")
        if not self.model:
            raise RuntimeError("Set REPLICATE_MODEL in the .env file (e.g. wan-video/wan-2.1-1.3b).")
        self.extra = json.loads(os.getenv("REPLICATE_EXTRA_INPUT", "{}"))
        self.seconds_field = os.getenv("REPLICATE_SECONDS_FIELD")

    def test_connection(self) -> dict:
        """Make a minimal lightweight API call to verify the token without charging money."""
        import replicate

        token = os.getenv("REPLICATE_API_TOKEN", "")
        if not token:
            return {"ok": False, "error": "REPLICATE_API_TOKEN is not set in .env."}
        try:
            client = replicate.Client(api_token=token)
            # Fetch the model metadata (free GET request, no hardware cost)
            m = client.models.get(self.model)
            return {
                "ok": True,
                "model": m.name or self.model,
                "owner": m.owner,
                "message": f"Successfully connected to Replicate! Model '{self.model}' is ready.",
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def generate(self, prompt, seconds, out_path, aspect_ratio="16:9", image_start_url=None, **kwargs):
        import httpx
        import replicate

        inputs = {"prompt": prompt}

        # Check model name for tailored field mappings
        model_lower = self.model.lower()
        if "wan" in model_lower:
            # Verified schema on replicate.com/wan-video/wan-2.1-1.3b
            inputs["aspect_ratio"] = aspect_ratio
            inputs["frame_num"] = 81  # 81 frames ≈ 5s at 16fps
            inputs["resolution"] = "480p"
            if image_start_url:
                inputs["image"] = image_start_url
        elif "ltx" in model_lower:
            # Verified schema on replicate.com/lightricks/ltx-video
            inputs["aspect_ratio"] = aspect_ratio
            fps = 24
            inputs["length"] = 8 * round(seconds * fps / 8) + 1  # 8n+1 constraint
            if image_start_url:
                inputs["image"] = image_start_url
        else:
            if self.seconds_field:
                inputs[self.seconds_field] = seconds

        # Merge user extra overrides
        inputs.update(self.extra)

        output = replicate.run(self.model, input=inputs)
        if isinstance(output, (list, tuple)):
            output = output[0]
        if hasattr(output, "read"):
            data = output.read()
        else:
            resp = httpx.get(str(output), timeout=240, follow_redirects=True)
            resp.raise_for_status()
            data = resp.content
        out_path.write_bytes(data)


class LocalLTXProvider(Provider):
    """Runs LTX-Video on your own NVIDIA GPU with diffusers."""

    def __init__(self):
        import torch
        from diffusers import LTXPipeline

        if not torch.cuda.is_available():
            raise RuntimeError("No NVIDIA GPU found. Use PROVIDER=replicate or PROVIDER=mock.")
        name = os.getenv("LOCAL_MODEL", "Lightricks/LTX-Video")
        self.pipe = LTXPipeline.from_pretrained(name, torch_dtype=torch.bfloat16)
        self.pipe.to("cuda")

    def test_connection(self) -> dict:
        import torch
        if torch.cuda.is_available():
            return {"ok": True, "message": f"NVIDIA GPU available: {torch.cuda.get_device_name(0)}"}
        return {"ok": False, "error": "No CUDA device detected."}

    def generate(self, prompt, seconds, out_path, **kwargs):
        from diffusers.utils import export_to_video

        fps = 24
        frames = 8 * round(seconds * fps / 8) + 1
        video = self.pipe(
            prompt=prompt,
            negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted",
            width=704,
            height=480,
            num_frames=frames,
            num_inference_steps=40,
        ).frames[0]
        export_to_video(video, str(out_path), fps=fps)


class StoryboardProvider(Provider):
    """Free storyboard video provider with zero GPU requirement.
    Uses Pollinations.ai (or local fallback), ffmpeg zoompan motion,
    HarfBuzz captions, and edge-tts voiceover.
    """

    def test_connection(self) -> dict:
        return {
            "ok": True,
            "message": "PROVIDER=storyboard is active. Generates 5-scene videos with pan/zoom motion, captions & edge-tts voiceover.",
        }

    def generate(self, prompt, seconds, out_path, scene_idx=0, aspect_ratio="16:9", caption="", **kwargs):
        from pipeline import (
            burn_caption_overlay,
            detect_voice,
            download_scene_image,
            generate_motion_clip,
            generate_voiceover,
            get_audio_duration,
            is_unwanted_caption_text,
            mux_clip_audio,
        )

        if aspect_ratio == "9:16":
            w, h = 1080, 1920
        elif aspect_ratio == "1:1":
            w, h = 1080, 1080
        else:
            w, h = 1280, 720
        temp_dir = out_path.parent / f"tmp_sb_{out_path.stem}"
        temp_dir.mkdir(exist_ok=True)
        import random
        seed = kwargs.get("seed") or random.randint(100000, 9999999)

        try:
            caption_lang = kwargs.get("caption_lang", "none")
            captions_enabled = str(caption_lang).lower().strip() not in ("none", "", "off", "disabled", "false", "0")
            if not captions_enabled or is_unwanted_caption_text(caption, prompt):
                caption = ""

            img_path = temp_dir / "frame.jpg"
            is_real = download_scene_image(
                prompt=prompt,
                out_path=img_path,
                width=w,
                height=h,
                seed=seed,
                scene_num=scene_idx + 1 if isinstance(scene_idx, int) else 1,
                caption=caption,
                skip_network=kwargs.get("skip_network", False),
                negative_prompt=kwargs.get("negative_prompt"),
            )

            tts_text = caption or kwargs.get("narration", "")
            if is_unwanted_caption_text(tts_text, prompt):
                tts_text = ""

            audio_path = temp_dir / "voice.mp3"
            has_voice = False
            if tts_text:
                voice = detect_voice(tts_text)
                tts_result = generate_voiceover(tts_text, voice, audio_path, skip_network=kwargs.get("skip_network", False))
                has_voice, voice_warn = tts_result if isinstance(tts_result, tuple) else (tts_result, "")
            audio_dur = get_audio_duration(audio_path) if has_voice else 0.0
            clip_dur = max(3.0, round(audio_dur + 0.5, 2)) if seconds <= 5 else float(seconds)

            from pipeline import animate_scene_image
            raw_clip = temp_dir / "raw_motion.mp4"
            animate_scene_image(
                image_path=img_path,
                out_clip_path=raw_clip,
                motion_instructions=kwargs.get("motion_instructions") or "",
                visual_prompt=prompt,
                width=w,
                height=h,
                duration=clip_dur,
                scene_idx=scene_idx if isinstance(scene_idx, int) else 0,
                aspect_ratio=aspect_ratio,
            )

            cap_clip = temp_dir / "cap_motion.mp4"
            if captions_enabled and caption and caption.strip():
                burn_caption_overlay(
                    raw_clip,
                    caption,
                    cap_clip,
                    width=w,
                    height=h,
                    duration=clip_dur,
                    crossfade_sec=0.25,
                    scene_idx=scene_idx if isinstance(scene_idx, int) else 0,
                )
            else:
                shutil.copy2(raw_clip, cap_clip)

            mux_clip_audio(cap_clip, audio_path if has_voice else None, out_path, duration=clip_dur)
            return is_real
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class PollinationsVideoProvider(Provider):
    """Uses official Pollinations Video API models (e.g. alibaba/wan-3.0, kandinskylab/kandinsky-6.0-lite).
    Requires active Pollen balance on the configured POLLINATIONS_API_KEY.
    """

    def __init__(self):
        self.api_key = (os.getenv("POLLINATIONS_API_KEY") or "").strip()
        if not self.api_key:
            raise RuntimeError("Set POLLINATIONS_API_KEY in the .env file.")
        self.model = os.getenv("POLLINATIONS_VIDEO_MODEL", "alibaba/wan-3.0")

    def test_connection(self) -> dict:
        import httpx
        if not self.api_key:
            return {"ok": False, "error": "POLLINATIONS_API_KEY is not configured in .env."}
        try:
            r = httpx.get("https://gen.pollinations.ai/models", timeout=15.0)
            if r.status_code == 200:
                return {"ok": True, "message": f"Connected to Pollinations. Video model: {self.model}"}
            return {"ok": False, "error": f"Pollinations models returned HTTP {r.status_code}"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def generate(self, prompt, seconds, out_path, aspect_ratio="16:9", image_start_url=None, **kwargs):
        import httpx
        import urllib.parse
        enc = urllib.parse.quote(prompt[:800])
        url = f"https://gen.pollinations.ai/image/{enc}"
        params = {"model": self.model, "seed": kwargs.get("seed", 42)}
        if image_start_url:
            params["image"] = image_start_url
        headers = {"Authorization": f"Bearer {self.api_key}"}

        with httpx.Client(timeout=180.0, follow_redirects=True) as client:
            resp = client.get(url, params=params, headers=headers)
            if resp.status_code == 200 and len(resp.content) > 1000:
                out_path.write_bytes(resp.content)
            elif resp.status_code == 402:
                raise RuntimeError(
                    "Pollinations video generation returned HTTP 402 (Insufficient Balance). "
                    "To enable Pollinations AI video, top up at https://enter.pollinations.ai/top-up "
                    "or configure REPLICATE_API_TOKEN in .env."
                )
            else:
                raise RuntimeError(f"Pollinations video generation failed (HTTP {resp.status_code}): {resp.text[:300]}")


def get_provider() -> Provider:
    name = os.getenv("PROVIDER", "storyboard").lower()
    if name == "storyboard":
        return StoryboardProvider()
    if name in ("pollinations_video", "pollinations-video"):
        return PollinationsVideoProvider()
    if name == "mock":
        return MockProvider()
    if name == "replicate":
        return ReplicateProvider()
    if name == "local":
        return LocalLTXProvider()
    raise RuntimeError(f"Unknown PROVIDER '{name}'. Use storyboard, pollinations_video, mock, replicate or local.")

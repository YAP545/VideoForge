"""Video Model Adapter for VideoForge.
Provides dynamic model selection, capabilities pre-flight checking,
truthful motion engine reporting, and video file temporal inspection.
"""
import unicode_compat
import json
import logging
import os
import subprocess
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class VideoEngineStatus:
    engine_id: str
    name: str
    is_available: bool
    genuine_video: bool
    requires_payment: bool
    status_message: str
    model_name: str = ""


class VideoModelAdapter:
    """Universal adapter coordinating genuine AI video generation (Replicate, Pollinations Video, Local LTX)
    and high-fidelity cinematic camera motion (FFmpeg).
    """

    def __init__(self):
        self._cached_capabilities: Optional[Dict[str, Any]] = None
        self._cloud_quota_exhausted: bool = False

    def check_capabilities(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Inspects all configured providers, models, tokens, and hardware."""
        if self._cached_capabilities and not force_refresh:
            return self._cached_capabilities

        provider_env = os.getenv("PROVIDER", "storyboard").lower().strip()
        rep_token = (os.getenv("REPLICATE_API_TOKEN") or "").strip()
        poll_key = (os.getenv("POLLINATIONS_API_KEY") or "").strip()
        poll_video_model = os.getenv("POLLINATIONS_VIDEO_MODEL", "alibaba/wan-3.0").strip()

        engines: Dict[str, VideoEngineStatus] = {}

        # 1. Replicate Engine (Wan 2.1 / LTX-Video)
        rep_model = os.getenv("REPLICATE_MODEL", "wan-video/wan-2.1-1.3b")
        if rep_token:
            engines["replicate"] = VideoEngineStatus(
                engine_id="replicate",
                name="Replicate Cloud Video",
                is_available=True,
                genuine_video=True,
                requires_payment=True,
                status_message=f"Replicate API token configured. Model: {rep_model}",
                model_name=rep_model,
            )
        else:
            engines["replicate"] = VideoEngineStatus(
                engine_id="replicate",
                name="Replicate Cloud Video",
                is_available=False,
                genuine_video=True,
                requires_payment=True,
                status_message="REPLICATE_API_TOKEN is not configured in .env",
                model_name=rep_model,
            )

        # 2. Pollinations Video Engine (Wan 3.0 / Kandinsky 6.0)
        if poll_key:
            # Check balance capability without throwing
            engines["pollinations_video"] = VideoEngineStatus(
                engine_id="pollinations_video",
                name="Pollinations Video",
                is_available=True,
                genuine_video=True,
                requires_payment=True,
                status_message=f"POLLINATIONS_API_KEY configured. Model: {poll_video_model} (requires paid pollen balance for cloud AI video)",
                model_name=poll_video_model,
            )
        else:
            engines["pollinations_video"] = VideoEngineStatus(
                engine_id="pollinations_video",
                name="Pollinations Video",
                is_available=False,
                genuine_video=True,
                requires_payment=True,
                status_message="POLLINATIONS_API_KEY is not configured in .env",
                model_name=poll_video_model,
            )

        # 3. Free Cloud AI Video Diffusion (Hugging Face ZeroGPU via Gradio Client)
        enable_free_cloud = os.getenv("ENABLE_FREE_CLOUD_VIDEO", "1").strip().lower() in ("1", "true", "yes")
        try:
            import gradio_client
            has_gradio_client = True
        except ImportError:
            has_gradio_client = False

        engines["free_cloud_diffusion"] = VideoEngineStatus(
            engine_id="free_cloud_diffusion",
            name="Free Cloud AI Diffusion (ZeroGPU)",
            is_available=has_gradio_client and enable_free_cloud,
            genuine_video=True,
            requires_payment=False,
            status_message="Free cloud AI video diffusion via Hugging Face ZeroGPU (Lightricks/ltx-video-distilled). $0.00 cost, zero disk space.",
            model_name="Lightricks/ltx-video-distilled",
        )

        # 4. Local Diffusers LTX-Video (NVIDIA GPU)
        has_cuda = False
        cuda_device = "None"
        try:
            import torch
            has_cuda = torch.cuda.is_available()
            if has_cuda:
                cuda_device = torch.cuda.get_device_name(0)
        except Exception:
            has_cuda = False

        local_model = os.getenv("LOCAL_MODEL", "Lightricks/LTX-Video")
        if has_cuda:
            engines["local_gpu"] = VideoEngineStatus(
                engine_id="local_gpu",
                name="Local LTX-Video",
                is_available=True,
                genuine_video=True,
                requires_payment=False,
                status_message=f"NVIDIA GPU detected ({cuda_device}). Model: {local_model}",
                model_name=local_model,
            )
        else:
            engines["local_gpu"] = VideoEngineStatus(
                engine_id="local_gpu",
                name="Local LTX-Video",
                is_available=False,
                genuine_video=True,
                requires_payment=False,
                status_message="No NVIDIA CUDA GPU detected (running in CPU mode)",
                model_name=local_model,
            )

        # 4. FFmpeg Cinematic Camera Motion (Always available, zero cost)
        engines["ffmpeg_camera"] = VideoEngineStatus(
            engine_id="ffmpeg_camera",
            name="FFmpeg Cinematic Camera Motion",
            is_available=True,
            genuine_video=False,  # Truthful reporting: camera motion over still images
            requires_payment=False,
            status_message="Local FFmpeg engine: pan, zoom, tilt, and framing over high-resolution still images.",
            model_name="ffmpeg-zoompan",
        )

        # Determine primary candidate
        if provider_env == "replicate" and engines["replicate"].is_available:
            active_engine = "replicate"
        elif provider_env == "pollinations_video" and engines["pollinations_video"].is_available:
            active_engine = "pollinations_video"
        elif provider_env == "local" and engines["local_gpu"].is_available:
            active_engine = "local_gpu"
        elif rep_token:
            active_engine = "replicate"
        else:
            active_engine = "ffmpeg_camera"

        res = {
            "configured_provider": provider_env,
            "active_engine": active_engine,
            "genuine_video_capable": engines[active_engine].genuine_video,
            "requires_payment": engines[active_engine].requires_payment,
            "active_description": engines[active_engine].status_message,
            "engines": {k: v.__dict__ for k, v in engines.items()},
        }
        self._cached_capabilities = res
        return res

    def generate_motion(
        self,
        image_path: Path,
        out_clip_path: Path,
        prompt: str,
        motion_instructions: str = "",
        width: int = 1080,
        height: int = 1920,
        duration: float = 4.0,
        fps: int = 24,
        scene_idx: int = 0,
        camera: str = "medium",
        emotion: str = "calm",
        cues: Optional[List[Dict[str, Any]]] = None,
        aspect_ratio: str = "16:9",
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Dynamically routes motion generation to genuine video AI if configured and funded,
        or truthful FFmpeg camera motion fallback.
        Returns: (success, engine_used, metadata)
        """
        import httpx
        from pipeline import generate_motion_clip

        caps = self.check_capabilities()
        rep_token = (os.getenv("REPLICATE_API_TOKEN") or "").strip()
        poll_key = (os.getenv("POLLINATIONS_API_KEY") or "").strip()
        provider_env = os.getenv("PROVIDER", "storyboard").lower().strip()

        # Route 1: Replicate
        if rep_token and (provider_env in ("replicate", "all")):
            try:
                import replicate
                model_name = os.getenv("REPLICATE_MODEL", "wan-video/wan-2.1-1.3b")
                if on_progress:
                    on_progress(f"Scene {scene_idx + 1}: Generating genuine AI video with Replicate ({model_name})...")
                print(f"[video_adapter] Running Replicate I2V model={model_name} for scene {scene_idx + 1}")
                client = replicate.Client(api_token=rep_token)
                with open(image_path, "rb") as img_f:
                    inputs = {
                        "image": img_f,
                        "prompt": motion_instructions or prompt,
                        "aspect_ratio": aspect_ratio,
                    }
                    output = client.run(model_name, input=inputs)
                    if isinstance(output, (list, tuple)):
                        output = output[0]
                    if hasattr(output, "read"):
                        data = output.read()
                    else:
                        resp = httpx.get(str(output), timeout=240, follow_redirects=True)
                        resp.raise_for_status()
                        data = resp.content
                    out_clip_path.write_bytes(data)
                    if out_clip_path.exists() and out_clip_path.stat().st_size > 1000:
                        return True, "replicate_wan_i2v", {
                            "genuine_video": True,
                            "motion_type": "ai_video_diffusion",
                            "model": model_name,
                        }
            except Exception as e:
                logger.warning(f"[video_adapter] Replicate I2V failed: {e}")

        # Route 2: Pollinations Video
        if poll_key and (provider_env in ("pollinations_video", "pollinations-video")):
            model = os.getenv("POLLINATIONS_VIDEO_MODEL", "alibaba/wan-3.0").strip()
            enc_prompt = urllib.parse.quote((motion_instructions or prompt)[:2500])
            url = f"https://gen.pollinations.ai/image/{enc_prompt}"
            params = {"model": model, "seed": 42 + scene_idx}
            headers = {"Authorization": f"Bearer {poll_key}"}
            try:
                if on_progress:
                    on_progress(f"Scene {scene_idx + 1}: Checking Pollinations video model ({model})...")
                with httpx.Client(timeout=120.0, follow_redirects=True) as client:
                    resp = client.get(url, params=params, headers=headers)
                    if resp.status_code == 200 and len(resp.content) > 1000:
                        out_clip_path.write_bytes(resp.content)
                        return True, f"pollinations_{model}", {
                            "genuine_video": True,
                            "motion_type": "ai_video_diffusion",
                            "model": model,
                        }
                    elif resp.status_code == 402:
                        notice = (
                            f"Notice: Pollinations video generation returned HTTP 402 (Insufficient Balance). "
                            f"Model {model} requires paid Pollen credits. "
                            f"Using high-resolution Pollinations Flux still image with FFmpeg camera motion navigation."
                        )
                        print(f"[video_adapter] {notice}")
                        if on_progress:
                            on_progress(notice)
            except Exception as e:
                logger.warning(f"[video_adapter] Pollinations video call failed: {e}")

        # Route 2b: Free Cloud AI Video Diffusion (ZeroGPU via Gradio Client)
        enable_free_cloud = os.getenv("ENABLE_FREE_CLOUD_VIDEO", "1").strip().lower() in ("1", "true", "yes")
        if enable_free_cloud and not self._cloud_quota_exhausted:
            try:
                from gradio_client import Client, handle_file
                if on_progress:
                    on_progress(f"Scene {scene_idx + 1}: Generating genuine motion with Free Cloud AI Diffusion (LTX-Video)...")
                print(f"[video_adapter] Running Free Cloud Diffusion (LTX-Video) for scene {scene_idx + 1}")
                client = Client("Lightricks/ltx-video-distilled")
                clean_action = motion_instructions or prompt or "The subject moves naturally with fluid cinematic motion"
                
                # Determine aspect ratio dimensions for LTX (e.g. 512x704 for vertical, 704x512 for horizontal)
                if height >= width:
                    ltx_w, ltx_h = 512, 704
                else:
                    ltx_w, ltx_h = 704, 512

                res = client.predict(
                    prompt=clean_action[:250],
                    negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted",
                    input_image_filepath=handle_file(str(image_path)),
                    input_video_filepath=None,
                    height_ui=ltx_h,
                    width_ui=ltx_w,
                    mode="image-to-video",
                    duration_ui=max(2, int(duration)),
                    ui_frames_to_use=9,
                    seed_ui=42 + scene_idx,
                    randomize_seed=True,
                    ui_guidance_scale=1.0,
                    improve_texture_flag=True,
                    api_name="/image_to_video",
                )
                generated_video = res[0] if isinstance(res, (tuple, list)) and res else res
                vid_path_str = generated_video.get("video") if isinstance(generated_video, dict) else generated_video if isinstance(generated_video, str) else None
                if vid_path_str and Path(vid_path_str).is_file() and Path(vid_path_str).stat().st_size > 1000:
                        # Transcode and conform to exact target resolution and duration using FFmpeg
                        temp_in = Path(vid_path_str)
                        cmd = [
                            "ffmpeg", "-y",
                            "-i", str(temp_in),
                            "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1",
                            "-t", str(duration),
                            "-r", str(fps),
                            "-c:v", "libx264",
                            "-pix_fmt", "yuv420p",
                            str(out_clip_path)
                        ]
                        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
                        if out_clip_path.exists() and out_clip_path.stat().st_size > 1000:
                            print(f"[video_adapter] Free Cloud Diffusion generated clip successfully for scene {scene_idx + 1}")
                            return True, "free_cloud_ltx_video", {
                                "genuine_video": True,
                                "motion_type": "ai_video_diffusion",
                                "model": "Lightricks/ltx-video-distilled",
                                "cost": "$0.00 (ZeroGPU)",
                            }
            except Exception as e:
                err_str = str(e).lower()
                if any(w in err_str for w in ["quota", "exceeded", "rate limit", "zerogpu", "gpu limit", "429"]):
                    self._cloud_quota_exhausted = True
                    print(f"[video_adapter] ZeroGPU quota exhausted. Short-circuiting subsequent cloud video calls.")
                logger.warning(f"[video_adapter] Free Cloud Diffusion call failed ({e}), falling back to FFmpeg camera motion.")
                print(f"[video_adapter] Free Cloud Diffusion fallback notice: {e}")
                if on_progress:
                    on_progress(f"Scene {scene_idx + 1}: Notice - Free cloud AI video quota currently exhausted. Generating smooth FFmpeg camera motion (pan/zoom/tilt) over high-resolution still image.")
        elif enable_free_cloud and self._cloud_quota_exhausted:
            if on_progress:
                on_progress(f"Scene {scene_idx + 1}: Notice - Free cloud AI video quota currently exhausted. Generating smooth FFmpeg camera motion (pan/zoom/tilt) over high-resolution still image.")

        # Route 3: FFmpeg Cinematic Camera Motion (Truthful, 100% Reliable, Artifact-Free)
        print(f"[video_adapter] Using FFmpeg cinematic camera motion for scene {scene_idx + 1}")
        if on_progress:
            on_progress(f"Scene {scene_idx + 1} of 5: Disclosed Engine - FFmpeg cinematic camera motion (still image navigation, not AI character movement)...")
        generate_motion_clip(
            image_path=image_path,
            out_clip_path=out_clip_path,
            scene_idx=scene_idx,
            width=width,
            height=height,
            fps=fps,
            duration=duration,
            camera=camera,
            emotion=emotion,
            cues=cues,
        )
        return True, "ffmpeg_camera_motion", {
            "genuine_video": False,
            "motion_type": "camera_motion",
            "model": "ffmpeg-zoompan",
            "notice": "Camera navigation around high-resolution still image (pan/zoom/tilt).",
        }


def inspect_video_motion(clip_path: Path) -> Dict[str, Any]:
    """Inspects video file headers and sample frames to quantify temporal movement
    and verify output integrity.
    """
    if not clip_path.exists():
        return {"ok": False, "error": f"File does not exist: {clip_path}"}

    file_size = clip_path.stat().st_size
    if file_size < 1000:
        return {"ok": False, "error": f"Corrupt or empty video file: {file_size} bytes"}

    # Probe duration, resolution, stream info using ffprobe
    duration = 0.0
    width = 0
    height = 0
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration:stream=width,height,r_frame_rate",
            "-of", "json",
            str(clip_path)
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10, encoding="utf-8", errors="replace")
        if res.returncode == 0:
            info = json.loads(res.stdout)
            duration = float(info.get("format", {}).get("duration", 0.0))
            streams = info.get("streams", [])
            if streams:
                width = int(streams[0].get("width", 0))
                height = int(streams[0].get("height", 0))
    except Exception as probe_err:
        logger.warning(f"ffprobe check failed ({probe_err}), continuing.")

    # Extract 2 sample frames (at 20% and 80% duration) to calculate pixel variance
    frame_diff_mean = 0.0
    try:
        import tempfile
        from PIL import Image, ImageChops, ImageStat

        with tempfile.TemporaryDirectory() as td:
            f1 = Path(td) / "f1.png"
            f2 = Path(td) / "f2.png"
            t1 = max(0.2, duration * 0.25)
            t2 = max(0.5, duration * 0.75)
            subprocess.run(["ffmpeg", "-y", "-ss", str(t1), "-i", str(clip_path), "-vframes", "1", str(f1)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
            subprocess.run(["ffmpeg", "-y", "-ss", str(t2), "-i", str(clip_path), "-vframes", "1", str(f2)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
            if f1.exists() and f2.exists():
                im1 = Image.open(f1).convert("RGB")
                im2 = Image.open(f2).convert("RGB")
                diff = ImageChops.difference(im1, im2)
                stat = ImageStat.Stat(diff)
                frame_diff_mean = sum(stat.mean) / len(stat.mean)
    except Exception as diff_err:
        logger.warning(f"Frame difference measurement error: {diff_err}")

    has_temporal_motion = frame_diff_mean > 1.5

    return {
        "ok": True,
        "valid": True,
        "path": str(clip_path),
        "file_size_bytes": file_size,
        "duration_seconds": round(duration, 2),
        "duration": round(duration, 2),
        "resolution": f"{width}x{height}",
        "width": width,
        "height": height,
        "frame_diff_mean": round(frame_diff_mean, 2),
        "has_temporal_motion": has_temporal_motion,
    }


# Global adapter instance
video_adapter = VideoModelAdapter()

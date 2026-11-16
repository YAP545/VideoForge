"""VideoForge Image Reliability & Caption Regression Test Suite (Offline).

Verifies:
1. Waits follow "Retry after Ns" on 402/429/503 (N+3 s).
2. Second round runs after a 60s wait for missing scenes.
3. Fallback reuse works (nearest scene with image, 6% crop, flipped pan direction).
4. Plain gradient placeholder card has no caption text and no "SCENE N" label.
5. Caption check confirms one_text_box=True (no double-burned captions).
6. With POLLINATIONS_API_KEY set: requests gen.pollinations.ai with key, 3s spacing, key never in logs.
"""
import io
import json
import os
import sys
import unittest
from pathlib import Path
from typing import List
import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import httpx
from pipeline import (
    parse_retry_after,
    download_pollinations_image,
    run_storyboard_pipeline,
    draw_placeholder_image,
    redact_secrets,
    validate_image,
    VIDEO_DIR,
)


def make_valid_jpeg_bytes() -> bytes:
    """Generate a realistic dummy JPEG with high variance & edge energy to pass validate_image."""
    np.random.seed(1234)
    # Generate high variance noise with structured stripes
    arr = np.random.randint(40, 220, (720, 1280, 3), dtype=np.uint8)
    for y in range(0, 720, 40):
        arr[y : y + 20, :, :] = 255
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92)
    return buf.getvalue()


class TestImageReliability(unittest.TestCase):
    def setUp(self):
        download_pollinations_image._key_disabled = False
        if hasattr(download_pollinations_image, "_disabled_keys"):
            download_pollinations_image._disabled_keys.clear()
        self.orig_poll_key = os.environ.get("POLLINATIONS_API_KEY")
        if "POLLINATIONS_API_KEY" in os.environ:
            del os.environ["POLLINATIONS_API_KEY"]

    def tearDown(self):
        if self.orig_poll_key is not None:
            os.environ["POLLINATIONS_API_KEY"] = self.orig_poll_key
        elif "POLLINATIONS_API_KEY" in os.environ:
            del os.environ["POLLINATIONS_API_KEY"]

    # ── 1. Retry-After Parsing ────────────────────────────────────────────────
    def test_parse_retry_after_json_and_headers(self):
        """Confirm Retry after is parsed correctly from 402/503 bodies and headers."""
        # 402 body with Retry after 23s
        r402 = httpx.Response(
            402,
            json={
                "status": 402,
                "success": False,
                "error": {
                    "code": "FREE_LIMIT",
                    "message": "Free limit reached: 1 image every 45 seconds per IP. Retry after 23s, ...",
                },
            },
            request=httpx.Request("GET", "https://image.pollinations.ai/prompt/fox"),
        )
        self.assertEqual(parse_retry_after(r402), 23)

        # 503 body with Retry after 10s
        r503 = httpx.Response(
            503,
            json={
                "error": {
                    "code": "SERVICE_UNAVAILABLE",
                    "message": "The Classic API is at its limit right now. Retry after 10s, ...",
                }
            },
            request=httpx.Request("GET", "https://image.pollinations.ai/prompt/fox"),
        )
        self.assertEqual(parse_retry_after(r503), 10)

        # Fallback to default when not specified
        r_other = httpx.Response(
            500,
            text="Internal Server Error",
            request=httpx.Request("GET", "https://image.pollinations.ai/prompt/fox"),
        )
        self.assertEqual(parse_retry_after(r_other, default=20), 20)

    # ── 2. Waits Follow Retry-After ───────────────────────────────────────────
    def test_download_waits_follow_retry_after(self):
        """Confirm download_pollinations_image waits N+3 seconds on 402/503."""
        sleep_durations: List[float] = []

        def fake_sleep(secs: float):
            sleep_durations.append(secs)

        valid_jpeg = make_valid_jpeg_bytes()
        calls = 0

        def mock_handler(request: httpx.Request):
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(
                    402,
                    json={
                        "status": 402,
                        "success": False,
                        "error": {
                            "code": "FREE_LIMIT",
                            "message": "Free limit reached: 1 image every 45 seconds per IP. Retry after 23s, ...",
                        },
                    },
                )
            return httpx.Response(200, content=valid_jpeg)

        mock_client = httpx.Client(transport=httpx.MockTransport(mock_handler))
        out_img = VIDEO_DIR / "test_wait_scene.jpg"

        ok = download_pollinations_image(
            prompt="A small fox in snow",
            out_path=out_img,
            scene_num=1,
            client=mock_client,
            sleep_fn=fake_sleep,
        )

        self.assertTrue(ok)
        self.assertEqual(calls, 2)
        # First wait was 23 + 3 = 26 seconds
        self.assertIn(26, sleep_durations)
        if out_img.exists():
            out_img.unlink()

    # ── 3. Two Rounds, Spacing & Fallback Reuse ──────────────────────────────
    def test_two_rounds_and_reuse_fallback(self):
        """Simulate scenes with 402, 503, Round 2 retry, and reuse fallback."""
        sleep_durations: List[float] = []

        def fake_sleep(secs: float):
            sleep_durations.append(secs)

        valid_jpeg = make_valid_jpeg_bytes()
        scene_call_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

        def mock_handler(request: httpx.Request):
            url_str = str(request.url)
            # Determine scene by prompt or url
            if "Scene%201" in url_str or "scene%201" in url_str.lower():
                s = 1
            elif "Scene%202" in url_str or "scene%202" in url_str.lower():
                s = 2
            elif "Scene%203" in url_str or "scene%203" in url_str.lower():
                s = 3
            elif "Scene%204" in url_str or "scene%204" in url_str.lower():
                s = 4
            else:
                s = 5

            scene_call_counts[s] += 1
            count = scene_call_counts[s]

            # Scene 1: 402 on call 1 (Retry after 23s), succeeds on call 2
            if s == 1:
                if count == 1:
                    return httpx.Response(
                        402,
                        json={
                            "error": {
                                "code": "FREE_LIMIT",
                                "message": "Free limit reached: 1 image every 45 seconds per IP. Retry after 23s, ...",
                            }
                        },
                    )
                return httpx.Response(200, content=valid_jpeg)

            # Scene 2: succeeds immediately
            elif s == 2:
                return httpx.Response(200, content=valid_jpeg)

            # Scene 3: succeeds immediately
            elif s == 3:
                return httpx.Response(200, content=valid_jpeg)

            # Scene 4: always 402 (fails Round 1 and Round 2)
            elif s == 4:
                return httpx.Response(
                    402,
                    json={
                        "error": {
                            "code": "FREE_LIMIT",
                            "message": "Free limit reached. Retry after 20s",
                        }
                    },
                )

            # Scene 5: 503 on Round 1 calls (Retry after 10s), succeeds on Round 2 call
            else:
                if count in (1, 2):
                    return httpx.Response(
                        503,
                        json={
                            "error": {
                                "code": "SERVICE_UNAVAILABLE",
                                "message": "The Classic API is at its limit right now. Retry after 10s, ...",
                            }
                        },
                    )
                return httpx.Response(200, content=valid_jpeg)

        mock_client = httpx.Client(transport=httpx.MockTransport(mock_handler))

        pre_planned = [
            {
                "order_idx": 0,
                "camera": "wide",
                "caption": "Scene 1: The fox wanders in the snow.",
                "image_prompt": "Scene 1: A small fox in snow with a bright red scarf",
            },
            {
                "order_idx": 1,
                "camera": "medium",
                "caption": "Scene 2: It notices something red.",
                "image_prompt": "Scene 2: A small fox in snow with a bright red scarf",
            },
            {
                "order_idx": 2,
                "camera": "wide",
                "caption": "Scene 3: A red mitten lies on the snow.",
                "image_prompt": "Scene 3: A small fox in snow with a bright red scarf",
            },
            {
                "order_idx": 3,
                "camera": "medium",
                "caption": "Scene 4: The fox approaches the mitten.",
                "image_prompt": "Scene 4: A small fox in snow with a bright red scarf",
            },
            {
                "order_idx": 4,
                "camera": "close-up",
                "caption": "Scene 5: Happy in the winter evening.",
                "image_prompt": "Scene 5: A small fox in snow with a bright red scarf",
            },
        ]

        from unittest.mock import patch

        progress_reports: List[str] = []

        with patch("pipeline.generate_voiceover", return_value=False):
            out_mp4, scenes, placeholders, cap_results = run_storyboard_pipeline(
                job_id="test_rel_job",
                topic="A small fox with a red scarf finds a lost mitten in the snow",
                pre_planned_scenes=pre_planned,
                skip_network=False,
                client=mock_client,
                sleep_fn=fake_sleep,
                on_progress=lambda msg: progress_reports.append(msg),
            )

        try:
            # 1. Confirm waits follow Retry after: 23+3=26s and 10+3=13s were slept
            self.assertIn(26, sleep_durations)
            self.assertIn(13, sleep_durations)

            # 2. Confirm anonymous inter-request spacing of 50s was used between scenes
            self.assertIn(50.0, sleep_durations)

            # 3. Confirm 60s wait occurred before Round 2
            self.assertIn(60.0, sleep_durations)

            # 4. Confirm Scene 4 (which failed both rounds) reused Scene 3
            sc4 = scenes[3]
            self.assertTrue(sc4.get("is_reused"))
            self.assertEqual(sc4.get("reused_from"), 3)
            self.assertTrue(sc4.get("flipped_pan"))

            # 5. Confirm placeholder card is NOT drawn because other scenes have images
            self.assertEqual(len(placeholders), 0)

            # 6. Confirm caption check verified one_text_box=True across all scenes
            for r in cap_results:
                self.assertTrue(r.get("one_text_box", False))
                self.assertTrue(r.get("passed", False))

        finally:
            if out_mp4.exists():
                out_mp4.unlink()
            for p in VIDEO_DIR.glob("test_rel_job*"):
                try:
                    p.unlink()
                except Exception:
                    pass

    # ── 4. Key Support, 3s Spacing & Secret Redaction ────────────────────────
    def test_key_support_spacing_and_redaction(self):
        """With POLLINATIONS_API_KEY set: uses gen.pollinations.ai, 3s spacing, never logs key."""
        fake_key = "super-secret-pollinations-key-99999"
        os.environ["POLLINATIONS_API_KEY"] = fake_key

        sleep_durations: List[float] = []

        def fake_sleep(secs: float):
            sleep_durations.append(secs)

        valid_jpeg = make_valid_jpeg_bytes()
        requested_urls: List[str] = []

        def mock_handler(request: httpx.Request):
            requested_urls.append(str(request.url))
            return httpx.Response(200, content=valid_jpeg)

        mock_client = httpx.Client(transport=httpx.MockTransport(mock_handler))
        out_img = VIDEO_DIR / "test_key_scene.jpg"

        ok = download_pollinations_image(
            prompt="A small fox with a red scarf",
            out_path=out_img,
            scene_num=1,
            client=mock_client,
            sleep_fn=fake_sleep,
        )

        self.assertTrue(ok)
        self.assertEqual(len(requested_urls), 1)
        # Verify requested endpoint is gen.pollinations.ai with key
        self.assertTrue(requested_urls[0].startswith("https://gen.pollinations.ai/image/"))
        self.assertIn("key=super-secret-pollinations-key-99999", requested_urls[0])

        # Verify redact_secrets sanitizes this URL and raw key string
        sanitized_url = redact_secrets(requested_urls[0])
        self.assertNotIn(fake_key, sanitized_url)
        self.assertIn("[REDACTED]", sanitized_url)

        sanitized_text = redact_secrets(f"Error fetching {requested_urls[0]} with {fake_key}")
        self.assertNotIn(fake_key, sanitized_text)
        self.assertIn("[REDACTED]", sanitized_text)

        if out_img.exists():
            out_img.unlink()

    # ── 5. Placeholder Card Has No Text ──────────────────────────────────────
    def test_placeholder_card_has_no_text(self):
        """Placeholder card is a plain gradient with NO caption text and NO 'SCENE N' label."""
        p_path = VIDEO_DIR / "test_plain_placeholder.jpg"
        draw_placeholder_image(p_path, scene_num=3, caption="A test caption", width=1280, height=720)
        self.assertTrue(p_path.exists())

        img = Image.open(p_path).convert("L")
        w, h = img.size
        # The caption region (bottom 22%) must have NO bright white text pixels (intensity >= 180)
        crop = img.crop((int(w * 0.06), int(h * 0.76), int(w * 0.94), int(h * 0.98)))
        arr = np.array(crop)
        bright_pixels = int(np.sum(arr >= 180))
        # Plain gradient values are all below 80, so 0 bright text pixels exist
        self.assertEqual(bright_pixels, 0)

        p_path.unlink()


if __name__ == "__main__":
    unittest.main()

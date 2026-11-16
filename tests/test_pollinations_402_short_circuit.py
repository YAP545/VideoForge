"""Tests for Pollinations HTTP 402 Insufficient Balance handling:
1. Two-step fallback: paid key 402 → anonymous endpoint; anonymous 402 → full stop.
2. When BOTH paid key AND anonymous return 402 paywall, all subsequent scenes make ZERO requests.
3. Never saving 402 error responses or corrupt JSON payloads as images.
4. Cleanly halting generation with exact error message when all providers are exhausted.
5. Verifying mark_job_failed records status='failed' in the database.
"""
import io
import json
import os
import sys
import unittest
from pathlib import Path
from typing import List
from unittest.mock import patch, MagicMock

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import httpx
from pipeline import (
    download_pollinations_image,
    run_storyboard_pipeline,
    is_pollinations_exhausted,
    reset_pollinations_exhausted,
    VIDEO_DIR,
)
from app import mark_job_failed
from models import SessionLocal, Video


class TestPollinations402ShortCircuit(unittest.TestCase):
    def setUp(self):
        reset_pollinations_exhausted()
        download_pollinations_image._key_disabled = False
        if hasattr(download_pollinations_image, "_disabled_keys"):
            download_pollinations_image._disabled_keys.clear()
        self.tmp_files = []
        # Remove any stale POLLINATIONS_API_KEY so tests control it explicitly
        self._orig_key = os.environ.get("POLLINATIONS_API_KEY")
        os.environ.pop("POLLINATIONS_API_KEY", None)

    def tearDown(self):
        reset_pollinations_exhausted()
        if self._orig_key is not None:
            os.environ["POLLINATIONS_API_KEY"] = self._orig_key
        else:
            os.environ.pop("POLLINATIONS_API_KEY", None)
        for p in self.tmp_files:
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass

    # ── 1. Paid key 402 → anon fallback → anon also 402 → full stop ──────────
    def test_402_both_endpoints_short_circuits_all(self):
        """When BOTH paid key AND anonymous endpoint return 402 insufficient-balance,
        _POLLINATIONS_402_EXHAUSTED is set after Scene 1 and Scene 2 makes ZERO requests.
        """
        request_count = 0
        requested_urls = []

        def mock_both_402(request: httpx.Request):
            nonlocal request_count
            request_count += 1
            requested_urls.append(str(request.url))
            # Both paid and anonymous return account-paywall 402 (no FREE_LIMIT)
            return httpx.Response(
                402,
                json={
                    "error": {
                        "message": "Insufficient balance. Available balance: 0.0000.",
                        "code": "INSUFFICIENT_BALANCE",
                    }
                },
            )

        os.environ["POLLINATIONS_API_KEY"] = "fake-zero-balance-key"
        client = httpx.Client(transport=httpx.MockTransport(mock_both_402))
        img_p1 = VIDEO_DIR / "test_sc1.jpg"
        img_p2 = VIDEO_DIR / "test_sc2.jpg"
        self.tmp_files.extend([img_p1, img_p2])

        # Scene 1: paid key 402, then anonymous fallback also 402
        with patch("pipeline.download_free_cloud_image", return_value=False):
            ok1 = download_pollinations_image(
                prompt="Scene 1: Young man walking in rain",
                out_path=img_p1,
                scene_num=1,
                client=client,
                max_attempts=1,
                draw_placeholder_on_fail=False,
            )

        self.assertFalse(ok1)
        # 2 requests: 1 to paid key (gen.pollinations.ai) + 1 to anonymous (image.pollinations.ai)
        self.assertEqual(request_count, 2)
        # Both exhausted → full exhaustion must be set
        self.assertTrue(is_pollinations_exhausted())

        requests_before_scene2 = request_count

        # Scene 2: fully short-circuited; ZERO additional network requests
        with patch("pipeline.download_free_cloud_image", return_value=False):
            ok2 = download_pollinations_image(
                prompt="Scene 2: Young man near lamp post",
                out_path=img_p2,
                scene_num=2,
                client=client,
                max_attempts=1,
                draw_placeholder_on_fail=False,
            )

        self.assertFalse(ok2)
        # No new requests made for Scene 2
        self.assertEqual(request_count, requests_before_scene2)

    # ── 2. Paid key 402 → anon succeeds → image saved ────────────────────────
    def test_paid_key_402_falls_back_to_anonymous_success(self):
        """When paid key returns 402 but anonymous endpoint succeeds, image is saved."""
        import numpy as np
        from PIL import Image as PilImage

        def make_jpeg():
            arr = (255 * (0.5 + 0.5 * __import__('numpy').random.rand(64, 64, 3))).astype(__import__('numpy').uint8)
            buf = io.BytesIO()
            PilImage.fromarray(arr).save(buf, "JPEG", quality=85)
            return buf.getvalue()

        jpeg_bytes = make_jpeg()
        call_count = [0]

        def mock_handler(request: httpx.Request):
            call_count[0] += 1
            url = str(request.url)
            if "gen.pollinations.ai" in url:
                return httpx.Response(
                    402,
                    json={"error": {"message": "Insufficient balance. 0.0000.", "code": "INSUFFICIENT_BALANCE"}},
                )
            # Anonymous endpoint succeeds
            return httpx.Response(200, content=jpeg_bytes, headers={"content-type": "image/jpeg"})

        os.environ["POLLINATIONS_API_KEY"] = "fake-zero-key"
        client = httpx.Client(transport=httpx.MockTransport(mock_handler))
        out = VIDEO_DIR / "test_anon_success.jpg"
        self.tmp_files.append(out)

        with patch("pipeline.download_free_cloud_image", return_value=False):
            ok = download_pollinations_image(
                prompt="A mountain at sunset",
                out_path=out,
                scene_num=1,
                client=client,
                max_attempts=1,
                draw_placeholder_on_fail=False,
            )

        self.assertTrue(ok)
        self.assertTrue(out.exists())
        self.assertGreater(out.stat().st_size, 500)
        # Paid key is marked exhausted but anonymous is NOT fully exhausted
        self.assertFalse(is_pollinations_exhausted())

    # ── 3. 402 error JSON never saved as image ────────────────────────────────
    def test_402_error_response_never_saved_as_image(self):
        """Verify that an HTTP 402 JSON payload is never written to disk as an image file."""
        error_json_bytes = json.dumps({
            "error": "Insufficient balance. Available balance: 0.0000."
        }).encode("utf-8")

        def mock_handler(request: httpx.Request):
            return httpx.Response(
                402,
                content=error_json_bytes,
                headers={"content-type": "application/json"},
            )

        client = httpx.Client(transport=httpx.MockTransport(mock_handler))
        out_path = VIDEO_DIR / "test_corrupt_check.jpg"
        self.tmp_files.append(out_path)

        with patch("pipeline.download_free_cloud_image", return_value=False):
            ok = download_pollinations_image(
                prompt="Rainy evening",
                out_path=out_path,
                scene_num=1,
                client=client,
                draw_placeholder_on_fail=False,
            )

        self.assertFalse(ok)
        # Confirm file was not created or is empty
        self.assertFalse(out_path.exists())

    # ── 4. run_storyboard_pipeline halts with descriptive error ───────────────
    def test_run_storyboard_halts_with_descriptive_error_on_402(self):
        """When both paid key AND anonymous endpoint fail with 402 (no FREE_LIMIT),
        run_storyboard_pipeline raises RuntimeError describing the exact failure.
        """
        def mock_402(request: httpx.Request):
            return httpx.Response(
                402,
                json={"error": "Insufficient balance. Available balance: 0.0000."},
            )

        os.environ["POLLINATIONS_API_KEY"] = "fake-zero-key"
        client = httpx.Client(transport=httpx.MockTransport(mock_402))
        scenes = [
            {"order_idx": i, "caption": f"Scene {i+1}", "image_prompt": f"Scene {i+1} prompt"}
            for i in range(5)
        ]

        with patch("pipeline.download_free_cloud_image", return_value=False):
            with self.assertRaises(RuntimeError) as ctx:
                run_storyboard_pipeline(
                    job_id="test_402_job",
                    topic="Hindi romantic shayari",
                    pre_planned_scenes=scenes,
                    skip_network=False,
                    client=client,
                    sleep_fn=lambda s: None,
                )

        err_msg = str(ctx.exception)
        # Should mention Pollinations 402 balance problem
        self.assertTrue(
            "Pollinations" in err_msg and "402" in err_msg or "unavailable" in err_msg.lower(),
            f"Expected 402 error info in: {err_msg}"
        )

    # ── 5. mark_job_failed sets status='failed' with descriptive message ──────
    def test_mark_job_failed_updates_video_status_and_message(self):
        """Verify mark_job_failed records video.status='failed' with the exact error message."""
        db = SessionLocal()
        try:
            test_vid = Video(
                id="test_vid_402",
                user_id="default_user",
                title="Test Video 402",
                topic="Hindi poetry scene",
                style="Cinematic",
                status="generating",
            )
            db.merge(test_vid)
            db.commit()

            failure_msg = (
                "pipeline: Image generation unavailable: Pollinations returned HTTP 402 "
                "(Insufficient balance - available balance is 0.0000 pollen)."
            )
            mark_job_failed(
                video_id="test_vid_402",
                user_message=failure_msg,
                technical_error="HTTP 402 Pollen balance exhausted",
                db=db,
            )

            updated = db.query(Video).filter(Video.id == "test_vid_402").first()
            self.assertIsNotNone(updated)
            self.assertEqual(updated.status, "failed")
            self.assertNotEqual(updated.status, "done")
            self.assertEqual(updated.error, failure_msg)
        finally:
            v = db.query(Video).filter(Video.id == "test_vid_402").first()
            if v:
                db.delete(v)
                db.commit()
            db.close()


if __name__ == "__main__":
    unittest.main()

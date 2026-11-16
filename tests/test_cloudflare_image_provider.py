"""
Tests Cloudflare Workers AI image provider logic safely with mocked responses.
Ensures zero credits spent, zero live network requests made.
"""
import base64
import io
import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image

import sys
sys.path.insert(0, r"C:\Users\yasha\Downloads\videoforge\videoforge")


class TestCloudflareImageProvider(unittest.TestCase):
    def setUp(self):
        from pipeline import reset_cloudflare_quota
        reset_cloudflare_quota()
        self.tmp_dir = Path(r"C:\Users\yasha\Downloads\videoforge\videoforge\videos\test_tmp")
        self.tmp_dir.mkdir(exist_ok=True, parents=True)
        self.out_img = self.tmp_dir / "test_cf_scene.jpg"

    def tearDown(self):
        from pipeline import reset_cloudflare_quota
        reset_cloudflare_quota()
        if self.out_img.exists():
            self.out_img.unlink()

    def create_dummy_png_bytes(self, size=(1024, 1024)):
        im = Image.new("RGB", size, color=(200, 100, 50))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()

    def test_cloudflare_config_check_safe(self):
        from pipeline import check_cloudflare_config
        with patch.dict(os.environ, {
            "CLOUDFLARE_ACCOUNT_ID": "test_account_12345",
            "CLOUDFLARE_API_TOKEN": "cfat_test_secret_token_abc"
        }):
            cfg = check_cloudflare_config()
            self.assertTrue(cfg["configured"])
            self.assertEqual(cfg["status"], "configured")
            self.assertEqual(cfg["model"], "@cf/black-forest-labs/flux-1-schnell")
            self.assertNotIn("cfat_test_secret_token_abc", str(cfg))

    def test_cloudflare_json_base64_response_decoded_and_resized(self):
        from pipeline import download_cloudflare_image
        png_bytes = self.create_dummy_png_bytes((1024, 1024))
        b64_str = base64.b64encode(png_bytes).decode("utf-8")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "application/json"}
        mock_resp.json.return_value = {
            "result": {"image": b64_str},
            "success": True,
            "errors": [],
            "messages": []
        }
        mock_resp.content = b'{"result": ...}'

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_resp
            mock_client_cls.return_value.__enter__.return_value = mock_client

            with patch.dict(os.environ, {
                "CLOUDFLARE_ACCOUNT_ID": "test_acct",
                "CLOUDFLARE_API_TOKEN": "test_token"
            }):
                # Request 1280x720 video dimensions
                success = download_cloudflare_image("A majestic lion in the savannah", self.out_img, width=1280, height=720, scene_num=1)
                self.assertTrue(success)
                self.assertTrue(self.out_img.exists())
                with Image.open(self.out_img) as saved_im:
                    self.assertEqual(saved_im.format, "JPEG")
                    self.assertEqual(saved_im.size, (1280, 720))

    def test_cloudflare_raw_binary_response_decoded_and_saved(self):
        from pipeline import download_cloudflare_image
        png_bytes = self.create_dummy_png_bytes()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "image/png"}
        mock_resp.content = png_bytes

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_resp
            mock_client_cls.return_value.__enter__.return_value = mock_client

            with patch.dict(os.environ, {
                "CLOUDFLARE_ACCOUNT_ID": "test_acct",
                "CLOUDFLARE_API_TOKEN": "test_token"
            }):
                success = download_cloudflare_image("A serene mountain at dawn", self.out_img, scene_num=2)
                self.assertTrue(success)
                self.assertTrue(self.out_img.exists())

    def test_cloudflare_error_handling_and_quota_stop(self):
        from pipeline import download_cloudflare_image
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.headers = {"content-type": "application/json"}
        mock_resp.text = '{"errors": [{"message": "Rate limit exceeded"}]}'
        mock_resp.content = b'error'

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_resp
            mock_client_cls.return_value.__enter__.return_value = mock_client

            with patch.dict(os.environ, {
                "CLOUDFLARE_ACCOUNT_ID": "test_acct",
                "CLOUDFLARE_API_TOKEN": "test_token"
            }):
                success = download_cloudflare_image("A futuristic city", self.out_img, scene_num=3, draw_placeholder_on_fail=False)
                self.assertFalse(success)
                self.assertFalse(self.out_img.exists())
                # Verify that subsequent call immediately stops without requesting network
                subsequent = download_cloudflare_image("Another prompt", self.out_img, scene_num=4)
                self.assertFalse(subsequent)
                self.assertEqual(mock_client.post.call_count, 1)

    def test_estimate_neurons_native_formula(self):
        from pipeline import estimate_cloudflare_neurons
        # FLUX.1 Schnell native 1024x1024 @ 4 steps = 4 tiles * 4.80 + 4 tiles * 4 steps * 9.60 = 172.8
        self.assertEqual(estimate_cloudflare_neurons(1024, 1024, 4), 172.80)
        self.assertEqual(estimate_cloudflare_neurons(1280, 720, 4), 172.80)

    def test_conservative_local_safety_cap(self):
        import pipeline
        from pipeline import download_cloudflare_image
        pipeline.reset_cloudflare_quota()
        # Simulate local usage reaching 7400 Neurons
        pipeline._CLOUDFLARE_NEURONS_USED_TODAY = 7400.0
        # A new request costs 172.8 Neurons -> total 7572.8 > 7500.0 safety cap
        with patch.dict(os.environ, {
            "CLOUDFLARE_ACCOUNT_ID": "test_acct",
            "CLOUDFLARE_API_TOKEN": "test_token"
        }):
            # Should halt locally without making any network call
            with patch("httpx.Client") as mock_client_cls:
                success = download_cloudflare_image("Test prompt", self.out_img, scene_num=1)
                self.assertFalse(success)
                self.assertEqual(mock_client_cls.call_count, 0)
                self.assertTrue(pipeline._CLOUDFLARE_QUOTA_EXHAUSTED)


    def test_cloudflare_auth_error_halts_and_reports_clear_message(self):
        """HTTP 401 code 10000 should set _CLOUDFLARE_AUTH_ERROR, report actionable guidance,
        and immediately halt subsequent scene calls without making redundant requests.
        """
        import pipeline
        from pipeline import download_cloudflare_image
        pipeline.reset_cloudflare_quota()

        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.headers = {"content-type": "application/json"}
        mock_resp.text = '{"result":null,"success":false,"errors":[{"code":10000,"message":"Authentication error"}],"messages":[]}'
        mock_resp.json.return_value = {
            "result": None,
            "success": False,
            "errors": [{"code": 10000, "message": "Authentication error"}],
            "messages": [],
        }
        mock_resp.content = mock_resp.text.encode()

        req_tracker = {}
        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_resp
            mock_client_cls.return_value.__enter__.return_value = mock_client

            with patch.dict(os.environ, {
                "CLOUDFLARE_ACCOUNT_ID": "test_acct_123",
                "CLOUDFLARE_API_TOKEN": "cfat_test_token_456"
            }):
                success_s1 = download_cloudflare_image("Scene 1 prompt", self.out_img, scene_num=1, request_tracker=req_tracker)
                self.assertFalse(success_s1)
                self.assertTrue(pipeline._CLOUDFLARE_AUTH_ERROR)
                self.assertIn("Authentication Error", req_tracker.get("cloudflare_error", ""))
                self.assertIn("10000", req_tracker.get("cloudflare_error", ""))
                self.assertEqual(mock_client.post.call_count, 1)

                # Subsequent call must halt immediately with 0 extra network calls
                success_s2 = download_cloudflare_image("Scene 2 prompt", self.out_img, scene_num=2, request_tracker=req_tracker)
                self.assertFalse(success_s2)
                self.assertEqual(mock_client.post.call_count, 1)


if __name__ == "__main__":
    unittest.main()

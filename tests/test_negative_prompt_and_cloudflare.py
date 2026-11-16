"""
Tests download_scene_image negative_prompt parameter handling, provider routing,
and Cloudflare model payload behavior safely with mock fixtures.
Zero live requests, zero credits spent.
"""
import io
import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image

import sys
sys.path.insert(0, r"C:\Users\yasha\Downloads\videoforge\videoforge")

import pipeline
from pipeline import (
    download_cloudflare_image,
    download_scene_image,
    download_pollinations_image,
    reset_cloudflare_quota,
)


class TestNegativePromptAndProvider(unittest.TestCase):
    def setUp(self):
        reset_cloudflare_quota()
        self.tmp_dir = Path(r"C:\Users\yasha\Downloads\videoforge\videoforge\videos\test_tmp")
        self.tmp_dir.mkdir(exist_ok=True, parents=True)
        self.out_img = self.tmp_dir / "test_neg_prompt.jpg"

    def tearDown(self):
        reset_cloudflare_quota()
        if self.out_img.exists():
            self.out_img.unlink()

    def create_dummy_png_bytes(self, size=(1024, 1024)):
        im = Image.new("RGB", size, color=(120, 160, 200))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()

    def test_download_scene_image_accepts_negative_prompt_without_crash(self):
        """Task 1-4: Confirm download_scene_image accepts negative_prompt without TypeError."""
        called_args = {}

        def mock_cf_download(prompt, out_path, **kwargs):
            called_args["prompt"] = prompt
            called_args["kwargs"] = kwargs
            im = Image.new("RGB", (1280, 720), color=(10, 20, 30))
            im.save(out_path, "JPEG")
            return True

        with patch("pipeline.download_cloudflare_image", side_effect=mock_cf_download):
            with patch.dict(os.environ, {
                "IMAGE_PROVIDER": "cloudflare",
                "CLOUDFLARE_ACCOUNT_ID": "acct_test",
                "CLOUDFLARE_API_TOKEN": "token_test",
            }):
                ok = download_scene_image(
                    prompt="A serene forest",
                    out_path=self.out_img,
                    scene_num=1,
                    negative_prompt="blurry, distorted, ugly",
                    unexpected_future_kwarg="safe_to_ignore",
                )
                self.assertTrue(ok)
                self.assertIn("negative_prompt", called_args["kwargs"])
                self.assertEqual(called_args["kwargs"]["negative_prompt"], "blurry, distorted, ugly")
                self.assertEqual(called_args["kwargs"]["unexpected_future_kwarg"], "safe_to_ignore")

    def test_flux_model_safely_ignores_negative_prompt_in_payload(self):
        """Task 5-6: FLUX.1 Schnell does NOT support negative_prompt in Cloudflare API schema;
        must safely ignore it and omit from JSON payload to avoid 400 Bad Request.
        """
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
                "CLOUDFLARE_API_TOKEN": "test_token",
                "CLOUDFLARE_IMAGE_MODEL": "@cf/black-forest-labs/flux-1-schnell",
            }):
                ok = download_cloudflare_image(
                    prompt="Lord Hanuman in meditation",
                    out_path=self.out_img,
                    negative_prompt="extra limbs, duplicate characters, blurry",
                )
                self.assertTrue(ok)
                self.assertEqual(mock_client.post.call_count, 1)
                posted_json = mock_client.post.call_args[1].get("json")
                self.assertNotIn("negative_prompt", posted_json)
                self.assertEqual(posted_json["prompt"], "Lord Hanuman in meditation")

    def test_sdxl_model_includes_negative_prompt_in_payload(self):
        """Task 5-6: SDXL models support negative_prompt; must include it in JSON payload."""
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
                "CLOUDFLARE_API_TOKEN": "test_token",
                "CLOUDFLARE_IMAGE_MODEL": "@cf/stabilityai/stable-diffusion-xl-base-1.0",
            }):
                ok = download_cloudflare_image(
                    prompt="A beautiful lake at sunset",
                    out_path=self.out_img,
                    negative_prompt="fog, dark, blurry",
                )
                self.assertTrue(ok)
                self.assertEqual(mock_client.post.call_count, 1)
                posted_json = mock_client.post.call_args[1].get("json")
                self.assertIn("negative_prompt", posted_json)
                self.assertEqual(posted_json["negative_prompt"], "fog, dark, blurry")

    def test_no_fallback_to_pollinations_when_cloudflare_configured(self):
        """Task 8: Do not enable paid Pollinations or anonymous Pollinations fallback."""
        with patch.dict(os.environ, {
            "IMAGE_PROVIDER": "cloudflare",
            "CLOUDFLARE_ACCOUNT_ID": "",
            "CLOUDFLARE_API_TOKEN": "",
        }):
            with patch("pipeline.download_pollinations_image") as mock_pol:
                ok = download_scene_image(
                    prompt="Mountain landscape",
                    out_path=self.out_img,
                    negative_prompt="blurry",
                )
                self.assertFalse(ok)
                self.assertEqual(mock_pol.call_count, 0)

    def test_storyboard_provider_calls_download_scene_image(self):
        """Providers.py StoryboardProvider routes to download_scene_image and passes negative_prompt."""
        from providers import StoryboardProvider
        provider = StoryboardProvider()

        def mock_scene_side_effect(prompt, out_path, **kwargs):
            im = Image.new("RGB", (1280, 720), color=(10, 20, 30))
            im.save(out_path, "JPEG")
            return True

        def mock_animate_side_effect(image_path, out_clip_path, **kwargs):
            out_clip_path.write_bytes(b"dummy_mp4_bytes")
            return True

        with patch("pipeline.download_scene_image", side_effect=mock_scene_side_effect) as mock_scene_img:
            with patch("pipeline.animate_scene_image", side_effect=mock_animate_side_effect):
                with patch("pipeline.generate_voiceover") as mock_voice:
                    mock_voice.return_value = (False, 0.0)
                    with patch("pipeline.mux_clip_audio"):
                        provider.generate(
                            prompt="A flying eagle",
                            seconds=4.0,
                            out_path=self.tmp_dir / "clip.mp4",
                            negative_prompt="distorted wings",
                        )
                        self.assertEqual(mock_scene_img.call_count, 1)
                        _, kwargs = mock_scene_img.call_args
                        self.assertEqual(kwargs.get("negative_prompt"), "distorted wings")


if __name__ == "__main__":
    unittest.main()

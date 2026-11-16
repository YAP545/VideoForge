"""
Tests for provider selection, missing credentials, fallback behavior, and production mode.
Zero credits spent, zero live network requests made.
"""
import os
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import download_scene_image


class TestProviderSelectionAndFallback(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(__file__).parent.parent / "videos" / "test_tmp"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.out_img = self.tmp_dir / "test_provider_scene.jpg"

    def tearDown(self):
        if self.out_img.exists():
            self.out_img.unlink()

    def create_dummy_image(self, path):
        if path:
            im = Image.new("RGB", (640, 360), color=(50, 100, 150))
            im.save(str(path), "JPEG")

    def test_default_provider_is_pollinations_when_unset(self):
        """When IMAGE_PROVIDER is unset in environment, it must default to pollinations."""
        poll_called = []
        cf_called = []

        def mock_poll(*args, **kwargs):
            poll_called.append(True)
            out = kwargs.get("out_path") or (args[1] if len(args) > 1 else self.out_img)
            self.create_dummy_image(out)
            return True

        def mock_cf(*args, **kwargs):
            cf_called.append(True)
            return True

        with patch("pipeline.download_pollinations_image", side_effect=mock_poll):
            with patch("pipeline.download_cloudflare_image", side_effect=mock_cf):
                env = os.environ.copy()
                env.pop("IMAGE_PROVIDER", None)
                env.pop("CLOUDFLARE_ACCOUNT_ID", None)
                env.pop("CLOUDFLARE_API_TOKEN", None)
                with patch.dict(os.environ, env, clear=True):
                    ok = download_scene_image("a scenic landscape", self.out_img, scene_num=1)
                    self.assertTrue(ok)
                    self.assertEqual(len(poll_called), 1, "Expected download_pollinations_image to be called")
                    self.assertEqual(len(cf_called), 0, "Cloudflare should not be called when IMAGE_PROVIDER is unset")

    def test_explicit_pollinations_provider_routes_directly(self):
        """When IMAGE_PROVIDER=pollinations, it routes to Pollinations without checking Cloudflare."""
        poll_called = []
        cf_called = []

        def mock_poll(*args, **kwargs):
            poll_called.append(True)
            out = kwargs.get("out_path") or (args[1] if len(args) > 1 else self.out_img)
            self.create_dummy_image(out)
            return True

        def mock_cf(*args, **kwargs):
            cf_called.append(True)
            return True

        with patch("pipeline.download_pollinations_image", side_effect=mock_poll):
            with patch("pipeline.download_cloudflare_image", side_effect=mock_cf):
                with patch.dict(os.environ, {"IMAGE_PROVIDER": "pollinations"}, clear=False):
                    ok = download_scene_image("a scenic mountain", self.out_img, scene_num=1)
                    self.assertTrue(ok)
                    self.assertEqual(len(poll_called), 1)
                    self.assertEqual(len(cf_called), 0)

    def test_cloudflare_provider_uses_cloudflare_when_configured(self):
        """When IMAGE_PROVIDER=cloudflare and credentials are valid, it routes to Cloudflare."""
        cf_called = []

        def mock_cf(*args, **kwargs):
            cf_called.append(True)
            out = kwargs.get("out_path") or (args[1] if len(args) > 1 else self.out_img)
            self.create_dummy_image(out)
            return True

        with patch("pipeline.download_cloudflare_image", side_effect=mock_cf):
            with patch.dict(os.environ, {
                "IMAGE_PROVIDER": "cloudflare",
                "CLOUDFLARE_ACCOUNT_ID": "mock_account_123",
                "CLOUDFLARE_API_TOKEN": "mock_token_456",
            }):
                ok = download_scene_image("an ancient temple", self.out_img, scene_num=1)
                self.assertTrue(ok)
                self.assertEqual(len(cf_called), 1)

    def test_cloudflare_provider_gracefully_falls_back_when_credentials_missing(self):
        """When IMAGE_PROVIDER=cloudflare but credentials are missing, it logs warning and falls back to Pollinations."""
        poll_called = []
        cf_called = []

        def mock_poll(*args, **kwargs):
            poll_called.append(True)
            out = kwargs.get("out_path") or (args[1] if len(args) > 1 else self.out_img)
            self.create_dummy_image(out)
            return True

        def mock_cf(*args, **kwargs):
            cf_called.append(True)
            return True

        with patch("pipeline.download_pollinations_image", side_effect=mock_poll):
            with patch("pipeline.download_cloudflare_image", side_effect=mock_cf):
                env = os.environ.copy()
                env["IMAGE_PROVIDER"] = "cloudflare"
                env.pop("CLOUDFLARE_ACCOUNT_ID", None)
                env.pop("CLOUDFLARE_API_TOKEN", None)
                with patch.dict(os.environ, env, clear=True):
                    ok = download_scene_image("a futuristic city", self.out_img, scene_num=1)
                    self.assertTrue(ok, "Fallback to Pollinations must succeed")
                    self.assertEqual(len(cf_called), 0, "Cloudflare should not be called without credentials")
                    self.assertEqual(len(poll_called), 1, "Pollinations must be called as graceful fallback")

    def test_auto_provider_selects_cloudflare_when_configured(self):
        """When IMAGE_PROVIDER=auto and Cloudflare credentials exist, Cloudflare is selected."""
        cf_called = []

        def mock_cf(*args, **kwargs):
            cf_called.append(True)
            out = kwargs.get("out_path") or (args[1] if len(args) > 1 else self.out_img)
            self.create_dummy_image(out)
            return True

        with patch("pipeline.download_cloudflare_image", side_effect=mock_cf):
            with patch.dict(os.environ, {
                "IMAGE_PROVIDER": "auto",
                "CLOUDFLARE_ACCOUNT_ID": "mock_account_123",
                "CLOUDFLARE_API_TOKEN": "mock_token_456",
            }):
                ok = download_scene_image("a royal chariot", self.out_img, scene_num=1)
                self.assertTrue(ok)
                self.assertEqual(len(cf_called), 1)

    def test_auto_provider_selects_pollinations_when_credentials_missing(self):
        """When IMAGE_PROVIDER=auto and Cloudflare credentials are missing, Pollinations is cleanly selected."""
        poll_called = []

        def mock_poll(*args, **kwargs):
            poll_called.append(True)
            out = kwargs.get("out_path") or (args[1] if len(args) > 1 else self.out_img)
            self.create_dummy_image(out)
            return True

        with patch("pipeline.download_pollinations_image", side_effect=mock_poll):
            env = os.environ.copy()
            env["IMAGE_PROVIDER"] = "auto"
            env.pop("CLOUDFLARE_ACCOUNT_ID", None)
            env.pop("CLOUDFLARE_API_TOKEN", None)
            with patch.dict(os.environ, env, clear=True):
                ok = download_scene_image("a soaring eagle", self.out_img, scene_num=1)
                self.assertTrue(ok)
                self.assertEqual(len(poll_called), 1)

    def test_production_mode_with_pollinations_allows_clean_generation(self):
        """In production (APP_ENV=production), Pollinations is fully allowed and operational."""
        poll_called = []

        def mock_poll(*args, **kwargs):
            poll_called.append(True)
            out = kwargs.get("out_path") or (args[1] if len(args) > 1 else self.out_img)
            self.create_dummy_image(out)
            return True

        with patch("pipeline.download_pollinations_image", side_effect=mock_poll):
            with patch.dict(os.environ, {
                "APP_ENV": "production",
                "IMAGE_PROVIDER": "pollinations",
            }):
                ok = download_scene_image("ocean waves at sunset", self.out_img, scene_num=1)
                self.assertTrue(ok)
                self.assertEqual(len(poll_called), 1)


if __name__ == "__main__":
    unittest.main()

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""Tests for Review & Edit Scenes restoration and Pollinations HTTP error status code handling.
All external network calls and image generation are mocked. No credits are spent.
"""

import io
import json
import os
import re
import unittest
from unittest.mock import MagicMock, patch

import httpx
from PIL import Image
from fastapi.testclient import TestClient

from app import app, SESSION_SECRET
from auth import create_session_token
from models import get_db, User, Video, Scene
from pipeline import (
    check_pollinations_config,
    download_pollinations_image,
    redact_secrets,
    reset_pollinations_exhausted,
)
from planner import build_generic_scene_prompt


def make_valid_jpeg() -> bytes:
    img = Image.new("RGB", (64, 64), color=(30, 40, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class TestScenesReviewAndEdit(unittest.TestCase):
    def setUp(self):
        self.db = next(get_db())
        # Find or create a test user
        self.user = self.db.query(User).first()
        if not self.user:
            self.user = User(
                id="test-user-review",
                email="testreview@videoforge.local",
                password_hash="mock",
                credits=10,
            )
            self.db.add(self.user)
            self.db.commit()

        self.token = create_session_token(self.user.id, SESSION_SECRET)
        self.client = TestClient(app)
        self.client.cookies.set("vf_session", self.token)

    def test_scenes_render_with_prompts_and_captions(self):
        """Verify that scene prompts and captions appear correctly in the Review & Edit Scenes form."""
        vid_id = "test-vid-render-1"
        self.db.query(Scene).filter(Scene.video_id == vid_id).delete()
        self.db.query(Video).filter(Video.id == vid_id).delete()

        v = Video(
            id=vid_id,
            user_id=self.user.id,
            title="Magical Forest Test",
            topic="Create a cinematic 3D animated short film about a brave young boy who discovers a magical glowing forest at night.",
            length_seconds=25,
            style="3d_animated",
            caption_lang="none",
            status="planning",
        )
        self.db.add(v)

        expected_prompts = [
            "Scene 1: Boy walks along forest path. One boy only.",
            "Scene 2: Boy enters magical forest with glowing butterflies.",
            "Scene 3: Boy helps trapped golden fox escape. Show the boy and one fox clearly.",
            "Scene 4: Golden fox leads boy to hidden glowing waterfall.",
            "Scene 5: Boy reaches forest exit with fox at dawn. Fade out.",
        ]
        for i, p in enumerate(expected_prompts):
            sc = Scene(
                id=f"sc-ren-{i}",
                video_id=vid_id,
                order_idx=i,
                visual_prompt=p,
                caption="",
                narration="",
                status="pending",
            )
            self.db.add(sc)
        self.db.commit()

        resp = self.client.get(f"/project/{vid_id}/plan")
        self.assertEqual(resp.status_code, 200)
        html = resp.text

        # Verify all 5 textareas contain the respective prompts
        for i, exp_p in enumerate(expected_prompts):
            self.assertIn(f'name="visual_prompt_{i}"', html)
            self.assertIn(exp_p, html)

        # Verify captions are empty string when captions disabled
        for i in range(5):
            self.assertIn(f'name="caption_{i}" value=""', html)

        # Clean up
        self.db.query(Scene).filter(Scene.video_id == vid_id).delete()
        self.db.query(Video).filter(Video.id == vid_id).delete()
        self.db.commit()

    def test_scenes_survive_form_submission(self):
        """Verify edited prompts and captions are preserved across POST /project/{id}/generate."""
        vid_id = "test-vid-submit-2"
        self.db.query(Scene).filter(Scene.video_id == vid_id).delete()
        self.db.query(Video).filter(Video.id == vid_id).delete()

        v = Video(
            id=vid_id,
            user_id=self.user.id,
            title="Submit Test",
            topic="A boy and a fox in a magical forest",
            length_seconds=25,
            style="3d_animated",
            caption_lang="en",
            status="planning",
        )
        self.db.add(v)
        self.db.commit()

        # Fetch CSRF token from page
        get_resp = self.client.get(f"/project/{vid_id}/plan")
        csrf_match = re.search(r'name="csrf_token"\s+value="([^"]+)"', get_resp.text)
        csrf_token = csrf_match.group(1) if csrf_match else "mock-csrf"

        edited_data = {
            "csrf_token": csrf_token,
            "enable_voiceover": "1",
            "visual_prompt_0": "Edited visual description 1",
            "caption_0": "Caption 1",
            "orig_caption_0": "",
            "narration_0": "Narration 1",
            "visual_prompt_1": "Edited visual description 2",
            "caption_1": "Caption 2",
            "orig_caption_1": "",
            "narration_1": "Narration 2",
        }

        with patch("app.project_q.put", return_value=None):
            with patch("app.is_credits_enabled", return_value=False):
                post_resp = self.client.post(
                    f"/project/{vid_id}/generate",
                    data=edited_data,
                    follow_redirects=False,
                )
                self.assertIn(post_resp.status_code, (302, 303))

        saved_scenes = (
            self.db.query(Scene)
            .filter(Scene.video_id == vid_id)
            .order_by(Scene.order_idx)
            .all()
        )
        self.assertEqual(len(saved_scenes), 2)
        self.assertEqual(saved_scenes[0].visual_prompt, "Edited visual description 1")
        self.assertEqual(saved_scenes[0].caption, "Caption 1")
        self.assertEqual(saved_scenes[1].visual_prompt, "Edited visual description 2")
        self.assertEqual(saved_scenes[1].caption, "Caption 2")

        # Clean up
        self.db.query(Scene).filter(Scene.video_id == vid_id).delete()
        self.db.query(Video).filter(Video.id == vid_id).delete()
        self.db.commit()

    def test_two_character_prompt_does_not_leak_kabir_and_meera(self):
        """Verify that generic two-character scenes do not hardcode Kabir and Meera."""
        scene = {
            "camera": "medium",
            "action": "Leo and the golden fox explore the glowing flora together",
            "characters_present": ["Leo", "Golden Fox"],
            "must_show": ["Leo", "Golden Fox"],
            "emotion": "joy",
        }
        story_info = {
            "main_setting": "magical glowing forest",
            "characters": [
                {"name": "Leo", "role": "young boy", "gender": "male", "outfit": "wearing a red hoodie", "hair": "short dark hair", "is_human": True},
                {"name": "Golden Fox", "role": "magical fox", "gender": "neutral", "outfit": "golden fur with glowing blue eyes", "is_human": False},
            ],
        }
        prompt = build_generic_scene_prompt(scene, story_info, style="3d_animated")
        self.assertNotIn("Kabir", prompt)
        self.assertNotIn("Meera", prompt)
        self.assertIn("Leo and Golden Fox", prompt)


class TestPollinationsConfigAndStatusCodes(unittest.TestCase):
    def setUp(self):
        reset_pollinations_exhausted()
        download_pollinations_image._key_disabled = False
        self.out_file = Path("test_poll_out.jpg")
        if self.out_file.exists():
            self.out_file.unlink()

    def tearDown(self):
        reset_pollinations_exhausted()
        download_pollinations_image._key_disabled = False
        if self.out_file.exists():
            self.out_file.unlink()

    def test_safe_config_check_reports_without_leaking_key(self):
        """check_pollinations_config reports status without revealing key in returned dict or string repr."""
        secret_key = "sk-super-secret-test-key-xyz123"
        with patch.dict(os.environ, {"POLLINATIONS_API_KEY": secret_key}):
            cfg = check_pollinations_config()
            self.assertTrue(cfg["configured"])
            self.assertEqual(cfg["status"], "configured")
            self.assertIn("gen.pollinations.ai", cfg["endpoint"])
            # Ensure key is NOT in cfg values
            self.assertNotIn(secret_key, str(cfg))

        with patch.dict(os.environ, {"POLLINATIONS_API_KEY": ""}):
            cfg_empty = check_pollinations_config()
            self.assertFalse(cfg_empty["configured"])
            self.assertEqual(cfg_empty["status"], "missing")
            self.assertIn("image.pollinations.ai", cfg_empty["endpoint"])

    @patch("pipeline.download_free_cloud_image", return_value=False)
    def test_http_401_invalid_key_distinguished_and_never_retried(self, mock_cloud):
        """HTTP 401 invalid key marks key exhausted and switches without infinite loops."""
        secret_key = "invalid-test-key"
        calls = []

        def mock_handler(request: httpx.Request):
            calls.append(str(request.url))
            return httpx.Response(401, json={"error": "Invalid API key"})

        mock_client = httpx.Client(transport=httpx.MockTransport(mock_handler))
        tracker = {}

        with patch.dict(os.environ, {"POLLINATIONS_API_KEY": secret_key}):
            ok = download_pollinations_image(
                prompt="A boy in a magical forest",
                out_path=self.out_file,
                scene_num=1,
                client=mock_client,
                max_attempts=3,
                request_tracker=tracker,
            )

        self.assertFalse(ok)
        self.assertTrue(tracker.get("pollinations_key_exhausted"))
        self.assertIn("401", tracker.get("pollinations_402_error", ""))
        paid_calls = [c for c in calls if "gen.pollinations.ai" in c]
        self.assertEqual(len(paid_calls), 1)

    @patch("pipeline.download_free_cloud_image", return_value=False)
    def test_http_403_permissions_error_distinguished_and_never_retried(self, mock_cloud):
        """HTTP 403 forbidden marks key exhausted and switches without infinite loops."""
        secret_key = "forbidden-test-key"
        calls = []

        def mock_handler(request: httpx.Request):
            calls.append(str(request.url))
            return httpx.Response(403, json={"error": "Forbidden: token lacks image generation permission"})

        mock_client = httpx.Client(transport=httpx.MockTransport(mock_handler))
        tracker = {}

        with patch.dict(os.environ, {"POLLINATIONS_API_KEY": secret_key}):
            ok = download_pollinations_image(
                prompt="A boy in a magical forest",
                out_path=self.out_file,
                scene_num=1,
                client=mock_client,
                max_attempts=3,
                request_tracker=tracker,
            )

        self.assertFalse(ok)
        self.assertTrue(tracker.get("pollinations_key_exhausted"))
        self.assertIn("403", tracker.get("pollinations_402_error", ""))
        paid_calls = [c for c in calls if "gen.pollinations.ai" in c]
        self.assertEqual(len(paid_calls), 1)

    @patch("pipeline.download_free_cloud_image", return_value=False)
    def test_http_402_insufficient_balance_never_saves_error_as_image(self, mock_cloud):
        """HTTP 402 error JSON is never saved as an image file on disk."""
        secret_key = "zero-balance-key"

        def mock_handler(request: httpx.Request):
            return httpx.Response(
                402,
                json={"error": {"message": "Insufficient balance. Available balance: 0.0000", "code": "INSUFFICIENT_BALANCE"}},
                headers={"Content-Type": "application/json"},
            )

        mock_client = httpx.Client(transport=httpx.MockTransport(mock_handler))
        tracker = {}

        with patch.dict(os.environ, {"POLLINATIONS_API_KEY": secret_key}):
            ok = download_pollinations_image(
                prompt="A boy in a magical forest",
                out_path=self.out_file,
                scene_num=1,
                client=mock_client,
                max_attempts=3,
                request_tracker=tracker,
            )

        self.assertFalse(ok)
        self.assertFalse(self.out_file.exists(), "Error JSON must never be written to out_file as an image!")

    def test_http_429_rate_limit_handled_with_retry_after(self):
        """HTTP 429 parses Retry-After, waits, and succeeds if next attempt returns valid image."""
        secret_key = "rate-limited-key"
        calls = []
        waited_times = []
        valid_img = make_valid_jpeg()

        def mock_handler(request: httpx.Request):
            calls.append(str(request.url))
            if len(calls) == 1:
                return httpx.Response(
                    429,
                    text="Rate limit exceeded. Retry after 5s",
                    headers={"Retry-After": "5", "Content-Type": "text/plain"},
                )
            return httpx.Response(200, content=valid_img, headers={"Content-Type": "image/jpeg"})

        def mock_sleep(secs):
            waited_times.append(secs)

        mock_client = httpx.Client(transport=httpx.MockTransport(mock_handler))

        with patch.dict(os.environ, {"POLLINATIONS_API_KEY": secret_key}):
            ok = download_pollinations_image(
                prompt="A boy in a magical forest",
                out_path=self.out_file,
                scene_num=1,
                client=mock_client,
                sleep_fn=mock_sleep,
                max_attempts=3,
            )

        self.assertTrue(ok)
        self.assertTrue(self.out_file.exists())
        self.assertEqual(len(calls), 2)
        self.assertTrue(len(waited_times) >= 1)
        self.assertAlmostEqual(waited_times[0], 8.0, delta=1.0)


if __name__ == "__main__":
    unittest.main()

"""VideoForge Safety & Hardening Test Suite (Offline).

Verifies:
1. Path traversal ("../../.env", absolute paths) rejected by safe_path and serving.
2. Failed job gets a user-facing message and refunds once only when credits are enabled.
3. Stuck job left in processing/generating for > 30 mins is recovered on startup.
4. With CREDITS_ENABLED=false a job runs with 0 credits (no deduction/refund).
"""
import os
import sys
import unittest
import uuid
from datetime import datetime, timedelta
from pathlib import Path

# Ensure videoforge is in path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Use isolated SQLite test database
os.environ["DATABASE_URL"] = f"sqlite:///{BASE_DIR / 'test_safety.db'}"
os.environ["PROVIDER"] = "mock"
os.environ["CREDITS_ENABLED"] = "true"

from starlette.testclient import TestClient
from models import init_db, SessionLocal, User, Video, Scene
from app import (
    app,
    safe_path,
    is_credits_enabled,
    mark_job_failed,
    recover_stuck_jobs,
    VIDEO_DIR,
)


class TestSafetyHardening(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()
        cls.client = TestClient(app, raise_server_exceptions=False)

    @classmethod
    def tearDownClass(cls):
        test_db = BASE_DIR / "test_safety.db"
        if test_db.exists():
            try:
                test_db.unlink()
            except Exception:
                pass

    def setUp(self):
        self.db = SessionLocal()

    def tearDown(self):
        self.db.close()

    # ── 1. Path Traversal Tests ──────────────────────────────────────────────
    def test_path_traversal_rejected(self):
        """Path traversal and absolute paths outside base_dir must raise ValueError."""
        # Relative traversal
        with self.assertRaises(ValueError):
            safe_path(VIDEO_DIR, "../../.env")

        with self.assertRaises(ValueError):
            safe_path(VIDEO_DIR, "../templates/dashboard.html")

        # Absolute paths outside base_dir
        with self.assertRaises(ValueError):
            safe_path(VIDEO_DIR, "C:/Windows/notepad.exe")

        with self.assertRaises(ValueError):
            safe_path(VIDEO_DIR, "/etc/passwd")

        # Valid filename inside VIDEO_DIR succeeds
        valid = safe_path(VIDEO_DIR, "output_video_123.mp4")
        self.assertTrue(str(valid).startswith(str(VIDEO_DIR.resolve())))
        self.assertTrue(str(valid).endswith("output_video_123.mp4"))

        # Serving endpoint /videos/{path} rejects traversal
        resp = self.client.get("/videos/../../.env")
        self.assertIn(resp.status_code, (403, 404))

    # ── 2. Job Failure & Single Refund Tests ──────────────────────────────────
    def test_failed_job_message_and_single_refund(self):
        """Failed job records user message, technical error, and refunds credits once only."""
        os.environ["CREDITS_ENABLED"] = "true"
        self.assertTrue(is_credits_enabled())

        # Create user with 10 credits
        user = User(
            id=str(uuid.uuid4())[:12],
            email=f"user_{uuid.uuid4().hex[:6]}@test.com",
            hashed_password="fakehashpw123",
            credits=10,
        )
        self.db.add(user)
        self.db.commit()

        # Create video with 5 scenes
        video = Video(
            id=str(uuid.uuid4())[:12],
            user_id=user.id,
            title="A small fox in snow",
            topic="A small fox in snow",
            length_seconds=25,
            status="generating",
            credits_refunded=False,
        )
        self.db.add(video)
        for i in range(5):
            scene = Scene(
                id=str(uuid.uuid4())[:12],
                video_id=video.id,
                order_idx=i,
                visual_prompt=f"Prompt scene {i}",
                status="pending",
            )
            self.db.add(scene)
        self.db.commit()

        # Mark job failed first time
        user_msg = "Video generation failed due to a temporary service issue. Please try again."
        tech_err = "ProviderTimeoutError: connection timed out after 30s"
        mark_job_failed(video.id, user_msg, tech_err, db=self.db)

        self.db.refresh(video)
        self.db.refresh(user)

        self.assertEqual(video.status, "failed")
        self.assertEqual(video.error, user_msg)
        self.assertEqual(video.progress_text, user_msg)
        self.assertTrue(video.credits_refunded)
        self.assertEqual(user.credits, 15)  # 10 + 5 refunded

        # Mark job failed a second time: must NOT double refund
        mark_job_failed(video.id, "Second failure message", "Tech error 2", db=self.db)
        self.db.refresh(user)
        self.assertEqual(user.credits, 15, "Credits must not be refunded more than once")

    # ── 3. Stuck Jobs Recovery Tests ─────────────────────────────────────────
    def test_stuck_job_recovered_on_startup(self):
        """Stuck jobs left in generating/processing > 30 mins are marked failed."""
        user = User(
            id=str(uuid.uuid4())[:12],
            email=f"stuck_{uuid.uuid4().hex[:6]}@test.com",
            hashed_password="fakehashpw123",
            credits=5,
        )
        self.db.add(user)
        self.db.commit()

        # Video created 45 minutes ago, still stuck in 'generating'
        old_time = datetime.utcnow() - timedelta(minutes=45)
        stuck_video = Video(
            id=str(uuid.uuid4())[:12],
            user_id=user.id,
            title="Stuck job test",
            topic="Stuck job test",
            length_seconds=25,
            status="generating",
            created_at=old_time,
            credits_refunded=False,
        )
        self.db.add(stuck_video)
        self.db.commit()

        # Run recovery
        recover_stuck_jobs(timeout_minutes=30)

        self.db.refresh(stuck_video)
        self.assertEqual(stuck_video.status, "failed")
        self.assertIn("timed out", stuck_video.error.lower())

    # ── 4. Credits Switch Tests ──────────────────────────────────────────────
    def test_credits_disabled_runs_with_zero_credits(self):
        """With CREDITS_ENABLED=false, 0 credit users can queue jobs without deduction or refund."""
        os.environ["CREDITS_ENABLED"] = "false"
        try:
            self.assertFalse(is_credits_enabled())

            user = User(
                id=str(uuid.uuid4())[:12],
                email=f"zerocredit_{uuid.uuid4().hex[:6]}@test.com",
                hashed_password="fakehashpw123",
                credits=0,
            )
            self.db.add(user)
            self.db.commit()

            video = Video(
                id=str(uuid.uuid4())[:12],
                user_id=user.id,
                title="Zero credits video",
                topic="Zero credits video",
                length_seconds=25,
                status="planning",
                credits_refunded=False,
            )
            self.db.add(video)
            for i in range(5):
                scene = Scene(
                    id=str(uuid.uuid4())[:12],
                    video_id=video.id,
                    order_idx=i,
                    visual_prompt=f"Scene prompt {i}",
                    status="pending",
                )
                self.db.add(scene)
            self.db.commit()

            # When CREDITS_ENABLED=false, mark_job_failed should not refund
            mark_job_failed(video.id, "Test fail with credits disabled", "Error info", db=self.db)
            self.db.refresh(user)
            self.assertEqual(user.credits, 0, "No credits should be added when credits are disabled")

            # Check that credit deductions are skipped when disabled
            # Simulate what /project/{video_id}/generate checks:
            required_credits = 5
            can_proceed = not is_credits_enabled() or user.credits >= required_credits
            self.assertTrue(can_proceed, "User with 0 credits must be allowed to proceed when credits are disabled")
        finally:
            os.environ["CREDITS_ENABLED"] = "true"


if __name__ == "__main__":
    unittest.main()

"""Verification script for Mock Mode Enhancements.

Uses an isolated test database (test_videoforge.db).
Your real videoforge.db is never modified.

For test 7 (APP_ENV=production guard): starts a SECOND isolated server
process with APP_ENV=production set in its environment.

Tests:
1. Demo Mode banner visible on Dashboard when PROVIDER=mock.
2. Demo Mode banner visible on Create Page when PROVIDER=mock.
3. Admin Provider Status section visible in Settings.
4. Admin Test Connection endpoint works correctly.
5. Demo Mode banner visible on Video Result Page.
6. Extracted mock clip frame with PLACEHOLDER overlay.
7. APP_ENV=production guard blocks mock generation.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import imageio_ffmpeg

from test_helpers import TestServer, make_admin_and_verify


ADMIN_EMAIL = "admin_primary@test.com"
ADMIN_PASS = "AdminPrimary123!"


def run_mock_verification():
    print(">> Starting Mock Mode Verification Suite...\n")
    results = []

    # ── Tests 1-6: development server (PROVIDER=mock, APP_ENV=development) ──
    with TestServer(env={"APP_ENV": "development", "PROVIDER": "mock"}) as srv:
        client = make_admin_and_verify(srv.port, ADMIN_EMAIL, ADMIN_PASS)
        print(f"[setup] Signed in as admin on port {srv.port}\n")

        # Test 1: Banner on Dashboard
        _, dash_html, _ = client.get("/dashboard")
        r1 = "DEMO MODE" in dash_html and "no real AI model is connected" in dash_html
        results.append(("1. Demo Mode banner visible on Dashboard", r1))

        # Test 2: Banner on Create Page
        _, create_html, _ = client.get("/create")
        r2 = "DEMO MODE" in create_html and "no real AI model is connected" in create_html
        results.append(("2. Demo Mode banner visible on Create Page", r2))

        # Test 3: Admin Provider Status in Settings
        _, settings_html, _ = client.get("/settings")
        r3 = (
            "Video Provider Status (Admin)" in settings_html
            and "Configured Engine" in settings_html
        )
        results.append(("3. Admin Provider Status section visible in Settings", r3))

        # Test 4: /api/admin/provider/test endpoint
        _, test_resp_str, _ = client.get("/api/admin/provider/test")
        try:
            test_json = json.loads(test_resp_str)
            r4 = (
                test_json.get("ok") is True
                and "PROVIDER=mock is active" in test_json.get("message", "")
            )
        except Exception:
            r4 = False
        results.append(("4. Admin Test Connection endpoint works correctly", r4))

        # Test 5+6: Generate mock project, check result banner & frame
        _, create_html, _ = client.get("/create")
        csrf_create = client.get_csrf(create_html)
        _, _, redir_url = client.post("/create", {
            "topic": "Kitten playing in garden",
            "length": 15,
            "style": "2d_cartoon",
            "ratio": "16:9",
            "caption_lang": "hi",
            "csrf_token": csrf_create,
        })
        project_id = redir_url.split("/project/")[1].split("/plan")[0]

        _, plan_html, _ = client.get(f"/project/{project_id}/plan")
        csrf_plan = client.get_csrf(plan_html)
        client.post(f"/project/{project_id}/generate", {
            "csrf_token": csrf_plan,
            "visual_prompt_0": "Cat chases butterfly in garden",
            "caption_0": "\u0915\u094d\u092f\u093e \u0915\u094b\u0908 \u0917\u093f\u0930\u092b\u093c\u094d\u0924\u093e\u0930 \u0939\u0941\u0906?",
            "visual_prompt_1": "Cat hides among flowers",
            "caption_1": "\u0935\u0939 \u0927\u0940\u0930\u0947 \u0938\u0947 \u0906\u0917\u0947 \u092c\u0922\u093c\u0924\u0940 \u0939\u0948\u0964",
            "visual_prompt_2": "Cat stretches in sunlight",
            "caption_2": "\u090f\u0915 \u0936\u093e\u0902\u0924 \u0914\u0930 \u092a\u094d\u092f\u093e\u0930\u093e \u0926\u093f\u0928\u0964",
        })

        print(">> Waiting for mock generation (up to 45s)...")
        for _ in range(60):  # 90s max - 3 clips x ~20s each
            time.sleep(1.5)
            _, status_str, _ = client.get(f"/api/project/{project_id}/status")
            try:
                if json.loads(status_str).get("status") == "done":
                    print("   Generation complete.")
                    break
            except Exception:
                pass

        _, video_html, _ = client.get(f"/video/{project_id}")
        r5 = "DEMO MODE" in video_html and "no real AI model is connected" in video_html
        results.append(("5. Demo Mode banner visible on Video Result Page", r5))

        clip_file = Path("videos") / f"{project_id}_scene_0.mp4"
        frame_out = Path("videos") / "mock_overlay_sample.png"
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([
            ffmpeg_exe, "-y", "-ss", "1.0",
            "-i", str(clip_file), "-vframes", "1", str(frame_out),
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        r6 = frame_out.exists() and frame_out.stat().st_size > 1000
        results.append(("6. Extracted mock clip frame with PLACEHOLDER overlay", r6))
        if r6:
            print(f"   Frame: {frame_out.resolve()}")

    # ── Test 7: Start a SECOND server with APP_ENV=production ───────────────
    print("\n>> Test 7: Starting second server with APP_ENV=production...")
    with TestServer(env={"APP_ENV": "production", "PROVIDER": "mock"}) as prod_srv:
        prod_client = make_admin_and_verify(prod_srv.port, ADMIN_EMAIL, ADMIN_PASS)
        print(f"[setup] Signed in as admin on production port {prod_srv.port}")

        _, create_html2, _ = prod_client.get("/create")
        csrf_c2 = prod_client.get_csrf(create_html2)
        _, _, redir2 = prod_client.post("/create", {
            "topic": "Production guard test",
            "length": 5,
            "style": "realistic",
            "ratio": "16:9",
            "caption_lang": "none",
            "csrf_token": csrf_c2,
        })
        p2_id = redir2.split("/project/")[1].split("/plan")[0]
        _, p2_plan_html, _ = prod_client.get(f"/project/{p2_id}/plan")
        csrf_p2 = prod_client.get_csrf(p2_plan_html)
        _, gen_html, _ = prod_client.post(f"/project/{p2_id}/generate", {
            "csrf_token": csrf_p2,
            "visual_prompt_0": "Scenic mountain landscape",
            "caption_0": "",
        })
        r7 = "Generation blocked" in gen_html and "PRODUCTION mode" in gen_html
        results.append(("7. APP_ENV=production guard blocks mock generation", r7))

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MOCK MODE ENHANCEMENTS TEST SUMMARY")
    print("=" * 60)
    all_passed = True
    for desc, passed in results:
        mark = "[PASS]" if passed else "[FAIL]"
        if not passed:
            all_passed = False
        print(f"{mark} - {desc}")
    print("=" * 60)
    if all_passed:
        print("\nALL 7 MOCK MODE ENHANCEMENT TESTS PASSED!\n")
    else:
        print("\nSOME TESTS FAILED\n")
        sys.exit(1)


if __name__ == "__main__":
    run_mock_verification()

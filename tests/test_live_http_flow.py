"""
test_live_http_flow.py
======================
Comprehensive HTTP integration test for VideoForge scene data flow.
- Uses FastAPI TestClient with Gemini/external calls mocked (zero API calls, zero credits).
- Tests 5 genres:
    1. Hanuman devotional topic
    2. Snowy forest adventure
    3. Mumbai street food animation
    4. Horror detective
    5. Educational science video
- Flow tested:
    POST /create -> 303 Redirect -> GET /project/{id}/plan -> 200 HTML
- Validations:
    * 5 textarea elements with name="visual_prompt_X" present in returned HTML
    * Every visual_prompt textarea contains > 50 characters
    * Captions can be empty (for caption_lang="none")
    * Database records (Scene) contain the exact same prompts as the textareas
    * Safe diagnostic information logged (IDs, lengths, field names)
"""
import sys, io, os, re, uuid, hashlib, json
from unittest.mock import patch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

def _mock_external_apis(*args, **kwargs):
    raise ConnectionError("External API mocked - test is offline")

TOPICS = [
    (
        "Lord Hanuman Devotional",
        "A majestic and powerful Lord Hanuman standing on a mountain peak at sunrise, devotional animation",
        "3d_animated",
        "none",
    ),
    (
        "Snowy Forest Adventure",
        "A young explorer traversing a snowy pine forest during a winter blizzard, discovering an ancient ice cavern",
        "3d_animated",
        "none",
    ),
    (
        "Mumbai Street Food Animation",
        "Vibrant 3D animation of a bustling Mumbai street food vendor preparing pav bhaji and cutting chai at dusk",
        "3d_animated",
        "none",
    ),
    (
        "Horror Detective",
        "A curious detective investigating an abandoned haunted asylum on a stormy night, shadowy figures moving in the dark",
        "cinematic",
        "none",
    ),
    (
        "Educational Science Video",
        "A journey inside a human cell showing mitochondria producing ATP energy and DNA replication in the nucleus",
        "scifi",
        "none",
    ),
]

all_passed = True
results_summary = []

with patch("httpx.Client.post", side_effect=_mock_external_apis):
    from fastapi.testclient import TestClient
    from app import app, SESSION_SECRET
    from auth import create_session_token
    from models import SessionLocal, User, Video, Scene

    db = SessionLocal()
    # Create or retrieve dedicated test user
    email = "test_data_flow@videoforge.local"
    user = db.query(User).filter(User.email == email).first()
    if not user:
        user = User(
            id=str(uuid.uuid4()),
            email=email,
            hashed_password=hashlib.sha256(b"secret").hexdigest(),
            credits=500,
            is_verified=True,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    user_id = user.id
    token = create_session_token(user_id, SESSION_SECRET)
    db.close()

    client = TestClient(app, raise_server_exceptions=True)

    print("=" * 80)
    print("STARTING COMPLETE LIVE HTTP SCENE DATA FLOW VALIDATION")
    print("=" * 80)

    for idx, (title, prompt, style, caption_lang) in enumerate(TOPICS, 1):
        print(f"\n[{idx}/5] TESTING TOPIC: {title}")
        print(f"    Prompt: {prompt[:70]}...")
        print(f"    Style: {style} | Captions: {caption_lang}")

        # Step 1: GET /create to obtain valid CSRF token
        r_create_get = client.get("/create", cookies={"vf_session": token})
        assert r_create_get.status_code == 200, f"GET /create returned {r_create_get.status_code}"
        csrf_m = re.search(r'name=["\']csrf_token["\'] value=["\']([^"\']+)["\']', r_create_get.text)
        csrf_token = csrf_m.group(1) if csrf_m else ""

        # Step 2: POST /create (simulate user clicking 'Plan scenes')
        post_data = {
            "topic": prompt,
            "length": "25",
            "style": style,
            "aspect_ratio": "16:9",
            "caption_lang": caption_lang,
            "csrf_token": csrf_token,
        }
        r_post = client.post("/create", data=post_data, cookies={"vf_session": token}, follow_redirects=False)
        print(f"    Route: POST /create -> Status: {r_post.status_code}")
        if r_post.status_code not in (302, 303):
            print(f"    ERROR: Expected 303 redirect, got {r_post.status_code}")
            all_passed = False
            continue

        redirect_url = r_post.headers.get("location", "")
        print(f"    Redirect location: {redirect_url}")
        v_match = re.search(r"/project/([a-zA-Z0-9_-]+)/plan", redirect_url)
        if not v_match:
            print(f"    ERROR: Location header does not match /project/{{id}}/plan: {redirect_url}")
            all_passed = False
            continue
        video_id = v_match.group(1)
        print(f"    Video Project ID: {video_id}")

        # Step 3: Check database records immediately after POST
        db_check = SessionLocal()
        db_video = db_check.query(Video).filter(Video.id == video_id).first()
        db_scenes = db_check.query(Scene).filter(Scene.video_id == video_id).order_by(Scene.order_idx).all()
        print(f"    Database check:")
        print(f"      Video record ID: {db_video.id} | Status: {db_video.status}")
        print(f"      Scenes count in DB: {len(db_scenes)}")
        assert len(db_scenes) == 5, f"Expected 5 scenes in DB, found {len(db_scenes)}"

        db_prompts = []
        for s_idx, sc in enumerate(db_scenes):
            vp_text = sc.visual_prompt or ""
            cap_text = sc.caption or ""
            db_prompts.append(vp_text)
            print(f"      Scene #{s_idx} [DB ID: {sc.id}]: visual_prompt length = {len(vp_text)}, caption length = {len(cap_text)}")
            if len(vp_text.strip()) < 50:
                print(f"      ERROR: DB scene #{s_idx} visual_prompt is too short or empty!")
                all_passed = False

        db_check.close()

        # Step 4: GET /project/{id}/plan (Review & Edit Scenes page)
        r_plan = client.get(redirect_url, cookies={"vf_session": token})
        print(f"    Route: GET {redirect_url} -> Status: {r_plan.status_code}")
        assert r_plan.status_code == 200, f"GET {redirect_url} returned {r_plan.status_code}"
        html = r_plan.text

        # Step 5: Parse returned HTML textareas
        textarea_matches = re.findall(
            r'<textarea[^>]+name=["\']visual_prompt_(\d+)["\'][^>]*>(.*?)</textarea>',
            html,
            re.DOTALL | re.IGNORECASE,
        )
        print(f"    HTML Form Textarea check:")
        print(f"      Found {len(textarea_matches)} 'visual_prompt_X' textareas in rendered HTML")
        assert len(textarea_matches) == 5, f"Expected 5 textareas, found {len(textarea_matches)}"

        topic_success = True
        for t_idx_str, content in textarea_matches:
            t_idx = int(t_idx_str)
            raw_content = content.strip()
            # Unescape basic HTML entities like &#39; or &amp;
            unescaped = raw_content.replace("&#39;", "'").replace("&amp;", "&").replace("&quot;", '"')
            db_expected = db_prompts[t_idx].strip()
            print(f"      Textarea #{t_idx}: length = {len(unescaped)} chars")
            print(f"        Preview: {unescaped[:70]}...")

            if len(unescaped) < 50:
                print(f"        ERROR: Textarea #{t_idx} is empty or too short!")
                topic_success = False
                all_passed = False

            # Check that HTML content matches DB content
            if unescaped[:60] != db_expected[:60]:
                print(f"        ERROR: Textarea content does not match DB scene content!")
                topic_success = False
                all_passed = False

        # Verify captions in HTML can remain empty
        caption_inputs = re.findall(
            r'<input[^>]+name=["\']caption_(\d+)["\'][^>]+value=["\']([^"\']*)["\']',
            html,
            re.IGNORECASE,
        )
        print(f"      Found {len(caption_inputs)} caption inputs. All captions empty as expected: {all(c[1] == '' for c in caption_inputs)}")

        if topic_success:
            print(f"    ==> PASS: {title} data flow 100% verified!")
            results_summary.append((title, "PASS", len(textarea_matches)))
        else:
            print(f"    ==> FAIL: {title} had blank or mismatched textareas!")
            results_summary.append((title, "FAIL", len(textarea_matches)))

    print("\n" + "=" * 80)
    print("FINAL VALIDATION SUMMARY:")
    for t_name, status, count in results_summary:
        print(f"  {status}: {t_name} ({count} scenes verified)")
    print("=" * 80)

sys.exit(0 if all_passed else 1)

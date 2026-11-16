"""Test storyboard pipeline caption visibility, 3D style injection, accessories,
and camera angle diversity for topic:
'A small fox with a red scarf finds a lost mitten in the snow'
"""
import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw

BASE = Path(__file__).resolve().parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from planner import (
    BANNED_PROMPT_WORDS,
    CAMERA_ANGLES,
    STYLE_AVOID_TEXT,
    STYLE_PRESET_PROMPTS,
    generate_character_sheet,
)
from pipeline import (
    check_video_captions,
    plan_storyboard_scenes,
    run_storyboard_pipeline,
)

def test_fox_caption_pipeline():
    topic = "A small fox with a red scarf finds a lost mitten in the snow"
    test_folder = BASE / "test_fox_captions"
    test_folder.mkdir(exist_ok=True)

    print(f"[TEST] 1. Planning scenes for topic: '{topic}' with 3D default style")
    scenes = plan_storyboard_scenes(topic, style="3d_animated", caption_lang="en")
    assert len(scenes) == 5, f"Expected 5 scenes, got {len(scenes)}"
    print("[PASS] Planned 5 scenes.")

    style_3d = STYLE_PRESET_PROMPTS["3d_animated"]
    avoid_text = STYLE_AVOID_TEXT

    # Verify style text, avoid text, accessory, and camera angle diversity
    angles_seen = []
    for i, sc in enumerate(scenes, 1):
        prompt = sc["image_prompt"]
        print(f"\n--- Scene {i} Prompt ({len(prompt.split())} words) ---")
        print(prompt)
        print(f"Caption: {sc['caption']}")

        # 1. Style text at START
        assert prompt.startswith(style_3d), f"Scene {i} prompt does NOT start with style_3d!\nPrompt: {prompt[:80]}"

        # 2. Style text and avoid text at END
        expected_end = f"{style_3d}. {avoid_text}."
        assert prompt.endswith(expected_end), f"Scene {i} prompt does NOT end with expected style & avoid text!\nEnd: {prompt[-120:]}"

        # 3. Accessory presence: 'red scarf' in ALL scene prompts
        assert "red scarf" in prompt.lower() or "scarf" in prompt.lower(), (
            f"Scene {i} prompt missing red scarf accessory: {prompt}"
        )

        # 4. Zero banned words
        prompt_lower = prompt.lower()
        for banned in BANNED_PROMPT_WORDS:
            assert banned not in prompt_lower, f"Scene {i} contains banned word '{banned}': {prompt}"

        # 5. Camera angles
        matched_angle = None
        for ak in ["wide", "low angle", "over-the-shoulder", "high angle", "close-up"]:
            if ak in prompt_lower:
                matched_angle = ak
                break
        assert matched_angle is not None, f"Scene {i} missing camera angle: {prompt}"
        angles_seen.append(matched_angle)

    # 5 unique camera angles
    assert len(set(angles_seen)) == 5, f"Expected 5 unique camera angles, got {angles_seen}"
    print("\n[PASS] All 5 prompts have style at start/end, avoid text, red scarf accessory, and 5 distinct camera angles.")

    # Mock download to quickly create high-variance test frames
    def mock_download_fn(p: str, out_path: Path, scene_num: int) -> bool:
        img = Image.new("RGB", (1280, 720), (25, 35, 55))
        draw = ImageDraw.Draw(img)
        # Background snowy landscape
        draw.rectangle([0, 450, 1280, 720], fill=(210, 225, 240))
        for x in range(0, 1280, 25):
            draw.line([(x, 400), (x + 20, 720)], fill=(160, 185, 210), width=2)
        # Illustrated fox silhouette & red scarf
        draw.ellipse([540, 360, 740, 560], fill=(225, 110, 35))
        draw.polygon([(560, 370), (590, 300), (620, 370)], fill=(210, 90, 25))
        draw.polygon([(660, 370), (690, 300), (720, 370)], fill=(210, 90, 25))
        # Bright red scarf
        draw.rectangle([560, 470, 720, 520], fill=(230, 35, 35))
        # Lost mitten
        draw.ellipse([800, 550, 880, 630], fill=(200, 40, 40))
        draw.text((60, 60), f"Scene {scene_num}: Fox with Red Scarf", fill=(255, 255, 255))
        img.save(str(out_path), "JPEG", quality=92)
        return True

    job_id = "test_fox_winter"
    print("\n[TEST] 2. Running storyboard pipeline...")
    out_mp4, final_scenes, placeholder_scenes, caption_results = run_storyboard_pipeline(
        job_id=job_id,
        topic=topic,
        style="3d_animated",
        aspect_ratio="16:9",
        pre_planned_scenes=scenes,
        download_fn=mock_download_fn,
        skip_network=True,
        caption_samples_dir=test_folder,
    )

    print(f"\n[TEST] 3. Pipeline completed.")
    print(f"Output video: {out_mp4}")
    assert out_mp4.exists(), f"Output video not found: {out_mp4}"
    assert out_mp4.stat().st_size > 10000, f"Output video too small: {out_mp4.stat().st_size}"

    print("\n[TEST] 4. Caption Verification Results:")
    for res in caption_results:
        print(f"  Scene {res['scene']}: '{res.get('caption', '')}' at {res.get('timestamp')}s -> "
              f"{res.get('bright_pixels')} bright pixels [PASSED={res.get('passed')}] (Sample: {res.get('sample_path')})")
        assert res["passed"] is True, f"Scene {res['scene']} caption check FAILED: {res}"
        assert res["bright_pixels"] >= 50, f"Scene {res['scene']} bright pixels too low: {res['bright_pixels']}"
        assert Path(res["sample_path"]).exists(), f"Sample PNG missing: {res['sample_path']}"

    print("\nALL VERIFICATIONS PASSED SUCCESSFULLY.")

if __name__ == "__main__":
    test_fox_caption_pipeline()

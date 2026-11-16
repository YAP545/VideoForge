import sys
import os
from pathlib import Path
from PIL import Image, ImageDraw

# Add videoforge directory to sys.path
BASE_DIR = Path(__file__).parent / "videoforge"
sys.path.insert(0, str(BASE_DIR))

# Ensure UTF-8 console output for Windows
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from pipeline import run_storyboard_pipeline
from planner import (
    extract_story_sheets,
    get_accessory_lock_and_avoids,
    VALID_CAMERAS,
)

TOPICS = [
    "A small fox with a red scarf finds a lost mitten in the snow",
    "A paper boat sailing down a rainy street",
    "एक छोटा हाथी मुंबई की सड़क पर गुब्बारा ढूंढता है",
]

def fake_download_fn(prompt: str, out_path: Path, scene_num: int) -> bool:
    """Creates a distinct colorful test image with scene number and details."""
    w, h = 1280, 720
    colors = [
        (45, 85, 125),
        (125, 65, 45),
        (55, 115, 65),
        (105, 55, 115),
        (120, 100, 40),
    ]
    bg = colors[(scene_num - 1) % len(colors)]
    img = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)
    # Add high-contrast lines and textures for edge detection
    for y in range(50, 650, 40):
        draw.line([(50, y), (1230, y)], fill=(200, 220, 240), width=2)
    draw.rectangle([100, 100, 400, 400], fill=(220, 200, 180), outline=(255, 255, 255), width=3)
    draw.ellipse([600, 200, 900, 500], fill=(240, 150, 80), outline=(0, 0, 0), width=3)
    draw.text((120, 120), f"TEST SCENE {scene_num}", fill=(255, 255, 255))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path), "JPEG", quality=90)
    return True


def fake_gemini_fn(image_path: Path, must_show: list, accessories: list) -> dict:
    """Simulates Gemini picture check returning passed."""
    return {
        "must_show_visible": True,
        "accessory_colors_ok": True,
        "readable": True,
    }

def verify_camera_rules(scenes: list, props: list) -> tuple:
    cams = [s.get("camera", "") for s in scenes]
    wide_count = sum(1 for c in cams if "wide" in c.lower())
    med_count = sum(1 for c in cams if "medium" in c.lower())
    close_count = sum(1 for c in cams if "close" in c.lower())
    
    if wide_count < 2:
        return False, f"Expected >=2 wide, got {wide_count} (cameras: {cams})"
    if med_count < 2:
        return False, f"Expected >=2 medium, got {med_count} (cameras: {cams})"
    if close_count > 1:
        return False, f"Expected <=1 close-up, got {close_count} (cameras: {cams})"
        
    for i in range(len(cams) - 1):
        if cams[i] == cams[i + 1]:
            return False, f"Adjacent identical cameras at scenes {i+1} and {i+2}: {cams[i]}"
            
    # Check close-up with props
    prop_names = [p.get("name", "").lower() for p in props if isinstance(p, dict)]
    for i, s in enumerate(scenes):
        cam = s.get("camera", "")
        ms = " ".join(str(m) for m in s.get("must_show", [])).lower()
        if "close" in cam.lower() and any(pn in ms for pn in prop_names if pn):
            return False, f"Scene {i+1} is close-up but must_show contains prop: {ms}"
            
    return True, "All camera rules passed"

def main():
    test_out_dir = Path("test_output_frames")
    test_out_dir.mkdir(exist_ok=True)
    all_passed = True

    for t_idx, topic in enumerate(TOPICS, 1):
        print(f"\n{'='*70}")
        print(f"TOPIC {t_idx}: {topic}")
        print(f"{'='*70}")
        
        job_id = f"test_run_{t_idx}"
        sample_dir = test_out_dir / f"topic_{t_idx}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Run pipeline with fake download and fake gemini
        mp4_path, scenes, placeholders, cap_results = run_storyboard_pipeline(
            job_id=job_id,
            topic=topic,
            style="3d_animated",
            aspect_ratio="16:9",
            caption_lang="hi" if t_idx == 3 else "en",
            download_fn=fake_download_fn,
            gemini_fn=fake_gemini_fn,
            caption_samples_dir=sample_dir,
            skip_network=True,
        )
        
        sheets = scenes[0].get("sheets") if scenes and "sheets" in scenes[0] else extract_story_sheets(topic)
        lock_text, _ = get_accessory_lock_and_avoids(sheets.get("accessories", []))
        
        # Verify 5 scenes
        assert len(scenes) == 5, f"Expected 5 scenes, got {len(scenes)}"
        
        # Verify camera rules
        cam_ok, cam_msg = verify_camera_rules(scenes, sheets.get("props", []))
        print(f"Camera rules: {cam_msg}")
        if not cam_ok:
            all_passed = False
            
        # Print per scene requirements
        for s_idx, sc in enumerate(scenes, 1):
            cam = sc.get("camera", "medium")
            must_show = sc.get("must_show", [])
            pic_chk = sc.get("picture_check", "passed")
            cap_bright = cap_results[s_idx - 1].get("bright_pixels", 0) if s_idx <= len(cap_results) else 0
            cap_pass = cap_results[s_idx - 1].get("passed", False) if s_idx <= len(cap_results) else False
            
            print(
                f"  Scene {s_idx}: camera={cam:<8} | must_show={str(must_show):<35} | "
                f"accessory_lock='{lock_text}' | picture_check={pic_chk:<7} | "
                f"caption_pixels={cap_bright} (passed={cap_pass})"
            )
            if not cap_pass:
                print(f"    WARNING: Caption failed for scene {s_idx}")
                all_passed = False
                
        assert mp4_path.exists() and mp4_path.stat().st_size > 10000, f"MP4 missing or too small: {mp4_path}"
        print(f"MP4 output verified: {mp4_path} ({mp4_path.stat().st_size} bytes)")

    print(f"\n{'='*70}")
    if all_passed:
        print("ALL 3 TEST TOPICS PASSED SUCCESSFULLY!")
    else:
        print("SOME TESTS FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    main()

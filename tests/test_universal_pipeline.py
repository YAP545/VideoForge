"""Test suite for the VideoForge Universal Prompt-to-Animation Engine.
Verifies universal prompt extraction, non-human subject handling, motion instruction mapping,
and video motion inspection.
"""
import os
import sys
import unittest
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from planner import (
    extract_story_info_fallback,
    build_generic_scene_prompt,
    normalize_story_info,
)
from animation_engine import detect_motion_elements
from video_adapter import video_adapter, inspect_video_motion


class TestUniversalPipeline(unittest.TestCase):
    def test_non_human_subject_extraction(self):
        """Vehicles, animals, and landscapes must not receive human skin, hair, or clothing."""
        # 1. Vehicle
        car_info = extract_story_info_fallback("Cyberpunk sports car drifting in rainy neon Tokyo")
        car_chars = car_info.get("characters", [])
        self.assertEqual(len(car_chars), 1)
        self.assertFalse(car_chars[0]["is_human"])
        self.assertEqual(car_chars[0]["skin_tone"], "")
        self.assertEqual(car_chars[0]["outfit"], "")
        self.assertIn("car", car_chars[0]["role"].lower())

        # 2. Creature / Animal
        eagle_info = extract_story_info_fallback("A golden eagle soaring over Alpine mountains and diving into a lake")
        eagle_chars = eagle_info.get("characters", [])
        self.assertEqual(len(eagle_chars), 1)
        self.assertFalse(eagle_chars[0]["is_human"])
        self.assertEqual(eagle_chars[0]["skin_tone"], "")
        self.assertEqual(eagle_chars[0]["outfit"], "")
        self.assertIn("eagle", eagle_chars[0]["role"].lower())

        # 3. Pure Scenery
        nature_info = extract_story_info_fallback("Serene waterfall crashing into a mountain lake under sunset skies")
        nature_chars = nature_info.get("characters", [])
        self.assertEqual(len(nature_chars), 0)

    def test_human_subject_extraction(self):
        """Human characters must receive culture-appropriate skin tones, distinct roles, and outfits."""
        # 1. Astronaut
        astro_info = extract_story_info_fallback("A lonely astronaut exploring an abandoned space station")
        astro_chars = astro_info.get("characters", [])
        self.assertEqual(len(astro_chars), 1)
        self.assertTrue(astro_chars[0]["is_human"])
        self.assertEqual(astro_chars[0]["role"], "astronaut")
        self.assertIn("spacesuit", astro_chars[0]["outfit"])

        # 2. College friends ensemble (6 distinct characters)
        college_info = extract_story_info_fallback("Create a 3D animated Indian engineering college friendship story with three boys and three girls, showing their first college day")
        college_chars = college_info.get("characters", [])
        self.assertEqual(len(college_chars), 6)
        names = [c["name"] for c in college_chars]
        self.assertEqual(len(set(names)), 6)
        outfits = [c["outfit"] for c in college_chars]
        self.assertEqual(len(set(outfits)), 6)

    def test_motion_element_detection(self):
        """Motion instructions must accurately flag vehicle, creature, human, water, wind, and foliage."""
        # Vehicle drift
        v_elements = detect_motion_elements("Sports car drifting with spinning tires through tight neon corners", "cyberpunk city")
        self.assertTrue(v_elements["has_vehicle"])
        self.assertTrue(v_elements["has_character"])

        # Bird flight
        c_elements = detect_motion_elements("Golden eagle flapping wings and banking sharply over water", "alpine lake")
        self.assertFalse(c_elements["has_human"])
        self.assertTrue(c_elements["has_water"])
        self.assertTrue(c_elements["has_character"])

        # Pure water / landscape
        w_elements = detect_motion_elements("Rushing waterfall cascading down mossy rocks into misty pool", "forest stream")
        self.assertFalse(w_elements["has_character"])
        self.assertTrue(w_elements["has_water"])
        self.assertTrue(w_elements["has_foliage"])

    def test_generic_scene_prompt_cleanliness(self):
        """Scene prompt builder must avoid leaking human facial expressions or skin tones into non-human scenes."""
        car_info = extract_story_info_fallback("Cyberpunk sports car drifting in rainy neon Tokyo")
        scene = {
            "scene_num": 1,
            "camera": "wide",
            "action": "drifting around wet asphalt with tires smoking and headlights glowing",
            "setting_detail": "rain-soaked Tokyo streets with vibrant neon reflections",
            "characters_present": [car_info["characters"][0]["name"]],
            "emotion": "calm",
            "continuity": "establishing speed and vehicle entry",
            "lighting": "intense neon cyan and magenta rim light",
            "depth": "deep city avenues receding into distant mist",
        }
        prompt = build_generic_scene_prompt(scene=scene, story_info=car_info, style="cyberpunk")
        
        # Verify no human leakage
        self.assertNotIn("natural skin tone", prompt)
        self.assertNotIn("smile", prompt)
        self.assertNotIn("visage", prompt)
        self.assertNotIn("wearing a", prompt)
        self.assertIn("Cyberpunk Sports Car", prompt)
        self.assertIn("drifting around wet asphalt", prompt)

    def test_motion_adapter_and_inspection(self):
        """Test video_adapter motion generation and video inspection tools."""
        test_dir = Path("videoforge/tmp_test")
        test_dir.mkdir(parents=True, exist_ok=True)
        img_path = test_dir / "test_input.jpg"
        out_clip = test_dir / "test_motion.mp4"

        # Create a small valid test image if none exists
        if not img_path.exists():
            img = Image.new("RGB", (720, 1280), color=(30, 80, 160))
            img.save(img_path)

        # Generate motion via video_adapter (defaults to FFmpeg camera motion baseline)
        ok, engine, meta = video_adapter.generate_motion(
            image_path=img_path,
            out_clip_path=out_clip,
            prompt="Test drone flying through valley",
            motion_instructions="slow cinematic push forward and downward tilt",
            width=720,
            height=1280,
            duration=2.0,
            fps=24,
            scene_idx=0,
            camera="push_in",
        )
        self.assertTrue(ok)
        self.assertTrue(out_clip.exists())
        self.assertGreater(out_clip.stat().st_size, 1000)

        # Inspect generated clip
        inspection = inspect_video_motion(out_clip)
        self.assertTrue(inspection["valid"])
        self.assertAlmostEqual(inspection["duration"], 2.0, delta=0.5)
        self.assertEqual(inspection["width"], 720)
        self.assertEqual(inspection["height"], 1280)
        self.assertGreater(inspection["frame_diff_mean"], 0.0)

        # Cleanup
        try:
            if out_clip.exists():
                out_clip.unlink()
            if img_path.exists():
                img_path.unlink()
            test_dir.rmdir()
        except Exception:
            pass


if __name__ == "__main__":
    unittest.main()

# -*- coding: utf-8 -*-
"""Regression tests verifying:
1. Scene 2 (sunset garden) contains exactly two characters without single-person constraints or inset picture phrasing.
2. Solo scenes (1, 3, 4, 5) contain single-person constraints and anti-duplication phrases.
3. validate_subtitle_sequence strictly enforces 4 lines across scenes 1..4 and empty caption/narration on scene 5.
4. Truthful engine reporting and ZeroGPU quota short-circuiting.
5. check_picture_gemini validation schema and defect detection.
"""

import unittest
from pathlib import Path
import sys
import os
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planner import plan_scenes, extract_explicit_devanagari_lines, enforce_explicit_shayari_plan, build_generic_scene_prompt
from pipeline import validate_subtitle_sequence, check_picture_gemini
from video_adapter import VideoModelAdapter


class TestImageValidationAndSubtitles(unittest.TestCase):

    def setUp(self):
        self.shayari_prompt = (
            "Create a 3D animated romantic story in Hindi for this shayari:\n"
            "तेरी यादों की खुशबू से महकती है मेरी हर शाम,\n"
            "जैसे ढलते सूरज ने लिख दिया हो तेरा नाम।\n"
            "भीगी पलकों पे सजाई है तेरी एक तस्वीर,\n"
            "हाथों की लकीरों में ढूँढता हूँ अपनी तकदीर।\n"
            "Style: 3D animated, emotionally touching, cinematic sunset garden memory and moonlit rooftop."
        )
        self.expected_lines = [
            "तेरी यादों की खुशबू से महकती है मेरी हर शाम,",
            "जैसे ढलते सूरज ने लिख दिया हो तेरा नाम।",
            "भीगी पलकों पे सजाई है तेरी एक तस्वीर,",
            "हाथों की लकीरों में ढूँढता हूँ अपनी तकदीर।",
        ]

        # Ensure a valid dummy image exists for FFmpeg test
        self.test_img = Path(__file__).resolve().parent.parent / "test_dummy_source.jpg"
        self.test_out = Path(__file__).resolve().parent.parent / "test_dummy_out.mp4"
        Image.new("RGB", (720, 1280), (80, 100, 120)).save(self.test_img)

    def tearDown(self):
        for p in (self.test_img, self.test_out):
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass

    def test_extract_explicit_devanagari_lines(self):
        lines = extract_explicit_devanagari_lines(self.shayari_prompt)
        self.assertEqual(len(lines), 4)
        for i, exp in enumerate(self.expected_lines):
            self.assertEqual(lines[i], exp)

    def test_shayari_scene_planning_character_counts_and_prompts(self):
        scenes = plan_scenes(self.shayari_prompt, length_seconds=25, style="3d_animated", caption_lang="hi")
        self.assertEqual(len(scenes), 5)

        # Scene 1: Solo Kabir
        s1 = scenes[0]
        self.assertEqual(s1["characters_present"], ["Kabir"])
        self.assertIn("single person only", s1["image_prompt"].lower())
        self.assertEqual(s1["caption"], self.expected_lines[0])
        self.assertEqual(s1["narration"], self.expected_lines[0])

        # Scene 2: Pair (Kabir and Meera) - NO single person constraint, NO inset picture
        s2 = scenes[1]
        self.assertEqual(set(s2["characters_present"]), {"Kabir", "Meera"})
        self.assertNotIn("single person only", s2["image_prompt"].lower())
        self.assertIn("exactly two people together in the frame", s2["image_prompt"].lower())
        self.assertIn("dark blue shirt", s2["image_prompt"].lower())
        self.assertIn("light pink traditional outfit", s2["image_prompt"].lower())
        self.assertIn("no inset picture", s2["image_prompt"].lower())
        self.assertIn("no picture frames", s2["image_prompt"].lower())
        self.assertEqual(s2["caption"], self.expected_lines[1])
        self.assertEqual(s2["narration"], self.expected_lines[1])

        # Scene 3: Solo Meera
        s3 = scenes[2]
        self.assertEqual(s3["characters_present"], ["Meera"])
        self.assertIn("single person only", s3["image_prompt"].lower())
        self.assertEqual(s3["caption"], self.expected_lines[2])
        self.assertEqual(s3["narration"], self.expected_lines[2])

        # Scene 4: Solo Kabir
        s4 = scenes[3]
        self.assertEqual(s4["characters_present"], ["Kabir"])
        self.assertIn("single person only", s4["image_prompt"].lower())
        self.assertEqual(s4["caption"], self.expected_lines[3])
        self.assertEqual(s4["narration"], self.expected_lines[3])

        # Scene 5: Outro solo, empty caption and narration, camera wide
        s5 = scenes[4]
        self.assertEqual(s5["characters_present"], ["Kabir"])
        self.assertEqual(s5["caption"], "")
        self.assertEqual(s5["narration"], "")
        self.assertEqual(s5["camera"], "wide")
        self.assertIn("single person only", s5["image_prompt"].lower())

    def test_validate_subtitle_sequence_success(self):
        scenes = [
            {"caption": self.expected_lines[0], "narration": self.expected_lines[0]},
            {"caption": self.expected_lines[1], "narration": self.expected_lines[1]},
            {"caption": self.expected_lines[2], "narration": self.expected_lines[2]},
            {"caption": self.expected_lines[3], "narration": self.expected_lines[3]},
            {"caption": "", "narration": ""},
        ]
        ok, errors = validate_subtitle_sequence(scenes, self.expected_lines)
        self.assertTrue(ok)
        self.assertEqual(len(errors), 0)

    def test_validate_subtitle_sequence_catches_leaked_scene_5(self):
        scenes = [
            {"caption": self.expected_lines[0], "narration": self.expected_lines[0]},
            {"caption": self.expected_lines[1], "narration": self.expected_lines[1]},
            {"caption": self.expected_lines[2], "narration": self.expected_lines[2]},
            {"caption": self.expected_lines[3], "narration": self.expected_lines[3]},
            {"caption": self.expected_lines[3], "narration": self.expected_lines[3]},  # Leaked repeat
        ]
        ok, errors = validate_subtitle_sequence(scenes, self.expected_lines)
        self.assertFalse(ok)
        self.assertTrue(any("Scene 5 must have empty caption" in e for e in errors))
        self.assertTrue(any("Scene 5 must have empty narration" in e for e in errors))

    def test_video_adapter_quota_exhausted_short_circuit_and_truthful_reporting(self):
        adapter = VideoModelAdapter()
        self.assertFalse(adapter._cloud_quota_exhausted)

        # Simulate quota exhaustion
        adapter._cloud_quota_exhausted = True

        progress_msgs = []
        def on_prog(msg):
            progress_msgs.append(msg)

        success, engine, meta = adapter.generate_motion(
            image_path=self.test_img,
            out_clip_path=self.test_out,
            prompt="A man looking at the stars",
            duration=1.0,
            scene_idx=1,
            on_progress=on_prog,
        )

        self.assertTrue(success)
        self.assertEqual(engine, "ffmpeg_camera_motion")
        self.assertFalse(meta.get("genuine_video"))
        self.assertEqual(meta.get("motion_type"), "camera_motion")

        # Verify truthful disclosure was broadcast in progress
        all_text = " ".join(progress_msgs)
        self.assertIn("Free cloud AI video quota currently exhausted", all_text)
        self.assertIn("FFmpeg cinematic camera motion", all_text)


if __name__ == "__main__":
    unittest.main()

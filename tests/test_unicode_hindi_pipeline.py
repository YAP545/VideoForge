"""Comprehensive Unicode & Global Hindi Compatibility Test Suite for VideoForge.
Verifies pure Hindi, Hindi romantic shayari, mixed Hindi-English, emojis, conjuncts (संयुक्त अक्षर),
Devanagari text shaping, edge-tts Hindi voiceover, and video export without charmap encoding errors.
"""
import os
import sys
import unittest
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
import unicode_compat

from planner import (
    extract_story_info_fallback,
    build_generic_scene_prompt,
    plan_scenes_template,
)
from pipeline import (
    detect_voice,
    generate_voiceover,
    burn_caption_overlay,
    burn_timed_captions,
    mux_clip_audio,
)
from compositor import (
    get_font_path,
    shape_and_render_multiline_prompt,
    export_srt,
)
from video_adapter import inspect_video_motion


class TestUnicodeHindiPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path("videoforge/tmp_test_unicode")
        cls.test_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        import shutil
        if cls.test_dir.exists():
            try:
                shutil.rmtree(cls.test_dir, ignore_errors=True)
            except Exception:
                pass

    def test_1_pure_hindi_prompt(self):
        """Pure Hindi with Devanagari text, matras, and Hindi punctuation."""
        prompt = "तेरी यादों का भी अपना एक फ़साना है।"
        
        # 1. Voice detection
        voice = detect_voice(prompt)
        self.assertIn("hi-IN", voice)

        # 2. Planning & extraction
        info = extract_story_info_fallback(prompt)
        self.assertTrue(len(info.get("story_elements", [])) >= 3)

        # 3. HarfBuzz caption rendering
        font_path = get_font_path()
        cap_img = shape_and_render_multiline_prompt(prompt, font_path, font_size=36, width=1080, height=1920)
        self.assertEqual(cap_img.size, (1080, 1920))
        self.assertEqual(cap_img.mode, "RGBA")

        # 4. SRT Export
        srt_path = self.test_dir / "pure_hindi.srt"
        scenes = [{"caption": prompt}]
        export_srt(scenes, 4.0, srt_path)
        self.assertTrue(srt_path.exists())
        srt_text = srt_path.read_text(encoding="utf-8")
        self.assertIn(prompt, srt_text)

    def test_2_hindi_romantic_shayari(self):
        """Hindi romantic shayari with voiceover generation and multiline captions."""
        shayari = "मोहब्बत की राहों में दर्द ही दर्द मिला, फिर भी ये दिल तुझपे ही फ़िदा रहा।"
        
        voice = detect_voice(shayari)
        self.assertIn("hi-IN", voice)

        # Generate audio
        audio_path = self.test_dir / "shayari_voice.mp3"
        word_bounds = []
        ok, warn = generate_voiceover(shayari, voice, audio_path, word_boundaries=word_bounds)
        self.assertTrue(ok)
        self.assertTrue(audio_path.exists())
        self.assertGreater(audio_path.stat().st_size, 1000)

        # Verify multiline caption wrapping
        font_path = get_font_path()
        cap_img = shape_and_render_multiline_prompt(shayari, font_path, font_size=42, width=720, height=1280)
        self.assertIsNotNone(cap_img)

    def test_3_mixed_hindi_english(self):
        """Mixed Hindi-English text (Hinglish + English story keywords)."""
        mixed_prompt = "Delhi metro station par ek cute couple coffee shop mein baatein kar rahe hain, romantic shayari."
        voice = detect_voice(mixed_prompt)
        self.assertIn("hi-IN", voice)

        scenes = plan_scenes_template(mixed_prompt, 25, "cinematic", "hi")
        self.assertEqual(len(scenes), 5)
        # Verify Hindi captions in template plan
        has_devanagari = any(any("\u0900" <= ch <= "\u097F" for ch in sc.get("caption", "")) for sc in scenes)
        self.assertTrue(has_devanagari)

    def test_4_hindi_punctuation_emojis_conjuncts(self):
        """Punctuation, emojis, and complex संयुक्त अक्षर (conjunct ligatures)."""
        complex_text = "कर्मण्येवाधिकारस्ते मा फलेषु कदाचन। 🕉️✨ — श्रीकृष्ण (ज्ञान, भक्ति, और कर्मयोग)"
        
        # Verify text survives print and file writing without charmap crash
        test_file = self.test_dir / "complex_hindi.txt"
        with open(test_file, "w") as f:
            f.write(complex_text)
        read_back = open(test_file).read()
        self.assertEqual(complex_text, read_back)

        # HarfBuzz Devanagari text shaping with emojis
        font_path = get_font_path()
        cap_img = shape_and_render_multiline_prompt(complex_text, font_path, font_size=32, width=1080, height=1920)
        self.assertEqual(cap_img.size, (1080, 1920))

    def test_5_english_regression(self):
        """English-only prompts must continue to work without any regression."""
        en_prompt = "A majestic golden eagle soaring over Alpine mountains and diving into a lake"
        voice = detect_voice(en_prompt)
        self.assertIn("en-US", voice)

        info = extract_story_info_fallback(en_prompt)
        self.assertEqual(len(info["characters"]), 1)
        self.assertFalse(info["characters"][0]["is_human"])

    def test_6_end_to_end_clip_assembly_hindi(self):
        """Generates a complete scene clip with Hindi caption overlay and Hindi TTS voiceover."""
        hindi_narration = "तेरी यादों का भी अपना एक फ़साना है।"
        voice = detect_voice(hindi_narration)
        
        # 1. Base test image
        img_path = self.test_dir / "hindi_scene_img.jpg"
        img = Image.new("RGB", (720, 1280), color=(18, 24, 45))
        img.save(img_path)

        # 2. Base motion clip via ffmpeg
        motion_clip = self.test_dir / "hindi_motion.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1",
            "-i", str(img_path),
            "-t", "2.0",
            "-r", "24",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(motion_clip)
        ]
        import subprocess
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        self.assertTrue(motion_clip.exists())

        # 3. Burn Hindi caption with HarfBuzz text shaping
        captioned_clip = self.test_dir / "hindi_captioned.mp4"
        burn_ok = burn_caption_overlay(
            clip_path=motion_clip,
            caption=hindi_narration,
            out_path=captioned_clip,
            width=720,
            height=1280,
        )
        self.assertTrue(burn_ok)
        self.assertTrue(captioned_clip.exists())

        # 4. Generate Hindi voiceover
        audio_path = self.test_dir / "hindi_voice.mp3"
        tts_ok, _ = generate_voiceover(hindi_narration, voice, audio_path)
        self.assertTrue(tts_ok)
        self.assertTrue(audio_path.exists())

        # 5. Mux audio and video
        final_clip = self.test_dir / "hindi_final_scene.mp4"
        mux_clip_audio(captioned_clip, audio_path, final_clip, duration=2.0)
        self.assertTrue(final_clip.exists())

        # 6. Verify with inspect_video_motion
        inspection = inspect_video_motion(final_clip)
        self.assertTrue(inspection["valid"])
        self.assertAlmostEqual(inspection["duration"], 2.0, delta=0.5)
        self.assertEqual(inspection["width"], 720)
        self.assertEqual(inspection["height"], 1280)


if __name__ == "__main__":
    unittest.main()

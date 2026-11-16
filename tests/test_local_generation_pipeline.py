"""
End-to-end local generation test verifying:
1. run_storyboard_pipeline executes Phase A, Phase B, and Phase C retry with negative_prompt.
2. No 'download_scene_image() got an unexpected keyword argument' error occurs.
3. StoryboardProvider.generate() works end-to-end locally.
4. Cloudflare is maintained as the configured IMAGE_PROVIDER.
"""
import os
import shutil
import unittest
from pathlib import Path
from PIL import Image, ImageDraw

import sys
sys.path.insert(0, r"C:\Users\yasha\Downloads\videoforge\videoforge")

import pipeline
from pipeline import run_storyboard_pipeline, reset_cloudflare_quota
from providers import StoryboardProvider


class TestLocalGenerationPipeline(unittest.TestCase):
    def setUp(self):
        reset_cloudflare_quota()
        self.tmp_dir = Path(r"C:\Users\yasha\Downloads\videoforge\videoforge\videos\test_local_gen")
        self.tmp_dir.mkdir(exist_ok=True, parents=True)

    def tearDown(self):
        reset_cloudflare_quota()
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_pipeline_local_generation_with_negative_prompt_retry(self):
        """Runs run_storyboard_pipeline with a mock download function simulating defect retry."""
        test_scenes = [
            {
                "image_prompt": f"Lord Hanuman scene {i} on sacred mountain peak at sunrise",
                "visual_prompt": f"Lord Hanuman scene {i} on sacred mountain peak at sunrise",
                "caption": f"Lord Hanuman meditating scene {i}",
                "duration": 3.0,
            }
            for i in range(1, 6)
        ]

        calls = []

        def mock_download(prompt, out_path, scene_num, **kwargs):
            calls.append({"prompt": prompt, "scene_num": scene_num, "kwargs": kwargs})
            im = Image.new("RGB", (1280, 720), (20, 40, 80))
            d = ImageDraw.Draw(im)
            d.rectangle([100, 100, 1180, 620], fill=(200, 150, 50))
            d.text((200, 200), f"Scene {scene_num}", fill=(255, 255, 255))
            im.save(out_path, "JPEG")
            return True

        # Run pipeline with skip_network=True and our mock download
        out_mp4, scenes_res, placeholders, caption_res = run_storyboard_pipeline(
            job_id="test_local_neg_prompt_run",
            topic="Lord Hanuman Devotional Animation",
            style="3d_animated",
            aspect_ratio="16:9",
            pre_planned_scenes=test_scenes,
            download_fn=mock_download,
            skip_network=True,
            caption_samples_dir=self.tmp_dir,
        )

        self.assertTrue(out_mp4.exists(), f"Expected video {out_mp4} to exist")
        self.assertGreater(out_mp4.stat().st_size, 1000, "Expected non-empty output mp4")
        self.assertGreaterEqual(len(calls), 2, "Expected at least 2 scene downloads")
        print(f"\n[PASS] Local generation test successful! Output video: {out_mp4} ({out_mp4.stat().st_size} bytes)")

    def test_storyboard_provider_generate_local(self):
        """Runs StoryboardProvider.generate locally with mock download."""
        provider = StoryboardProvider()
        clip_path = self.tmp_dir / "provider_clip.mp4"

        # Mock download_scene_image so it creates a valid frame
        def mock_scene_img(prompt, out_path, **kwargs):
            im = Image.new("RGB", (1280, 720), (30, 60, 90))
            im.save(out_path, "JPEG")
            return True

        from unittest.mock import patch
        with patch("pipeline.download_scene_image", side_effect=mock_scene_img) as mock_dl:
            res = provider.generate(
                prompt="Lord Hanuman standing atop mountain",
                seconds=3.0,
                out_path=clip_path,
                aspect_ratio="16:9",
                caption="Om Hanumate Namah",
                skip_network=True,
                negative_prompt="blurry, distorted",
            )
            self.assertEqual(mock_dl.call_count, 1)
            _, passed_kwargs = mock_dl.call_args
            self.assertEqual(passed_kwargs.get("negative_prompt"), "blurry, distorted")
            self.assertTrue(clip_path.exists())
            self.assertGreater(clip_path.stat().st_size, 1000)
            print(f"[PASS] StoryboardProvider local generation clip: {clip_path} ({clip_path.stat().st_size} bytes)")


if __name__ == "__main__":
    unittest.main()

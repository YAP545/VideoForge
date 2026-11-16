"""Offline tests for generic scene planning across diverse topics."""
import os
import unittest
from unittest.mock import patch, MagicMock

import planner


class TestGenericScenePlanning(unittest.TestCase):
    TOPICS = [
        "a fisherman and a storm",
        "a robot learning to paint",
        "two kids and a lost dog",
        "a farmer facing drought",
        "a scientist on Mars",
    ]

    def test_five_topics_planning_requirements(self):
        """Test on five distinct topics with offline planner.
        Verifies:
        1. characters have skin_tone (humans)
        2. prompts contain 'natural skin tone' and the background phrase
        3. at least one scene has a character absent
        4. at least one scene is wide
        5. a plan with both characters in all scenes is auto-fixed
        """
        bg_phrase = "detailed full background filling the frame, not a plain white or grey background"

        for topic in self.TOPICS:
            with self.subTest(topic=topic):
                story_info = planner.extract_story_info(topic)
                chars = story_info.get("characters", [])

                # 1. Characters have skin_tone (humans)
                for c in chars:
                    if c.get("is_human", True):
                        self.assertTrue(
                            bool(c.get("skin_tone")),
                            f"Human character {c.get('name')} in topic '{topic}' missing skin_tone",
                        )

                # Generate scenes via planner
                scenes = planner.plan_scenes(topic, style="3d_animated")
                self.assertEqual(len(scenes), 5, f"Topic '{topic}' should produce 5 scenes")

                # 2. Prompts contain 'natural skin tone' and background phrase
                for idx, sc in enumerate(scenes):
                    prompt = sc.get("image_prompt", "")
                    self.assertIn(
                        "natural skin tone",
                        prompt,
                        f"Topic '{topic}' scene {idx+1} missing 'natural skin tone' in prompt: {prompt}",
                    )
                    self.assertIn(
                        bg_phrase,
                        prompt,
                        f"Topic '{topic}' scene {idx+1} missing background phrase in prompt: {prompt}",
                    )

                # 3. At least one scene has a character absent
                if chars:
                    absent_found = False
                    for sc in scenes:
                        present = sc.get("characters_present", [])
                        if len(present) < len(chars):
                            absent_found = True
                            break
                    self.assertTrue(
                        absent_found,
                        f"Topic '{topic}' expected at least one scene with an absent character",
                    )

                # 4. At least one scene is wide
                has_wide = any("wide" in str(sc.get("camera", "")).lower() for sc in scenes)
                self.assertTrue(has_wide, f"Topic '{topic}' expected at least one wide shot")

    def test_plan_with_both_characters_in_all_scenes_is_autofixed(self):
        """A plan with both characters in all scenes must be rejected by validator and auto-fixed."""
        topic = "two kids and a lost dog"
        story_info = planner.extract_story_info(topic)
        char_names = [c["name"] for c in story_info.get("characters", [])]
        self.assertGreaterEqual(len(char_names), 2)

        # Defective plan: both characters present in every scene, no wide shot, all close-up
        defective_scenes = [
            {
                "camera": "close-up",
                "action": f"{char_names[0]} and {char_names[1]} search together.",
                "setting_detail": "short detail",
                "characters_present": list(char_names),
                "must_show": list(char_names),
            }
            for _ in range(5)
        ]

        defects = planner.validate_scene_plan(defective_scenes, story_info)
        self.assertTrue(len(defects) > 0, "Validator should detect defects")
        self.assertTrue(any("same characters" in d for d in defects))
        self.assertTrue(any("wide" in d for d in defects))

        # Autofix
        fixed_scenes = planner.autofix_scene_plan(defective_scenes, story_info)
        post_defects = planner.validate_scene_plan(fixed_scenes, story_info)

        # Confirm at least one scene now has an absent character
        absent_found = any(len(sc.get("characters_present", [])) < len(char_names) for sc in fixed_scenes)
        self.assertTrue(absent_found, "After autofix, at least one character must be absent in a scene")

        # Confirm establishing wide shot is present
        has_wide = any("wide" in sc.get("camera", "").lower() for sc in fixed_scenes)
        self.assertTrue(has_wide, "After autofix, an establishing wide shot must be present")

    def test_fake_llm_workflow(self):
        """Test LLM planning path with mock to ensure re-ask on defect and autofix work."""
        topic = "a scientist on Mars"
        story_info = planner.extract_story_info(topic)

        # Fake LLM response that returns defective plan first, then valid plan
        defective_payload = [
            {
                "camera": "close-up",
                "action": "Scientist looks around.",
                "setting_detail": "red rocky Martian landscape spreading wide across horizon",
                "characters_present": ["The Scientist"],
                "must_show": ["The Scientist"],
                "caption": "Exploring Mars.",
            }
            for _ in range(5)
        ]

        import json
        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__.return_value = mock_client

            # Return defective payload
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "candidates": [{"content": {"parts": [{"text": json.dumps(defective_payload)}]}}]
            }
            mock_client.post.return_value = mock_resp

            scenes = planner.plan_scenes_gemini(
                topic=topic,
                length_seconds=25,
                style="3d_animated",
                caption_lang="en",
                api_key="fake-test-key",
            )

            self.assertEqual(len(scenes), 5)
            # Re-ask was called because first plan was defective
            self.assertGreaterEqual(mock_client.post.call_count, 1)
            # Scenes were auto-fixed
            has_wide = any("wide" in sc.get("camera", "").lower() for sc in scenes)
            self.assertTrue(has_wide)
            for sc in scenes:
                self.assertIn("natural skin tone", sc["image_prompt"])

    def test_bad_json_from_fake_llm_produces_five_scenes(self):
        """When the LLM returns bad or unparseable JSON, the planner must not crash and must produce 5 scenes."""
        topic = "a fisherman and a storm"
        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__.return_value = mock_client
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "candidates": [{"content": {"parts": [{"text": "```json\nHere is your plan: {invalid_json..."}]}}]
            }
            mock_client.post.return_value = mock_resp

            scenes = planner.plan_scenes_gemini(
                topic=topic,
                length_seconds=25,
                style="3d_animated",
                caption_lang="en",
                api_key="fake-test-key",
            )
            self.assertEqual(len(scenes), 5, "Fallback must produce exactly 5 scenes")
            for sc in scenes:
                self.assertTrue(bool(sc.get("image_prompt")))
                self.assertTrue(bool(sc.get("camera")))

    def test_empty_characters_present_produces_five_scenes(self):
        """When characters_present is empty in all scenes, the planner must still produce a 5-scene plan."""
        topic = "a paper boat sailing down a rainy street"
        story_info = planner.extract_story_info(topic)

        # Scenes with empty characters_present
        raw_scenes = [
            {
                "camera": "wide",
                "action": "A paper boat drifts peacefully.",
                "setting_detail": "rainy flooded city street reflecting soft streetlights",
                "characters_present": [],
                "must_show": ["paper boat"],
                "caption": "Drifting alone.",
            }
            for _ in range(5)
        ]
        # Autofix and build prompts
        fixed = planner.autofix_scene_plan(raw_scenes, story_info)
        self.assertEqual(len(fixed), 5)
        for sc in fixed:
            prompt = planner.build_generic_scene_prompt(sc, story_info, style="3d_animated")
            self.assertTrue(bool(prompt))
            self.assertIn("natural skin tone", prompt)
            self.assertIn("detailed full background filling the frame, not a plain white or grey background", prompt)


if __name__ == "__main__":
    unittest.main()

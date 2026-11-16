"""
test_universal_scene_generation.py
Full regression test suite for the universal scene generation system.
Run via: venv/Scripts/python.exe tests/test_universal_scene_generation.py
"""
import sys, os
# Clear API keys to test pure deterministic fallback speed and reliability offline
os.environ.pop("GEMINI_API_KEY", None)
os.environ.pop("ANTHROPIC_API_KEY", None)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import planner

TOPICS = [
    ("devotional", "Hanuman devotional animation about his journey to Lanka"),
    ("romance",    "Romantic couple animation at sunset by the beach"),
    ("horror",     "Horror animation in abandoned haunted house at night"),
    ("comedy",     "Funny cartoon animation with talking animals in the jungle"),
    ("action",     "Action superhero animation fighting villains in the city"),
    ("children",   "Children animal story about a baby elephant learning to fly"),
    ("scifi",      "Sci-fi animation on a space station orbiting Earth"),
]

passed = 0
failed = 0

def test(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        print(f"  [PASS] {name}")
        passed += 1
    else:
        print(f"  [FAIL] {name}" + (f" - {detail}" if detail else ""))
        failed += 1

print("=" * 68)
print("UNIVERSAL SCENE GENERATION - REGRESSION TESTS")
print("=" * 68)

# -- Test 1: Genre detection accuracy -----------------------------------------
print("\n[1] Genre detection accuracy")
for expected_genre, topic in TOPICS:
    detected = planner.detect_genre(topic)
    test(f"detect_genre({topic[:35]}...)", detected == expected_genre,
         f"expected={expected_genre!r}, got={detected!r}")

# -- Test 2: 5 scenes produced for every topic --------------------------------
print("\n[2] 5 scenes produced per topic")
for expected_genre, topic in TOPICS:
    scenes = planner.plan_scenes_template(topic, 25, "3d_animated", "none")
    test(f"{expected_genre}: exactly 5 scenes", len(scenes) == 5,
         f"got {len(scenes)}")

# -- Test 3: No empty visual_prompt ------------------------------------------
print("\n[3] No empty visual_prompt fields")
for expected_genre, topic in TOPICS:
    scenes = planner.plan_scenes_template(topic, 25, "3d_animated", "none")
    empty = [i+1 for i, s in enumerate(scenes) if not (s.get("visual_prompt") or "").strip()]
    test(f"{expected_genre}: all visual_prompts non-empty", len(empty) == 0,
         f"empty scenes: {empty}")

# -- Test 4: Actions are distinct across 5 scenes ----------------------------
print("\n[4] Scene actions are distinct (not duplicated)")
for expected_genre, topic in TOPICS:
    scenes = planner.plan_scenes_template(topic, 25, "3d_animated", "none")
    actions = [s.get("action", "")[:60] for s in scenes]
    unique = len(set(actions))
    test(f"{expected_genre}: >=4 distinct actions (got {unique})", unique >= 4,
         f"actions={actions}")

# -- Test 5: Actions are genre-appropriate (keyword check) -------------------
print("\n[5] Actions reference genre-appropriate content")
genre_markers = {
    "devotional": ["reverence", "divine", "strength", "victorious", "obstacle"],
    "romance":    ["meet", "laugh", "walk", "heartfelt", "sunset"],
    "horror":     ["arrives", "disturbing", "backs away", "runs", "escapes"],
    "comedy":     ["arrives", "shock", "absurd", "ridiculous", "laughs"],
    "action":     ["surveys", "confronts", "combat", "decisive", "victorious"],
    "children":   ["wakes", "discovers", "tries", "learns", "celebrates"],
    "scifi":      ["gazes", "detects", "races", "breakthrough", "viewport"],
}
for expected_genre, topic in TOPICS:
    scenes = planner.plan_scenes_template(topic, 25, "3d_animated", "none")
    actions_combined = " ".join(s.get("action", "").lower() for s in scenes)
    markers = genre_markers.get(expected_genre, [])
    matched = [m for m in markers if m in actions_combined]
    test(f"{expected_genre}: genre markers in actions ({len(matched)}/{len(markers)})",
         len(matched) >= 3, f"matched={matched}")

# -- Test 6: Shayari path does NOT trigger for non-shayari topics -------------
print("\n[6] Shayari path isolation (non-shayari topics not contaminated)")
for expected_genre, topic in TOPICS:
    scenes = planner.plan_scenes_template(topic, 25, "3d_animated", "none")
    combined = " ".join(s.get("visual_prompt", "") + " " + s.get("action", "") for s in scenes)
    contaminated = any(name in combined for name in ["Kabir", "Meera", "shayari"])
    test(f"{expected_genre}: no Kabir/Meera/shayari contamination",
         not contaminated, f"contaminated text: {combined[:100]}")

# -- Test 7: Captions off -> all captions empty -------------------------------
print("\n[7] Captions disabled -> empty caption fields")
for expected_genre, topic in TOPICS:
    scenes = planner.plan_scenes_template(topic, 25, "3d_animated", "none")
    nonempty_caps = [i+1 for i, s in enumerate(scenes) if (s.get("caption") or "").strip()]
    test(f"{expected_genre}: all captions empty (caption_lang=none)", len(nonempty_caps) == 0,
         f"scenes with captions: {nonempty_caps}")

# -- Test 8: Hindi captions when caption_lang='hi' ---------------------------
print("\n[8] Hindi captions produced when caption_lang='hi'")
_, topic = TOPICS[0]  # devotional
scenes_hi = planner.plan_scenes_template(topic, 25, "3d_animated", "hi")
has_caps = any((s.get("caption") or "").strip() for s in scenes_hi)
test("devotional (hi): at least one caption present", has_caps)

# -- Test 9: Characters from topic appear in actions -------------------------
print("\n[9] Topic character appears in scene actions")
char_checks = [
    ("devotional", "Hanuman devotional animation about his journey to Lanka", "Hanuman"),
    ("scifi",      "Sci-fi animation on a space station orbiting Earth", None),
]
for genre, topic, expected_name in char_checks:
    scenes = planner.plan_scenes_template(topic, 25, "3d_animated", "none")
    combined = " ".join(s.get("action", "") for s in scenes)
    if expected_name:
        test(f"{genre}: {expected_name!r} in actions", expected_name in combined,
             f"combined[:120]={combined[:120]}")
    else:
        has_char = bool(scenes[0].get("characters_present"))
        test(f"{genre}: at least one character present", has_char)

# -- Test 10: Valid camera values ---------------------------------------------
print("\n[10] Camera values are valid")
VALID_CAMS = {"wide", "medium", "close-up"}
for expected_genre, topic in TOPICS:
    scenes = planner.plan_scenes_template(topic, 25, "3d_animated", "none")
    bad = [s.get("camera") for s in scenes if s.get("camera") not in VALID_CAMS]
    test(f"{expected_genre}: all cameras valid", len(bad) == 0, f"invalid: {bad}")

# -- Test 11: All 10 supported style presets parse without error --------------
print("\n[11] All style presets produce valid prompts")
styles = list(planner.STYLE_PRESET_PROMPTS.keys())
topic_test = "A fantasy adventure with a brave hero"
for style in styles:
    try:
        scenes = planner.plan_scenes_template(topic_test, 25, style, "none")
        ok = len(scenes) == 5 and all((s.get("visual_prompt") or "").strip() for s in scenes)
        test(f"style={style!r}", ok)
    except Exception as ex:
        test(f"style={style!r}", False, str(ex))

# -- Summary ------------------------------------------------------------------
total = passed + failed
print()
print("=" * 68)
print(f"RESULTS: {passed}/{total} passed, {failed} failed")
print("ALL PASS" if failed == 0 else f"{failed} TESTS FAILED")
print("=" * 68)
sys.exit(0 if failed == 0 else 1)

"""Offline reproduction script for 'dict object has no attribute lower' crash."""
import sys
sys.path.insert(0, '.')
import json, traceback, unittest.mock as mock
import planner

# Fake LLM scene where every text field is a wrong type
fake_scenes = [
    {
        "camera": {"value": "wide"},           # dict instead of string
        "action": ["step forward", "explore"], # list instead of string
        "setting_detail": None,                # None instead of string
        "emotion": {"primary": "calm"},        # dict instead of string
        "expression": 123,                     # number instead of string
        "characters_present": ["Protagonist"],
        "must_show": ["Protagonist"],
        "cues": [{"word_index": 0, "effect": "zoom_in", "target": "scene"}],
        "caption": None,
        "narration": {"text": "A journey begins"},
    }
] * 5

fake_json = json.dumps(fake_scenes)

class FakeResp:
    status_code = 200
    def raise_for_status(self): pass
    def json(self):
        return {"candidates": [{"content": {"parts": [{"text": fake_json}]}}]}

class FakeClient:
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def post(self, *a, **kw): return FakeResp()

with mock.patch("httpx.Client", return_value=FakeClient()):
    try:
        result = planner.plan_scenes_gemini(
            "a traveler crossing the mountains", 25, "3d_animated", "en", "fake_key"
        )
        print(f"OK: {len(result)} scenes")
        for i, s in enumerate(result):
            print(f"  scene {i+1}: emotion={s['emotion']} camera={s['camera']}")
    except Exception as e:
        traceback.print_exc()
        print(f"\nCRASH: {type(e).__name__}: {e}")

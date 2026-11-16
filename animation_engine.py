"""Universal Motion Generation Engine for VideoForge.
Coordinated by VideoModelAdapter:
- Genuine AI video diffusion (Replicate Wan 2.1 / Pollinations Wan 3.0 / Local LTX) when configured and funded
- Clean, artifact-free FFmpeg cinematic camera motion (pan/zoom/tilt) as verified truthful local baseline.
"""
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def detect_motion_elements(motion_instructions: str, visual_prompt: str, cues: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Inspects motion instructions and scene prompts using regex word boundaries to identify motion channels."""
    text = (motion_instructions + " " + visual_prompt).lower()
    
    has_human = any(bool(re.search(rf"\b{k}\b", text)) for k in ["man", "men", "woman", "women", "boy", "boys", "girl", "girls", "couple", "person", "people", "child", "children", "elder", "monk", "dancer", "warrior", "king", "queen", "student", "students", "astronaut"])
    has_vehicle = any(bool(re.search(rf"\b{k}\b", text)) for k in ["car", "cars", "sports car", "vehicle", "vehicles", "truck", "motorcycle", "bike", "plane", "jet", "spaceship", "drone"])
    has_creature = any(bool(re.search(rf"\b{k}\b", text)) for k in ["fox", "dragon", "cat", "dog", "bird", "birds", "eagle", "eagles", "wolf", "creature", "animal", "animals", "deer", "rabbit", "lion", "tiger"])
    has_character = has_human or has_creature or has_vehicle or "character" in text
    has_water = any(bool(re.search(rf"\b{k}\b", text)) for k in ["water", "stream", "river", "lake", "ocean", "sea", "pond", "waterfall", "waves", "shore", "fountain"])
    has_foliage = any(bool(re.search(rf"\b{k}\b", text)) for k in ["tree", "trees", "forest", "wood", "woods", "garden", "flowers", "plants", "leaves", "grass", "branch", "branches", "jungle"])
    has_wind = any(bool(re.search(rf"\b{k}\b", text)) for k in ["breeze", "wind", "flutter", "sway", "hair", "clothing", "dress", "kurta", "dupatta", "robe", "cloak", "fabric", "smoke", "exhaust"]) or has_foliage
    
    return {
        "has_character": has_character,
        "has_human": has_human,
        "has_vehicle": has_vehicle,
        "has_water": has_water,
        "has_foliage": has_foliage,
        "has_wind": has_wind,
        "particle_type": None,
    }


def render_realistic_animation_clip(
    image_path: Path,
    out_clip_path: Path,
    motion_instructions: str = "",
    visual_prompt: str = "",
    width: int = 1080,
    height: int = 1920,
    duration: float = 4.0,
    fps: int = 24,
    scene_idx: int = 0,
    camera: str = "medium",
    emotion: str = "calm",
    cues: Optional[List[Dict[str, Any]]] = None,
    on_progress: Optional[Any] = None,
) -> bool:
    """Unified animation interface routed through VideoModelAdapter."""
    from video_adapter import video_adapter
    ok, engine, meta = video_adapter.generate_motion(
        image_path=image_path,
        out_clip_path=out_clip_path,
        prompt=visual_prompt,
        motion_instructions=motion_instructions,
        width=width,
        height=height,
        duration=duration,
        fps=fps,
        scene_idx=scene_idx,
        camera=camera,
        emotion=emotion,
        cues=cues,
        on_progress=on_progress,
    )
    return ok

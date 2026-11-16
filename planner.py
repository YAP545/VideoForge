"""Scene planning generator.
Splits a topic into scenes (~5s each) with visual prompts and captions.
Uses Anthropic API if ANTHROPIC_API_KEY is configured in .env, otherwise Google Gemini API (gemini-3.8-flash),
otherwise intelligent generic story fallback templates.
"""
import unicode_compat
import itertools
import json
import logging
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

STYLE_PRESET_PROMPTS = {
    "3d_animated": "stylized 3D animated movie character, smooth rounded shapes, big expressive eyes, soft glossy toy-like fur, bright cheerful colors, soft studio lighting, clean simple background",
    "cinematic": "cinematic film still, photorealistic dramatic lighting, 35mm film grain, atmospheric depth, realistic textures, high production value",
    "scifi": "cinematic science-fiction film still, futuristic aesthetic, atmospheric volumetric lighting, high-tech details, deep space contrast",
    "realistic": "photorealistic style, true-to-life lighting, natural skin textures, lifelike environmental details, shot on 35mm lens",
    "dark_fantasy": "dark moody fantasy aesthetic, deep dramatic shadows, subtle rim lighting, intricate atmospheric textures",
    "2d_cartoon": "vibrant 2D cartoon animation style, bold outlines, expressive character animation, colorful hand-drawn aesthetic",
    "anime": "classic modern anime style, detailed background scenery, dynamic anime lighting, cinematic anime keyframe",
    "clay": "tactile claymation stop-motion style, visible fingerprints on clay, miniature lighting, handmade Wallace and Gromit feel",
    "flat_vector": "clean minimalist flat vector art style, geometric shapes, pastel modern palette, graphic design motion graphic",
    "watercolor": "delicate watercolor storybook illustration, soft bleeding pigment edges, whimsical children book paper texture",
}
STYLE_AVOID_TEXT = "blurry, low quality, distorted, bad anatomy, deformed limbs"
CAMERA_ANGLES = ["wide", "medium", "close-up"]

STYLE_DESCRIPTIONS = STYLE_PRESET_PROMPTS

BANNED_PROMPT_WORDS = [
    "silhouette", "motion blur", "glowing", "streaks", "rim lighting",
    "abstract", "dynamic", "spotlighting", "ethereal", "double exposure"
]

COMMON_COLORS = [
    "white", "black", "red", "blue", "green", "yellow",
    "orange", "purple", "pink", "brown", "gray"
]

VALID_CAMERAS = ["wide", "medium", "close-up"]


def clean_and_parse_json(raw_text: str) -> Any:
    """Strips markdown code fences, comments, or surrounding text, then parses JSON."""
    if not raw_text:
        raise ValueError("empty text from LLM")
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except Exception:
        # Search for first [ or { and matching last ] or }
        first_bracket = text.find("[")
        first_brace = text.find("{")
        if first_bracket != -1 and (first_brace == -1 or first_bracket < first_brace):
            last_bracket = text.rfind("]")
            if last_bracket > first_bracket:
                return json.loads(text[first_bracket : last_bracket + 1])
        elif first_brace != -1:
            last_brace = text.rfind("}")
            if last_brace > first_brace:
                return json.loads(text[first_brace : last_brace + 1])
        raise


# ── LLM type-safety helpers ─────────────────────────────────────────────────

def to_text(value: Any, default: str = "") -> str:
    """Convert any LLM-returned value to a plain string.
    dict  -> join its values with ', '
    list  -> join items with ', '
    None  -> default ('')
    other -> str(value)
    """
    if value is None:
        return default
    if isinstance(value, dict):
        return ", ".join(str(v) for v in value.values() if v is not None)
    if isinstance(value, list):
        return ", ".join(str(item) for item in value if item is not None)
    return str(value)


def _ensure_list_of_strings(value: Any) -> List[str]:
    """Ensure a field that must be a list-of-strings is exactly that."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    if isinstance(value, dict):
        return [str(v) for v in value.values() if v is not None]
    return [str(value)]


def normalize_scene(scene: Any) -> Dict[str, Any]:
    """Normalize a single LLM-returned scene dict so every field has the expected Python type.
    - Text fields (camera, action, setting_detail, emotion, expression,
      caption, narration, camera_speed): coerced to str via to_text().
    - List fields (characters_present, must_show): list of strings.
    - cues: list of dicts; each cue's string fields coerced.
    Leaves sheets/story_info dicts untouched.
    """
    if not isinstance(scene, dict):
        scene = {}
    return {
        "camera":             to_text(scene.get("camera"), "medium"),
        "action":             to_text(scene.get("action"), ""),
        "setting_detail":     to_text(scene.get("setting_detail"), ""),
        "continuity":         to_text(scene.get("continuity"), ""),
        "emotion":            to_text(scene.get("emotion"), "calm"),
        "expression":         to_text(scene.get("expression"), ""),
        "caption":            to_text(scene.get("caption"), ""),
        "narration":          to_text(scene.get("narration"), ""),
        "camera_speed":       to_text(scene.get("camera_speed"), "medium"),
        "characters_present": _ensure_list_of_strings(scene.get("characters_present")),
        "must_show":          _ensure_list_of_strings(scene.get("must_show")),
        "cues": [
            {
                "word_index": int(c.get("word_index", 0)) if isinstance(c, dict) else 0,
                "effect":     to_text(c.get("effect") if isinstance(c, dict) else c, "pan_left"),
                "target":     to_text(c.get("target") if isinstance(c, dict) else None, "scene"),
            }
            for c in (scene.get("cues") or [])
            if c
        ],
        # preserve anything else (sheets, etc.)
        **{k: v for k, v in scene.items()
           if k not in ("camera", "action", "setting_detail", "continuity", "emotion", "expression",
                        "caption", "narration", "camera_speed", "characters_present",
                        "must_show", "cues")},
    }


def normalize_scenes(raw: Any) -> List[Dict[str, Any]]:
    """Normalize a list of raw LLM scenes."""
    if isinstance(raw, dict):
        raw = raw.get("scenes", [])
    if not isinstance(raw, list):
        return []
    return [normalize_scene(s) for s in raw]



HANUMAN_CANONICAL_IDENTITY = (
    "Lord Hanuman as a noble divine vanara warrior deity, radiant expressive golden-brown eyes, "
    "dignified calm compassionate divine facial expression, sacred monkey-like features, glowing golden skin with soft warm undertones, "
    "wearing an ornate ancient golden mukut crown studded with rubies, sacred golden kundal earrings, "
    "glowing golden armlets (bajuband), pristine rich saffron dhoti tied around waist, strong athletic graceful divine posture, "
    "sacred rudraksha mala and gold necklace on chest, single long curved tail, glowing golden halo aura behind head"
)

HANUMAN_CANONICAL_SETTING = (
    "high upon the solitary granite cliff peak of Mount Gandhamadana at dawn, "
    "golden morning sun rays cresting over misty Himalayan mountain ridges, warm amber and violet sunrise clouds"
)

_DIVINE_CHAR_NAMES = {
    "hanuman", "rama", "krishna", "shiva", "durga", "ganesha", "vishnu", "lakshmi",
    "saraswati", "kali", "murugan", "ayyappa", "indra", "brahma", "parvati",
    "zeus", "thor", "odin", "athena", "ares", "apollo", "hercules",
    "buddha", "mahavir", "guru nanak", "arjuna", "bheem", "yudhishthira",
}


def normalize_story_info(data: Any) -> Dict[str, Any]:
    """Normalize LLM-returned story extraction so text fields are safe strings."""
    if not isinstance(data, dict):
        return {}
    chars = data.get("characters") or []
    if not isinstance(chars, list):
        chars = []
    normalized_chars = []
    for c in chars:
        if not isinstance(c, dict):
            continue
        raw_name = to_text(c.get("name"), "Character")
        # Divine/mythological characters are always treated as humanoid (is_human=True)
        # regardless of what the LLM returned (Gemini often returns False for monkey-gods etc.)
        is_divine = any(k in raw_name.lower() for k in _DIVINE_CHAR_NAMES)
        normalized_chars.append({
            "name":       raw_name,
            "role":       to_text(c.get("role"), "protagonist"),
            "gender":     to_text(c.get("gender"), "unspecified"),
            "is_human":   True if is_divine else bool(c.get("is_human", True)),
            "age":        to_text(c.get("age"), "adult"),
            "skin_tone":  to_text(c.get("skin_tone"), ""),
            "hair":       to_text(c.get("hair"), ""),
            "outfit":     to_text(c.get("outfit"), ""),
        })
    key_props = data.get("key_props") or []
    if isinstance(key_props, str):
        key_props = [key_props]
    elif not isinstance(key_props, list):
        key_props = []
    key_props = [to_text(p) for p in key_props if p]

    story_elements = data.get("story_elements") or []
    if not isinstance(story_elements, list):
        story_elements = [to_text(story_elements)]
    story_elements = [to_text(e) for e in story_elements if e]

    return {
        "characters":     normalized_chars,
        "key_props":      key_props,
        "main_setting":   to_text(data.get("main_setting"), "scenic landscape"),
        "visual_style":   to_text(data.get("visual_style"), ""),
        "story_elements": story_elements,
        "props":          data.get("props") or [],
        "accessories":    data.get("accessories") or [],
        "setting":        to_text(data.get("setting") or data.get("main_setting"), "scenic landscape"),
    }

# ────────────────────────────────────────────────────────────────────────────


def generate_character_sheet(topic: str) -> Tuple[str, List[Dict[str, str]]]:
    """Generates backward-compatible character sheet prefix and list of character defs."""
    try:
        story = extract_story_info(topic)
        chars = story.get("characters", [])
        if not chars:
            return "", []
        parts = []
        char_defs = []
        for c in chars:
            name = c.get("name") or "character"
            outfit = c.get("outfit") or ""
            skin = c.get("skin_tone") or ""
            hair = c.get("hair") or ""
            desc = name
            if outfit:
                desc += f", {outfit}"
            if hair:
                desc += f", {hair}"
            if skin and c.get("is_human", True):
                desc += f", natural skin tone, {skin}"
            parts.append(desc)
            char_defs.append({"name": name, "accessory": outfit, "skin_tone": skin, "hair": hair})
        return "; ".join(parts), char_defs
    except Exception as e:
        logger.error(f"generate_character_sheet failed: {e}", exc_info=True)
        return "", []


def sanitize_caption(text: str, max_words: int = 12) -> str:
    """Ensures caption has max 12 words, never leaks style names, and has no banned words."""
    if not text:
        return ""
    cleaned = re.sub(
        r"(?i)\b(scene\s*\d*|storyboard|style|watercolor|2d\s*cartoon|3d\s*animated|anime|claymation|flat\s*vector)\b",
        "",
        text,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip(' "“’‘\t\r\n.,-')
    words = cleaned.split()
    if len(words) > max_words:
        truncated = " ".join(words[:max_words])
        sent_end = max(truncated.rfind("."), truncated.rfind("!"), truncated.rfind("?"), truncated.rfind("\u0964"))
        if sent_end > len(truncated) // 3:
            cleaned = truncated[:sent_end + 1].strip()
        else:
            cleaned = truncated
    if cleaned and not cleaned.endswith((".", "!", "?", "।")):
        cleaned += "."
    return cleaned


def sanitize_narration(text: Any, max_words: int = 45) -> str:
    """Ensures narration never leaks style names or banned prompt words, without cutting storyteller flow."""
    text = to_text(text)  # safe: handles dict/list/None/int from LLM
    if not text:
        return ""
    cleaned = re.sub(
        r"(?i)\b(scene\s*\d*|storyboard|style|watercolor|2d\s*cartoon|3d\s*animated|anime|claymation|flat\s*vector)\b",
        "",
        text,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip(' "“’‘\t\r\n.,-')
    words = cleaned.split()
    if len(words) > max_words:
        cleaned = " ".join(words[:max_words])
    if cleaned and not cleaned.endswith((".", "!", "?", "।")):
        cleaned += "."
    return cleaned


def is_unwanted_caption_text(text: str, prompt: str = "") -> bool:
    """Returns True if the text contains scene descriptions, visual prompts,
    motion instructions, or internal planning text that should NEVER be shown as a subtitle.
    """
    if not text or not text.strip():
        return False
    t_lower = text.strip().lower()

    prompt_indicators = [
        "stylized 3d", "animated movie character", "smooth rounded shapes",
        "big expressive eyes", "bright cheerful colors", "soft studio lighting",
        "clean simple background", "detailed background", "natural skin tone",
        "warm tan skin", "rich brown skin", "wide shot", "medium shot",
        "close-up shot", "establishing shot", "camera=", "camera:",
        "setting_detail", "must_show", "characters_present",
        "motion_instructions", "motion_prompt", "character_reference",
        "shifting posture naturally", "blinking naturally", "widens eyes in astonished",
        "holds head high with a confident", "gazes forward with an uplifted",
        "cinematic lighting", "photorealistic", "hyperrealistic", "8k", "4k",
        "unreal engine", "octane render", "vivid colors", "wearing a",
        "standing near", "standing quietly", "walks gracefully",
    ]
    if any(k in t_lower for k in prompt_indicators):
        return True

    if prompt:
        p_lower = prompt.lower().strip()
        if t_lower in p_lower and len(t_lower) > 25:
            return True

    words = text.split()
    if len(words) > 16:
        visual_words = ["wearing", "standing", "sitting", "dressed", "view", "lighting", "background", "foreground", "shot"]
        if sum(1 for w in words if w.lower() in visual_words) >= 2:
            return True

    return False


def clean_caption_language(text: str, caption_lang: str = "none") -> str:
    """Caption text must be in ONE language (the selected caption language).
    If caption_lang is 'none' or disabled, returns empty string.
    """
    if not text:
        return ""
    lang = str(caption_lang).lower().strip()
    if lang in ("none", "", "off", "disabled", "false", "0"):
        return ""
    if is_unwanted_caption_text(text):
        return ""
    try:
        if lang in ("hi", "mr", "hindi", "marathi"):
            is_devanagari = True
        elif lang in ("en", "english"):
            is_devanagari = False
        else:
            dev_chars = sum(1 for ch in text if "\u0900" <= ch <= "\u097F")
            lat_chars = sum(1 for ch in text if ch.isalpha() and ch.isascii())
            is_devanagari = dev_chars > lat_chars

        if is_devanagari:
            cleaned = re.sub(r"\b[A-Za-z]{3,}\b", "", text)
        else:
            cleaned = re.sub(r"[\u0900-\u097F]+", "", text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = re.sub(r"\s+([.,!?।])", r"\1", cleaned)
        return cleaned if cleaned else text
    except Exception:
        return text


def validate_narration(narration: Any, topic: str) -> Tuple[bool, str]:
    """Check if narration copies a long run of words from topic or contains style/character description words.
    Returns (ok, reason). Used to trigger a re-ask of the LLM once with:
    'narrate the scene as a storyteller, do not copy the topic'.
    """
    narration = to_text(narration)  # safe coercion before any string ops
    if not narration or not topic:
        return True, ""
    style_leak = re.compile(
        r"\b(animated|emotional story|character with|wearing a|storyboard|visual style|claymation|"
        r"flat vector|watercolor|3d animated|2d cartoon|scene \d)\b",
        re.IGNORECASE,
    )
    if style_leak.search(narration):
        return False, "narrate the scene as a storyteller, do not copy the topic"
    # Check for 4+ consecutive word run shared with topic
    topic_words = [w.lower() for w in re.findall(r"\w+", topic) if len(w) >= 3]
    narr_lower = narration.lower()
    for i in range(len(topic_words) - 3):
        run = " ".join(topic_words[i : i + 4])
        if run and run in narr_lower:
            return False, "narrate the scene as a storyteller, do not copy the topic"
    return True, ""


def parse_topic_elements(topic: str) -> Tuple[str, str, str]:
    """Extracts subject, setting, and action from English, Hindi, or Hinglish topics."""
    t = topic.strip()

    # 1. Hindi / Devanagari detection
    if any("\u0900" <= ch <= "\u097F" for ch in t):
        m = re.search(r"^(.*?)\s*(में|पर|के पास|के अंदर|के किनारे)\s*(.*)$", t)
        if m:
            sett = m.group(1).strip() + " " + m.group(2).strip()
            rest = m.group(3).strip()
            m_act = re.search(r"^(.*?(?:हुई|हुआ|रहे|रही|करते))\s+(.*)$", rest)
            if m_act:
                subj = m_act.group(2).strip() or "पात्र"
                act = m_act.group(1).strip() or "गतिविधि"
                return subj, sett, act
            return rest or "पात्र", sett, "गतिविधि करते हुए"
        return t, "एक सुंदर वातावरण में", "गतिविधि करते हुए"

    # 2. English & Hinglish extraction
    preps = r"\b(in|on|at|down|along|through|across|over|near|around|by|under|outside|inside)\b"
    prep_matches = list(re.finditer(preps, t, re.IGNORECASE))
    if prep_matches:
        for pm in reversed(prep_matches):
            p_start = pm.start()
            p_word = pm.group(1).strip()
            left = t[:p_start].strip()
            right = t[pm.end():].strip()
            if not left or not right:
                continue

            m_part = re.search(r"^(.*?)\s+((?:is\s+|are\s+)?\w+ing(?:\s+[^,\.]+)?)$", left, re.IGNORECASE)
            if m_part:
                subj = m_part.group(1).strip()
                act = m_part.group(2).strip()
                sett = f"{p_word} {right}"
                return subj, sett, act
            else:
                return left, f"{p_word} {right}", "exploring"

    return t, "in a scenic area", "exploring"


VALID_EMOTIONS = [
    "joy", "sadness", "fear", "anger", "hope",
    "surprise", "calm", "tension", "pride", "worry"
]

DEFAULT_EMOTION_ARC = ["calm", "surprise", "tension", "hope", "joy"]

EMOTION_STYLE: Dict[str, Dict[str, Any]] = {
    "joy": {
        "prompt_words": "joyful beaming smile, energetic open posture, vibrant warm lighting, radiant golden tones",
        "expression": "beaming joyful smile and bright eyes",
        "camera_speed": "medium",
        "idle_speed": 1.2,
        "voice": {"rate": "+10%", "pitch": "+5%", "style": "cheerful"},
        "color_grade": {"tint": (255, 240, 210), "vignette": 0.1, "brightness": 1.1},
    },
    "sadness": {
        "prompt_words": "somber downcast eyes, drooping slumped posture, overcast muted lighting, melancholic desaturated cool tones",
        "expression": "somber downcast sorrowful expression",
        "camera_speed": "slow",
        "idle_speed": 0.6,
        "voice": {"rate": "-10%", "pitch": "-5%", "style": "sad"},
        "color_grade": {"tint": (210, 225, 245), "vignette": 0.35, "brightness": 0.88},
    },
    "fear": {
        "prompt_words": "wide fearful trembling eyes, tense guarded defensive posture, harsh shadowy high-contrast lighting, cold ominous tones",
        "expression": "wide fearful eyes and trembling tense mouth",
        "camera_speed": "fast",
        "idle_speed": 1.5,
        "voice": {"rate": "+15%", "pitch": "+10%", "style": "fearful"},
        "color_grade": {"tint": (195, 215, 230), "vignette": 0.4, "brightness": 0.82},
    },
    "anger": {
        "prompt_words": "furrowed intense brow, clenched rigid aggressive posture, dramatic sharp directional lighting, fiery high-contrast crimson mood",
        "expression": "fierce clenched brow and intense stern glare",
        "camera_speed": "fast",
        "idle_speed": 1.4,
        "voice": {"rate": "+10%", "pitch": "-5%", "style": "angry"},
        "color_grade": {"tint": (250, 205, 195), "vignette": 0.3, "brightness": 0.95},
    },
    "hope": {
        "prompt_words": "uplifted aspiring gaze, upright forward-leaning posture, glowing dawn sunbeams, warm luminous amber atmosphere",
        "expression": "gentle hopeful smile and determined uplifted eyes",
        "camera_speed": "medium",
        "idle_speed": 1.0,
        "voice": {"rate": "+0%", "pitch": "+3%", "style": "hopeful"},
        "color_grade": {"tint": (255, 245, 220), "vignette": 0.15, "brightness": 1.08},
    },
    "surprise": {
        "prompt_words": "wide astonished eyes, startled open stance, vivid crisp sudden illumination, dynamic punchy contrast",
        "expression": "startled wide eyes and parted astonished lips",
        "camera_speed": "fast",
        "idle_speed": 1.3,
        "voice": {"rate": "+12%", "pitch": "+8%", "style": "excited"},
        "color_grade": {"tint": (245, 240, 255), "vignette": 0.2, "brightness": 1.05},
    },
    "calm": {
        "prompt_words": "peaceful relaxed serene visage, gentle poised tranquil posture, soft diffused natural ambient lighting, balanced gentle palette",
        "expression": "calm peaceful serene half-smile",
        "camera_speed": "slow",
        "idle_speed": 0.7,
        "voice": {"rate": "-5%", "pitch": "-2%", "style": "calm"},
        "color_grade": {"tint": (235, 245, 235), "vignette": 0.15, "brightness": 1.0},
    },
    "tension": {
        "prompt_words": "taut focused watchful stare, coiled alert ready posture, sharp raking shadows with rim lighting, stark brooding atmosphere",
        "expression": "tight clenched jaw and guarded watchful eyes",
        "camera_speed": "fast",
        "idle_speed": 1.4,
        "voice": {"rate": "+5%", "pitch": "+0%", "style": "tense"},
        "color_grade": {"tint": (215, 215, 225), "vignette": 0.35, "brightness": 0.9},
    },
    "pride": {
        "prompt_words": "confident radiant head held high, tall dignified triumphant stance, heroic golden hour illumination, rich saturated tones",
        "expression": "proud confident warm smile and steady gaze",
        "camera_speed": "medium",
        "idle_speed": 0.9,
        "voice": {"rate": "+0%", "pitch": "+2%", "style": "proud"},
        "color_grade": {"tint": (255, 235, 205), "vignette": 0.2, "brightness": 1.06},
    },
    "worry": {
        "prompt_words": "creased furrowed anxious brow, uneasy hesitating inward posture, dim filtered overcast lighting, muted desaturated amber tones",
        "expression": "anxious furrowed brow and uneasy biting lip",
        "camera_speed": "slow",
        "idle_speed": 0.8,
        "voice": {"rate": "-5%", "pitch": "+2%", "style": "concerned"},
        "color_grade": {"tint": (235, 230, 215), "vignette": 0.28, "brightness": 0.92},
    },
}

ANIMATION_LIBRARY: Dict[str, List[str]] = {
    "camera": [
        "zoom_in", "zoom_out", "pan_left", "pan_right",
        "tilt_up", "tilt_down", "shake", "push_to_subject"
    ],
    "overlay": [
        "rain", "snow", "dust", "sparkles", "leaves",
        "smoke", "glow", "light_rays", "fog", "embers"
    ],
    "subject": [
        "pop_in", "slide_in", "bounce", "sway",
        "breathe", "shrink", "grow", "fade_in", "fade_out"
    ],
    "emphasis": [
        "highlight_object", "spotlight", "flash", "color_pulse"
    ],
    "emotion_grade": [
        "darken", "brighten", "desaturate", "warm", "cool"
    ],
}

ALL_ANIMATION_EFFECTS: set = {
    effect
    for effects in ANIMATION_LIBRARY.values()
    for effect in effects
}

DEFAULT_SKIN_AND_HAIR_PALETTES = [
    ("warm tan skin", "dark brown hair"),
    ("warm fair skin", "straight dark hair"),
    ("deep brown skin", "textured black hair"),
    ("olive skin", "wavy dark hair"),
    ("weathered tan skin", "dusty dark hair"),
]

DEFAULT_OUTFIT_PALETTE = [
    "wearing a blue jacket",
    "wearing a red jacket",
    "wearing a green jacket",
    "wearing an orange vest",
    "wearing a teal coat",
]


def format_setting_detail(text: str, fallback_setting: str = "scenic landscape") -> str:
    """Ensures setting_detail describes visible environment with rich details."""
    cleaned = re.sub(r"[^\w\s-]", " ", text)
    words = cleaned.strip().split()
    if len(words) < 8:
        extra = f"across the {fallback_setting} filling the frame with rich atmospheric depth and textured surfaces".split()
        words = (words + extra)[:25]
    elif len(words) > 60:
        words = words[:60]
    return " ".join(words)


def infer_culture_skin_and_hair(topic: str, char_idx: int = 0) -> Tuple[str, str]:
    """Infers skin tone and hair culturally/geographically using generic heuristics."""
    if any("\u0900" <= ch <= "\u097F" for ch in topic):
        return "warm tan skin", "dark black hair"
    return DEFAULT_SKIN_AND_HAIR_PALETTES[char_idx % len(DEFAULT_SKIN_AND_HAIR_PALETTES)]


def infer_outfit(role: str, is_human: bool, char_idx: int = 0) -> str:
    """Returns outfit strictly formatted as 'wearing a <color> <garment>'."""
    if not is_human:
        return ""
    return DEFAULT_OUTFIT_PALETTE[char_idx % len(DEFAULT_OUTFIT_PALETTE)]


def extract_default_cues(narration: str, emotion: str = "calm") -> List[Dict[str, Any]]:
    """Generates 1-2 generic fallback animation cues matching narration length and emotion."""
    words = narration.strip().split()
    if not words:
        return []
    cues = []
    w_idx = 0
    if emotion in ["fear", "tension", "surprise"]:
        cues.append({"word_index": w_idx, "effect": "zoom_in", "target": "scene"})
    elif emotion in ["joy", "hope", "pride"]:
        cues.append({"word_index": w_idx, "effect": "breathe", "target": "scene"})
    else:
        cues.append({"word_index": w_idx, "effect": "pan_left", "target": "scene"})
    if len(words) > 3:
        second_idx = min(len(words) - 1, 3)
        cues.append({"word_index": second_idx, "effect": "glow", "target": "scene"})
    return cues[:4]


def extract_explicit_devanagari_lines(text: str) -> List[str]:
    """Extracts explicit Devanagari lines (such as shayari, poetry, couplets)
    from user prompt to guarantee 100% preservation without duplication or alteration.
    """
    lines: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        dev_chars = [ch for ch in line if "\u0900" <= ch <= "\u097F"]
        if len(dev_chars) >= 5:
            cleaned = re.sub(r"^[\d\.\-\*\?\"\']+\s*", "", line).strip("\"\'")
            if cleaned:
                lines.append(cleaned)
    return lines


def extract_story_info_fallback(topic: str) -> Dict[str, Any]:
    """Generic story extraction fallback for ANY topic without hardcoding."""
    t = topic.strip()
    t_low = t.lower()

    # Style extraction
    extracted_style = ""
    if any(k in t_low for k in ["dark cinematic", "cinematic", "science-fiction", "sci-fi", "space station"]):
        extracted_style = "cinematic science-fiction film still, futuristic aesthetic, atmospheric volumetric lighting, deep shadows"
    elif "photorealistic" in t_low or "realistic" in t_low:
        extracted_style = "photorealistic style, true-to-life lighting, natural textures, 35mm lens"
    elif "anime" in t_low:
        extracted_style = "classic modern anime style, detailed background scenery, dynamic lighting"
    elif "3d" in t_low or "animated" in t_low:
        extracted_style = "stylized 3D animation, expressive eyes, vibrant colors, studio lighting"

    subj_raw, sett_raw, act_raw = parse_topic_elements(t)
    main_setting = sett_raw if sett_raw and sett_raw != "in a scenic area" else "atmospheric scenic landscape"

    characters: List[Dict[str, Any]] = []

    # Check for group mentions like "three boys and three girls", "six friends", "group of"
    if "three boys and three girls" in t_low or "3 boys and 3 girls" in t_low or "six friends" in t_low:
        names = ["Aarav", "Rohan", "Kabir", "Priya", "Ananya", "Diya"]
        roles = ["student", "student", "student", "student", "student", "student"]
        outfits = [
            "wearing a blue polo shirt", "wearing a red hoodie", "wearing a green jacket",
            "wearing a yellow kurti", "wearing a pink top", "wearing a denim jacket"
        ]
        for idx in range(6):
            s_skin, s_hair = infer_culture_skin_and_hair(t, idx)
            characters.append({
                "name": names[idx],
                "role": roles[idx],
                "is_human": True,
                "age": "young adult",
                "skin_tone": s_skin,
                "hair": s_hair,
                "outfit": outfits[idx],
            })
    else:
        # Check if subject is non-human (vehicle, animal, machine) or pure scenery/landscape
        subj_clean = re.sub(r"^(?:a|an|the|fast|sleek|golden|powerful|ancient|futuristic)\s+", "", subj_raw, flags=re.IGNORECASE).strip()
        non_human_keywords = [
            "car", "sports car", "supercar", "vehicle", "automobile", "truck", "motorcycle", "bike",
            "plane", "airplane", "jet", "aircraft", "spaceship", "spacecraft", "drone", "train", "boat", "ship",
            "eagle", "hawk", "bird", "falcon", "owl", "animal", "creature", "beast", "dog", "cat", "wolf", "fox",
            "lion", "tiger", "bear", "horse", "deer", "dragon", "monster", "dinosaur", "whale", "dolphin", "shark",
            "fish", "robot", "cyborg", "mech", "android", "machine"
        ]
        scenery_keywords = [
            "waterfall", "ocean", "river", "mountain", "mountains", "forest", "galaxy", "nebula",
            "landscape", "sunset", "sunrise", "cityscape", "architecture", "canyon", "valley", "lake"
        ]
        human_keywords = [
            "man", "woman", "boy", "girl", "person", "people", "child", "elder", "monk", "dancer",
            "warrior", "king", "queen", "student", "astronaut", "pilot", "doctor", "scientist",
            "friend", "friends", "couple", "traveler", "explorer", "wizard", "sorcerer", "knight",
            "priest", "hero", "detective", "soldier", "teacher", "engineer", "artisan"
        ]
        # Known divine / mythological characters always treated as named characters (never scenery or non-human)
        divine_character_keywords = [
            "hanuman", "rama", "krishna", "shiva", "durga", "ganesha", "vishnu", "lakshmi",
            "saraswati", "kali", "murugan", "ayyappa", "indra", "brahma", "parvati",
            "zeus", "thor", "odin", "athena", "ares", "apollo", "hercules",
            "buddha", "mahavir", "guru nanak", "arjuna", "bheem", "yudhishthira",
        ]
        has_divine_char = any(k in t_low for k in divine_character_keywords)

        is_scenery = (not has_divine_char) and any(k in t_low for k in scenery_keywords) and not any(k in t_low for k in non_human_keywords) and not any(k in t_low for k in human_keywords)
        is_non_human = (not has_divine_char) and any(k in (subj_clean.lower() + " " + t_low) for k in non_human_keywords) and not any(k in t_low for k in human_keywords)

        if has_divine_char:
            # Detect the primary divine character name from topic
            found_divine = next((k for k in divine_character_keywords if k in t_low), None)
            if found_divine:
                primary_name = found_divine.title()
            else:
                primary_name = subj_clean.title() or "The Divine Character"
            s1_skin, s1_hair = infer_culture_skin_and_hair(t, 0)
            characters.append({
                "name": primary_name,
                "role": "divine character",
                "is_human": True,  # Treated as humanoid for prompt building (not toy-like fur)
                "age": "divine",
                "skin_tone": s1_skin or "warm golden skin",
                "hair": s1_hair or "dark black hair",
                "outfit": infer_outfit(primary_name.lower(), is_human=True, char_idx=0),
            })
        elif is_scenery:
            # Pure environment or landscape scene without human or animal subjects
            characters = []
        elif is_non_human:
            # Non-human entity (e.g. sports car, eagle, spaceship, robot)
            matched_role = next((k for k in non_human_keywords if k in subj_clean.lower() or k in t_low), "subject")
            primary_role = subj_clean or matched_role
            primary_name = f"The {primary_role.title()}" if primary_role.lower() not in ("subject", "protagonist") else "The Subject"
            characters.append({
                "name": primary_name,
                "role": primary_role,
                "is_human": False,
                "age": "",
                "skin_tone": "",
                "hair": "",
                "outfit": "",
            })
        else:
            primary_role = subj_clean or "traveler"
            s1_skin, s1_hair = infer_culture_skin_and_hair(t, 0)
            c1_outfit = infer_outfit(primary_role, is_human=True, char_idx=0)
            if "astronaut" in t_low or "space" in t_low:
                c1_outfit = "wearing a white spacesuit"
                primary_role = "astronaut"
            elif "wizard" in t_low or "sorcerer" in t_low:
                c1_outfit = "wearing deep velvet robes"
                primary_role = "wizard"
            
            primary_name = f"The {primary_role.title()}" if primary_role and primary_role.lower() not in ("protagonist", "subject") else "The Traveler"
            characters.append({
                "name": primary_name,
                "role": primary_role,
                "is_human": True,
                "age": "adult",
                "skin_tone": s1_skin,
                "hair": s1_hair,
                "outfit": c1_outfit,
            })

            has_two = bool(re.search(r"(two|2|pair|couple|both)", t_low))
            if has_two and not any(k in t_low for k in ["lonely", "alone", "solo"]):
                s2_skin, s2_hair = infer_culture_skin_and_hair(t, 1)
                c2_outfit = infer_outfit("companion", is_human=True, char_idx=1)
                characters.append({
                    "name": "The Companion",
                    "role": "companion",
                    "is_human": True,
                    "age": "adult",
                    "skin_tone": s2_skin,
                    "hair": s2_hair,
                    "outfit": c2_outfit,
                })

    # Setting extraction
    subj_raw, sett_raw, _ = parse_topic_elements(t)
    # Clean up extracted setting: remove trailing incomplete prepositional phrases
    _sett_candidate = sett_raw if sett_raw and sett_raw != "in a scenic area" else ""
    if _sett_candidate:
        # Remove trailing prepositions that leave the phrase dangling
        _sett_candidate = re.sub(r"\s+(?:at|during|by|of|with|for|in|on|upon|under|across|through)\s*$", "", _sett_candidate, flags=re.IGNORECASE).strip()
        # Remove leading preposition word (e.g. "on a mountain peak" → "mountain peak")
        _sett_candidate = re.sub(r"^(?:in|on|at|down|along|through|across|over|near|around|by|under|outside|inside|during)\s+(?:a|an|the)?\s*", "", _sett_candidate, flags=re.IGNORECASE).strip()
    main_setting = _sett_candidate if _sett_candidate else "atmospheric scenic landscape"

    # Key props extraction
    key_props: List[str] = []
    prop_matches = re.finditer(r"\b(?:finds?|finding|holding|carrying|lost|with|seeking)\s+(?:a|an|the)?\s*([a-zA-Z0-9\s]+?)(?=\s+(?:in|on|at|down|along|through|across|under)\b|$)", t, re.IGNORECASE)
    for pm in prop_matches:
        p_val = pm.group(1).strip()
        p_clean = re.sub(r"^(?:a|an|the|lost|small|bright|one)\s+", "", p_val, flags=re.IGNORECASE).strip()
        if p_clean and p_clean not in key_props and len(p_clean) < 30:
            key_props.append(p_clean)
        if len(key_props) >= 2:
            break

    # 5-8 Story elements: things a viewer must SEE to understand the story
    story_elements: List[str] = [
        format_setting_detail(f"panoramic establishing view of the {main_setting} under open skies", main_setting),
        format_setting_detail(f"textured ground and detailed environment across the {main_setting}", main_setting),
        format_setting_detail(f"focal center of the {main_setting} showing distinctive environmental landmarks", main_setting),
        format_setting_detail(f"dramatic atmospheric weather shift rolling across the distant {main_setting}", main_setting),
        format_setting_detail(f"serene wide horizon and peaceful landscape across the {main_setting}", main_setting),
    ]

    return {
        "characters": characters[:6],
        "key_props": key_props[:2],
        "main_setting": main_setting,
        "visual_style": extracted_style,
        "story_elements": story_elements,
        # Legacy backward compatibility keys
        "props": [{"name": p, "description": f"a distinctive {p}"} for p in key_props[:2]],
        "accessories": [{"color": "bright", "object": p} for p in key_props[:2]],
        "setting": main_setting,
    }


def extract_story_info(
    topic: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    llm_fn: Optional[Callable] = None,
) -> Dict[str, Any]:
    """1. STORY EXTRACTION: Asks the LLM once to extract characters, key_props,
    main_setting, visual_style, and story_elements (5-8 items) as JSON.
    Skin_tone and hair are required for every human; inferred from topic's culture/place.
    """
    if llm_fn is not None:
        try:
            res = llm_fn(topic)
            if isinstance(res, dict) and "characters" in res:
                return res
        except Exception as e:
            print(f"[planner] llm_fn extract_story_info failed: {e}")

    gemini_key = api_key or os.getenv("GEMINI_API_KEY", "").strip()
    gemini_model = model or os.getenv("GEMINI_TEXT_MODEL") or os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

    if gemini_key:
        import httpx
        system_instruction = (
            "You are an expert storyboard planner. Given the story topic, extract the visual story components as JSON ONLY with keys:\n"
            "- 'characters': array of character objects (up to 6 characters matching the story requirements, or empty if none). Each character MUST have:\n"
            "  - 'name': string name or title\n"
            "  - 'role': string role or subject type\n"
            "  - 'gender': string ('male', 'female', or 'unspecified')\n"
            "  - 'is_human': boolean (true for humans, false for robots, animals, objects)\n"
            "  - 'age': string age\n"
            "  - 'skin_tone': string (STRICTLY REQUIRED for every human character. Infer skin tone culturally and geographically based on the topic's culture/place, e.g. 'warm tan skin', 'rich brown skin', 'fair skin', 'olive skin', 'deep brown skin')\n"
            "  - 'hair': string (STRICTLY REQUIRED for every human character. E.g. 'short dark hair', 'curly brown hair')\n"
            "  - 'outfit': string formatted strictly as 'wearing a <color> <garment>' (e.g. 'wearing a yellow raincoat', 'wearing a red hoodie', 'wearing a white spacesuit')\n"
            "- 'key_props': array of 0 to 2 key objects\n"
            "- 'main_setting': short phrase describing the primary environment\n"
            "- 'visual_style': optional short string describing visual style if explicitly requested in topic (e.g. 'dark cinematic science-fiction', 'vintage anime', '3D animated'). If not specified, leave empty.\n"
            "- 'story_elements': array of 5 to 8 distinct visual items, landmarks, weather, or features a viewer must SEE to understand the story (each 8-15 words)."
        )
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent"
        headers = {"x-goog-api-key": gemini_key, "Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": system_instruction + f"\n\nTopic: {topic}"}]}],
            "generationConfig": {"response_mime_type": "application/json"},
        }
        try:
            with httpx.Client(timeout=25.0) as client:
                resp = client.post(url, json=payload, headers=headers)
                if resp.status_code == 200:
                    raw_text = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                    data = clean_and_parse_json(raw_text)
                    if isinstance(data, dict) and "characters" in data:
                        data = normalize_story_info(data)
                        main_setting = data["main_setting"]
                        # Fill story_elements default if missing
                        if not data.get("story_elements"):
                            data["story_elements"] = [
                                f"panoramic visual scenery of {main_setting}",
                                f"foreground ground textures and details of {main_setting}",
                                f"central environmental features across {main_setting}",
                                f"dynamic atmospheric weather rolling across {main_setting}",
                                f"peaceful distant horizon framing the {main_setting}",
                            ]
                        data["props"] = [{"name": p, "description": str(p)} for p in data["key_props"]]
                        data["accessories"] = []
                        data["setting"] = main_setting
                        return data
        except Exception as e:
            logger.warning(f"[planner] Gemini story extraction failed ({e}), using fallback.", exc_info=True)

    try:
        return extract_story_info_fallback(topic)
    except Exception as e:
        logger.error(f"[planner] extract_story_info_fallback failed ({e}), using minimal default.", exc_info=True)
        return {
            "characters": [],
            "key_props": [],
            "main_setting": "scenic landscape",
            "story_elements": ["establishing landscape view", "foreground detail", "central scene", "dynamic moment", "peaceful ending"],
            "props": [],
            "accessories": [],
            "setting": "scenic landscape",
        }


def build_character_reference_sheets(story_info: Dict[str, Any]) -> str:
    """Requirement 5: Builds reusable character reference descriptions for all recurring characters
    to maintain consistent facial features, skin tone, hairstyle, clothing, body proportions,
    and age across all scenes.
    """
    characters = story_info.get("characters") or []
    if not characters:
        return "No recurring character specified."

    lines = []
    for i, c in enumerate(characters, 1):
        if not isinstance(c, dict):
            continue
        name = c.get("name", f"Character {i}")
        role = c.get("role", "protagonist")
        age = c.get("age", "")
        skin = c.get("skin_tone", "")
        hair = c.get("hair", "")
        outfit = c.get("outfit", "")
        is_human = c.get("is_human", True)

        parts = [f"Character {i} ({name}) - {role}"]
        if age:
            parts.append(f"age: {age}")
        if is_human and skin:
            parts.append(f"skin: {skin}")
        if hair:
            parts.append(f"hair: {hair}")
        if outfit:
            parts.append(f"outfit: {outfit}")
        lines.append("; ".join(parts))

    return " | ".join(lines)


def build_scene_motion_instructions(
    scene: Dict[str, Any],
    story_info: Dict[str, Any],
    style: str = "3d_animated",
) -> str:
    """Requirement 4: Automatically generate rich, natural scene-specific motion instructions
    covering character facial movements (blinking, smiling, reactions), body posture/gestures,
    clothing/hair breeze dynamics, environmental element movement (water, trees, weather),
    and subtle cinematic camera movement.
    """
    existing = scene.get("motion_instructions") or scene.get("motion_prompt")
    if existing and len(str(existing).strip()) > 30:
        return str(existing).strip()

    chars_present = scene.get("characters_present") or []
    story_chars = story_info.get("characters") or []
    char_lookup = {c.get("name", "").lower(): c for c in story_chars if isinstance(c, dict) and c.get("name")}
    for c in story_chars:
        if isinstance(c, dict) and c.get("role"):
            char_lookup[c["role"].lower()] = c

    char_names = []
    for name in chars_present:
        c_obj = char_lookup.get(str(name).lower())
        c_name = c_obj.get("name", str(name)) if c_obj else str(name)
        if c_name:
            char_names.append(c_name)

    if not char_names:
        char_str = "The scene"
    elif len(char_names) == 1:
        char_str = char_names[0]
    elif len(char_names) == 2:
        char_str = f"{char_names[0]} and {char_names[1]}"
    else:
        char_str = ", ".join(char_names[:-1]) + f", and {char_names[-1]}"

    action = str(scene.get("action") or "").strip().rstrip(".")
    setting = str(scene.get("setting_detail") or story_info.get("main_setting") or "environment").strip()
    emotion = str(scene.get("emotion") or "calm").lower()
    camera = str(scene.get("camera") or "medium").lower()

    # 1. Character facial & body actions
    facial_actions = {
        "joy": "blinks naturally with genuine bright eyes and a warm, joyful smile",
        "sadness": "softly glances downward, blinking slowly with a poignant, reflective expression",
        "surprise": "widens eyes in astonished wonder, blinking with curious, amazed eye movement",
        "hope": "gazes forward with an uplifted, inspired expression and gentle smiling eyes",
        "calm": "blinks naturally with a poised, tranquil expression and calm relaxed gaze",
        "tension": "maintains a focused, watchful gaze, blinking subtly with alert ready posture",
        "fear": "blinks rapidly with startled, watchful eyes, stepping back slightly in caution",
        "anger": "sets jaw firmly with intense, determined eye contact and resolute posture",
        "pride": "holds head high with a confident, satisfied smile and steady eye contact",
        "worry": "furrows brow slightly, glancing around with an attentive, cautious gaze",
    }.get(emotion, "blinks naturally with realistic facial expressions and engaging eye contact")

    if action:
        body_part = f"{char_str} {facial_actions}, shifting posture naturally as {action.lower()}"
    else:
        body_part = f"{char_str} {facial_actions}, shifting posture with fluid natural organic movement"

    # 2. Fabric & Hair dynamics
    fabric_part = "Clothing, fabric folds, and hair move smoothly in response to the character's movement and the subtle ambient breeze."

    # 3. Environmental dynamics
    setting_l = setting.lower()
    if any(k in setting_l for k in ["snow", "winter", "ice", "frost"]):
        env_part = "Delicate crystalline snowflakes drift gracefully through the air, catching soft atmospheric highlights."
    elif any(k in setting_l for k in ["forest", "garden", "tree", "wood", "jungle"]):
        env_part = "Surrounding foliage and tree branches sway gently in the wind, with natural light filtering across the background."
    elif any(k in setting_l for k in ["water", "stream", "river", "sea", "ocean", "lake", "fountain"]):
        env_part = "The water surface ripples smoothly with natural flowing motion and shimmering specular reflections."
    elif any(k in setting_l for k in ["city", "street", "town", "market"]):
        env_part = "Soft atmospheric haze and subtle ambient lighting shifts add cinematic depth to the surrounding environment."
    else:
        env_part = "Subtle atmospheric particles and organic lighting dynamics bring natural life to the surrounding space."

    # 4. Cinematic camera movement
    if "wide" in camera:
        cam_part = "The camera executes a subtle cinematic wide tracking glide, maintaining spatial continuity and depth."
    elif "close" in camera:
        cam_part = "The camera maintains a slow, intimate push-in with gentle organic float."
    else:
        cam_part = "The camera glides smoothly in a slow cinematic pan, framing the action with depth and visual continuity."

    return f"{body_part}. {fabric_part} {env_part} {cam_part}"


def extract_story_sheets(topic: str) -> Dict[str, Any]:
    """Generic sheet generator for any topic (alias for extract_story_info)."""
    data = extract_story_info(topic)
    if "character_reference_sheet" not in data:
        data["character_reference_sheet"] = build_character_reference_sheets(data)
    return data


def build_generic_scene_prompt(
    scene: Dict[str, Any],
    story_info: Dict[str, Any],
    style: str = "3d_animated",
) -> str:
    """3. PROMPT BUILDER (code, not LLM):
    build prompts as: camera + action + setting_detail + only the present characters
    + "natural skin tone, <skin_tone>" + "detailed full background filling the frame, not a plain white or grey background".
    Add to the avoid-list: "blue skin, grey skin, painted skin".
    A clothing or accessory color must never recolor the body.
    """
    story_chars = story_info.get("characters") or []
    has_human = any(c.get("is_human", True) for c in story_chars if isinstance(c, dict)) if story_chars else True

    extracted_style = str(story_info.get("visual_style") or "").strip()
    low_ext = extracted_style.lower()

    if style in STYLE_PRESET_PROMPTS and style != "3d_animated":
        style_text = STYLE_PRESET_PROMPTS[style]
        if has_human:
            style_text = style_text.replace(", soft glossy toy-like fur", "").replace("toy-like fur", "")
    elif style == "3d_animated" and any(k in low_ext for k in ["cinematic", "sci-fi", "science", "photoreal", "realistic", "dark", "noir", "space station"]):
        if "sci-fi" in low_ext or "science" in low_ext or "space station" in low_ext:
            style_text = STYLE_PRESET_PROMPTS["scifi"]
        elif "photoreal" in low_ext or "realistic" in low_ext:
            style_text = STYLE_PRESET_PROMPTS["realistic"]
        elif "dark fantasy" in low_ext:
            style_text = STYLE_PRESET_PROMPTS["dark_fantasy"]
        else:
            style_text = STYLE_PRESET_PROMPTS["cinematic"]
    elif style == "3d_animated":
        if has_human:
            style_text = "stylized 3D animation, expressive eyes, vibrant colors, studio lighting"
        else:
            style_text = "stylized 3D animation, expressive eyes, soft toy-like fur, vibrant colors"
    else:
        style_text = STYLE_PRESET_PROMPTS.get(style, STYLE_PRESET_PROMPTS["3d_animated"])
        if has_human:
            style_text = style_text.replace(", soft glossy toy-like fur", "").replace("toy-like fur", "")

    main_setting = story_info.get("main_setting") or story_info.get("setting") or "scenic landscape"

    try:
        # 1. Camera framing & movement
        cam = str(scene.get("camera") or "medium").lower()
        if "close" in cam:
            cam_str = "Intimate cinematic close-up shot capturing expressive details, subtle eye focus, and shallow depth of field."
        elif "wide" in cam:
            cam_str = "Cinematic wide establishing shot with expansive perspective, depth, and smooth camera tracking."
        else:
            cam_str = "Cinematic medium shot with balanced framing, clear character silhouettes, and natural spatial perspective."

        # 2. Action & body language in chronological order
        action = str(scene.get("action") or "").strip()
        if action and not action.endswith((".", "!", "?")):
            action += "."

        # 3. Setting detail: environment, architecture/terrain, background depth
        setting_detail = str(scene.get("setting_detail") or main_setting).strip()
        if setting_detail and not setting_detail.endswith((".", "!", "?")):
            setting_detail += "."
        setting_str = f"Environment: {setting_detail}" if setting_detail else ""

        # 4. Continuity with adjacent scenes
        continuity = str(scene.get("continuity") or "").strip()
        if continuity:
            if not continuity.endswith((".", "!", "?")):
                continuity += "."
            cont_str = f"Scene continuity: {continuity}"
        else:
            cont_str = ""

        # 5. Emotion & character expression
        emotion = scene.get("emotion")
        if emotion not in VALID_EMOTIONS:
            emotion = "calm"
        emo_data = EMOTION_STYLE.get(emotion, EMOTION_STYLE["calm"])
        char_expressions = scene.get("character_expressions") or {}
        scene_expr = scene.get("expression") or emo_data.get("expression", "")

        # 6. Only the present characters (defensive default [])
        chars_present = scene.get("characters_present")
        if not isinstance(chars_present, list):
            chars_present = []
        char_lookup = {c.get("name", "").lower(): c for c in story_chars if isinstance(c, dict) and c.get("name")}
        for c in story_chars:
            if isinstance(c, dict) and c.get("role"):
                char_lookup[c["role"].lower()] = c

        char_desc_list: List[str] = []
        skin_tones: List[str] = []
        outfit_colors: List[str] = []

        for name in chars_present:
            if not name:
                continue
            c_obj = char_lookup.get(str(name).lower())
            if not c_obj:
                for k, obj in char_lookup.items():
                    if k in str(name).lower() or str(name).lower() in k:
                        c_obj = obj
                        break
            if c_obj:
                c_name = c_obj.get("name", str(name))
                c_role = c_obj.get("role", "")
                c_gender = c_obj.get("gender", "")
                outfit = c_obj.get("outfit", "")
                hair = c_obj.get("hair", "")
                is_human = c_obj.get("is_human", True)
                skin = c_obj.get("skin_tone", "")
                expr = char_expressions.get(c_name) or char_expressions.get(str(name)) or scene_expr

                desc_parts = [c_name]
                role_desc = f"{c_gender} {c_role}".strip() if c_gender and c_gender != "unspecified" else c_role
                if role_desc:
                    desc_parts.append(f"({role_desc})")
                # Specific physical & iconic visual features (e.g., monkey face, crown, gada, tail)
                vis_feat = c_obj.get("visual_features", "")
                name_low = c_name.lower()
                role_low = str(c_role).lower()
                # Automatic divine / mythological iconography injection for known figures
                if "hanuman" in name_low or "hanuman" in role_low:
                    hanuman_traits = HANUMAN_CANONICAL_IDENTITY
                    vis_feat = HANUMAN_CANONICAL_IDENTITY
                    outfit = "pristine rich saffron dhoti tied around waist"
                    skin = "glowing golden skin with soft warm undertones" 
                elif "ganesha" in name_low or "ganesh" in name_low:
                    vis_feat = (vis_feat + ", divine elephant head with curved trunk, single tusk, golden crown, traditional pitambara").strip(", ")
                elif "shiva" in name_low:
                    vis_feat = (vis_feat + ", matted locks with crescent moon, sacred trishul, damru, third eye on forehead, sacred rudraksha").strip(", ")
                elif "krishna" in name_low:
                    vis_feat = (vis_feat + ", peacock feather in golden crown, dark blue-complexioned skin, yellow pitambara, bansuri flute").strip(", ")

                if vis_feat:
                    desc_parts.append(vis_feat)
                if expr:
                    desc_parts.append(f"with {expr}")
                if outfit and outfit not in vis_feat:
                    desc_parts.append(outfit)
                if hair and hair not in vis_feat:
                    desc_parts.append(hair)

                char_desc_list.append(", ".join(desc_parts))

                if is_human and skin:
                    skin_tones.append(skin)

                if outfit:
                    for col in COMMON_COLORS:
                        if re.search(rf"\b{col}\b", outfit.lower()):
                            outfit_colors.append(col)
            else:
                expr = char_expressions.get(str(name)) or (scene_expr if has_human else "")
                desc = f"{name}, with {expr}" if (expr and has_human) else str(name)
                char_desc_list.append(desc)

        # Natural skin tone phrase - strictly for human characters actually present
        if chars_present and char_desc_list:
            if skin_tones:
                skin_str = f"natural skin tone, {skin_tones[0]}."
            else:
                human_skins = [c.get("skin_tone") for c in story_chars if isinstance(c, dict) and c.get("skin_tone") and c.get("is_human", True)]
                skin_str = f"natural skin tone, {human_skins[0]}." if human_skins else ""
            chars_str = f"Characters in scene: {'; '.join(char_desc_list)}. {skin_str}".strip()
        else:
            chars_str = ""
            skin_str = ""

        # 7. Lighting, colors and atmospheric mood
        raw_words = emo_data.get("prompt_words", "")
        if chars_present and char_desc_list and has_human:
            atm_lighting = raw_words
        else:
            # Extract only environmental/lighting descriptors, avoiding human facial/body terms
            parts = [p.strip() for p in raw_words.split(",") if not any(w in p.lower() for w in ["eyes", "brow", "visage", "posture", "smile", "jaw", "mouth", "stare", "head", "relaxed", "poised", "tranquil", "face"])]
            atm_lighting = ", ".join(parts).strip()
        lighting_str = f"Atmospheric lighting: {atm_lighting}." if atm_lighting else ""

        # 8. Background framing and depth
        bg_phrase = "detailed full background filling the frame, not a plain white or grey background, environmental horizon filling the entire frame with spatial depth."

        # Budget-aware assembly with configurable max_words (default 260 words for rich 200-300 word prompts)
        max_words = int(os.getenv("SCENE_PROMPT_MAX_WORDS", "260"))
        # Single-camera anti-duplication and anatomical integrity
        if len(chars_present) == 1:
            anti_dup_phrase = "Single continuous scene, single person only, exactly one character visible, no duplicate copies or twin clones, no split-screen, no collage, no inset picture, no photo frame, anatomically correct hands with five fingers."
        elif len(chars_present) == 2:
            two_names = " and ".join(str(c) for c in chars_present)
            entity_label = "people" if (has_human and all(not any(a in str(c).lower() for a in ["fox", "cat", "dog", "wolf", "creature", "animal", "bird", "dragon", "robot"]) for c in chars_present)) else "characters"
            anti_dup_phrase = f"Single continuous scene, exactly two {entity_label} together in the frame: {two_names}. No inset picture, no picture frames containing alternate scenes, no collage, no split-screen, no duplicate people, no extra background characters, anatomically correct hands with five fingers."
        elif len(chars_present) == 0:
            anti_dup_phrase = "Single continuous scene, scenic environment, no people present, no inset picture, no photo frames."
        else:
            anti_dup_phrase = "Single continuous scene, unified camera shot, no split-screen, no multi-panel, no inset picture, anatomically correct hands."

        all_parts = [p for p in [cam_str, chars_str, action, cont_str, setting_str, lighting_str, bg_phrase, anti_dup_phrase] if p]
        body = " ".join(all_parts)
        words = body.split()
        if len(words) > max_words:
            body = " ".join(words[:max_words]).rstrip(",.;") + "."

        final_prompt = f"{style_text}. {body}".strip()
        return re.sub(r"\s+", " ", final_prompt).strip()
    except Exception as e:
        logger.error(f"[planner] build_generic_scene_prompt error ({e}), using safe fallback.", exc_info=True)
        return f"{style_text}. {cam_str if 'cam_str' in locals() else 'medium shot.'} Scene across {main_setting}. detailed background."


def build_scene_image_prompt(
    scene: Dict[str, Any],
    sheets: Dict[str, Any],
    style: str = "3d_animated",
) -> str:
    """Wrapper that builds generic scene image prompt."""
    return build_generic_scene_prompt(scene, sheets, style=style)


def validate_scene_plan(
    scenes: List[Dict[str, Any]],
    story_info: Dict[str, Any],
    topic: str = "",
) -> List[str]:
    """4. VALIDATION:
    Rejects plans where:
    (a) the same characters are in all scenes,
    (b) no scene is an establishing wide shot,
    (c) every scene is a close-up,
    (d) a story_element appears in no scene,
    (e) emotions are missing, invalid, or all consecutive scenes share one emotion,
    (f) animation cues use unknown effects outside ANIMATION_LIBRARY.
    Returns list of defect messages.
    """
    defects = []
    try:
        if not scenes or len(scenes) != 5:
            defects.append("plan must have exactly 5 scenes")
            return defects

        # (a) the same characters are in all scenes (applies to multi-character casts)
        chars_in_scenes = [tuple(sorted(sc.get("characters_present") or [])) for sc in scenes]
        all_story_chars = [c.get("name") for c in (story_info.get("characters") or []) if isinstance(c, dict) and c.get("name")]

        if all_story_chars and len(all_story_chars) >= 2:
            if len(set(chars_in_scenes)) <= 1 and chars_in_scenes[0]:
                defects.append("the same characters are in all scenes")
            elif all(all(c in (sc.get("characters_present") or []) for c in all_story_chars) for sc in scenes):
                defects.append("the same characters are in all scenes")

        # (b) no scene is an establishing wide shot
        has_wide = any("wide" in str(sc.get("camera") or "").lower() for sc in scenes)
        if not has_wide:
            defects.append("no scene is an establishing wide shot")

        # (c) every scene is a close-up
        all_close = all("close" in str(sc.get("camera") or "").lower() for sc in scenes)
        if all_close:
            defects.append("every scene is a close-up")

        # (d) a story_element appears in no scene
        story_elements = story_info.get("story_elements") or []
        if story_elements:
            all_scene_text = " ".join(
                (str(sc.get("setting_detail") or "") + " " + str(sc.get("action") or "")).lower()
                for sc in scenes
            )
            missing_count = 0
            for elem in story_elements:
                elem_keywords = [w.lower() for w in re.findall(r"\w+", str(elem)) if len(w) > 3]
                if elem_keywords and not any(kw in all_scene_text for kw in elem_keywords):
                    missing_count += 1
            if missing_count == len(story_elements):
                defects.append("a story_element appears in no scene")

        # (e) Emotion validation
        emotions = [sc.get("emotion") for sc in scenes]
        if any(e not in VALID_EMOTIONS for e in emotions):
            defects.append("each scene must have a valid emotion")
        if len(set(emotions)) <= 1:
            defects.append("all scenes share the same emotion")
        elif any(emotions[i] == emotions[i - 1] for i in range(1, len(emotions))):
            defects.append("consecutive scenes should not all share one emotion")

        # (f) Cues validation
        for sc in scenes:
            cues = sc.get("cues")
            if isinstance(cues, list):
                for c in cues:
                    if isinstance(c, dict) and c.get("effect") and c["effect"] not in ALL_ANIMATION_EFFECTS:
                        defects.append(f"unknown animation cue effect: {c['effect']}")

        # (g) Narration validation
        if topic:
            for sc in scenes:
                nar = sc.get("narration") or sc.get("caption", "")
                ok, reason = validate_narration(nar, topic)
                if not ok:
                    defects.append(reason)
                    break
    except Exception as e:
        logger.error(f"[planner] validate_scene_plan error ({e}), skipping validation.", exc_info=True)

    return defects


def autofix_scene_plan(
    scenes: List[Dict[str, Any]],
    story_info: Dict[str, Any],
    topic: str = "",
) -> List[Dict[str, Any]]:
    """Auto-fixes plan defects in code without failing:
    - Drops a character from a scene if same characters in all scenes.
    - Forces an establishing wide shot on scene 1.
    - Rebalances cameras away from all close-up.
    - Fills setting_detail with 8-15 word phrases drawn from story_elements.
    - Fixes missing or duplicate consecutive emotions to follow story arc.
    - Filters cues to only library effects, dropping unknown effects, capped at 4.
    """
    try:
        main_setting = story_info.get("main_setting") or story_info.get("setting") or "scenic landscape"
        story_chars = [c.get("name") for c in (story_info.get("characters") or []) if isinstance(c, dict) and c.get("name")]
        story_elements = story_info.get("story_elements") or []

        # Enforce defensive defaults for all scenes
        for sc in scenes:
            if not sc.get("camera"):
                sc["camera"] = "medium"
            if not sc.get("setting_detail"):
                sc["setting_detail"] = main_setting
            if not isinstance(sc.get("characters_present"), list):
                sc["characters_present"] = []
            if not isinstance(sc.get("must_show"), list):
                sc["must_show"] = []

        # Fix (b) & (c): Camera
        current_cameras = [str(sc.get("camera") or "medium").lower() for sc in scenes]
        if not any("wide" in c for c in current_cameras):
            scenes[0]["camera"] = "wide"
        if all("close" in c for c in current_cameras):
            scenes[0]["camera"] = "wide"
            scenes[1]["camera"] = "medium"
            scenes[2]["camera"] = "wide"
            scenes[3]["camera"] = "medium"
            scenes[4]["camera"] = "close-up"

        # Fix (a): the same characters are in all scenes (multi-character casts)
        if story_chars and len(story_chars) >= 2:
            chars_in_scenes = [tuple(sorted(sc.get("characters_present") or [])) for sc in scenes]
            if len(set(chars_in_scenes)) <= 1 or all(all(c in (sc.get("characters_present") or []) for c in story_chars) for sc in scenes):
                scenes[0]["characters_present"] = [story_chars[0]]
                scenes[1]["characters_present"] = [story_chars[1]]

        # Fix (d): story_elements in setting_detail
        for idx, sc in enumerate(scenes):
            current_detail = str(sc.get("setting_detail") or "").strip()
            words = current_detail.split()
            if len(words) < 8 or len(words) > 15:
                elem = story_elements[idx % len(story_elements)] if story_elements else f"scenic view across the {main_setting}"
                sc["setting_detail"] = format_setting_detail(str(elem), main_setting)

        # Fix (e): Emotions (story arc & no consecutive duplicate emotions)
        for idx, sc in enumerate(scenes):
            emo = sc.get("emotion")
            if emo not in VALID_EMOTIONS:
                sc["emotion"] = DEFAULT_EMOTION_ARC[idx % len(DEFAULT_EMOTION_ARC)]
            elif idx > 0 and sc["emotion"] == scenes[idx - 1]["emotion"]:
                prev_emo = scenes[idx - 1]["emotion"]
                alt_emotions = [e for e in DEFAULT_EMOTION_ARC if e != prev_emo]
                sc["emotion"] = alt_emotions[idx % len(alt_emotions)]
            sc["expression"] = EMOTION_STYLE[sc["emotion"]]["expression"]
            sc["camera_speed"] = EMOTION_STYLE[sc["emotion"]]["camera_speed"]

        if len(set(sc.get("emotion") for sc in scenes)) <= 1:
            for idx, sc in enumerate(scenes):
                sc["emotion"] = DEFAULT_EMOTION_ARC[idx % len(DEFAULT_EMOTION_ARC)]
                sc["expression"] = EMOTION_STYLE[sc["emotion"]]["expression"]
                sc["camera_speed"] = EMOTION_STYLE[sc["emotion"]]["camera_speed"]

        # Fix (f): Cues (drop unknown effects, cap to 4, fallback if empty)
        for sc in scenes:
            raw_cues = sc.get("cues")
            if not isinstance(raw_cues, list):
                raw_cues = []
            valid_cues = []
            for c in raw_cues:
                if isinstance(c, dict) and c.get("effect") in ALL_ANIMATION_EFFECTS:
                    valid_cues.append({
                        "word_index": int(c.get("word_index", 0)),
                        "effect": str(c["effect"]),
                        "target": str(c.get("target") or "scene"),
                    })
            if not valid_cues:
                valid_cues = extract_default_cues(sc.get("narration") or sc.get("caption", ""), sc["emotion"])
            sc["cues"] = valid_cues[:4]

        # Fix (g): Narration that copies topic
        if topic:
            for idx, sc in enumerate(scenes):
                nar = sc.get("narration") or sc.get("caption", "")
                ok, _ = validate_narration(nar, topic)
                if not ok:
                    sc["narration"] = f"A compelling chapter of the voyage unfolds across {main_setting}."

        return scenes
    except Exception as e:
        logger.error(f"[planner] autofix_scene_plan error ({e}), returning scenes.", exc_info=True)
        return scenes


def sanitize_image_prompt(
    raw_prompt: str,
    character_sheet_prefix: str = "",
    style: str = "3d_animated",
    camera_angle: str = "",
    char_defs: Optional[List[Dict[str, str]]] = None,
    max_words: int = 110,
) -> str:
    """Backward-compatible prompt sanitizer."""
    style_text = STYLE_PRESET_PROMPTS.get(style, STYLE_PRESET_PROMPTS["3d_animated"])
    avoid_text = STYLE_AVOID_TEXT

    body = raw_prompt.strip()
    for st in STYLE_PRESET_PROMPTS.values():
        body = re.sub(rf"(?i){re.escape(st)}\.?", "", body)
    body = re.sub(rf"(?i){re.escape(avoid_text)}\.?", "", body)

    for bw in BANNED_PROMPT_WORDS:
        body = re.sub(rf"(?i)\b{re.escape(bw)}\b", "", body)

    if camera_angle and not any(ang in body.lower() for ang in ["wide", "close-up", "medium", "low angle", "high angle"]):
        body = f"{body}, {camera_angle}"

    body = re.sub(r"\s+", " ", body).strip(" ,.-")
    result = f"{style_text}. {character_sheet_prefix} {body}. Negative prompt: {avoid_text}."
    return re.sub(r"\s+", " ", result).strip()


def verify_and_fix_character_sheet_features(
    prompt: str,
    char_defs: List[Dict[str, str]],
) -> str:
    """Backward-compatible accessory verifier."""
    fixed = prompt
    for c in char_defs:
        acc = c.get("accessory", "")
        if acc and acc.lower() not in fixed.lower():
            fixed += f", wearing {acc}"
    return fixed


def check_scene_plan_diversity(scenes: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """Checks if each scene has a distinct camera and distinct action."""
    cameras = [sc.get("camera", "") for sc in scenes if sc.get("camera")]
    if len(cameras) == len(scenes):
        for i in range(len(cameras) - 1):
            if cameras[i] == cameras[i + 1]:
                return False, "Adjacent identical cameras detected"
    return True, "OK"


def validate_and_autofix_camera_plan(
    scenes: List[Dict[str, Any]],
    props: Optional[List[Any]] = None,
) -> List[Dict[str, Any]]:
    """Enforces: at least 2 wide, at least 2 medium, at most 1 close-up,
    no adjacent identical cameras, never close-up if must_show contains prop.
    """
    valid_cams = ["wide", "medium", "close-up"]
    for sc in scenes:
        if sc.get("camera") not in valid_cams:
            sc["camera"] = "medium"

    prop_names = []
    if props:
        for p in props:
            if isinstance(p, dict):
                prop_names.append(p.get("name", "").lower())
            elif isinstance(p, str):
                prop_names.append(p.lower())

    for sc in scenes:
        must = [str(m).lower() for m in sc.get("must_show", [])]
        if any(pn in m for pn in prop_names for m in must if pn):
            if sc.get("camera") == "close-up":
                sc["camera"] = "medium"

    for i in range(len(scenes) - 1):
        if scenes[i].get("camera") == scenes[i + 1].get("camera"):
            if scenes[i].get("camera") == "wide":
                scenes[i + 1]["camera"] = "medium"
            elif scenes[i].get("camera") == "medium":
                scenes[i + 1]["camera"] = "wide" if i + 1 < 3 else "close-up"
            else:
                scenes[i + 1]["camera"] = "medium"

    wides = sum(1 for sc in scenes if sc.get("camera") == "wide")
    mediums = sum(1 for sc in scenes if sc.get("camera") == "medium")
    closeups = sum(1 for sc in scenes if sc.get("camera") == "close-up")

    if wides < 2 or mediums < 2 or closeups > 1:
        scenes[0]["camera"] = "wide"
        scenes[1]["camera"] = "medium"
        scenes[2]["camera"] = "wide"
        scenes[3]["camera"] = "medium"
        scenes[4]["camera"] = "medium" if any(pn in str(scenes[4].get("must_show", "")).lower() for pn in prop_names if pn) else "close-up"

    return scenes



def enforce_explicit_shayari_plan(
    scenes: List[Dict[str, Any]],
    story_info: Dict[str, Any],
    explicit_lines: List[str],
    style: str = "3d_animated",
) -> List[Dict[str, Any]]:
    """Enforces exact 4-line Devanagari shayari structure and 5-scene character count and camera rules:
    - Scene 1: Solo male protagonist (Kabir in dark blue shirt) on rooftop at night, Line 1
    - Scene 2: Pair scene (Kabir in dark blue shirt, Meera in light pink traditional outfit) in sunset garden memory, Line 2 (NO inset picture, NO photo frame)
    - Scene 3: Solo female protagonist (Meera) beside rain-streaked window at night, Line 3
    - Scene 4: Solo male protagonist (Kabir) on rooftop looking at city lights, Line 4
    - Scene 5: Outro wide shot under starry sky with camera pull-back and NO caption/narration (instrumental and fade to black)
    """
    if not explicit_lines or len(explicit_lines) != 4 or len(scenes) < 5:
        return scenes

    male_name = "Kabir"
    female_name = "Meera"
    chars = story_info.get("characters") or []
    m_found = None
    f_found = None
    for c in chars:
        if isinstance(c, dict):
            g = str(c.get("gender", "")).lower()
            r = str(c.get("role", "")).lower()
            if g == "male" or (re.search(r"\b(man|boy|male|guy)\b", r) and "woman" not in r):
                if not m_found:
                    m_found = c
            elif g == "female" or re.search(r"\b(woman|girl|female)\b", r):
                if not f_found:
                    f_found = c
    if m_found:
        m_found["name"] = "Kabir"
        m_found["outfit"] = "wearing a dark blue shirt"
        m_found["hair"] = "short black hair"
        male_name = "Kabir"
    else:
        chars.append({"name": "Kabir", "role": "young man", "gender": "male", "outfit": "wearing a dark blue shirt", "hair": "short black hair", "is_human": True, "skin_tone": "warm wheatish skin"})
    if f_found:
        f_found["name"] = "Meera"
        f_found["outfit"] = "wearing a light pink traditional outfit"
        f_found["hair"] = "long black hair"
        female_name = "Meera"
    else:
        chars.append({"name": "Meera", "role": "young woman", "gender": "female", "outfit": "wearing a light pink traditional outfit", "hair": "long black hair", "is_human": True, "skin_tone": "warm fair skin"})

    scenes[0]["characters_present"] = [male_name]
    scenes[0]["must_show"] = [male_name]
    scenes[0]["camera"] = "wide"
    scenes[0]["emotion"] = "calm"
    scenes[0]["caption"] = explicit_lines[0]
    scenes[0]["narration"] = explicit_lines[0]

    scenes[1]["characters_present"] = [male_name, female_name]
    scenes[1]["must_show"] = [male_name, female_name]
    scenes[1]["action"] = f"In a warm golden sunset memory, {male_name} and {female_name} walk beside each other in a sunlit garden, exchanging loving smiles and gently holding hands."
    scenes[1]["setting_detail"] = "A lush sunlit garden during golden hour sunset filled with blooming flowers, warm golden sunbeams, and natural foliage."
    scenes[1]["camera"] = "medium"
    scenes[1]["emotion"] = "joy"
    scenes[1]["caption"] = explicit_lines[1]
    scenes[1]["narration"] = explicit_lines[1]

    scenes[2]["characters_present"] = [female_name]
    scenes[2]["must_show"] = [female_name]
    scenes[2]["action"] = f"{female_name} sits quietly beside a rain-streaked bedroom window at night, holding an old printed photograph near her heart and gazing toward the moon."
    scenes[2]["setting_detail"] = "A cozy room beside a large rain-streaked window at night, with raindrops running down the glass pane under cool blue moonlight."
    scenes[2]["camera"] = "medium"
    scenes[2]["emotion"] = "sadness"
    scenes[2]["caption"] = explicit_lines[2]
    scenes[2]["narration"] = explicit_lines[2]

    scenes[3]["characters_present"] = [male_name]
    scenes[3]["must_show"] = [male_name]
    scenes[3]["action"] = f"{male_name} stands on the moonlit rooftop holding an old printed photograph near his chest, looking across the city skyline with deep longing."
    scenes[3]["setting_detail"] = "The quiet concrete rooftop at night overlooking distant city lights with cool blue moonlight and gentle shadows."
    scenes[3]["camera"] = "medium"
    scenes[3]["emotion"] = "sadness"
    scenes[3]["caption"] = explicit_lines[3]
    scenes[3]["narration"] = explicit_lines[3]

    scenes[4]["characters_present"] = [male_name]
    scenes[4]["must_show"] = [male_name]
    scenes[4]["action"] = f"{male_name} lowers his phone, tilts his head up toward the vast star-filled sky, and stands completely peaceful in quiet acceptance of separation as the camera pulls back."
    scenes[4]["setting_detail"] = "A breathtaking wide cinematic view of the rooftop under a vast starry night sky and Milky Way with glowing city lights below."
    scenes[4]["camera"] = "wide"
    scenes[4]["emotion"] = "calm"
    scenes[4]["caption"] = ""
    scenes[4]["narration"] = ""

    for idx, sc in enumerate(scenes[:5]):
        p = build_generic_scene_prompt(sc, story_info, style=style)
        sc["image_prompt"] = p
        sc["visual_prompt"] = p

    return scenes


def plan_scenes_anthropic(
    topic: str,
    length_seconds: int,
    style: str,
    caption_lang: str,
    api_key: str,
) -> List[Dict[str, Any]]:
    """Anthropic Claude scene planning adhering to generic requirements."""
    import httpx

    num_scenes = 5
    story_info = extract_story_info(topic)

    ext_style = str(story_info.get("visual_style") or "").strip()
    low_ext = ext_style.lower()
    if style in STYLE_PRESET_PROMPTS and style != "3d_animated":
        style_prompt = STYLE_PRESET_PROMPTS[style]
    elif style == "3d_animated" and any(k in low_ext for k in ["cinematic", "sci-fi", "science", "photoreal", "realistic", "dark", "noir", "space station"]):
        if "sci-fi" in low_ext or "science" in low_ext or "space station" in low_ext:
            style_prompt = STYLE_PRESET_PROMPTS["scifi"]
        elif "photoreal" in low_ext or "realistic" in low_ext:
            style_prompt = STYLE_PRESET_PROMPTS["realistic"]
        elif "dark fantasy" in low_ext:
            style_prompt = STYLE_PRESET_PROMPTS["dark_fantasy"]
        else:
            style_prompt = STYLE_PRESET_PROMPTS["cinematic"]
    else:
        style_prompt = STYLE_PRESET_PROMPTS.get(style, STYLE_PRESET_PROMPTS["3d_animated"])

    system_prompt = (
        "You are an expert storyboard artist and animation director.\n"
        "Divide the story topic into exactly 5 chronological scenes maintaining strict narrative and visual continuity.\n"
        "STRICT REQUIREMENTS FOR EACH SCENE:\n"
        "- camera: 'wide', 'medium', or 'close-up'. (Rule: at least 2 wide, at least 2 medium, at most 1 close-up, no two neighboring scenes with same camera, scene 1 must be establishing wide shot, never close-up if must_show has prop)\n"
        "- action: 2 to 3 chronological descriptive sentences detailing specific actions, interactions, body language, and staging\n"
        "- setting_detail: 25 to 50 words describing the specific environment, background architecture/terrain, atmospheric depth, lighting direction, and color palette\n"
        "- continuity: 1 sentence detailing visual continuity and narrative connection with adjacent scenes\n"
        "- characters_present: list of character names present in this scene. Rules:\n"
        "  * MUST ONLY use character names established in Extracted story components. Do NOT invent new character names.\n"
        "  * For ensemble/friendship stories (e.g. three boys and three girls), feature appropriate sub-groups across scenes ensuring both boys and girls appear throughout the story.\n"
        "  * For a solo protagonist story, keep the protagonist present across all or most scenes where the narrative requires their exploration or action.\n"
        "  * If a scene is a pure landscape, establishing environment, or empty spacecraft/room, set characters_present to []\n"
        "- emotion: one of 'joy', 'sadness', 'fear', 'anger', 'hope', 'surprise', 'calm', 'tension', 'pride', 'worry'. Follow story arc: consecutive scenes must NOT share the same emotion.\n"
        "- expression: character expression matching the emotion\n"
        "- cues: list of 0 to 4 animation cues {word_index: int, effect: string, target: string} where effect MUST be from ANIMATION_LIBRARY\n"
        "- must_show: list of items that must be visible in this scene\n"
        "- caption: short story caption (MAX 12 WORDS) in the specified caption language\n"
        "- narration: short spoken voiceover line matching caption\n"
        "Output ONLY a valid JSON array of 5 scene objects."
    )
    user_content = (
        f"Topic: {topic}\n"
        f"Length: {num_scenes} scenes.\n"
        f"Extracted story components: {json.dumps(story_info, ensure_ascii=False)}\n"
        f"Visual style: {style_prompt}.\n"
        f"Caption language: {'Hindi (Devanagari script)' if caption_lang == 'hi' else 'English'}."
    )
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1500,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_content}],
    }
    scenes: List[Dict[str, Any]] = []
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post("https://api.anthropic.com/v1/messages", json=payload, headers=headers)
            resp.raise_for_status()
            raw_text = resp.json()["content"][0]["text"].strip()
            data = clean_and_parse_json(raw_text)
            scenes = normalize_scenes(data)

            # Validate and re-ask once if defects
            defects = validate_scene_plan(scenes, story_info, topic=topic)
            if defects:
                fix_content = (
                    f"Your previous plan had the following defect(s): {'; '.join(defects)}.\n"
                    "Please regenerate the exact 5 scenes as a valid JSON array, strictly resolving these defects."
                )
                re_messages = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": raw_text},
                    {"role": "user", "content": fix_content},
                ]
                try:
                    re_resp = client.post(
                        "https://api.anthropic.com/v1/messages",
                        json={**payload, "messages": re_messages},
                        headers=headers,
                    )
                    if re_resp.status_code == 200:
                        re_text = re_resp.json()["content"][0]["text"].strip()
                        re_data = clean_and_parse_json(re_text)
                        re_scenes = normalize_scenes(re_data)
                        if len(re_scenes) == 5:
                            scenes = re_scenes
                except Exception as e:
                    logger.warning(f"[planner] Anthropic plan regeneration failed ({e}), will autofix in code.")
    except Exception as e:
        logger.warning(f"[planner] Anthropic planning error ({e}), re-raising.", exc_info=True)
        raise

    if len(scenes) < 5:
        raise RuntimeError(f"planner unavailable: Anthropic returned {len(scenes)} scenes instead of 5")

    scenes = scenes[:5]
    defects = validate_scene_plan(scenes, story_info, topic=topic)
    if defects:
        scenes = autofix_scene_plan(scenes, story_info, topic=topic)

    validate_and_autofix_camera_plan(scenes, story_info.get("props", []))

    res = []
    for idx, s in enumerate(scenes):
        p = build_generic_scene_prompt(s, story_info, style=style)
        is_captions_off = str(caption_lang).lower().strip() in ("none", "", "off", "disabled", "false", "0")
        cap = "" if is_captions_off else clean_caption_language(sanitize_caption(s.get("caption", "")), caption_lang)
        if is_unwanted_caption_text(cap, p):
            cap = ""
        nar = sanitize_narration(s.get("narration") or (cap if cap else s.get("action", "")))
        if is_unwanted_caption_text(nar, p):
            nar = ""
        emo = s.get("emotion") or "calm"
        item = {
            "image_prompt": p,
            "visual_prompt": p,
            "caption": cap,
            "narration": nar,
            "camera": s.get("camera", "medium"),
            "must_show": s.get("must_show", []),
            "action": s.get("action", ""),
            "setting_detail": s.get("setting_detail", ""),
            "characters_present": s.get("characters_present", []),
            "emotion": emo,
            "expression": s.get("expression") or EMOTION_STYLE.get(emo, EMOTION_STYLE["calm"])["expression"],
            "camera_speed": s.get("camera_speed") or EMOTION_STYLE.get(emo, EMOTION_STYLE["calm"])["camera_speed"],
            "cues": s.get("cues", []),
            "sheets": story_info,
            "motion_instructions": s.get("motion_instructions") or build_scene_motion_instructions(s, story_info, style=style),
            "motion_prompt": s.get("motion_instructions") or build_scene_motion_instructions(s, story_info, style=style),
            "character_reference": story_info.get("character_reference_sheet", ""),
        }
        logger.info(
            f"[planner] Scene {idx+1}: characters_present={item['characters_present']}, "
            f"camera={item['camera']}, setting_detail='{item['setting_detail']}', "
            f"emotion={item['emotion']}, cues={len(item['cues'])}"
        )
        res.append(item)
    return res


def plan_scenes_gemini(
    topic: str,
    length_seconds: int,
    style: str,
    caption_lang: str,
    api_key: str,
) -> List[Dict[str, Any]]:
    """Scene planning using Google Gemini API adhering to generic requirements."""
    import httpx

    model = os.getenv("GEMINI_TEXT_MODEL") or os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    is_hi = (caption_lang == "hi") or any("\u0900" <= ch <= "\u097F" for ch in topic) or any(k in topic.lower() for k in ["shayari", "hindi", "shairi"])
    lang_name = "Hindi (Devanagari script)" if is_hi else "English"

    try:
        story_info = extract_story_info(topic, api_key=api_key, model=model)

        ext_style = str(story_info.get("visual_style") or "").strip()
        low_ext = ext_style.lower()
        if style in STYLE_PRESET_PROMPTS and style != "3d_animated":
            style_prompt = STYLE_PRESET_PROMPTS[style]
        elif style == "3d_animated" and any(k in low_ext for k in ["cinematic", "sci-fi", "science", "photoreal", "realistic", "dark", "noir", "space station"]):
            if "sci-fi" in low_ext or "science" in low_ext or "space station" in low_ext:
                style_prompt = STYLE_PRESET_PROMPTS["scifi"]
            elif "photoreal" in low_ext or "realistic" in low_ext:
                style_prompt = STYLE_PRESET_PROMPTS["realistic"]
            elif "dark fantasy" in low_ext:
                style_prompt = STYLE_PRESET_PROMPTS["dark_fantasy"]
            else:
                style_prompt = STYLE_PRESET_PROMPTS["cinematic"]
        else:
            style_prompt = STYLE_PRESET_PROMPTS.get(style, STYLE_PRESET_PROMPTS["3d_animated"])

        system_instruction = (
            "You are an expert storyboard artist and animation director.\n"
            "Divide the story topic into exactly 5 chronological scenes maintaining strict narrative and visual continuity.\n"
            "STRICT REQUIREMENTS FOR EACH SCENE:\n"
            "- camera: 'wide', 'medium', or 'close-up'. (Rule: at least 2 wide, at least 2 medium, at most 1 close-up, no two neighboring scenes with same camera, scene 1 must be establishing wide shot, never close-up if must_show has prop)\n"
            "- action: 2 to 3 chronological descriptive sentences detailing specific actions, interactions, body language, and staging\n"
            "- setting_detail: 25 to 50 words describing the specific environment, background architecture/terrain, atmospheric depth, lighting direction, and color palette\n"
            "- continuity: 1 sentence detailing visual continuity and narrative connection with adjacent scenes\n"
            "- characters_present: list of character names present in this scene. Rules:\n"
            "  * MUST ONLY use character names established in Extracted story components. Do NOT invent new character names.\n"
            "  * For ensemble/friendship stories (e.g. three boys and three girls), feature appropriate sub-groups across scenes ensuring both boys and girls appear throughout the story.\n"
            "  * For a solo protagonist story, keep the protagonist present across all or most scenes where the narrative requires their exploration or action.\n"
            "  * If a scene is a pure landscape, establishing environment, or empty spacecraft/room, set characters_present to []\n"
            "- emotion: one of 'joy', 'sadness', 'fear', 'anger', 'hope', 'surprise', 'calm', 'tension', 'pride', 'worry'. Follow story arc: consecutive scenes must NOT share the same emotion.\n"
            "- expression: character expression matching the emotion\n"
            "- cues: list of 0 to 4 animation cues {word_index: int, effect: string, target: string} where effect MUST be from ANIMATION_LIBRARY\n"
            "- must_show: list of items that must be visible in this scene\n"
            "- caption: short story caption (MAX 12 WORDS) in the specified caption language\n"
            "- narration: spoken voiceover line matching caption\n"
            "- motion_instructions: 2 to 3 vivid sentences describing camera movement (pan, zoom, tilt) and environmental dynamics\n"
            "Output ONLY a valid JSON array of 5 scene objects."
        )
        explicit_lines = extract_explicit_devanagari_lines(topic)
        explicit_note = ""
        if explicit_lines:
            lines_fmt = "\n".join(f"  Line {i+1}: {l}" for i, l in enumerate(explicit_lines))
            explicit_note = (
                f"\n\nCRITICAL NARRATION & SHAYARI RULES:\n"
                f"The user provided exact original lines:\n{lines_fmt}\n"
                f"- Scene 1 narration & caption MUST be exactly Line 1.\n"
                f"- Scene 2 narration & caption MUST be exactly Line 2.\n"
                f"- Scene 3 narration & caption MUST be exactly Line 3.\n"
                f"- Scene 4 narration & caption MUST be exactly Line 4.\n"
                f"- Scene 5 is the cinematic and musical outro under the stars; its narration & caption MUST be empty (\"\") so Line 4 is NOT repeated.\n"
                f"- Single unified frame only: each scene MUST describe ONE camera angle, never describe split-screens, collages, or returning to a previous scene within one prompt.\n"
                f"- Character continuity: young Indian man (24yo, short black hair, dark blue shirt) and young Indian woman (long black hair, light pink traditional outfit).\n"
            )

        user_content = (
            f"Topic: {topic}\n"
            f"Extracted story components: {json.dumps(story_info)}\n"
            f"Animation style: {style_prompt}\n"
            f"Caption language: {lang_name}.{explicit_note}"
        )

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": system_instruction + "\n\n" + user_content}]}],
            "generationConfig": {"response_mime_type": "application/json"},
        }

        scenes: List[Dict[str, Any]] = []
        with httpx.Client(timeout=30.0) as client:
            last_call_err = None
            raw_text = ""
            for attempt in range(2):
                try:
                    resp = client.post(url, json=payload, headers=headers)
                    resp.raise_for_status()
                    raw_text = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                    data = clean_and_parse_json(raw_text)
                    scenes = normalize_scenes(data)
                    if len(scenes) >= 5:
                        break
                except Exception as call_err:
                    last_call_err = call_err
                    if attempt == 0:
                        logger.warning(f"[planner] Gemini planning attempt 1 failed ({call_err}), retrying once...")
                        time.sleep(1.0)
            if not scenes and last_call_err:
                raise last_call_err

            # Validate and re-ask once if defects
            defects = validate_scene_plan(scenes, story_info, topic=topic)
            if defects:
                fix_prompt = (
                    f"Your previous plan had the following defect(s): {'; '.join(defects)}.\n"
                    "Please regenerate the exact 5 scenes as a valid JSON array, strictly resolving these defects."
                )
                re_payload = {
                    "contents": [
                        {"parts": [{"text": system_instruction + "\n\n" + user_content}]},
                        {"parts": [{"text": raw_text}]},
                        {"parts": [{"text": fix_prompt}]},
                    ],
                    "generationConfig": {"response_mime_type": "application/json"},
                }
                try:
                    re_resp = client.post(url, json=re_payload, headers=headers)
                    if re_resp.status_code == 200:
                        re_text = re_resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                        re_data = clean_and_parse_json(re_text)
                        re_scenes = normalize_scenes(re_data)
                        if len(re_scenes) == 5:
                            scenes = re_scenes
                except Exception as e:
                    logger.warning(f"[planner] Gemini plan regeneration failed ({e}), will autofix in code.")

        if len(scenes) < 5:
            raise RuntimeError(f"planner unavailable: Gemini returned {len(scenes)} scenes instead of 5")

        scenes = scenes[:5]
        defects = validate_scene_plan(scenes, story_info, topic=topic)
        if defects:
            scenes = autofix_scene_plan(scenes, story_info, topic=topic)

        validate_and_autofix_camera_plan(scenes, story_info.get("props", []))

        explicit_lines = extract_explicit_devanagari_lines(topic)
        if explicit_lines and len(explicit_lines) == 4:
            male_name = "Kabir"
            female_name = "Meera"
            chars = story_info.get("characters") or []
            m_found = None
            f_found = None
            for c in chars:
                if isinstance(c, dict):
                    g = str(c.get("gender", "")).lower()
                    r = str(c.get("role", "")).lower()
                    if g == "male" or (re.search(r"\b(man|boy|male|guy)\b", r) and "woman" not in r):
                        if not m_found:
                            m_found = c
                    elif g == "female" or re.search(r"\b(woman|girl|female)\b", r):
                        if not f_found:
                            f_found = c
            if m_found:
                male_name = m_found.get("name", "Kabir")
                m_found["outfit"] = "wearing a dark blue shirt"
                m_found["hair"] = "short black hair"
            if f_found:
                female_name = f_found.get("name", "Meera")
                f_found["outfit"] = "wearing a light pink traditional outfit"
                f_found["hair"] = "long black hair"

            # Strict 5-scene character count & narrative mapping
            scenes[0]["characters_present"] = [male_name]
            scenes[1]["characters_present"] = [male_name, female_name]
            scenes[1]["action"] = f"In a warm golden sunset memory, {male_name} and {female_name} walk beside each other in a sunlit garden, exchanging loving smiles and gently holding hands."
            scenes[1]["setting_detail"] = "A lush sunlit garden during golden hour sunset filled with blooming flowers, warm golden sunbeams, and natural foliage."
            scenes[1]["camera"] = "medium"
            scenes[1]["emotion"] = "joy"
            scenes[2]["characters_present"] = [female_name]
            scenes[2]["action"] = f"{female_name} sits quietly beside a rain-streaked bedroom window at night, holding the old photograph near her heart and gazing toward the moon."
            scenes[2]["setting_detail"] = "A cozy room beside a large rain-streaked window at night, with raindrops running down the glass pane under cool blue moonlight."
            scenes[2]["camera"] = "medium"
            scenes[2]["emotion"] = "sadness"
            scenes[3]["characters_present"] = [male_name]
            scenes[3]["action"] = f"{male_name} stands on the moonlit rooftop holding the old photograph near his chest, looking across the city skyline with deep longing."
            scenes[3]["setting_detail"] = "The quiet concrete rooftop at night overlooking distant city lights with cool blue moonlight and gentle shadows."
            scenes[3]["camera"] = "medium"
            scenes[3]["emotion"] = "sadness"
            scenes[4]["characters_present"] = [male_name]
            scenes[4]["action"] = f"{male_name} lowers his phone, tilts his head up toward the vast star-filled sky, and stands completely peaceful in quiet acceptance of separation as the camera pulls back."
            scenes[4]["setting_detail"] = "A breathtaking wide cinematic view of the rooftop under a vast starry night sky and Milky Way with glowing city lights below."
            scenes[4]["camera"] = "wide"
            scenes[4]["emotion"] = "calm"

        res = []
        for idx, s in enumerate(scenes):
            p = build_generic_scene_prompt(s, story_info, style=style)
            is_captions_off = str(caption_lang).lower().strip() in ("none", "", "off", "disabled", "false", "0")
            if explicit_lines:
                if idx < len(explicit_lines):
                    cap = explicit_lines[idx]
                    nar = explicit_lines[idx]
                else:
                    cap = ""
                    nar = ""
            else:
                cap = "" if is_captions_off else clean_caption_language(sanitize_caption(s.get("caption", "")), caption_lang)
                if is_unwanted_caption_text(cap, p):
                    cap = ""
                nar = sanitize_narration(s.get("narration") or (cap if cap else s.get("action", "")))
                if is_unwanted_caption_text(nar, p):
                    nar = ""
            emo = s.get("emotion") or "calm"
            item = {
                "image_prompt": p,
                "visual_prompt": p,
                "caption": cap,
                "narration": nar,
                "camera": s.get("camera", "medium"),
                "must_show": s.get("must_show", []),
                "action": s.get("action", ""),
                "setting_detail": s.get("setting_detail", ""),
                "characters_present": s.get("characters_present", []),
                "emotion": emo,
                "expression": s.get("expression") or EMOTION_STYLE.get(emo, EMOTION_STYLE["calm"])["expression"],
                "camera_speed": s.get("camera_speed") or EMOTION_STYLE.get(emo, EMOTION_STYLE["calm"])["camera_speed"],
                "cues": s.get("cues", []),
                "sheets": story_info,
            }
            logger.info(
                f"[planner] Scene {idx+1}: characters_present={item['characters_present']}, "
                f"camera={item['camera']}, setting_detail='{item['setting_detail']}', "
                f"emotion={item['emotion']}, cues={len(item['cues'])}"
            )
            res.append(item)
        return res
    except Exception as e:
        # Fallback to template planning on any unrecoverable error
        return plan_scenes_template(topic, length_seconds, style, caption_lang)



# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GENRE DETECTION & SCENE-ARC TEMPLATES
# No story-specific hardcoding; covers all 14 animation categories.
# ═══════════════════════════════════════════════════════════════════════════════

GENRE_KEYWORDS: dict = {
    "devotional": [
        "hanuman", "rama", "krishna", "shiva", "durga", "ganesha", "vishnu",
        "deity", "devotional", "god", "goddess", "temple", "prayer", "divine",
        "mythological", "myth", "bhagwan", "puja", "mandir", "bhakti",
        "mahabharata", "ramayana", "vedic", "avatar", "deva", "devi",
    ],
    "horror": [
        "horror", "haunted", "ghost", "zombie", "demon", "monster", "scary",
        "terror", "nightmare", "creepy", "abandoned", "cursed", "dark figure",
        "possessed", "eerie", "spine-chilling", "paranormal",
    ],
    "romance": [
        "romantic", "romance", "couple", "love story", "lovers", "proposal",
        "anniversary", "date", "sunset together", "heartfelt", "falling in love",
    ],
    "comedy": [
        "funny", "comedy", "humorous", "silly", "slapstick", "cartoon comedy",
        "talking animals", "laugh", "hilarious", "clumsy", "absurd",
    ],
    "action": [
        "action", "superhero", "fighting", "battle", "hero", "villain",
        "martial arts", "combat", "chase", "explosion", "adventure quest",
        "warrior", "knight", "ninja", "samurai",
    ],
    "horror": [
        "horror", "haunted", "ghost", "zombie", "demon", "monster", "scary",
        "terror", "nightmare", "creepy", "abandoned", "cursed",
    ],
    "children": [
        "children", "child", "kids", "baby", "toddler", "nursery", "learning",
        "educational", "alphabet", "numbers", "animal story", "fairy tale",
        "bedtime story", "little", "young",
    ],
    "scifi": [
        "sci-fi", "science fiction", "space", "spaceship", "galaxy", "planet",
        "alien", "robot", "cyborg", "future", "futuristic", "laser", "warp",
        "star", "cosmos", "androids", "space station",
    ],
    "fantasy": [
        "fantasy", "magic", "wizard", "dragon", "elf", "dwarf", "unicorn",
        "enchanted", "mystical", "quest", "sorcerer", "dungeon", "magical",
        "potion", "spell", "fairy", "kingdom", "realm",
    ],
    "animals": [
        "elephant", "lion", "tiger", "fox", "wolf", "bear", "deer", "rabbit",
        "owl", "eagle", "whale", "dolphin", "panda", "cheetah", "gorilla",
        "wildlife", "jungle", "safari", "nature documentary", "animal story",
        "animal adventure",
    ],
    "educational": [
        "educational", "teach", "learning", "science", "history", "geography",
        "math", "tutorial", "explained", "documentary", "discover",
    ],
}

# Scene arc templates: 5 beats per genre.
# Each beat has: camera, emotion, action_template, setting_role, char_focus
# {c1} = primary character name, {setting} = main_setting, {prop} = first prop or ''
GENRE_ARCS: dict = {
    "devotional": [
        {"camera": "wide",     "emotion": "calm",    "char_focus": "alone",
         "action_template": "{c1} stands in peaceful reverence at the sacred {setting}, hands folded, eyes closed in prayer",
         "setting_role": "establish the sacred grounds"},
        {"camera": "medium",   "emotion": "hope",    "char_focus": "alone",
         "action_template": "{c1} receives a divine calling, a radiant beam of golden light descends from above",
         "setting_role": "show the divine celestial sky opening"},
        {"camera": "wide",     "emotion": "tension", "char_focus": "alone",
         "action_template": "{c1} faces a great obstacle or adversary with courage and determination",
         "setting_role": "show the challenging terrain or battlefield"},
        {"camera": "medium",   "emotion": "joy",     "char_focus": "alone",
         "action_template": "{c1} performs an extraordinary feat of strength, devotion, or divine power",
         "setting_role": "dramatic moment of triumph or miracle"},
        {"camera": "close-up", "emotion": "calm",    "char_focus": "alone",
         "action_template": "{c1} returns victorious, bowing in gratitude as the divine light blesses the scene",
         "setting_role": "peaceful conclusion with golden glow"},
    ],
    "horror": [
        {"camera": "wide",     "emotion": "calm",    "char_focus": "alone",
         "action_template": "{c1} arrives at the {setting} on a dark stormy night, unaware of the danger lurking",
         "setting_role": "establish the ominous abandoned location"},
        {"camera": "medium",   "emotion": "surprise","char_focus": "alone",
         "action_template": "{c1} notices disturbing signs — strange sounds, moving shadows, flickering lights",
         "setting_role": "show unsettling environmental details"},
        {"camera": "close-up", "emotion": "tension", "char_focus": "alone",
         "action_template": "{c1} backs away in fear as a terrifying entity reveals itself from the darkness",
         "setting_role": "tense confrontation point"},
        {"camera": "medium",   "emotion": "tension", "char_focus": "alone",
         "action_template": "{c1} runs frantically through the {setting} as the horror pursues relentlessly",
         "setting_role": "chaotic chase through dangerous environment"},
        {"camera": "wide",     "emotion": "hope",    "char_focus": "alone",
         "action_template": "{c1} finally escapes into open daylight, breathing heavily, the {setting} silent behind",
         "setting_role": "relief of escape with horror left behind"},
    ],
    "romance": [
        {"camera": "wide",     "emotion": "calm",    "char_focus": "together",
         "action_template": "{c1} and their partner meet for the first time at the beautiful {setting}, eyes meeting across the space",
         "setting_role": "romantic establishing shot"},
        {"camera": "medium",   "emotion": "joy",     "char_focus": "together",
         "action_template": "The two share a warm laugh and conversation, growing closer as the mood brightens",
         "setting_role": "intimate connection scene"},
        {"camera": "wide",     "emotion": "hope",    "char_focus": "together",
         "action_template": "They walk side by side through the {setting}, holding hands as the golden light wraps around them",
         "setting_role": "romantic walk together"},
        {"camera": "close-up", "emotion": "joy",     "char_focus": "together",
         "action_template": "A heartfelt moment — a gentle touch, a loving gaze, or a sweet gift exchanged",
         "setting_role": "emotional peak close-up"},
        {"camera": "medium",   "emotion": "calm",    "char_focus": "together",
         "action_template": "The couple stands together watching the sunset at {setting}, smiling peacefully at their shared journey",
         "setting_role": "warm romantic conclusion"},
    ],
    "comedy": [
        {"camera": "wide",     "emotion": "joy",     "char_focus": "alone",
         "action_template": "{c1} arrives at {setting} full of confidence, immediately stumbling into a comical mishap",
         "setting_role": "comedy setup establishing shot"},
        {"camera": "medium",   "emotion": "surprise","char_focus": "alone",
         "action_template": "{c1} reacts with exaggerated shock as things go hilariously wrong",
         "setting_role": "comedic reaction scene"},
        {"camera": "wide",     "emotion": "tension", "char_focus": "alone",
         "action_template": "{c1} attempts an absurd solution to the problem, making it even more chaotic",
         "setting_role": "escalating comedy chaos"},
        {"camera": "medium",   "emotion": "surprise","char_focus": "alone",
         "action_template": "{c1} faces the ultimate ridiculous consequence — a pie in the face, total collapse, or absurd transformation",
         "setting_role": "comedic climax moment"},
        {"camera": "wide",     "emotion": "joy",     "char_focus": "alone",
         "action_template": "{c1} laughs it off and takes a cheerful bow as everything resolves in the funniest way",
         "setting_role": "happy comedic conclusion"},
    ],
    "action": [
        {"camera": "wide",     "emotion": "tension", "char_focus": "alone",
         "action_template": "{c1} surveys the dangerous {setting}, preparing for the mission ahead with focused determination",
         "setting_role": "action hero establishing shot"},
        {"camera": "medium",   "emotion": "tension", "char_focus": "alone",
         "action_template": "{c1} confronts the villain or antagonistic force head-on with confidence and power",
         "setting_role": "hero vs villain standoff"},
        {"camera": "wide",     "emotion": "tension", "char_focus": "alone",
         "action_template": "{c1} launches into intense combat — punching, leaping, using special abilities across the battlefield",
         "setting_role": "explosive action battle"},
        {"camera": "medium",   "emotion": "hope",    "char_focus": "alone",
         "action_template": "{c1} delivers the decisive final blow, defeating the enemy in a spectacular moment",
         "setting_role": "triumph climax"},
        {"camera": "wide",     "emotion": "joy",     "char_focus": "alone",
         "action_template": "{c1} stands victorious at the {setting} as allies cheer and calm is restored",
         "setting_role": "heroic victory conclusion"},
    ],
    "children": [
        {"camera": "wide",     "emotion": "joy",     "char_focus": "alone",
         "action_template": "{c1} wakes up excited and runs toward {setting}, ready for a brand-new adventure",
         "setting_role": "cheerful colourful establishing shot"},
        {"camera": "medium",   "emotion": "surprise","char_focus": "alone",
         "action_template": "{c1} discovers something magical and wonderful — eyes wide with delight",
         "setting_role": "magical discovery moment"},
        {"camera": "wide",     "emotion": "hope",    "char_focus": "alone",
         "action_template": "{c1} tries to solve a fun challenge or puzzle, stumbling along with cute determination",
         "setting_role": "playful problem-solving scene"},
        {"camera": "medium",   "emotion": "joy",     "char_focus": "alone",
         "action_template": "{c1} learns an important lesson with a big smile — kindness, sharing, or bravery",
         "setting_role": "heartwarming lesson moment"},
        {"camera": "wide",     "emotion": "calm",    "char_focus": "alone",
         "action_template": "{c1} celebrates with friends and returns home happily as the {setting} glows warmly",
         "setting_role": "joyful resolution scene"},
    ],
    "scifi": [
        {"camera": "wide",     "emotion": "calm",    "char_focus": "alone",
         "action_template": "{c1} gazes out at the breathtaking expanse of {setting}, surrounded by advanced technology",
         "setting_role": "futuristic environment establishing shot"},
        {"camera": "medium",   "emotion": "tension", "char_focus": "alone",
         "action_template": "{c1} detects an anomaly — alarms flash, data streams, and a critical problem emerges",
         "setting_role": "crisis discovery moment"},
        {"camera": "wide",     "emotion": "tension", "char_focus": "alone",
         "action_template": "{c1} races through the {setting} activating systems and working against a critical countdown",
         "setting_role": "high-stakes action in sci-fi environment"},
        {"camera": "medium",   "emotion": "hope",    "char_focus": "alone",
         "action_template": "{c1} makes the breakthrough discovery or fires the decisive shot, resolving the crisis",
         "setting_role": "climactic sci-fi resolution"},
        {"camera": "wide",     "emotion": "calm",    "char_focus": "alone",
         "action_template": "{c1} stands at the viewport of {setting} as stars drift peacefully — the mission complete",
         "setting_role": "serene futuristic conclusion"},
    ],
    "fantasy": [
        {"camera": "wide",     "emotion": "calm",    "char_focus": "alone",
         "action_template": "{c1} enters the enchanted {setting} for the first time, surrounded by glowing magical elements",
         "setting_role": "magical world establishing shot"},
        {"camera": "medium",   "emotion": "surprise","char_focus": "alone",
         "action_template": "{c1} encounters a magical creature or discovers a mystical artefact that begins the quest",
         "setting_role": "magical discovery or encounter"},
        {"camera": "wide",     "emotion": "tension", "char_focus": "alone",
         "action_template": "{c1} faces the dark force or obstacle standing between them and their goal",
         "setting_role": "fantasy conflict in magical landscape"},
        {"camera": "medium",   "emotion": "hope",    "char_focus": "alone",
         "action_template": "{c1} unleashes or masters their magical power, turning the tide of the conflict",
         "setting_role": "magical power moment"},
        {"camera": "wide",     "emotion": "joy",     "char_focus": "alone",
         "action_template": "{c1} succeeds in the quest as the {setting} is restored to beauty, magic shimmering everywhere",
         "setting_role": "triumphant magical conclusion"},
    ],
    "animals": [
        {"camera": "wide",     "emotion": "calm",    "char_focus": "alone",
         "action_template": "{c1} roams through the {setting} at golden hour, exploring the natural environment",
         "setting_role": "wildlife establishing shot"},
        {"camera": "medium",   "emotion": "surprise","char_focus": "alone",
         "action_template": "{c1} encounters another creature or discovers a challenge in the wilderness",
         "setting_role": "encounter or discovery scene"},
        {"camera": "wide",     "emotion": "tension", "char_focus": "alone",
         "action_template": "{c1} navigates a difficult moment — a predator, a river crossing, or a stormy night",
         "setting_role": "survival challenge scene"},
        {"camera": "medium",   "emotion": "hope",    "char_focus": "alone",
         "action_template": "{c1} shows remarkable resilience, intelligence, or teamwork to overcome the challenge",
         "setting_role": "moment of animal triumph"},
        {"camera": "wide",     "emotion": "calm",    "char_focus": "alone",
         "action_template": "{c1} rests peacefully in the {setting} as the sun sets, the natural world quiet and beautiful",
         "setting_role": "serene wildlife conclusion"},
    ],
    "educational": [
        {"camera": "wide",     "emotion": "calm",    "char_focus": "alone",
         "action_template": "{c1} stands at {setting}, introducing the concept or subject with enthusiasm",
         "setting_role": "educational introduction shot"},
        {"camera": "medium",   "emotion": "surprise","char_focus": "alone",
         "action_template": "{c1} demonstrates the first key idea with clear visual examples and expressive gestures",
         "setting_role": "first concept explanation"},
        {"camera": "wide",     "emotion": "hope",    "char_focus": "alone",
         "action_template": "{c1} shows how the concept works in a real-world context with a vivid demonstration",
         "setting_role": "real-world application scene"},
        {"camera": "close-up", "emotion": "joy",     "char_focus": "alone",
         "action_template": "{c1} interacts with the key element or tool, highlighting the most important detail",
         "setting_role": "detail focus close-up"},
        {"camera": "medium",   "emotion": "calm",    "char_focus": "alone",
         "action_template": "{c1} summarises the lesson with a confident smile, reinforcing the main takeaway",
         "setting_role": "educational conclusion recap"},
    ],
    "generic": [
        {"camera": "wide",     "emotion": "calm",    "char_focus": "alone",
         "action_template": "{c1} arrives at the {setting}, taking in the grand sweeping vista with curiosity",
         "setting_role": "story-opening establishing wide shot"},
        {"camera": "medium",   "emotion": "surprise","char_focus": "alone",
         "action_template": "{c1} makes an unexpected discovery that changes the course of the story",
         "setting_role": "inciting incident scene"},
        {"camera": "wide",     "emotion": "tension", "char_focus": "alone",
         "action_template": "{c1} navigates the central challenge of the story with energy and determination",
         "setting_role": "main narrative conflict scene"},
        {"camera": "medium",   "emotion": "hope",    "char_focus": "alone",
         "action_template": "{c1} reaches the pivotal turning point and finds the resolution",
         "setting_role": "climax and breakthrough scene"},
        {"camera": "close-up", "emotion": "joy",     "char_focus": "alone",
         "action_template": "{c1} stands at the {setting} with a satisfied smile as the journey concludes",
         "setting_role": "warm story-ending close-up"},
    ],
}


def detect_genre(topic: str) -> str:
    """Detect animation genre from topic text. Returns genre key for GENRE_ARCS."""
    t = topic.lower()
    scores: dict[str, int] = {}
    for genre, keywords in GENRE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in t)
        if score:
            scores[genre] = score
    if not scores:
        return "generic"
    return max(scores, key=lambda g: scores[g])


def _build_arc_scene(
    beat: dict,
    idx: int,
    c1_name: str,
    setting: str,
    prop_name: str,
    chars: list,
    story_info: dict,
) -> dict:
    """Build one raw scene dict from a genre arc beat."""
    action = beat["action_template"].format(
        c1=c1_name,
        setting=setting,
        prop=prop_name or "the key object",
    )
    setting_detail = format_setting_detail(beat["setting_role"], setting)

    # Characters present
    char_focus = beat.get("char_focus", "alone")
    if char_focus == "together" and len(chars) >= 2:
        chars_present = [c.get("name") for c in chars[:2] if c.get("name")]
    else:
        chars_present = [c1_name] if c1_name else []

    must_show = list(chars_present)
    if prop_name and idx in (1, 2):
        must_show.append(prop_name)

    emo = beat.get("emotion", "calm")
    emo_data = EMOTION_STYLE.get(emo, EMOTION_STYLE["calm"])

    # Caption / narration
    # Strip leading articles from setting for grammatically correct captions
    _sett_clean = re.sub(r"^(?:a|an|the)\s+", "", setting, flags=re.IGNORECASE).strip()
    _char_clean = re.sub(r"^(?:the|a|an)\s+", "", c1_name, flags=re.IGNORECASE).strip() if c1_name else "our hero"
    captions_map = {
        0: {"en": f"{_char_clean} arrives at the {_sett_clean}, beginning an unforgettable story.", "hi": f"{_char_clean} ki kahani ki shuruat hoti hai."},
        1: {"en": f"A surprising discovery changes {_char_clean}'s journey forever.", "hi": f"Ek ajeeb khoj sab kuch badal deti hai."},
        2: {"en": f"{_char_clean} faces the central challenge with courage.", "hi": f"Chunauti ka saamna hota hai."},
        3: {"en": f"The decisive moment arrives — {_char_clean} stands strong.", "hi": f"Faisle ka pal aa gaya."},
        4: {"en": f"{_char_clean}'s journey reaches a peaceful and triumphant end.", "hi": f"Ek anmol safar ka shantipurn ant."},
    }
    cap_data = captions_map.get(idx, captions_map[0])

    return {
        "camera": beat["camera"],
        "action": action,
        "setting_detail": setting_detail,
        "characters_present": chars_present,
        "must_show": must_show,
        "emotion": emo,
        "expression": emo_data["expression"],
        "camera_speed": emo_data["camera_speed"],
        "cues": [{"word_index": idx % 3, "effect": ["pan_left","zoom_in","push_to_subject","breathe","glow"][idx], "target": "scene"}],
        "_cap_en": cap_data["en"],
        "_cap_hi": cap_data["hi"],
    }


def plan_scenes_template(
    topic: str,
    length_seconds: int = 25,
    style: str = "3d_animated",
    caption_lang: str = "none",
) -> List[Dict[str, Any]]:
    """Universal fallback planner for ANY animation topic/genre.

    Works without any LLM API key:
    - Detects genre (devotional, horror, romance, comedy, action, children,
      scifi, fantasy, animals, educational, generic).
    - Selects a genre-specific 5-scene narrative arc.
    - Fills arc templates with characters/setting extracted from the topic.
    - Builds full image prompts via build_generic_scene_prompt().
    - Captions in English or Hindi as appropriate.
    - Never produces empty Visual Description fields.
    """
    story_info = extract_story_info(topic)
    chars = story_info.get("characters", [])
    props = story_info.get("key_props", [])
    setting = story_info.get("main_setting", "scenic landscape")

    # Primary character name (clean up overly long parsed names)
    if chars and chars[0].get("name") and not str(chars[0]["name"]).lower().startswith("character"):
        raw_c1 = str(chars[0]["name"]).strip()
        # If the extracted name is basically the entire topic (or > 30 chars), clean it up
        if len(raw_c1) > 28:
            words = raw_c1.split()
            # If starts with 'The', take up to 2-3 words, or find primary entity
            if words[0].lower() == "the" and len(words) > 1:
                c1_name = " ".join(words[:2])
            else:
                c1_name = words[0]
        else:
            c1_name = raw_c1
    else:
        c1_name = "The Hero" 
    prop_name = props[0] if props else ""

    is_hindi = (caption_lang == "hi") or any("ऀ" <= ch <= "ॿ" for ch in topic)

    # ── Genre detection ────────────────────────────────────────────────────────
    genre = detect_genre(topic)
    arc = GENRE_ARCS.get(genre, GENRE_ARCS["generic"])

    # ── Build 5 raw scenes from arc beats ──────────────────────────────────────
    scenes_raw = []
    for idx, beat in enumerate(arc[:5]):
        raw = _build_arc_scene(beat, idx, c1_name, setting, prop_name, chars, story_info)
        cap_en = raw.pop("_cap_en")
        cap_hi = raw.pop("_cap_hi")
        raw["caption"] = cap_hi if is_hindi else cap_en
        raw["narration"] = cap_hi if is_hindi else cap_en
        scenes_raw.append(raw)

    # Pad to 5 if arc was shorter (shouldn't happen but guard anyway)
    generic_arc = GENRE_ARCS["generic"]
    while len(scenes_raw) < 5:
        idx = len(scenes_raw)
        beat = generic_arc[idx % len(generic_arc)]
        raw = _build_arc_scene(beat, idx, c1_name, setting, prop_name, chars, story_info)
        cap_en = raw.pop("_cap_en")
        cap_hi = raw.pop("_cap_hi")
        raw["caption"] = cap_hi if is_hindi else cap_en
        raw["narration"] = cap_hi if is_hindi else cap_en
        scenes_raw.append(raw)

    # ── Validate and auto-fix ──────────────────────────────────────────────────
    defects = validate_scene_plan(scenes_raw, story_info)
    if defects:
        scenes_raw = autofix_scene_plan(scenes_raw, story_info)

    validate_and_autofix_camera_plan(scenes_raw, story_info.get("props", []))

    # ── Format final scene dicts with full image prompts ───────────────────────
    formatted = []
    for idx, s in enumerate(scenes_raw):
        p = build_generic_scene_prompt(s, story_info, style=style)
        is_captions_off = str(caption_lang).lower().strip() in ("none", "", "off", "disabled", "false", "0")
        cap = "" if is_captions_off else clean_caption_language(sanitize_caption(s.get("caption", "")), caption_lang)
        if is_unwanted_caption_text(cap, p):
            cap = ""
        nar = sanitize_narration(s.get("narration") or (cap if cap else s.get("action", "")))
        if is_unwanted_caption_text(nar, p):
            nar = ""
        emo = s.get("emotion") or "calm"
        item = {
            "image_prompt": p,
            "visual_prompt": p,
            "caption": cap,
            "narration": nar,
            "camera": s.get("camera", "medium"),
            "must_show": s.get("must_show", []),
            "action": s.get("action", ""),
            "setting_detail": s.get("setting_detail", ""),
            "characters_present": s.get("characters_present", []),
            "emotion": emo,
            "expression": s.get("expression") or EMOTION_STYLE.get(emo, EMOTION_STYLE["calm"])["expression"],
            "camera_speed": s.get("camera_speed") or EMOTION_STYLE.get(emo, EMOTION_STYLE["calm"])["camera_speed"],
            "cues": s.get("cues", []),
            "sheets": story_info,
            "motion_instructions": s.get("motion_instructions") or build_scene_motion_instructions(s, story_info, style=style),
            "motion_prompt": s.get("motion_instructions") or build_scene_motion_instructions(s, story_info, style=style),
            "character_reference": story_info.get("character_reference_sheet", ""),
        }
        logger.info(
            f"[planner] Scene {idx+1} (genre={genre}): characters_present={item['characters_present']}, "
            f"camera={item['camera']}, setting_detail='{item['setting_detail']}', "
            f"emotion={item['emotion']}, cues={len(item['cues'])}"
        )
        formatted.append(item)
    return formatted



def _raw_plan_scenes(
    topic: str,
    length_seconds: int = 25,
    style: str = "3d_animated",
    caption_lang: str = "none",
    fallback_to_template: bool = True,
) -> List[Dict[str, Any]]:
    """Scene planning entrypoint with fail-safe priority chain:
    1. Anthropic API
    2. Google Gemini API
    3. Generic story fallback template
    """
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if anthropic_key:
        try:
            return plan_scenes_anthropic(topic, length_seconds, style, caption_lang, anthropic_key)
        except Exception as e:
            if not fallback_to_template:
                raise RuntimeError(f"Anthropic planning failed ({e}). Please retry.")
            print(f"[planner] Anthropic planning failed ({e}), checking Gemini API...")

    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if gemini_key:
        try:
            return plan_scenes_gemini(topic, length_seconds, style, caption_lang, gemini_key)
        except Exception as e:
            if not fallback_to_template:
                raise RuntimeError(f"Gemini planning failed ({e}). Please retry.")
            print(f"[planner] Gemini planning failed ({e}), falling back to template.")

    # No API key available or both failed - retry once with template, then fail
    last_err = None
    for attempt in range(2):
        try:
            return plan_scenes_template(topic, length_seconds, style, caption_lang)
        except Exception as e:
            last_err = e
            if attempt == 0:
                print(f"[planner] Template planning failed ({e}), retrying once...")

    reason = str(last_err)[:200] if last_err else "no LLM API key configured"
    raise RuntimeError(f"planner unavailable: {reason}")


def plan_scenes(
    topic: str,
    length_seconds: int = 25,
    style: str = "3d_animated",
    caption_lang: str = "none",
    fallback_to_template: bool = True,
) -> List[Dict[str, Any]]:
    scenes = _raw_plan_scenes(topic, length_seconds=length_seconds, style=style, caption_lang=caption_lang, fallback_to_template=fallback_to_template)
    explicit_lines = extract_explicit_devanagari_lines(topic)
    if explicit_lines and len(explicit_lines) == 4:
        story_info = extract_story_info(topic)
        scenes = enforce_explicit_shayari_plan(scenes, story_info, explicit_lines, style=style)
    return scenes

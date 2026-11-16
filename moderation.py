"""A basic prompt filter. It is a starting point, not full protection.

Before you let the public use the site, add a real moderation service, rate limits
per user, and a way to report videos. Do not allow videos of real, named people.
"""
import re

_BLOCKED = [
    r"nude", r"naked", r"porn\w*", r"sexual\w*", r"erotic", r"rape\w*",
    r"gore", r"beheading", r"mutilat\w*", r"torture",
    r"child\s+abuse", r"suicide", r"self[- ]harm",
    r"bomb\s+making", r"how\s+to\s+make\s+a\s+bomb",
]
_PATTERN = re.compile(r"\b(" + "|".join(_BLOCKED) + r")\b", re.IGNORECASE)


def check_prompt(prompt: str):
    """Return (True, "") if allowed, else (False, reason)."""
    if _PATTERN.search(prompt):
        return False, "This prompt is not allowed. Please describe a different scene."
    return True, ""

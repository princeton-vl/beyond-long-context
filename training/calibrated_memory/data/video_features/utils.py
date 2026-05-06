"""Utilities shared across video feature datasets and builders."""

from __future__ import annotations

UNCERTAIN_KEYWORDS = (
    "uncertain",
    "idk",
    "none of the",
    "none-of-the",
    "none/",
    "none of above",
    "none above",
)


def is_uncertain_option(option: dict) -> bool:
    """Best-effort check whether an option represents the uncertain/IDK choice."""

    if option.get("is_uncertain"):
        return True
    label = str(option.get("label", "")).lower()
    return any(keyword in label for keyword in UNCERTAIN_KEYWORDS)

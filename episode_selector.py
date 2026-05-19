"""
Episode selection helpers.  Import one selector and assign it to episode_files
in identify_speaker_names.main():

    from episode_selector import unprocessed_episodes
    episode_files = unprocessed_episodes()
"""
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional
import json

BASE_DIR     = Path(__file__).resolve().parent
EPISODES_DIR = BASE_DIR / "episodes"
MAPPINGS_DIR = BASE_DIR / "speaker_mappings"


# ── Private helpers ─────────────────────────────────────────────

def _episode_number(stem: str) -> Optional[int]:
    try:
        return int(stem.rsplit("_", 1)[-1])
    except (ValueError, IndexError):
        return None


def _sorted(stems: list) -> list:
    return sorted(stems, key=lambda s: (_episode_number(s) or 0))


def _load_mapping(stem: str) -> Optional[dict]:
    path = MAPPINGS_DIR / f"{stem}_speaker_mapping_v2.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _names_similar(a: str, b: str, threshold: float = 0.85) -> bool:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold


def _is_clean(mapping: dict, candidates: list) -> bool:
    resolved = {v for v in mapping.values() if v != "UNCLEAR"}
    if any(v == "UNCLEAR" or "_unresolved" in v for v in mapping.values()):
        return False
    return not any(
        c for c in candidates
        if not any(_names_similar(c, r) for r in resolved)
    )


# ── Public selectors ─────────────────────────────────────────────

def all_episodes() -> list:
    """Every episode found in episodes/."""
    return _sorted([f.stem for f in EPISODES_DIR.glob("episode_*.json")])


def unprocessed_episodes() -> list:
    """Episodes whose mapping file has no llm_candidates yet."""
    result = []
    for ep in all_episodes():
        data = _load_mapping(ep)
        if data is None or "llm_candidates" not in data:
            result.append(ep)
    return result


def episodes_by_number(start: int, end: Optional[int] = None) -> list:
    """Episodes with number >= start (and <= end if given).

    Examples:
        episodes_by_number(133)        # episode 133 onwards
        episodes_by_number(133, 150)   # episodes 133–150
    """
    result = []
    for ep in all_episodes():
        n = _episode_number(ep)
        if n is None:
            continue
        if end is None:
            if n >= start:
                result.append(ep)
        else:
            if start <= n <= end:
                result.append(ep)
    return result


def partial_assignment_episodes() -> list:
    """Episodes that were processed but did not get a clean assignment."""
    result = []
    for ep in all_episodes():
        data = _load_mapping(ep)
        if data is None or "llm_candidates" not in data:
            continue
        if not _is_clean(data.get("mapping", {}), data["llm_candidates"]):
            result.append(ep)
    return result

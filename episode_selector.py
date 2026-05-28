"""
Episode selection helpers.  Import one selector and assign it to episode_files
in identify_speaker_names.main():

    from episode_selector import unprocessed_episodes
    episode_files = unprocessed_episodes()

Path configuration — change PODCAST_DIR (and the file-path builders below if the
naming convention differs) when switching to a different podcast series.
"""
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional
import json

BASE_DIR        = Path(__file__).resolve().parent
PODCAST_DIR     = BASE_DIR / "podcast_series" / "technovation"
EPISODES_DIR    = PODCAST_DIR / "episodes"
TRANSCRIPTS_DIR = PODCAST_DIR / "processed_AI_TRANSCRIBE"
MAPPINGS_DIR    = PODCAST_DIR / "speaker_mappings"

# File-path builders — adjust the f-string patterns if the podcast uses different filenames
def episode_path(stem: str)    -> Path: return EPISODES_DIR    / f"{stem}.json"
def transcript_path(stem: str) -> Path: return TRANSCRIPTS_DIR / f"{stem}_utterances.jsonl"
def mapping_path(stem: str)    -> Path: return MAPPINGS_DIR    / f"{stem}_speaker_mapping_v2.json"


# ── Private helpers ─────────────────────────────────────────────

def _episode_number(stem: str) -> Optional[int]:
    try:
        return int(stem.rsplit("_", 1)[-1])
    except (ValueError, IndexError):
        return None


def _sorted(stems: list) -> list:
    return sorted(stems, key=lambda s: (_episode_number(s) or 0))


def _load_mapping(stem: str) -> Optional[dict]:
    path = mapping_path(stem)
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
    """Every episode found in EPISODES_DIR."""
    return _sorted([f.stem for f in EPISODES_DIR.glob("episode_*.json")])


def unprocessed_episodes() -> list:
    """Episodes whose mapping file has no llm_candidates yet."""
    result = []
    for ep in all_episodes():
        data = _load_mapping(ep)
        if data is None or "llm_candidates" not in data:
            result.append(ep)
    return result


def episodes_by_number(start: int | list[int], end: Optional[int] = None) -> list:
    """Intersection of the requested episodes and those that exist in EPISODES_DIR.

    Examples:
        episodes_by_number(133)          # episode 133 onwards
        episodes_by_number(133, 150)     # episodes 133–150
        episodes_by_number([3, 7, 42])   # exactly those episodes
    """
    existing = {_episode_number(ep): ep for ep in all_episodes()}

    if isinstance(start, list):
        target = set(start)
    elif end is None:
        target = {n for n in existing if n >= start}
    else:
        target = set(range(start, end + 1))

    return _sorted([existing[n] for n in target if n in existing])


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

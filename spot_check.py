"""
Spot-check clean speaker assignments by printing a review card for a
systematic sample of episodes. For each sampled episode it shows the
mapping, the episode description, and the opening utterances — enough
to confirm or refute the assignment without opening any files manually.

Run after identify_speaker_names.py has been executed for all episodes.
"""
from episode_selector import MAPPINGS_DIR, episode_path, transcript_path
from bs4 import BeautifulSoup
import json

SAMPLE_EVERY     = 12   # take every Nth clean episode
UTTERANCES_SHOWN = 30   # opening utterances shown per episode


def _is_clean(mapping: dict, candidates: list) -> bool:
    resolved_values = {v for v in mapping.values() if v != "UNCLEAR"}
    has_unresolved  = any(v == "UNCLEAR" for v in mapping.values())
    unaccounted     = [c for c in candidates if c not in resolved_values]
    return not has_unresolved and not unaccounted


def _load_description(episode: str) -> str:
    path = episode_path(episode)
    if not path.exists():
        return "(description not found)"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return BeautifulSoup(
        data.get("description", data.get("content", "")),
        "html.parser"
    ).get_text().strip()


def _load_utterances(episode: str, limit: int) -> list[dict]:
    path = transcript_path(episode)
    if not path.exists():
        return []
    utterances = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            line = line.strip()
            if line:
                utterances.append(json.loads(line))
    return utterances


def main():
    mapping_files = sorted(MAPPINGS_DIR.glob("*_speaker_mapping_v2.json"))

    clean_episodes = []
    for path in mapping_files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "llm_candidates" not in data:
            continue
        mapping    = data.get("mapping", {})
        candidates = data["llm_candidates"]
        if _is_clean(mapping, candidates):
            episode = path.stem.removesuffix("_speaker_mapping_v2")
            clean_episodes.append((episode, data))

    # Every Nth, always including first and last for range coverage
    indices = sorted({0, len(clean_episodes) - 1} | set(range(0, len(clean_episodes), SAMPLE_EVERY)))
    sample  = [clean_episodes[i] for i in indices]

    print("=" * 70)
    print(f"SPOT-CHECK REVIEW  —  {len(sample)} of {len(clean_episodes)} clean episodes sampled")
    print("=" * 70)

    for episode, data in sample:
        mapping    = data["mapping"]
        candidates = data["llm_candidates"]

        description = _load_description(episode)
        utterances  = _load_utterances(episode, UTTERANCES_SHOWN)

        print(f"\n{'─' * 70}")
        print(f"Episode    : {episode}")
        print(f"Mapping    : {json.dumps(mapping, ensure_ascii=False)}")
        print(f"Candidates : {candidates}")
        print(f"\nDescription:\n{description[:500]}{'...' if len(description) > 500 else ''}")
        print(f"\nFirst {len(utterances)} utterances:")
        for utt in utterances:
            print(f"  [{utt['start']:.1f}s] {utt['speaker']}: {utt['text']}")

    print(f"\n{'=' * 70}")
    print(f"End of spot-check  ({len(sample)} episodes reviewed)")


if __name__ == "__main__":
    main()

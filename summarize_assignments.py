"""
Reads all speaker mapping JSONs and prints a summary of clean vs non-clean
LLM assignments. For non-clean episodes it shows the episode number, the
current mapping, and the LLM candidates that were found.

Run after identify_speaker_names.py has been executed for all episodes.
"""
from pathlib import Path
import json

BASE_DIR      = Path(__file__).resolve().parent
MAPPINGS_DIR  = BASE_DIR / "speaker_mappings"


def is_clean_assignment(mapping: dict, candidates: list) -> bool:
    resolved_values = {v for v in mapping.values() if v != "UNCLEAR"}
    has_unresolved  = any(v == "UNCLEAR" for v in mapping.values())
    unaccounted     = [c for c in candidates if c not in resolved_values]
    return not has_unresolved and not unaccounted


def main():
    mapping_files = sorted(MAPPINGS_DIR.glob("*_speaker_mapping_v2.json"))

    clean        = []
    non_clean    = []
    not_processed = []

    for path in mapping_files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        episode = data.get("episode_name", path.stem)

        if "llm_candidates" not in data:
            not_processed.append(episode)
            continue

        mapping    = data.get("mapping", {})
        candidates = data["llm_candidates"]

        if is_clean_assignment(mapping, candidates):
            clean.append(episode)
        else:
            non_clean.append({
                "episode":    episode,
                "mapping":    mapping,
                "candidates": candidates,
            })

    # ── Summary ────────────────────────────────────────────────────
    total_processed = len(clean) + len(non_clean)
    print("=" * 60)
    print("ASSIGNMENT SUMMARY")
    print("=" * 60)
    print(f"Processed episodes : {total_processed}")
    print(f"  Clean            : {len(clean)}")
    print(f"  Non-clean        : {len(non_clean)}")
    if not_processed:
        print(f"Not yet processed  : {len(not_processed)}")
    print()

    if non_clean:
        print("-" * 60)
        print("NON-CLEAN EPISODES")
        print("-" * 60)
        for entry in non_clean:
            print(f"\nEpisode : {entry['episode']}")
            print(f"Mapping : {json.dumps(entry['mapping'], ensure_ascii=False)}")
            print(f"LLM candidates: {entry['candidates']}")

    if not_processed:
        print("\n" + "-" * 60)
        print("NOT YET PROCESSED")
        print("-" * 60)
        print(", ".join(not_processed))


if __name__ == "__main__":
    main()

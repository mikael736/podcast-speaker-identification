# ===============================================================
# IMPORTS
# ===============================================================
from episode_selector import (
    all_episodes,
    unprocessed_episodes,
    episodes_by_number,
    partial_assignment_episodes,
)
from pathlib import Path
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
from dataclasses import dataclass, field
from dotenv import load_dotenv
from google import genai
from google.genai import types
from typing import Optional
import json
import os
import re
import time


# ===============================================================
# PATHS
# ===============================================================
BASE_DIR = Path(__file__).resolve().parent


# ===============================================================
# CONFIGURATION
# ===============================================================
load_dotenv()
GEMINI_API_KEY      = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL        = "gemini-3.1-flash-lite"
DEFAULT_MAX_TOKENS  = 256
DEFAULT_TEMPERATURE = 0.3


# ===============================================================
# HELPER FUNCTIONS
# ===============================================================
@dataclass
class Utterance:
    episode_name: str
    segment_id: int
    speaker: str
    start: float
    end: float
    text: str
    words: list = field(default_factory=list)


def names_are_similar(a: str, b: str, threshold: float = 0.85) -> bool:
    """Return True if two name strings are close enough to be the same person."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold


def load_utterances_jsonl(path) -> list[Utterance]:
    """Read a .jsonl file and return a list of Utterance objects."""
    utterances = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                utterances.append(Utterance(**json.loads(line)))
    return utterances


def format_utterances_to_text(utterances: list[Utterance]) -> str:
    """Convert utterances to a readable transcript block for the LLM prompt."""
    return "\n".join(
        f"[{utt.start:.1f}s - {utt.end:.1f}s] {utt.speaker}: {utt.text}"
        for utt in utterances
    )


def shorten_utterance_list(
    utterances: list[Utterance],
    max_duration: float = 180.0,
    min_utterances_per_speaker: int = 2,
    intro_context_before: int = 3
) -> list[Utterance]:
    """Select a representative subset of utterances for the LLM prompt."""
    if not utterances:
        return []

    unique_speakers = set(utt.speaker for utt in utterances if utt.speaker)
    if not unique_speakers:
        raise ValueError("No speaker labels found in utterances — cannot select a representative sample.")

    selected = []
    speaker_counts = {speaker: 0 for speaker in unique_speakers}
    total_duration = 0.0

    # First pass: collect early utterances up to the duration limit
    for utt in utterances:
        utterance_duration = utt.end - utt.start
        if total_duration + utterance_duration > max_duration:
            break
        selected.append(utt)
        if utt.speaker:
            speaker_counts[utt.speaker] += 1
        total_duration += utterance_duration

    # Second pass: for under-represented speakers, grab their intro window.
    # This captures whoever introduced the speaker (typically just before their
    # first utterance) together with their first words — the LLM needs both to
    # assign the name.
    selected_ids = {id(u) for u in selected}
    for speaker in unique_speakers:
        if speaker_counts[speaker] >= min_utterances_per_speaker:
            continue

        first_idx = next((i for i, u in enumerate(utterances) if u.speaker == speaker), None)
        if first_idx is None:
            continue

        # Window: N utterances before the speaker's first appearance + their first M utterances
        context_start = max(0, first_idx - intro_context_before)
        window = utterances[context_start : first_idx + min_utterances_per_speaker]

        for utt in window:
            if id(utt) not in selected_ids:
                selected.append(utt)
                selected_ids.add(id(utt))
                if utt.speaker:
                    speaker_counts[utt.speaker] += 1

    selected.sort(key=lambda x: x.start)
    return selected


def _parse_name(raw: str) -> Optional[str]:
    """Strip noise from a raw LLM name token and validate it. Returns None if invalid."""
    name = re.sub(r'^[-*\s"\']+', '', raw.strip()).rstrip("'\"")
    if not name or name.upper() == "UNCLEAR" or name == "None":
        return None
    return name


def _extract_candidates(
    episode_description: str,
    model_name: str = GEMINI_MODEL,
    api_key: str = GEMINI_API_KEY,
) -> list[str]:
    """Step 1: extract participant names from the episode description only."""
    system_prompt = (
        "You are an assistant that extracts the names of podcast participants from episode descriptions."
    )
    user_prompt = (
        "List the full names of all people who participate as speakers in this podcast episode. "
        "Only include actual participants — not people merely referenced, quoted, or discussed. "
        "Return only the person's name — no job titles, roles, company names, or any other description.\n\n"
        f"Episode Description:\n{episode_description}\n\n"
        "Respond with only a Python list, e.g. ['Jason Foster', 'Sarah Johnson'] "
        "or [] if no names are found. No other text."
    )
    try:
        client = genai.Client(api_key=api_key)
        result = client.models.generate_content(
            model=model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.1,
                max_output_tokens=200,
            ),
        )
        response = result.text.strip()
        print(f"Candidates from description: {response}")
    except Exception as e:
        raise RuntimeError(f"Gemini API call failed (step 1): {e}")

    match = re.search(r'\[([^\]]*)\]', response)
    if not match:
        return []
    content = match.group(1).strip()
    if not content:
        return []
    return [
        name for raw in content.split(',')
        if (name := _parse_name(raw)) is not None
    ]


def identify_speakers(
    utterances: list[Utterance],
    episode_description: str,
    model_name: str = GEMINI_MODEL,
    max_new_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    api_key: str = GEMINI_API_KEY
) -> Optional[tuple[dict[str, str], list[str]]]:
    """
    Two-step speaker identification:
      1. Extract candidate names from the episode description.
      2. Assign those candidates to SPEAKER_XX labels using transcript excerpts.
    Returns (speaker_mapping, candidates) or None if utterances are empty.
    """
    if not utterances:
        return None

    # "UNKNOWN" is a diarization artifact label, not a real speaker — exclude it
    unique_speakers = sorted(
        set(utt.speaker for utt in utterances if utt.speaker and utt.speaker != "UNKNOWN")
    )
    if not unique_speakers:
        return None

    # Step 1 — extract candidate names from description
    print("Step 1: extracting candidate names from description...")
    candidates = _extract_candidates(episode_description, model_name=model_name, api_key=api_key)
    if not candidates:
        print("No candidates found in description — all speakers marked UNCLEAR.")
        return ({s: "UNCLEAR" for s in unique_speakers}, [])

    # Step 2 — assign candidates to speaker labels using transcript
    sample_utterances = shorten_utterance_list(utterances)
    print(f"Step 2: assigning speakers. Using {len(sample_utterances)} of {len(utterances)} utterances.")

    sample_text    = format_utterances_to_text(sample_utterances)
    speaker_lines  = "\n".join(f"{s}: [name from list or UNCLEAR]" for s in unique_speakers)
    candidates_str = ", ".join(candidates)

    system_prompt = (
        "You are an assistant that matches diarization speaker labels to named participants "
        "in a podcast transcript."
    )
    user_prompt = f"""Assign each speaker label to the correct person from the participants list.

Participants: [{candidates_str}]

Transcript Excerpts:
{sample_text}

Respond ONLY in this exact format:
{speaker_lines}

Rules:
- Use only names from the Participants list, written exactly as given. Do not introduce new names.
- Use UNCLEAR if you cannot confidently match a speaker to any participant.
- If a speaker is clearly acting as a podcast host or interviewer but their name is not in the Participants list, use 'host'. Multiple speakers may be labeled 'host'.
- If a speaker is clearly a podcast intro/outro announcer, use 'Intro/Outro Voice'.

No explanation, no extra text."""

    print(f"Querying Gemini ({model_name}) for speaker assignment...")
    try:
        client = genai.Client(api_key=api_key)
        result = client.models.generate_content(
            model=model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                max_output_tokens=max_new_tokens,
            ),
        )
        response = result.text.strip()
        print(f"Assignment response: {response}")
    except Exception as e:
        raise RuntimeError(f"Gemini API call failed (step 2): {e}")

    # Parse SPEAKER_XX → name
    speaker_mapping = {}
    for speaker in unique_speakers:
        match = re.search(rf"{re.escape(speaker)}\s*:\s*([^\n]+)", response, re.IGNORECASE)
        if match:
            name = _parse_name(match.group(1))
            speaker_mapping[speaker] = name if name else "UNCLEAR"
            if not name:
                print(f"Speaker {speaker}: UNCLEAR")
        else:
            speaker_mapping[speaker] = "UNCLEAR"
            print(f"Could not find mapping for {speaker} in response")

    return (speaker_mapping, candidates) if speaker_mapping else None


# ===============================================================
# MAIN
# ===============================================================
def process_single_episode(utterances_path, episode_json_path, mapping_path):
    """Identify speakers and update the speaker mapping JSON."""
    print(f"Loading utterances from {utterances_path}")
    utterances = load_utterances_jsonl(utterances_path)

    print(f"Loading episode description from {episode_json_path}")
    with open(Path(episode_json_path), "r", encoding="utf-8") as f:
        episode_data = json.load(f)
    episode_description = BeautifulSoup(
        episode_data.get("description", episode_data.get("content", "")),
        "html.parser"
    ).get_text().strip()
    print(f"Episode description loaded ({len(episode_description)} characters)")

    print(f"Loading existing mapping from {mapping_path}")
    with open(Path(mapping_path), "r", encoding="utf-8") as f:
        mapping_data = json.load(f)

    print("\nIdentifying speakers...")
    result = identify_speakers(utterances, episode_description)
    if result is None:
        print("Speaker identification failed. Mapping file will not be updated.")
        return

    speaker_mapping, candidates = result

    # Auto-assign: if exactly 1 unclear speaker and 1 unaccounted candidate, assign them
    _resolved = {v for v in speaker_mapping.values() if v not in ("UNCLEAR", "Intro/Outro Voice")}
    _unclear  = [k for k, v in speaker_mapping.items() if v == "UNCLEAR"]
    _leftover = [c for c in candidates if not any(names_are_similar(c, r) for r in _resolved)]
    if len(_unclear) == 1 and len(_leftover) == 1:
        speaker_mapping[_unclear[0]] = _leftover[0]
        print(f"Auto-assigned: {_unclear[0]} → {_leftover[0]}")

    resolved   = {k: v for k, v in speaker_mapping.items() if v != "UNCLEAR"}
    unresolved = [k for k, v in speaker_mapping.items() if v == "UNCLEAR"]

    # is_clean: no unresolved speakers AND every candidate accounts for a known speaker
    unaccounted = [
        c for c in candidates
        if not any(names_are_similar(c, r) for r in resolved.values())
    ]
    is_clean = len(unresolved) == 0 and len(unaccounted) == 0

    # Always write candidates to the JSON for full transparency
    mapping_data["llm_candidates"] = candidates

    if is_clean:
        print(f"Clean assignment: {resolved}")
        mapping_data["mapping"] = resolved
    else:
        print(f"Partial assignment — {len(unresolved)} unresolved, {len(unaccounted)} unaccounted candidate(s).")

    if candidates:
        print(f"Candidates: {candidates}")

    with open(Path(mapping_path), "w", encoding="utf-8") as f:
        json.dump(mapping_data, f, indent=2, ensure_ascii=False)
    print(f"Mapping saved to {mapping_path}")


def main():
    print("=" * 60)
    print("SPEAKER IDENTIFICATION PROCESSOR")
    print("=" * 60)

    # Swap the selector to control which episodes are processed:
    #   all_episodes()                  – every episode
    #   unprocessed_episodes()          – no llm_candidates yet (default)
    #   episodes_by_number(133)         – episode 133 onwards
    #   episodes_by_number(133, 150)    – episodes 133–150
    #   partial_assignment_episodes()   – processed but not cleanly assigned
    episode_files = episodes_by_number(203,204)

    for episode in episode_files:
        print(f"\n--- Episode {episode} ---")
        for attempt in range(3):
            try:
                process_single_episode(
                    utterances_path=BASE_DIR / "processed_AI_TRANSCRIBE" / f"{episode}_utterances.jsonl",
                    episode_json_path=BASE_DIR / "episodes"              / f"{episode}.json",
                    mapping_path=BASE_DIR     / "speaker_mappings"       / f"{episode}_speaker_mapping_v2.json"
                )
                break
            except RuntimeError as e:
                if attempt == 2:
                    print(f"Skipping {episode} after 3 failed attempts: {e}")
                else:
                    print(f"Attempt {attempt + 1} failed, retrying in 15s... ({e})")
                    time.sleep(15)
        time.sleep(4)  # stay under 15 RPM (2 calls/episode × ~2s latency + 4s = ~8s/episode)

if __name__ == "__main__":
    main()

# ===============================================================
# IMPORTS
# ===============================================================
from pathlib import Path
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
from dataclasses import dataclass, field
from dotenv import load_dotenv
from groq import Groq
from typing import Optional
import json
import os
import re


# ===============================================================
# PATHS
# ===============================================================
BASE_DIR = Path(__file__).resolve().parent


# ===============================================================
# CONFIGURATION
# ===============================================================
load_dotenv()
GROQ_API_KEY        = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL          = "llama-3.3-70b-versatile"
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
    name = re.sub(r'^[-*\s]+', '', raw.strip())
    if not name or name.upper() == "UNCLEAR" or name == "None":
        return None
    if not re.match(r'^[A-Za-z\s\-\./]{2,50}$', name):
        return None
    return name


def identify_speakers(
    utterances: list[Utterance],
    episode_description: str,
    model_name: str = GROQ_MODEL,
    max_new_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    api_key: str = GROQ_API_KEY
) -> Optional[tuple[dict[str, str], list[str]]]:
    """
    Query the LLM to map SPEAKER_XX labels to names.
    Returns (speaker_mapping, candidates) where:
      - speaker_mapping: {SPEAKER_XX: name or 'UNCLEAR'}
      - candidates: all person names found (assigned or not), for transparency
    Returns None if utterances are empty.
    """
    if not utterances:
        return None

    unique_speakers = sorted(set(utt.speaker for utt in utterances if utt.speaker))
    if not unique_speakers:
        return None

    sample_utterances = shorten_utterance_list(utterances)
    print(f"Using {len(sample_utterances)} utterances (out of {len(utterances)}) for speaker identification")

    sample_text    = format_utterances_to_text(sample_utterances)
    speaker_lines  = "\n".join(f"{s}: [name or UNCLEAR]" for s in unique_speakers)
    format_example = f"{speaker_lines}\nCANDIDATES: [Name1, Name2, ...]"

    system_prompt = (
        "You are a helpful assistant that identifies speaker names from podcast transcripts. "
        "Be critical and only identify speakers if their names are clearly mentioned in the episode "
        "description or transcript. If you cannot confidently identify a speaker, use 'UNCLEAR'."
    )

    user_prompt = f"""Given the following episode description and sample transcript excerpts, identify the real names of the speakers.

Episode Description:
{episode_description}

Sample Transcript Excerpts:
{sample_text}

Respond ONLY in this exact format:
{format_example}

Rules:
- Assign each speaker using the name as written in the episode description where possible.
- One speaker may be a podcast intro/outro announcer (formal welcome, sponsor reads, sign-off). Label them 'Intro/Outro Voice', especially if all other speakers are already identified.
- If a speaker cannot be confidently identified, use 'UNCLEAR'.
- CANDIDATES must always be a Python list with square brackets containing only proper person names, e.g. CANDIDATES: [Pete Williams, Jason Foster] or CANDIDATES: [] if no names were found. List every person name found in the description or transcript, whether assigned or not, including spelling variants. Do not include explanations, notes, or any text that is not a person's name.

No explanation, no bullet points, no extra text."""

    print(f"Querying Groq ({model_name})...")
    try:
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        response = completion.choices[0].message.content.strip()
        print(f"Model response: {response}")
    except Exception as e:
        raise RuntimeError(f"Groq API call failed: {e}")

    # Parse SPEAKER_XX → name mappings
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

    # Parse CANDIDATES: [...] — all person names found
    candidates = []
    candidates_match = re.search(r'CANDIDATES:\s*\[?([^\]\n]*)\]?', response, re.IGNORECASE)
    if candidates_match:
        content = candidates_match.group(1).strip()
        if content:
            candidates = [
                name for raw in content.split(',')
                if (name := _parse_name(raw)) is not None
            ]

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

    episode_files = [
        f.stem  # e.g. "episode_4" from "episode_4.json"
        for f in (BASE_DIR / "episodes").iterdir()
        if f.suffix == ".json"
    ]

    for episode in episode_files:
        print(f"\n--- Episode {episode} ---")
        process_single_episode(
            utterances_path=BASE_DIR / "processed_AI_TRANSCRIBE" / f"{episode}_utterances.jsonl",
            episode_json_path=BASE_DIR / "episodes"              / f"{episode}.json",
            mapping_path=BASE_DIR     / "speaker_mappings"       / f"{episode}_speaker_mapping_v2.json"
        )

if __name__ == "__main__":
    main()

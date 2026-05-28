# podcast-speaker-identification

Resolves generic diarization labels (e.g. `SPEAKER_00`) to real person
names for a batch of podcast episodes, using a two-step LLM pipeline
via the Gemini API.

## How it works

1. **Load** – reads per-episode utterances (`.jsonl`) and episode
   metadata (`.json`).
2. **Extract candidates** – sends the episode title and description to
   Gemini and asks it to list the names of all participants (step 1).
3. **Sample** – selects a representative subset of utterances (≤ 180 s
   of audio, with intro-context for under-represented speakers) to keep
   the prompt short.
4. **Assign** – sends the sampled transcript together with the candidate
   list to Gemini and asks it to map each `SPEAKER_XX` label to a name
   from that list (step 2).
5. **Validate** – a mapping is written to the speaker mapping JSON only
   when all speakers are resolved and every candidate name is accounted
   for ("clean" assignment). Partial results are logged but not committed
   to `mapping`.
6. **Audit trail** – the raw candidate list is always saved to
   `llm_candidates` in the mapping file for transparency.

## Directory layout

```
identify_speaker_names/
├── identify_speaker_names.py    # main processing script
├── episode_selector.py          # path config + helpers for selecting which episodes to process
├── summarize_assignments.py     # prints a clean/non-clean summary across all episodes
├── spot_check.py                # samples clean assignments for manual review
└── podcast_series/              # one folder per podcast series (gitignored)
    └── <podcast_name>/
        ├── episodes/            # episode metadata JSON files
        ├── processed_AI_TRANSCRIBE/  # utterance JSONL files from diarization
        └── speaker_mappings/    # output: one JSON mapping file per episode
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install google-genai python-dotenv beautifulsoup4
```

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_key_here
```

## Usage

**Process episodes:**
```bash
python identify_speaker_names.py
```

The episode selector near the bottom of `main()` controls which episodes
are processed. Swap the active line to target a specific range or subset:

```python
all_episodes()                   # every episode
unprocessed_episodes()           # no llm_candidates yet (default)
episodes_by_number(133)          # episode 133 onwards
episodes_by_number(133, 150)     # episodes 133–150
partial_assignment_episodes()    # processed but not cleanly assigned
```

**Review overall assignment quality:**
```bash
python summarize_assignments.py
```

**Spot-check a sample of clean assignments:**
```bash
python spot_check.py
```

Prints a review card (mapping, description, opening utterances) for
every 12th clean episode. Pipe to a file to scroll through comfortably:

```bash
python spot_check.py > review.txt
```

## Output format

Each `speaker_mappings/episode_N_speaker_mapping_v2.json` gains two fields:

| Field | Description |
|---|---|
| `mapping` | `{SPEAKER_XX: name}` — only present when all speakers are cleanly resolved |
| `llm_candidates` | All participant names extracted from the episode description |

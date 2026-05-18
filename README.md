# podcast-speaker-identification

Resolves generic diarization labels (e.g. `SPEAKER_00`) to real person names for a batch of podcast episodes, using an LLM via the Groq API.

## How it works

1. **Load** – reads per-episode utterances (`.jsonl`) and episode metadata (`.json`).
2. **Sample** – selects a representative subset of utterances (≤ 180 s of audio, with intro-context for under-represented speakers) to keep the prompt short.
3. **Identify** – sends the sampled transcript and episode description to Groq (`llama-3.3-70b-versatile`) and asks it to map each `SPEAKER_XX` label to a name.
4. **Validate** – a mapping is written to the speaker mapping JSON only when all speakers are resolved and every candidate name is accounted for ("clean" assignment). Partial results are logged but not committed to `mapping`.
5. **Audit trail** – the raw list of candidate names found by the LLM is always saved to `llm_candidates` in the mapping file for transparency.

## Directory layout

```
identify_speaker_names/
├── identify_speaker_names.py       # main script
├── episodes/                       # episode metadata JSON files (gitignored)
├── processed_AI_TRANSCRIBE/        # utterance JSONL files from diarization (gitignored)
└── speaker_mappings/               # output: one JSON mapping file per episode (gitignored)
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install groq python-dotenv beautifulsoup4
```

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
```

## Usage

```bash
python identify_speaker_names.py
```

The script iterates over every `episodes/episode_*.json` file and processes the corresponding utterances and mapping files.

## Output format

Each `speaker_mappings/episode_N_speaker_mapping_v2.json` gains two fields:

| Field | Description |
|---|---|
| `mapping` | `{SPEAKER_XX: name}` — only present when all speakers are cleanly resolved |
| `llm_candidates` | All person names the LLM found (assigned or not) |

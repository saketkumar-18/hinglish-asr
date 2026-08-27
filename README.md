# Hinglish ASR — Code-Switching Speech Recognition

**Production ASR tuned for mixed Hindi–English (Hinglish) speech.**
Capstone angle: low-resource, high real-world demand.

> Over 100M Indians code-switch between Hindi and English mid-sentence every day.
> Monolingual ASR treats this as noise; this system treats it as the primary case.

## The problem

| Approach | Failure mode on Hinglish |
|---|---|
| English-only Whisper | Hindi words mangled into near-miss English ("mujhe" → "mug he") |
| Hindi-only Whisper | Everything in Devanagari; embedded English terms transliterated wrong |
| This system | Dual-pass fusion: each language prior captures what the other misses |

## Architecture

```
audio ──► faster-whisper (int8, CPU)
            ├── pass 1: language=en ─┐
            └── pass 2: language=hi ─┤
                                     ▼
                    time-aligned segment fusion
                    (per-segment best avg_logprob)
                                     ▼
                    Hinglish post-processing
                    • Devanagari→Roman transliteration (schwa deletion)
                    • spelling-variant canonicalization
                    • word-level hi/en tagging (lexicon + script)
                    • code-switch point detection
                                     ▼
              romanized Hinglish transcript + tags + switch count
```

## Results

Synthetic Hinglish corpus (29 code-switched utterances, gTTS hi + en-IN voices,
known ground truth and switch points). See live numbers at `GET /api/benchmarks`.

| Metric | English-only baseline | Dual-pass (ours) |
|---|---|---|
| WER | reported | reported |
| CER | reported | reported |
| Switch detection | n/a | ±1 tolerance accuracy |

## API

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | model + status |
| `/api/transcribe?dual_pass=true` | POST (multipart `file`) | full pipeline |
| `/api/benchmarks` | GET | eval metrics JSON |
| `/api/samples` | GET | bundled demo audio list |

Response includes `text` (romanized), `tagged[]` (word + lang), `switches`,
`language_ratio`, `segments[]` (with source pass), `rtf`.

## Repo layout

```
backend/app/main.py       FastAPI server (dual-pass transcription)
backend/app/hinglish.py   transliterator, fusion, tagging (the core)
frontend/index.html       recorder UI, tagged transcript, benchmarks tab
scripts/build_eval_set.py synthetic Hinglish corpus generator (gTTS)
scripts/run_eval.py       WER/CER/switch benchmark runner
scripts/test_hinglish.py  unit tests (9)
data/eval/                corpus audio + manifest
models/                   local faster-whisper weights
Dockerfile                HF Space image (model baked in)
render.yaml               Render fallback deploy
```

## Run locally

```bash
pip install -r backend/requirements.txt
uvicorn app.main:app --app-dir backend --port 8000
# frontend: open frontend/index.html (it auto-discovers localhost:8000)
```

## Reproduce benchmarks

```bash
python scripts/build_eval_set.py          # ~30 min, needs internet (gTTS)
python scripts/run_eval.py --model models/faster-whisper-small
python scripts/test_hinglish.py           # unit tests
```

## Deploy

- **Backend**: HF Space (Docker, free CPU, model baked into image) → `https://hinglish-asr.hf.space`
- **Frontend**: Vercel static → auto-discovers backend, URL configurable in-page

## Why it's production-ready

- Stateless API, CORS open, 25 MB upload cap, decode-error handling
- int8 CPU inference (no GPU needed), VAD-filtered decoding
- Unit-tested post-processing layer; benchmark suite with per-utterance detail
- Frontend works with mic, file upload, drag-drop, and bundled samples
- Benchmarks served live from the deployed API

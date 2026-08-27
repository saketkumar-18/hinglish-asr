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
                    • EN pass is default (best romanized spelling)
                    • HI pass wins when EN TRANSLATES instead of
                      transcribing (low word-recovery score)
                    • trailing Hindi-only segments appended
                                     ▼
                    Hinglish post-processing
                    • Devanagari/Urdu→Roman transliteration (schwa deletion)
                    • spelling-variant canonicalization
                    • word-level hi/en tagging (lexicon + script)
                    • code-switch point detection
                                     ▼
              romanized Hinglish transcript + tags + switch count
```

## Results

Synthetic Hinglish corpus (29 code-switched utterances, gTTS hi + en-IN voices,
known ground truth and switch points). Model: faster-whisper **small**, int8 CPU.
See live numbers at `GET /api/benchmarks`.

| Metric | English-only | Hindi-only | Dual-pass (ours) |
|---|---|---|---|
| WER | 0.557 | 0.566 | **0.461** (−17%) |
| CER | 0.334 | 0.228 | **0.163** (−51% vs EN) |
| Switch detection | n/a | n/a | 0.759 (±1 tolerance) |

The dual-pass system beats both monolingual baselines on WER and CER. The key
insight: Whisper's English pass often *translates* Hindi speech into English
("main office se ghar aa raha hoon" → "i am coming home from the office");
word-recovery scoring detects this and defers to the Hindi pass for those spans.

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
Dockerfile                Render image (base model baked in at build)
Dockerfile.hf             HF Space image (small model, boot-time download)
render.yaml               Render blueprint
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

- **Backend**: Render free tier (Docker, base model baked into image) → `https://hinglish-asr.onrender.com`
- **Frontend**: Vercel static → `https://hinglish-asr.vercel.app` (auto-discovers backend, URL configurable in-page)
- `Dockerfile.hf` is an alternate HF Space image (small model, downloaded at boot) if you prefer HF hosting.

## Why it's production-ready

- Stateless API, CORS open, 25 MB upload cap, decode-error handling
- int8 CPU inference (no GPU needed), VAD-filtered decoding
- Unit-tested post-processing layer; benchmark suite with per-utterance detail
- Frontend works with mic, file upload, drag-drop, and bundled samples
- Benchmarks served live from the deployed API

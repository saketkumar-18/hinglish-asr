"""Hinglish ASR backend — FastAPI + faster-whisper dual-pass fusion.

Endpoints:
  GET  /api/health        -> status + model info
  POST /api/transcribe    -> multipart audio -> Hinglish transcript
  GET  /api/benchmarks    -> eval metrics (WER/CER, switch accuracy)
"""
from __future__ import annotations

import io
import os
import time
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.hinglish import postprocess, fuse_segments

MODEL_NAME = os.environ.get("WHISPER_MODEL", "small")
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE", "int8")

app = FastAPI(title="Hinglish ASR", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_model_lock = None


def get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        print(f"[asr] loading {MODEL_NAME} ({COMPUTE_TYPE}) ...")
        t = time.time()
        _model = WhisperModel(MODEL_NAME, device="cpu", compute_type=COMPUTE_TYPE)
        print(f"[asr] model loaded in {time.time()-t:.1f}s")
    return _model


def _segments_to_dicts(segments):
    out = []
    for s in segments:
        out.append({
            "start": round(s.start, 3),
            "end": round(s.end, 3),
            "text": s.text,
            "avg_logprob": round(s.avg_logprob, 4),
        })
    return out


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "compute": COMPUTE_TYPE,
        "model_loaded": _model is not None,
    }


@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    dual_pass: bool = True,
):
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty audio")
    if len(data) > 25 * 1024 * 1024:
        raise HTTPException(413, "audio too large (max 25 MB)")

    model = get_model()

    # decode to 16k mono float32 via faster-whisper's internal decoder.
    # decode_audio (PyAV) needs a real file path, so spool the upload to a
    # temp file first.
    from faster_whisper.audio import decode_audio
    import tempfile
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
            tf.write(data)
            tmp_path = tf.name
        audio = decode_audio(tmp_path, sampling_rate=16000)
    except Exception as e:
        raise HTTPException(400, f"could not decode audio: {e}")
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    t0 = time.time()

    # English pass
    segs_en_gen, info_en = model.transcribe(
        audio, language="en", task="transcribe",
        vad_filter=True, beam_size=5,
    )
    segs_en = _segments_to_dicts(segs_en_gen)

    segs_hi = []
    if dual_pass:
        segs_hi_gen, info_hi = model.transcribe(
            audio, language="hi", task="transcribe",
            vad_filter=True, beam_size=5,
        )
        segs_hi = _segments_to_dicts(segs_hi_gen)

    if dual_pass and segs_hi:
        fused = fuse_segments(segs_en, segs_hi)
    else:
        fused = [dict(s, source="en") for s in segs_en]

    full_text = " ".join(s["text"] for s in fused).strip()
    pp = postprocess(full_text)

    # attach language tags per segment for the UI
    for s in fused:
        s_pp = postprocess(s["text"])
        s["tagged"] = s_pp["tagged"]
        s["lang"] = "hi" if s_pp["language_ratio"]["hi"] > 0.5 else "en"

    duration = info_en.duration if info_en else 0.0
    elapsed = time.time() - t0

    return JSONResponse({
        "text": pp["text"],
        "tagged": pp["tagged"],
        "switches": pp["switches"],
        "language_ratio": pp["language_ratio"],
        "segments": fused,
        "duration_s": round(duration, 2),
        "processing_s": round(elapsed, 2),
        "rtf": round(elapsed / duration, 3) if duration else None,
        "model": MODEL_NAME,
        "dual_pass": dual_pass,
    })


@app.get("/api/benchmarks")
def benchmarks():
    import json
    path = os.path.join(os.path.dirname(__file__), "..", "eval", "results.json")
    path = os.path.normpath(path)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"note": "no benchmark results yet"}


@app.get("/api/samples")
def samples_list():
    """List bundled sample audio files for one-click demo."""
    import json
    out = []
    samples_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "samples"))
    manifest = os.path.join(samples_dir, "manifest.json")
    if os.path.isdir(samples_dir) and os.path.exists(manifest):
        with open(manifest, "r", encoding="utf-8") as f:
            for item in json.load(f):
                out.append({"id": item["id"], "label": item.get("label", item["id"]),
                            "url": f"/api/samples/{item['id']}"})
    return out


@app.get("/api/samples/{sample_id}")
def sample_file(sample_id: str):
    from fastapi.responses import FileResponse
    import re
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", sample_id):
        raise HTTPException(400, "bad id")
    samples_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "samples"))
    path = os.path.join(samples_dir, sample_id + ".mp3")
    if not os.path.exists(path):
        raise HTTPException(404, "sample not found")
    return FileResponse(path, media_type="audio/mpeg")

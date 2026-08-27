"""Run the Hinglish ASR benchmark over the synthetic eval set.

Scores three configurations per utterance:
  1. en_only   — single English-pass Whisper (baseline)
  2. hi_only   — single Hindi-pass Whisper (baseline)
  3. dual      — dual-pass fusion + Hinglish post-processing (ours)

Metrics: WER, CER (jiwer), code-switch detection accuracy.
Writes backend/eval/results.json (served by GET /api/benchmarks).

Usage:
  python scripts/run_eval.py --model <path-or-name> [--limit N] [--workers 1]
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))

from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
import jiwer

from app.hinglish import postprocess, fuse_segments, transliterate, tag_words, count_switches

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(ROOT, "data", "eval")


def norm_text(s: str) -> str:
    """Normalize for scoring: lowercase, strip punctuation, collapse spaces."""
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def segs_to_dicts(segments):
    return [{"start": round(s.start, 3), "end": round(s.end, 3),
             "text": s.text, "avg_logprob": round(s.avg_logprob, 4)} for s in segments]


def transcribe_pass(model, audio, lang):
    segs, info = model.transcribe(audio, language=lang, task="transcribe",
                                  vad_filter=True, beam_size=5)
    return segs_to_dicts(segs), info


def hyp_from(segs, do_translit=False):
    text = " ".join(s["text"] for s in segs).strip()
    if do_translit:
        text = transliterate(text)
    return postprocess(text)["text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "faster-whisper-small"))
    ap.add_argument("--compute", default="int8")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    with open(os.path.join(EVAL_DIR, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    if args.limit:
        manifest = manifest[: args.limit]

    print(f"[eval] loading model {args.model} ...")
    t = time.time()
    model = WhisperModel(args.model, device="cpu", compute_type=args.compute)
    print(f"[eval] loaded in {time.time()-t:.1f}s")

    per_item = []
    agg = {"wer_en": [], "wer_hi": [], "wer_dual": [],
           "cer_en": [], "cer_hi": [], "cer_dual": [],
           "switch_ok": 0, "switch_total": 0}
    total_audio = 0.0
    total_proc = 0.0

    for item in manifest:
        path = os.path.join(EVAL_DIR, item["audio"])
        truth = norm_text(item["truth"])
        audio = decode_audio(path, sampling_rate=16000)
        t0 = time.time()

        segs_en, info = transcribe_pass(model, audio, "en")
        segs_hi, _ = transcribe_pass(model, audio, "hi")

        hyp_en = norm_text(hyp_from(segs_en))
        hyp_hi = norm_text(hyp_from(segs_hi, do_translit=True))
        fused = fuse_segments(segs_en, segs_hi)
        hyp_dual = norm_text(postprocess(" ".join(s["text"] for s in fused))["text"])

        proc = time.time() - t0
        dur = info.duration or 1.0
        total_audio += dur
        total_proc += proc

        def wer_cer(hyp):
            if not truth:
                return None, None
            if not hyp:
                return 1.0, 1.0
            return jiwer.wer(truth, hyp), jiwer.cer(truth, hyp)

        wer_e, cer_e = wer_cer(hyp_en)
        wer_h, cer_h = wer_cer(hyp_hi)
        wer_d, cer_d = wer_cer(hyp_dual)

        # switch detection on dual hypothesis
        pred_switches = count_switches(tag_words(hyp_dual))
        gt_switches = item["switches"]
        ok = abs(pred_switches - gt_switches) <= 1  # tolerant: ±1 switch
        agg["switch_ok"] += int(ok)
        agg["switch_total"] += 1

        for k, v in [("wer_en", wer_e), ("wer_hi", wer_h), ("wer_dual", wer_d),
                     ("cer_en", cer_e), ("cer_hi", cer_h), ("cer_dual", cer_d)]:
            if v is not None:
                agg[k].append(min(v, 1.0))

        per_item.append({
            "id": item["id"], "truth": item["truth"],
            "hyp_en": hyp_en, "hyp_hi": hyp_hi, "hyp": hyp_dual,
            "wer": round(wer_d, 4) if wer_d is not None else None,
            "wer_en_only": round(wer_e, 4) if wer_e is not None else None,
            "switches_gt": gt_switches, "switches_pred": pred_switches,
            "switch_ok": ok,
        })
        print(f"[{item['id']}] WER dual={wer_d:.3f} en={wer_e:.3f} hi={wer_h:.3f} "
              f"switches {gt_switches}->{pred_switches} ({'ok' if ok else 'MISS'}) [{proc:.1f}s]")

    def mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    results = {
        "meta": {
            "model": os.path.basename(os.path.normpath(args.model)),
            "compute": args.compute,
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_utterances": len(per_item),
            "audio_s": round(total_audio, 1),
            "processing_s": round(total_proc, 1),
            "rtf": round(total_proc / total_audio, 3) if total_audio else None,
        },
        "metrics": {
            "wer_dual": mean(agg["wer_dual"]),
            "cer_dual": mean(agg["cer_dual"]),
            "wer_en_only": mean(agg["wer_en"]),
            "cer_en_only": mean(agg["cer_en"]),
            "wer_hi_only": mean(agg["wer_hi"]),
            "cer_hi_only": mean(agg["cer_hi"]),
            "switch_accuracy": round(agg["switch_ok"] / agg["switch_total"], 4) if agg["switch_total"] else None,
            "switch_tolerance": "+/-1 switch",
        },
        "per_item": per_item,
    }

    out_path = os.path.join(ROOT, "backend", "eval", "results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    # also keep a copy in reports/
    rep = os.path.join(ROOT, "reports", f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(rep, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    m = results["metrics"]
    print("\n===== RESULTS =====")
    print(f"WER  dual={m['wer_dual']}  en-only={m['wer_en_only']}  hi-only={m['wer_hi_only']}")
    print(f"CER  dual={m['cer_dual']}  en-only={m['cer_en_only']}  hi-only={m['cer_hi_only']}")
    print(f"Switch accuracy: {m['switch_accuracy']} (tolerance ±1)")
    print(f"RTF: {results['meta']['rtf']} on CPU int8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

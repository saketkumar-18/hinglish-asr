"""Pick 4 eval utterances as bundled demo samples for the frontend."""
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL = os.path.join(ROOT, "data", "eval")
OUT = os.path.join(ROOT, "backend", "samples")
os.makedirs(OUT, exist_ok=True)

with open(os.path.join(EVAL, "manifest.json"), encoding="utf-8") as f:
    manifest = json.load(f)

PICKS = [
    ("eval_001", "Meeting mein presentation"),
    ("eval_006", "Mera laptop kharab"),
    ("eval_015", "Weekend par movie"),
    ("eval_021", "Online khana order"),
]

out_manifest = []
for pid, label in PICKS:
    item = next((m for m in manifest if m["id"] == pid), None)
    if not item:
        print(f"missing {pid}")
        continue
    src = os.path.join(EVAL, item["audio"])
    dst = os.path.join(OUT, pid + ".mp3")
    shutil.copyfile(src, dst)
    out_manifest.append({"id": pid, "label": label, "truth": item["truth"]})
    print(f"copied {pid} -> {dst}")

with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as f:
    json.dump(out_manifest, f, ensure_ascii=False, indent=2)
print(f"wrote {len(out_manifest)} samples")

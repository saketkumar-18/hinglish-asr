"""Download a faster-whisper model from the HF Hub (used at Docker build time)."""
import os
import sys
import urllib.request

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Systran/faster-whisper-base"
DEST = sys.argv[2] if len(sys.argv) > 2 else "/app/models/faster-whisper-base"

FILES = ["model.bin", "config.json", "tokenizer.json", "vocabulary.txt"]
BASE = f"https://huggingface.co/{MODEL}/resolve/main/"

os.makedirs(DEST, exist_ok=True)
for f in FILES:
    url = BASE + f
    out = os.path.join(DEST, f)
    print(f"downloading {url} -> {out}", flush=True)
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(url, out)
            break
        except Exception as e:
            print(f"  attempt {attempt + 1} failed: {e}", flush=True)
            if attempt == 2:
                raise
    print(f"  ok ({os.path.getsize(out)} bytes)", flush=True)
print("model download complete", flush=True)

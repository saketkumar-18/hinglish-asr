"""Robust parallel chunked downloader with per-chunk resume + retry."""
import os
import sys
import time
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = sys.argv[1]
DEST = sys.argv[2]
NCHUNKS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
MAX_RETRY = 60

req = urllib.request.Request(URL, method="HEAD")
with urllib.request.urlopen(req, timeout=30) as r:
    total = int(r.headers["Content-Length"])

os.makedirs(os.path.dirname(DEST) or ".", exist_ok=True)
chunk = (total + NCHUNKS - 1) // NCHUNKS
lock = threading.Lock()


def part_path(i):
    return DEST + f".part{i}"


def part_target(i):
    start = i * chunk
    end = min(start + chunk, total) - 1
    return start, end


def fetch(i):
    start, end = part_target(i)
    target = end - start + 1
    p = part_path(i)
    for attempt in range(MAX_RETRY):
        existing = os.path.getsize(p) if os.path.exists(p) else 0
        if existing >= target:
            return True
        s = start + existing
        try:
            req = urllib.request.Request(URL, headers={"Range": f"bytes={s}-{end}"})
            with urllib.request.urlopen(req, timeout=60) as r, open(p, "ab") as f:
                while True:
                    b = r.read(1 << 16)
                    if not b:
                        break
                    f.write(b)
        except Exception as e:
            with lock:
                print(f"  chunk{i} attempt{attempt} err: {type(e).__name__}", flush=True)
            time.sleep(1.5 * (attempt % 5 + 1))
    return os.path.getsize(p) >= target


ok = True
with ThreadPoolExecutor(max_workers=NCHUNKS) as ex:
    results = list(ex.map(fetch, range(NCHUNKS)))
ok = all(results)

if ok:
    with open(DEST, "wb") as out:
        for i in range(NCHUNKS):
            with open(part_path(i), "rb") as f:
                out.write(f.read())
            os.remove(part_path(i))
    size = os.path.getsize(DEST)
    print(f"assembled {DEST} size={size} expected={total} ok={size==total}")
    sys.exit(0 if size == total else 1)
else:
    print("INCOMPLETE - some chunks failed after retries")
    sys.exit(1)

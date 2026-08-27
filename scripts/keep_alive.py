"""Keep the Render free-tier backend awake (it sleeps after ~15 min idle).

Hits /api/health every run. Cheap, no LLM. Run via cron every 10 minutes.
"""
import urllib.request

URL = "https://hinglish-asr.onrender.com/api/health"

try:
    with urllib.request.urlopen(URL, timeout=30) as r:
        body = r.read().decode()
        print(f"keep-alive ok: {r.status} {body[:120]}")
except Exception as e:
    print(f"keep-alive failed: {e}")

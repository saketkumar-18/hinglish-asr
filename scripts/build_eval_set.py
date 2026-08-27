"""Build a synthetic Hinglish evaluation set (WAV, lossless chunk join).

Each item is a code-switched sentence defined as (chunk, language) pairs.
Hindi chunks are synthesized with gTTS lang='hi' (Devanagari), English chunks
with lang='en' tld='com.in' (Indian English). Each MP3 chunk is decoded to
16 kHz mono PCM, the chunks are concatenated with a short silence gap, and the
result is written as a WAV file (lossless — unlike raw MP3 concatenation, which
silently drops frames at boundaries and loses embedded English words).

Output:
  data/eval/manifest.json
  data/eval/audio/eval_XXX.wav
"""
import io
import json
import os
import struct
import sys
import time
import wave

import numpy as np
from gtts import gTTS

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))
from faster_whisper.audio import decode_audio  # MP3 -> float32 @16k

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data", "eval")
AUDIO_DIR = os.path.join(OUT_DIR, "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

SR = 16000
GAP_S = 0.12  # silence between chunks so boundaries are clean

SENTENCES = [
    {"chunks": [("मुझे कल", "hi"), ("meeting", "en"), ("में", "hi"), ("presentation", "en"), ("देनी है", "hi")],
     "truth": "mujhe kal meeting mein presentation deni hai"},
    {"chunks": [("यह", "hi"), ("app", "en"), ("बहुत अच्छा है", "hi")],
     "truth": "ye app bahut accha hai"},
    {"chunks": [("मैं", "hi"), ("office", "en"), ("से", "hi"), ("घर", "hi"), ("आ रहा हूँ", "hi")],
     "truth": "main office se ghar aa raha hoon"},
    {"chunks": [("क्या तुम मुझे", "hi"), ("email", "en"), ("भेज सकते हो", "hi")],
     "truth": "kya tum mujhe email bhej sakte ho"},
    {"chunks": [("मेरा", "hi"), ("laptop", "en"), ("खराब हो गया है", "hi")],
     "truth": "mera laptop kharab ho gaya hai"},
    {"chunks": [("आज", "hi"), ("weather", "en"), ("बहुत अच्छा है", "hi")],
     "truth": "aaj weather bahut accha hai"},
    {"chunks": [("हमें", "hi"), ("project", "en"), ("जल्दी", "hi"), ("submit", "en"), ("करना है", "hi")],
     "truth": "humein project jaldi submit karna hai"},
    {"chunks": [("इस", "hi"), ("restaurant", "en"), ("का खाना बहुत", "hi"), ("tasty", "en"), ("है", "hi")],
     "truth": "is restaurant ka khana bahut tasty hai"},
    {"chunks": [("मेरी", "hi"), ("train", "en"), ("लेट हो रही है", "hi")],
     "truth": "meri train late ho rahi hai"},
    {"chunks": [("तुमने", "hi"), ("assignment", "en"), ("पूरा किया या नहीं", "hi")],
     "truth": "tumne assignment poora kiya ya nahi"},
    {"chunks": [("मुझे", "hi"), ("Hindi", "en"), ("गाने बहुत पसंद हैं", "hi")],
     "truth": "mujhe hindi gaane bahut pasand hai"},
    {"chunks": [("कल", "hi"), ("interview", "en"), ("है तो मुझे तैयारी करनी है", "hi")],
     "truth": "kal interview hai to mujhe taiyari karni hai"},
    {"chunks": [("यह", "hi"), ("phone", "en"), ("बहुत महंगा है", "hi")],
     "truth": "ye phone bahut mehenga hai"},
    {"chunks": [("हम लोग", "hi"), ("weekend", "en"), ("पर", "hi"), ("movie", "en"), ("देखने जाएंगे", "hi")],
     "truth": "hum log weekend par movie dekhne jayenge"},
    {"chunks": [("मेरा", "hi"), ("internet", "en"), ("बहुत धीमा चल रहा है", "hi")],
     "truth": "mera internet bahut dheema chal raha hai"},
    {"chunks": [("आपको", "hi"), ("password", "en"), ("बदलना चाहिए", "hi")],
     "truth": "aapko password badalna chahiye"},
    {"chunks": [("यह", "hi"), ("book", "en"), ("बहुत", "hi"), ("interesting", "en"), ("है", "hi")],
     "truth": "ye book bahut interesting hai"},
    {"chunks": [("मुझे", "hi"), ("doctor", "en"), ("से", "hi"), ("appointment", "en"), ("लेना है", "hi")],
     "truth": "mujhe doctor se appointment lena hai"},
    {"chunks": [("घर पर", "hi"), ("wifi", "en"), ("नहीं चल रहा", "hi")],
     "truth": "ghar par wifi nahi chal raha"},
    {"chunks": [("मैंने", "hi"), ("online", "en"), ("खाना", "hi"), ("order", "en"), ("किया है", "hi")],
     "truth": "maine online khana order kiya hai"},
    {"chunks": [("तुम", "hi"), ("class", "en"), ("में आए थे या नहीं", "hi")],
     "truth": "tum class mein aaye the ya nahi"},
    {"chunks": [("मेरी", "hi"), ("battery", "en"), ("खत्म हो गई है", "hi")],
     "truth": "meri battery khatam ho gayi hai"},
    {"chunks": [("हमें", "hi"), ("team", "en"), ("के साथ", "hi"), ("discussion", "en"), ("करना है", "hi")],
     "truth": "humein team ke saath discussion karna hai"},
    {"chunks": [("यह", "hi"), ("song", "en"), ("बहुत अच्छा है फिर से चलाओ", "hi")],
     "truth": "ye song bahut accha hai phir se chalao"},
    {"chunks": [("मेरा", "hi"), ("flight", "en"), ("दो घंटे लेट है", "hi")],
     "truth": "mera flight do ghante late hai"},
    {"chunks": [("आज का", "hi"), ("match", "en"), ("बहुत", "hi"), ("exciting", "en"), ("था", "hi")],
     "truth": "aaj ka match bahut exciting tha"},
    {"chunks": [("मुझे", "hi"), ("English", "en"), ("बोलने में थोड़ी दिक्कत होती है", "hi")],
     "truth": "mujhe english bolne mein thodi dikkat hoti hai"},
    {"chunks": [("क्या इस", "hi"), ("hotel", "en"), ("में", "hi"), ("breakfast", "en"), ("मिलेगा", "hi")],
     "truth": "kya is hotel mein breakfast milega"},
    {"chunks": [("पहले", "hi"), ("data", "en"), ("बैकअप कर लो फिर", "hi"), ("install", "en"), ("करो", "hi")],
     "truth": "pehle data backup kar lo phir install karo"},
]


def synth_chunk_pcm(text: str, lang: str) -> np.ndarray:
    if lang == "hi":
        tts = gTTS(text=text, lang="hi", slow=False)
    else:
        tts = gTTS(text=text, lang="en", tld="co.in", slow=False)
    # write to a temp MP3 file, then decode via path (decode_audio wants a path)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
        tmp = tf.name
    try:
        tts.save(tmp)
        return decode_audio(tmp, sampling_rate=SR)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def write_wav(path: str, pcm: np.ndarray):
    pcm16 = np.clip(pcm, -1.0, 1.0)
    pcm16 = (pcm16 * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm16.tobytes())


def count_switches(chunks):
    n, prev = 0, None
    for _, lang in chunks:
        if prev is not None and lang != prev:
            n += 1
        prev = lang
    return n


def main():
    manifest = []
    gap = np.zeros(int(SR * GAP_S), dtype=np.float32)
    for item in SENTENCES:
        name = f"eval_{len(manifest) + 1:03d}"
        path = os.path.join(AUDIO_DIR, name + ".wav")
        if os.path.exists(path):
            print(f"[skip] {name}")
        else:
            parts, ok = [], True
            for text, lang in item["chunks"]:
                for attempt in range(4):
                    try:
                        parts.append(synth_chunk_pcm(text, lang))
                        break
                    except Exception as e:
                        print(f"  retry {attempt+1} {name} {text!r}: {e}")
                        time.sleep(2)
                else:
                    ok = False
            if not ok:
                print(f"[FAIL] {name}")
                continue
            joined = []
            for i, p in enumerate(parts):
                if i:
                    joined.append(gap)
                joined.append(p)
            write_wav(path, np.concatenate(joined))
            print(f"[ok] {name} ({os.path.getsize(path)} bytes)")
        manifest.append({
            "id": name, "audio": f"audio/{name}.wav",
            "truth": item["truth"], "switches": count_switches(item["chunks"]),
            "chunks": item["chunks"],
        })
        time.sleep(0.3)

    with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\nwrote {len(manifest)} items -> manifest.json")


if __name__ == "__main__":
    main()

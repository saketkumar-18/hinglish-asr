"""Build a synthetic Hinglish evaluation set.

Each item is a code-switched sentence defined as (chunk, language) pairs.
Hindi chunks are synthesized with gTTS lang='hi' (Devanagari text), English
chunks with lang='en' tld='com.in' (Indian English). Chunks are concatenated
into one MP3 per sentence, with a romanized ground-truth transcript for
WER/CER scoring and a known switch count for switch-detection accuracy.

Output:
  data/eval/manifest.json
  data/eval/audio/eval_XXX.mp3
"""
import json
import os
import sys
import time

from gtts import gTTS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data", "eval")
AUDIO_DIR = os.path.join(OUT_DIR, "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

# (Devanagari/English TTS text, language) chunks -> romanized ground truth
SENTENCES = [
    {
        "chunks": [("मुझे कल", "hi"), ("meeting", "en"), ("में", "hi"), ("presentation", "en"), ("देनी है", "hi")],
        "truth": "mujhe kal meeting mein presentation deni hai",
    },
    {
        "chunks": [("यह", "hi"), ("app", "en"), ("बहुत अच्छा है", "hi")],
        "truth": "ye app bahut accha hai",
    },
    {
        "chunks": [("मैं", "hi"), ("office", "en"), ("से", "hi"), ("घर", "hi"), ("आ रहा हूँ", "hi")],
        "truth": "main office se ghar aa raha hoon",
    },
    {
        "chunks": [("क्या तुम मुझे", "hi"), ("email", "en"), ("भेज सकते हो", "hi")],
        "truth": "kya tum mujhe email bhej sakte ho",
    },
    {
        "chunks": [("मेरा", "hi"), ("laptop", "en"), ("खराब हो गया है", "hi")],
        "truth": "mera laptop kharab ho gaya hai",
    },
    {
        "chunks": [("आज", "hi"), ("weather", "en"), ("बहुत अच्छा है", "hi")],
        "truth": "aaj weather bahut accha hai",
    },
    {
        "chunks": [("हमें", "hi"), ("project", "en"), ("जल्दी", "hi"), ("submit", "en"), ("करना है", "hi")],
        "truth": "humein project jaldi submit karna hai",
    },
    {
        "chunks": [("यह", "hi"), ("restaurant", "en"), ("का खाना बहुत tasty है", "hi")],
        "truth": "ye restaurant ka khana bahut tasty hai",
    },
    {
        "chunks": [("मेरी", "hi"), ("train", "en"), ("लेट हो रही है", "hi")],
        "truth": "meri train late ho rahi hai",
    },
    {
        "chunks": [("तुमने", "hi"), ("assignment", "en"), ("पूरा किया या नहीं", "hi")],
        "truth": "tumne assignment poora kiya ya nahi",
    },
    {
        "chunks": [("मुझे", "hi"), ("Hindi", "en"), ("गाने बहुत पसंद हैं", "hi")],
        "truth": "mujhe hindi gaane bahut pasand hai",
    },
    {
        "chunks": [("कल", "hi"), ("interview", "en"), ("है तो मुझे तैयारी करनी है", "hi")],
        "truth": "kal interview hai to mujhe taiyari karni hai",
    },
    {
        "chunks": [("यह", "hi"), ("phone", "en"), ("बहुत महंगा है", "hi")],
        "truth": "ye phone bahut mehenga hai",
    },
    {
        "chunks": [("हम लोग", "hi"), ("weekend", "en"), ("पर", "hi"), ("movie", "en"), ("देखने जाएंगे", "hi")],
        "truth": "hum log weekend par movie dekhne jayenge",
    },
    {
        "chunks": [("मेरा", "hi"), ("internet", "en"), ("बहुत धीमा चल रहा है", "hi")],
        "truth": "mera internet bahut dheema chal raha hai",
    },
    {
        "chunks": [("आपको", "hi"), ("password", "en"), ("बदलना चाहिए", "hi")],
        "truth": "aapko password badalna chahiye",
    },
    {
        "chunks": [("यह", "hi"), ("book", "en"), ("बहुत interesting है", "hi")],
        "truth": "ye book bahut interesting hai",
    },
    {
        "chunks": [("मुझे", "hi"), ("doctor", "en"), ("से", "hi"), ("appointment", "en"), ("लेना है", "hi")],
        "truth": "mujhe doctor se appointment lena hai",
    },
    {
        "chunks": [("घर पर", "hi"), ("wifi", "en"), ("नहीं चल रहा", "hi")],
        "truth": "ghar par wifi nahi chal raha",
    },
    {
        "chunks": [("मैंने", "hi"), ("online", "en"), ("खाना", "hi"), ("order", "en"), ("किया है", "hi")],
        "truth": "maine online khana order kiya hai",
    },
    {
        "chunks": [("तुम", "hi"), ("class", "en"), ("में आए थे या नहीं", "hi")],
        "truth": "tum class mein aaye the ya nahi",
    },
    {
        "chunks": [("मेरी", "hi"), ("battery", "en"), ("खत्म हो गई है", "hi")],
        "truth": "meri battery khatam ho gayi hai",
    },
    {
        "chunks": [("हमें", "hi"), ("team", "en"), ("के साथ", "hi"), ("discussion", "en"), ("करना है", "hi")],
        "truth": "humein team ke saath discussion karna hai",
    },
    {
        "chunks": [("यह", "hi"), ("song", "en"), ("बहुत अच्छा है, फिर से चलाओ", "hi")],
        "truth": "ye song bahut accha hai phir se chalao",
    },
    {
        "chunks": [("मेरा", "hi"), ("flight", "en"), ("दो घंटे लेट है", "hi")],
        "truth": "mera flight do ghante late hai",
    },
    {
        "chunks": [("आज का", "hi"), ("match", "en"), ("बहुत exciting था", "hi")],
        "truth": "aaj ka match bahut exciting tha",
    },
    {
        "chunks": [("मुझे", "hi"), ("English", "en"), ("बोलने में थोड़ी दिक्कत होती है", "hi")],
        "truth": "mujhe english bolne mein thodi dikkat hoti hai",
    },
    {
        "chunks": [("क्या इस", "hi"), ("hotel", "en"), ("में", "hi"), ("breakfast", "en"), ("मिलेगा", "hi")],
        "truth": "kya is hotel mein breakfast milega",
    },
    {
        "chunks": [("पहले", "hi"), ("data", "en"), ("बैकअप कर लो, फिर", "hi"), ("install", "en"), ("करो", "hi")],
        "truth": "pehle data backup kar lo phir install karo",
    },
]


def synth_chunk(text: str, lang: str) -> bytes:
    if lang == "hi":
        tts = gTTS(text=text, lang="hi", slow=False)
    else:
        tts = gTTS(text=text, lang="en", tld="com.in", slow=False)
    import io
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    return buf.getvalue()


def count_switches(chunks):
    n = 0
    prev = None
    for _, lang in chunks:
        if prev is not None and lang != prev:
            n += 1
        prev = lang
    return n


def main():
    manifest = []
    for i, item in enumerate(SENTENCES):
        if not item["truth"]:
            continue
        name = f"eval_{len(manifest) + 1:03d}"
        path = os.path.join(AUDIO_DIR, name + ".mp3")
        if os.path.exists(path):
            print(f"[skip] {name} exists")
        else:
            parts = []
            ok = True
            for text, lang in item["chunks"]:
                for attempt in range(3):
                    try:
                        parts.append(synth_chunk(text, lang))
                        break
                    except Exception as e:
                        print(f"  retry {attempt+1} for {name} chunk {text!r}: {e}")
                        time.sleep(2)
                else:
                    ok = False
            if not ok:
                print(f"[FAIL] {name}")
                continue
            with open(path, "wb") as f:
                for p in parts:
                    f.write(p)
            print(f"[ok] {name} ({os.path.getsize(path)} bytes)")
        manifest.append({
            "id": name,
            "audio": f"audio/{name}.mp3",
            "truth": item["truth"],
            "switches": count_switches(item["chunks"]),
            "chunks": item["chunks"],
        })
        time.sleep(0.4)  # be polite to the TTS endpoint

    with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\nwrote {len(manifest)} items -> {os.path.join(OUT_DIR, 'manifest.json')}")


if __name__ == "__main__":
    main()

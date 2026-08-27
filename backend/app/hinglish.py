"""Hinglish post-processing layer for code-switching ASR.

Components:
1. Devanagari -> Roman transliterator (common Hinglish spelling convention)
2. Variant canonicalization (Whisper misspellings -> canonical Hinglish forms)
3. Word-level language tagging (hi / en) via lexicon + script heuristics
4. Code-switch point detection
5. Dual-pass fusion helpers (time-aligned en/hi hypothesis selection)
"""
from __future__ import annotations

import re
import unicodedata
from typing import List, Dict, Tuple

# ---------------------------------------------------------------------------
# 1. Devanagari -> Roman transliteration
# ---------------------------------------------------------------------------

_CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    "ळ": "l", "क़": "q", "ख़": "kh", "ग़": "g",
    "ज़": "z", "ड़": "r", "ढ़": "rh", "फ़": "f", "य़": "y",
}

_MATRAS = {
    "ा": "aa", "ि": "i", "ी": "i", "ु": "u", "ू": "u",
    "ृ": "ri", "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
    "ं": "n", "ः": "h", "ँ": "n",
}

_VOWELS = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo",
    "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
    "ऑ": "o", "ऍ": "e",
}

_DIGITS = {
    "०": "0", "१": "1", "२": "2", "३": "3", "४": "4",
    "५": "5", "६": "6", "७": "7", "८": "8", "९": "9",
}

HALANT = "्"
DANDA = "।"


def has_devanagari(text: str) -> bool:
    return any("\u0900" <= ch <= "\u097F" for ch in text)


def has_arabic_urdu(text: str) -> bool:
    """Whisper's hi pass sometimes emits Urdu (Nastaliq) script for Hindi speech."""
    return any(("\u0600" <= ch <= "\u06FF") or ("\u0750" <= ch <= "\u077F") for ch in text)


# Urdu consonants -> romanized approximation (Hinglish-oriented)
_URDU = {
    "ا": "a", "آ": "aa", "ب": "b", "پ": "p", "ت": "t", "ٹ": "t", "ث": "s",
    "ج": "j", "چ": "ch", "ح": "h", "خ": "kh", "د": "d", "ڈ": "d", "ذ": "z",
    "ر": "r", "ڑ": "r", "ز": "z", "ژ": "zh", "س": "s", "ش": "sh", "ص": "s",
    "ض": "z", "ط": "t", "ظ": "z", "ع": "", "غ": "g", "ف": "f", "ق": "q",
    "ک": "k", "ك": "k", "گ": "g", "ل": "l", "م": "m", "ن": "n", "ں": "n",
    "و": "v", "ہ": "h", "ھ": "h", "ء": "", "ی": "y", "ي": "y", "ے": "e",
    "أ": "a", "إ": "i", "ؤ": "o", "ئ": "y",
}
_URDU_DIACRITICS = {
    "َ": "a", "ِ": "i", "ُ": "u", "ً": "an", "ٍ": "in", "ٌ": "un",
    "ّ": "", "ْ": "", "ٰ": "a", "ٓ": "",
}
_URDU_DIGITS = {
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
    "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
}


def transliterate_urdu_word(word: str) -> str:
    out = []
    for ch in word:
        if ch in _URDU:
            out.append(_URDU[ch])
        elif ch in _URDU_DIACRITICS:
            out.append(_URDU_DIACRITICS[ch])
        elif ch in _URDU_DIGITS:
            out.append(_URDU_DIGITS[ch])
        else:
            out.append(ch)
    rom = "".join(out)
    rom = re.sub(r"([a-z])\1{2,}", r"\1\1", rom)  # cap triple letters
    return rom


def transliterate_word(word: str) -> str:
    """Transliterate one Devanagari word to common Hinglish romanization.

    Applies word-final schwa deletion (कम -> kam, not kama) which matches
    how Hindi speakers actually romanize.
    """
    out: List[str] = []
    chars = list(word)
    n = len(chars)
    i = 0
    while i < n:
        ch = chars[i]
        if ch in _DIGITS:
            out.append(_DIGITS[ch])
            i += 1
        elif ch in _VOWELS:
            out.append(_VOWELS[ch])
            i += 1
        elif ch in _CONSONANTS:
            out.append(_CONSONANTS[ch])
            nxt = chars[i + 1] if i + 1 < n else ""
            if nxt == HALANT:
                i += 2  # explicit halant: no inherent vowel
                continue
            if nxt in _MATRAS:
                i += 1  # matra consumed below
                continue
            # inherent 'a' unless word-final (schwa deletion)
            if i + 1 < n:
                out.append("a")
            i += 1
        elif ch in _MATRAS:
            out.append(_MATRAS[ch])
            i += 1
        elif ch == HALANT:
            i += 1
        elif ch == DANDA:
            out.append(".")
            i += 1
        else:
            out.append(ch)
            i += 1
    rom = "".join(out)
    # tidy: collapse doubled inherent vowels from edge cases
    rom = re.sub(r"aa+", "aa", rom) if not re.search(r"[ाआ]", word) else rom
    return rom


def transliterate(text: str) -> str:
    """Transliterate any Devanagari or Urdu runs in text, leaving Latin alone."""
    if not has_devanagari(text) and not has_arabic_urdu(text):
        return text
    out = []
    buf = []
    buf_kind = None  # 'dev' or 'urdu'

    def flush():
        nonlocal buf, buf_kind
        if buf:
            w = "".join(buf)
            out.append(transliterate_word(w) if buf_kind == "dev" else transliterate_urdu_word(w))
            buf = []
            buf_kind = None

    for ch in text:
        if "\u0900" <= ch <= "\u097F":
            if buf_kind != "dev":
                flush()
                buf_kind = "dev"
            buf.append(ch)
        elif ("\u0600" <= ch <= "\u06FF") or ("\u0750" <= ch <= "\u077F"):
            if buf_kind != "urdu":
                flush()
                buf_kind = "urdu"
            buf.append(ch)
        else:
            flush()
            out.append(ch)
    flush()
    return "".join(out)


# ---------------------------------------------------------------------------
# 2. Variant canonicalization
# ---------------------------------------------------------------------------

_VARIANTS: Dict[str, str] = {
    # common Whisper romanization drift -> canonical Hinglish spelling
    "karnaa": "karna", "karana": "karna", "karna": "karna",
    "nahī": "nahi", "nahee": "nahi", "na he": "nahi", "nhi": "nahi", "nahin": "nahi",
    "kyaa": "kya", "kaya": "kya",
    "haan": "haan", "hā": "haan",
    "hain": "hai", "hay": "hai",
    "mey": "mein", "mai": "main",
    "aap": "aap", "āp": "aap",
    "thaa": "tha", "thee": "thi",
    "kuchh": "kuch", "kuch": "kuch",
    "acchaa": "accha", "achha": "accha", "acha": "accha",
    "bahut": "bahut", "bohot": "bahut", "bhaut": "bahut",
    "voh": "woh", "vo": "woh",
    "yeh": "ye", "yeh": "ye",
    "kaun": "kaun",
    "kabhi": "kabhi",
    "zaraa": "zara", "zra": "zara",
    "pataa": "pata", "pta": "pata",
    "chahiye": "chahiye", "chaahiye": "chahiye",
    "samajh": "samajh", "samjh": "samajh",
    "theek": "thik", "theak": "thik",
    "dhoondh": "dhundh", "dhund": "dhundh",
    "bataao": "batao", "batao": "batao",
    "chalaa": "chala", "chalee": "chali",
    "milaa": "mila", "milee": "mili",
    "huā": "hua", "hui": "hui",
    "rahaa": "raha", "rahee": "rahi",
    "wala": "wala", "waala": "wala",
    "bhaiya": "bhaiya", "bhai": "bhai",
}


def canonicalize_word(w: str) -> str:
    return _VARIANTS.get(w, w)


# ---------------------------------------------------------------------------
# 3. Word-level language tagging
# ---------------------------------------------------------------------------

# High-frequency romanized Hindi function/content words (lexicon for tagging).
HINDI_LEXICON = set("""
aap ab adha adhik agar aisa aise alag andar apna apne apni aapka aapke aapki
aaj aaram aashiq asali asli aur aurat aadmi aadhe aana aane aao aapko
baat baad baaki baar baat karna baba badhiya bagal bahar bahut baki banao banana
banda banda band bandh bando bana banane banaya banaye bane bano barabar bas
bata batao batana bataya bhai bhar bhed bhool bhukh bhookh bhi bich mein bigaad
bijli bilkul bojh bol bolna bolo bolte bolti bura bure buriya
chahiye chahiye tha chalna chalo chalein chalta chalti chand chhota chhoti chhutti
chinta chitthi chhoda chhod do chhodo chhodo chhutti chij cheez cheezein chinta
daal daalna daalo daam daant dada dadi dard darwaza de dena denge deni dekho
dekha dekhi dekhna dekhne dekh raha dekh rahi desh dhan dhire dhire dhundh dhundho
didi dil din do doctor dost dukaan duniya dusra dusre dusri
ek gaana gaane gaao gaav galti gande ghar ghoomne ghumaao ghadi ghanta ghodi
gussa guzra
haan haal hai hain hamara hamare hamari hame hum hume humein humara humare humari
hamesha har hatao hawa hazaar himmat hisaab ho hoga hoge hogi hoi hokar hota hoti
hote hu hua hue hui hun hoon
jab jab tak jaisa jaise jaisi jaldi janam janab jawab jeeb jee haan jee nahi
jyada jyaada jaldi kaam kaam karo kaash kab kabhi kachra kafi kaha kahin kaisa
kaise kaisi kal kam kameez kamre kan kaun kaunsa kaunsi karta karti karte karna
karne karo karta hoon karti hoon karte hain kar raha kar rahi kasam khaana khaana
khana khao kharab kharida kharidna khatam khatra khilona khola kholo khud khush
ki kitna kitne kitni kisi kisiko kisko kismat koi koshish krupaya kya kyon kyun
kyun ki kyunki
laga lagana lagao lagta lagti lagte lakdi lakshman lamba lambi log logon
maa maan maana maang maango maaf maaf karna mahal main maine malik mat matlab
mausam mein me mil mila mile mili milna milo mitti mubarak mujhe mujhko musibat
muft mein
na nahi nahin nahi hai na karo naam naya nayi naye ne neeche niche neta nila
paani pani padosi pakka pakka karna papa paise paisa palang pata pata hai pata
nahi patni pe pehle pehli phir phir se phool pichle pita pooch poochna poochho
pyaar pyaasi pyaasa
raat rahi raha rahe raho rakha rakho rakhta rakhti rang rasta rehna rehte raho
roz roti roshni
saaf saal saab sab sab kuch sabzi sach sahi sahara sahil sakta sakti sakte samajh
samajhna samjha samjho samay sapna sath saath sathi sawal sawaal shadi shakal
sham shayad shiksha shukriya sir sirf soch sochna socho sona sone soya soyi
subah suraj suru suru karna sunder
tab tab tak takleef talab tamatar thanda thandi thik thik hai thoda thodi thoda
sa thoda thoda tha thi thoda time
ummed un unka unke unki unko unn unhe unhein us uska uske uski usko utar utna
utni utna
waqt wala wali wapas wapis wahan wahi waise waisa wala welcome woh wo
ya yaad yaar yahan yahi yeh ye yehi
zaroor zaroorat zaroori zara zindagi
""".split())

# Small English function-word list to disambiguate ASCII words.
ENGLISH_LEXICON = set("""
a an the and or but if then else when where who whom whose which what why how
is am are was were be been being do does did done have has had having will would
shall should can could may might must i you he she it we they me him her us them
my your his its our their mine yours hers ours theirs this that these those
not no yes okay ok please thank thanks welcome sorry hello hi hey goodbye bye
in on at by for with without from to into onto upon over under above below
between among through during before after while since until about against
good bad great nice fine well better best worse worst big small large little
new old young first last next time day night morning evening week month year
today tomorrow yesterday now later soon here there everywhere somewhere
work working worked job office meeting email phone call message send sent
computer mobile internet wifi password login data file download upload
money price cost buy bought sell sold shop market food water tea coffee milk
bread rice help need want like love know knew known think thought see saw seen
look looked watch watched listen listened speak spoke spoken say said tell told
ask asked answer answered question problem solution idea plan start stop
""".split())


def tag_word(word: str) -> str:
    """Return 'hi' or 'en' for a word."""
    if has_devanagari(word) or has_arabic_urdu(word):
        return "hi"
    w = word.lower().strip(".,!?;:'\"()[]{}")
    if not w:
        return "en"
    if w in HINDI_LEXICON:
        return "hi"
    if w in ENGLISH_LEXICON:
        return "en"
    # ASCII word not in Hindi lexicon -> default English
    if all(ord(c) < 128 for c in w):
        return "en"
    return "hi"


def tag_words(text: str) -> List[Dict[str, str]]:
    words = re.findall(r"[\w\u0900-\u097F\u0600-\u06FF\u0750-\u077F]+(?:['’][\w\u0900-\u097F]+)?|[^\w\s\u0900-\u097F\u0600-\u06FF\u0750-\u077F]", text)
    out = []
    for w in words:
        if re.fullmatch(r"[^\w\s\u0900-\u097F\u0600-\u06FF\u0750-\u077F]", w):
            continue
        out.append({"word": w, "lang": tag_word(w)})
    return out


def count_switches(tagged: List[Dict[str, str]]) -> int:
    n = 0
    prev = None
    for t in tagged:
        if prev is not None and t["lang"] != prev:
            n += 1
        prev = t["lang"]
    return n


def language_ratio(tagged: List[Dict[str, str]]) -> Dict[str, float]:
    hi = sum(1 for t in tagged if t["lang"] == "hi")
    en = len(tagged) - hi
    total = max(len(tagged), 1)
    return {"hi": round(hi / total, 3), "en": round(en / total, 3)}


# ---------------------------------------------------------------------------
# 4. Full post-processing pipeline
# ---------------------------------------------------------------------------

def postprocess(text: str) -> Dict:
    """Run the full Hinglish post-processing pipeline on raw ASR text."""
    text = unicodedata.normalize("NFC", text or "").strip()
    roman = transliterate(text)
    # canonicalize word variants
    tokens = roman.split()
    canon = []
    for t in tokens:
        core = t.strip(".,!?;:'\"()[]{}")
        tail = t[len(core):] if core else t
        lead = t[: len(t) - len(t.lstrip(".,!?;:'\"()[]{}"))]
        canon.append(lead + canonicalize_word(core.lower()) + tail)
    roman = " ".join(canon)
    tagged = tag_words(roman)
    return {
        "text": roman,
        "tagged": tagged,
        "switches": count_switches(tagged),
        "language_ratio": language_ratio(tagged),
    }


# ---------------------------------------------------------------------------
# 5. Dual-pass fusion
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[\w\u0900-\u097F\u0600-\u06FF\u0750-\u077F]+")


def _content_words(text: str) -> List[str]:
    return [canonicalize_word(w.lower()) for w in _WORD_RE.findall(text)]


def _consonants(w: str) -> str:
    """Consonant skeleton: strip vowels. Handles transliteration vowel swaps."""
    return "".join(c for c in w.lower() if c not in "aeiou")


def _similar(a: str, b: str) -> bool:
    """Cheap fuzzy match tolerant of transliteration spelling variants."""
    if not a or not b:
        return False
    if a == b:
        return True
    import difflib
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    short = min(len(a), len(b)) <= 4
    if ratio >= (0.6 if short else 0.75):
        return True
    if len(a) >= 3 and len(b) >= 3 and (a.startswith(b[:3]) or b.startswith(a[:3])):
        # 3-char prefix match plus comparable length -> treat as same word
        if 0.6 <= len(a) / len(b) <= 1.6:
            return True
    # consonant-skeleton match (mi/me, eemel/email, doktar/doctor)
    ca, cb = _consonants(a), _consonants(b)
    if ca and cb:
        if ca == cb:
            return True
        if min(len(ca), len(cb)) >= 2 and difflib.SequenceMatcher(None, ca, cb).ratio() >= 0.8:
            return True
    return False


def translation_score(en_text: str, hi_text: str) -> float:
    """Fraction of Hindi-pass content words recoverable from the English pass.

    Whisper's English pass sometimes TRANSLATES Hindi speech into English
    ("main office se ghar aa raha hoon" -> "i am coming home from the office").
    A low recovery ratio signals translation, so the Hindi pass should win.
    """
    hi_words = _content_words(transliterate(hi_text))
    en_words = _content_words(en_text)
    if not hi_words:
        return 1.0
    if not en_words:
        return 0.0
    hits = 0
    for hw in hi_words:
        if any(_similar(hw, ew) for ew in en_words):
            hits += 1
    return hits / len(hi_words)


TRANSLATION_THRESHOLD = 0.45  # below this, the EN pass is translating, not transcribing


def fuse_segments(segs_en: List[Dict], segs_hi: List[Dict]) -> List[Dict]:
    """Time-aligned fusion of English-pass and Hindi-pass hypotheses.

    Strategy (validated on the synthetic Hinglish eval set):
      * The English pass usually produces the best romanized Hinglish, so it
        is the default source for each time span.
      * When the English pass TRANSLATES the speech instead of transcribing it
        (low word recovery vs the Hindi pass), the Hindi pass wins and its
        output is transliterated to Roman script.
      * Hindi-pass segments with no English overlap (e.g. trailing Hindi
        speech the EN pass dropped) are appended, transliterated.
    """
    chosen: List[Dict] = []
    used_hi = set()

    for se in segs_en:
        s0, s1 = se["start"], se["end"]
        best_hi = None
        best_overlap = 0.0
        for idx, sh in enumerate(segs_hi):
            ov = min(s1, sh["end"]) - max(s0, sh["start"])
            if ov > best_overlap:
                best_overlap = ov
                best_hi = (idx, sh)

        dur = max(s1 - s0, 1e-6)
        if best_hi is not None and best_overlap / dur >= 0.4:
            idx, sh = best_hi
            rec = translation_score(se["text"], sh["text"])
            if rec < TRANSLATION_THRESHOLD:
                # EN pass translated this span -> use Hindi pass
                chosen.append({
                    "start": s0, "end": s1,
                    "text": transliterate(sh["text"]).strip(),
                    "source": "hi",
                    "avg_logprob": sh.get("avg_logprob"),
                })
                used_hi.add(idx)
                continue
            # otherwise EN pass is transcribing fine -> keep it (better spelling)
        chosen.append({
            "start": s0, "end": s1, "text": se["text"].strip(),
            "source": "en",
            "avg_logprob": se.get("avg_logprob"),
        })

    # append Hindi-only segments with no English overlap (trailing Hindi speech)
    for idx, sh in enumerate(segs_hi):
        if idx in used_hi:
            continue
        overlaps = any(
            min(sh["end"], c["end"]) - max(sh["start"], c["start"]) > 0.2
            for c in chosen
        )
        if not overlaps and sh["text"].strip():
            chosen.append({
                "start": sh["start"], "end": sh["end"],
                "text": transliterate(sh["text"]).strip(),
                "source": "hi",
                "avg_logprob": sh.get("avg_logprob"),
            })

    chosen.sort(key=lambda c: c["start"])
    return [c for c in chosen if c["text"]]

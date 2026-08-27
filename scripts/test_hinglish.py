"""Unit tests for the Hinglish post-processing layer."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))

from app.hinglish import (
    transliterate, transliterate_word, has_devanagari, postprocess,
    tag_word, tag_words, count_switches, fuse_segments, canonicalize_word,
)


def test_transliterate_basic():
    assert transliterate_word("नमस्ते") == "namaste"
    assert transliterate_word("कम") == "kam"          # schwa deletion
    assert transliterate_word("काम") == "kaam"
    assert transliterate_word("हिंदी") == "hindi"
    assert has_devanagari("नमस्ते")
    assert not has_devanagari("hello")


def test_transliterate_mixed():
    out = transliterate("मैं office जा रहा हूँ")
    assert "office" in out
    assert not has_devanagari(out)
    assert "main" in out


def test_canonicalize():
    assert canonicalize_word("bohot") == "bahut"
    assert canonicalize_word("nahi") == "nahi"
    assert canonicalize_word("unknownword") == "unknownword"


def test_tag_word():
    assert tag_word("मैं") == "hi"
    assert tag_word("meeting") == "en"
    assert tag_word("kya") == "hi"
    assert tag_word("the") == "en"


def test_switch_count():
    tagged = tag_words("mujhe kal meeting mein presentation deni hai")
    # mujhe(hi) kal(hi) meeting(en) mein(hi) presentation(en) deni(hi) hai(hi)
    assert count_switches(tagged) == 4


def test_postprocess_pipeline():
    pp = postprocess("मैं office से ghar आ रह हूँ")
    assert not has_devanagari(pp["text"])
    assert pp["switches"] >= 1
    assert 0 <= pp["language_ratio"]["hi"] <= 1
    assert abs(pp["language_ratio"]["hi"] + pp["language_ratio"]["en"] - 1) < 0.01


def test_fuse_segments_prefers_hi_on_overlap():
    segs_en = [{"start": 0.0, "end": 2.0, "text": "main office ja raha hoon", "avg_logprob": -0.5}]
    segs_hi = [{"start": 0.0, "end": 2.0, "text": "मैं ऑफिस जा रहा हूँ", "avg_logprob": -0.3}]
    fused = fuse_segments(segs_en, segs_hi)
    assert len(fused) == 1
    assert fused[0]["source"] == "hi"
    assert not has_devanagari(fused[0]["text"])


def test_fuse_keeps_en_when_hi_worse():
    segs_en = [{"start": 0.0, "end": 2.0, "text": "send me the email", "avg_logprob": -0.2}]
    segs_hi = [{"start": 0.0, "end": 2.0, "text": "सेंड मी द ईमेल", "avg_logprob": -0.9}]
    fused = fuse_segments(segs_en, segs_hi)
    assert fused[0]["source"] == "en"
    assert fused[0]["text"] == "send me the email"


def test_fuse_appends_trailing_hi():
    segs_en = [{"start": 0.0, "end": 1.0, "text": "hello", "avg_logprob": -0.2}]
    segs_hi = [{"start": 3.0, "end": 5.0, "text": "धन्यवाद", "avg_logprob": -0.3}]
    fused = fuse_segments(segs_en, segs_hi)
    assert len(fused) == 2
    assert fused[1]["source"] == "hi"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)

"""Basic tests for the feature extractor.

    python -m pytest tests/ -q      (if pytest is installed)
    python tests/test_features.py   (otherwise)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from features import extract_features, extract_features_frame, FEATURE_NAMES
import pandas as pd


def test_all_features_present():
    feats = extract_features("http://example.com")
    assert set(feats.keys()) == set(FEATURE_NAMES)
    assert len(feats) == 25


def test_ip_detection():
    assert extract_features("http://192.168.0.1/login")["has_ip"] == 1
    assert extract_features("http://example.com/login")["has_ip"] == 0


def test_at_symbol_and_suspicious_tld():
    f = extract_features("http://user@phish.tk/verify")
    assert f["has_at_symbol"] == 1
    assert f["suspicious_tld"] == 1


def test_punycode_and_shortener():
    assert extract_features("http://xn--pypal-4ve.com")["has_punycode"] == 1
    assert extract_features("http://bit.ly/abc")["is_shortened"] == 1


def test_suspicious_keywords_counted():
    f = extract_features("http://site.com/login/verify/account/secure")
    assert f["suspicious_keyword_count"] >= 4


def test_scheme_invariance():
    # same URL should score the same with or without http://
    a = extract_features("example.com/login/verify")
    b = extract_features("http://example.com/login/verify")
    assert a == b


def test_malformed_input_does_not_crash():
    for bad in ["", "http://[::bad", "n o t a u r l", "???", "http://"]:
        feats = extract_features(bad)
        assert len(feats) == 25


def test_frame_extraction():
    s = pd.Series(["http://a.com", "http://b.tk/login"])
    frame = extract_features_frame(s)
    assert list(frame.columns) == FEATURE_NAMES
    assert len(frame) == 2


def _run_all():
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS  {name}")
            passed += 1
    print(f"\n{passed} tests passed.")


if __name__ == "__main__":
    _run_all()

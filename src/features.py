"""Turns a URL string into 25 numeric features.

Everything is worked out from the URL text only - no internet, no DNS, no
visiting the page - so the detector stays fully offline. The features are based
on common phishing-URL signals from the literature (Sahingoz et al. 2019;
Mohammad et al. 2014)."""

from __future__ import annotations

import math
import re
from collections import Counter
from urllib.parse import urlparse, unquote

import pandas as pd

from config import (
    SUSPICIOUS_KEYWORDS,
    SHORTENER_SERVICES,
    SUSPICIOUS_TLDS,
    SUSPICIOUS_EXTENSIONS,
    MULTI_LABEL_SUFFIXES,
)

# Compile the regexes once (faster when running over 800k rows).
_IPV4_RE = re.compile(
    r"(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d?\d)"
)
_HEX_IP_RE = re.compile(r"0x[0-9a-fA-F]+")
_SPECIAL_RE = re.compile(r"[^a-zA-Z0-9]")
_ENCODED_RE = re.compile(r"%[0-9a-fA-F]{2}")

# Column order - everything (training, prediction) relies on this being fixed.
FEATURE_NAMES = [
    "url_length", "num_dots", "num_hyphens", "has_ip", "suspicious_keyword_count",
    "num_subdomains", "domain_length", "path_length", "query_length",
    "num_slashes", "num_params",
    "num_digits", "digit_ratio", "num_special_chars", "char_entropy",
    "has_at_symbol", "has_double_slash_redirect", "https_in_path",
    "is_shortened", "has_punycode", "num_encoded_chars", "lookalike_char_count",
    "suspicious_tld", "has_suspicious_extension", "num_digits_in_domain",
]

# Digit/symbol swaps used in look-alike domains (paypa1 -> paypal, g00gle...).
_LOOKALIKE_MAP = {"0": "o", "1": "l", "3": "e", "5": "s", "@": "a", "$": "s"}


def _shannon_entropy(text: str) -> float:
    """Entropy of the string. Random-looking (auto-generated) URLs score higher."""
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def _split_host(hostname: str) -> tuple[str, int]:
    """Return (registered domain, number of subdomains). Uses a small suffix
    list so it works offline (handles things like co.uk)."""
    if not hostname:
        return "", 0
    hostname = hostname.strip(".").lower()
    labels = hostname.split(".")
    if len(labels) < 2:
        return hostname, 0

    if len(labels) >= 3 and ".".join(labels[-2:]) in MULTI_LABEL_SUFFIXES:
        registered = ".".join(labels[-3:])
        subdomain_labels = labels[:-3]
    else:
        registered = ".".join(labels[-2:])
        subdomain_labels = labels[:-2]
    return registered, len(subdomain_labels)


def _normalise(url: str) -> str:
    """Add http:// if the URL has no scheme, otherwise urlparse gets confused.
    (Lots of rows in the dataset have no scheme.) Doesn't fetch anything."""
    url = str(url).strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", url):
        return "http://" + url
    return url


def extract_features(url: str) -> dict:
    """Build the feature dictionary for one URL."""
    # Normalise first so the same URL scores the same with or without "http://".
    raw = _normalise(str(url).strip())
    try:
        parsed = urlparse(raw)
        hostname = parsed.hostname or ""
        path = parsed.path or ""
        query = parsed.query or ""
    except ValueError:
        # A few URLs are broken enough to crash urlparse, so split by hand.
        stripped = re.sub(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", "", raw)
        host_part = re.split(r"[/?#]", stripped, maxsplit=1)
        hostname = host_part[0].split("@")[-1].split(":")[0].lower()
        remainder = host_part[1] if len(host_part) > 1 else ""
        path, _, query = remainder.partition("?")
        path = "/" + path

    registered_domain, num_subdomains = _split_host(hostname)
    tld = hostname.rsplit(".", 1)[-1] if "." in hostname else ""
    decoded = unquote(raw)
    digits = [c for c in raw if c.isdigit()]
    domain_digits = [c for c in hostname if c.isdigit()]

    # Count characters that change after reversing look-alike swaps.
    deobfuscated = "".join(_LOOKALIKE_MAP.get(c, c) for c in hostname)
    lookalike_count = sum(1 for a, b in zip(hostname, deobfuscated) if a != b)

    has_ip = bool(_IPV4_RE.search(hostname) or _HEX_IP_RE.search(hostname))

    features = {
        "url_length": len(raw),
        "num_dots": raw.count("."),
        "num_hyphens": raw.count("-"),
        "has_ip": int(has_ip),
        "suspicious_keyword_count": sum(
            decoded.lower().count(k) for k in SUSPICIOUS_KEYWORDS
        ),
        "num_subdomains": num_subdomains,
        "domain_length": len(hostname),
        "path_length": len(path),
        "query_length": len(query),
        "num_slashes": raw.count("/"),
        "num_params": query.count("=") if query else 0,
        "num_digits": len(digits),
        "digit_ratio": (len(digits) / len(raw)) if raw else 0.0,
        "num_special_chars": len(_SPECIAL_RE.findall(raw)),
        "char_entropy": round(_shannon_entropy(raw), 4),
        "has_at_symbol": int("@" in raw),                 # hides the real host
        "has_double_slash_redirect": int(raw.rfind("//") > 6),
        "https_in_path": int("https" in (path + query).lower()),  # fake security
        "is_shortened": int(registered_domain in SHORTENER_SERVICES),
        "has_punycode": int("xn--" in hostname.lower()),
        "num_encoded_chars": len(_ENCODED_RE.findall(raw)),
        "lookalike_char_count": lookalike_count,
        "suspicious_tld": int(tld in SUSPICIOUS_TLDS),
        "has_suspicious_extension": int(
            any(path.lower().endswith(ext) for ext in SUSPICIOUS_EXTENSIONS)
        ),
        "num_digits_in_domain": len(domain_digits),
    }
    return features


def extract_features_frame(urls: pd.Series) -> pd.DataFrame:
    """Run extract_features over a whole column of URLs -> DataFrame."""
    records = [extract_features(u) for u in urls]
    frame = pd.DataFrame.from_records(records, index=urls.index)
    return frame[FEATURE_NAMES]


if __name__ == "__main__":
    # Quick test when running this file directly.
    samples = [
        "http://paypa1-secure-login.com/verify?account=123",
        "https://www.google.com/search?q=python",
        "http://192.168.0.1@malicious.tk/update.exe",
    ]
    for s in samples:
        print(s)
        for name, value in extract_features(s).items():
            print(f"    {name:28s} {value}")
        print()

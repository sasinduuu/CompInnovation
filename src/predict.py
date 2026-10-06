"""PhishGuard command-line detector (the app).

Give it a URL and it prints whether it's phishing, a trust score, and the
reasons. Works fully offline - it never opens the URL.

    python src/predict.py "http://paypal.com.secure-login.tk/verify"
    python src/predict.py            # asks you to type a URL
    python src/predict.py --demo     # runs some example URLs
"""

from __future__ import annotations

import sys
import pickle

import pandas as pd

from config import FINAL_MODEL_FILE, FEATURE_PIPELINE_FILE
from features import FEATURE_NAMES, extract_features

# Plain-English reason to show when a feature fires.
EXPLANATIONS = {
    "has_ip": "uses a raw IP address instead of a domain name",
    "has_at_symbol": "contains an '@' symbol, which hides the real destination",
    "has_double_slash_redirect": "has a '//' redirect inside the path",
    "has_punycode": "uses punycode (xn--), often a look-alike domain",
    "is_shortened": "uses a URL-shortening service that hides the destination",
    "suspicious_tld": "uses a top-level domain commonly abused for phishing",
    "has_suspicious_extension": "points to a suspicious downloadable file",
    "https_in_path": "places the word 'https' in the path to fake security",
    "num_subdomains": "has an unusually deep chain of sub-domains",
    "suspicious_keyword_count": "contains credential/action keywords "
                                "(login, verify, account, ...)",
    "lookalike_char_count": "substitutes digits/symbols for letters "
                            "(e.g. paypa1 for paypal)",
    "num_hyphens": "contains many hyphens, common in deceptive domains",
    "num_dots": "contains an unusually high number of dots",
}


def load_detector():
    """Load the saved model and scaler."""
    try:
        with open(FINAL_MODEL_FILE, "rb") as fh:
            bundle = pickle.load(fh)
        with open(FEATURE_PIPELINE_FILE, "rb") as fh:
            scaler = pickle.load(fh)
    except FileNotFoundError:
        sys.exit("Model files not found. Run `python src/train_models.py` first.")
    return bundle, scaler


def validate(url: str) -> str:
    """Check the input looks like a URL, else raise ValueError."""
    if url is None:
        raise ValueError("No URL provided.")
    url = url.strip().strip('"').strip("'")
    if not url:
        raise ValueError("Empty input - please enter a URL.")
    if len(url) < 4 or "." not in url and "/" not in url:
        raise ValueError(f"'{url}' does not look like a URL.")
    if any(c.isspace() for c in url):
        raise ValueError("A URL cannot contain spaces.")
    return url


def reasons_for(url: str, feats: dict) -> list[str]:
    """Collect the reasons (the flags + a few count-based ones) for a URL."""
    reasons = []
    for flag in ["has_ip", "has_at_symbol", "has_double_slash_redirect",
                 "has_punycode", "is_shortened", "suspicious_tld",
                 "has_suspicious_extension", "https_in_path"]:
        if feats[flag]:
            reasons.append(EXPLANATIONS[flag])
    if feats["num_subdomains"] >= 3:
        reasons.append(EXPLANATIONS["num_subdomains"])
    if feats["suspicious_keyword_count"] >= 2:
        reasons.append(EXPLANATIONS["suspicious_keyword_count"])
    if feats["lookalike_char_count"] >= 1:
        reasons.append(EXPLANATIONS["lookalike_char_count"])
    if feats["num_hyphens"] >= 3:
        reasons.append(EXPLANATIONS["num_hyphens"])
    if feats["num_dots"] >= 5:
        reasons.append(EXPLANATIONS["num_dots"])
    return reasons


def band(phishing_prob: float, threshold: float) -> str:
    """Turn the probability into a risk band. Tied to the model threshold so it
    can't disagree with the verdict."""
    if phishing_prob >= threshold:
        return "HIGH RISK - LIKELY PHISHING"
    if phishing_prob >= 0.40:
        return "SUSPICIOUS"
    return "LIKELY SAFE FROM PHISHING"


def analyse(url: str, bundle, scaler) -> dict:
    """Get the features, run the model, and build the result."""
    feats = extract_features(url)
    vec = pd.DataFrame([feats])[FEATURE_NAMES]
    phishing_prob = float(bundle["model"].predict_proba(scaler.transform(vec))[0, 1])
    is_phishing = phishing_prob >= bundle["threshold"]
    trust = int(round((1 - phishing_prob) * 100))
    return {
        "url": url,
        "phishing_probability": phishing_prob,
        "is_phishing": is_phishing,
        "trust_score": trust,
        "band": band(phishing_prob, bundle["threshold"]),
        "reasons": reasons_for(url, feats),
    }


def print_result(res: dict) -> None:
    print("=" * 68)
    print(f"URL analysed : {res['url']}")
    print(f"Phishing prob: {res['phishing_probability']:.1%}")
    print(f"Trust score  : {res['trust_score']}/100")
    verdict = "PHISHING" if res["is_phishing"] else "NOT PHISHING"
    print(f"Verdict      : {verdict}   [{res['band']}]")
    if res["reasons"]:
        print("Why:")
        for r in res["reasons"]:
            print(f"   - {r}")
    else:
        print("Why: no strong phishing signals were found in the URL.")
    if not res["is_phishing"]:
        print("Note: 'not phishing' is NOT the same as 'safe' - this tool only "
              "judges phishing, not malware or other threats.")
    print("(The target page was never visited - analysis is 100% offline.)")
    print("=" * 68)


DEMO_URLS = [
    "http://paypal.com.secure-login.verify-account.tk/signin",
    "http://192.168.10.5/wp-admin/appleid/verify.php",
    "https://xn--pypal-4ve.com/account/update",
    "http://bit.ly/3xRandomRedirect",
    "https://www.wikipedia.org/wiki/Machine_learning",
    "https://github.com/sasinduuu/CompInnovation",
]


def main(argv: list[str]) -> int:
    bundle, scaler = load_detector()
    print(f"PhishGuard detector ready (model: {bundle['model_name']}, "
          f"offline mode).")

    if len(argv) > 1 and argv[1] == "--demo":
        for url in DEMO_URLS:
            print_result(analyse(url, bundle, scaler))
        return 0

    if len(argv) > 1:
        urls = [" ".join(argv[1:])]
    else:
        try:
            raw = input("Enter a URL to analyse: ")
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            return 0
        urls = [raw]

    for raw in urls:
        try:
            url = validate(raw)
        except ValueError as exc:
            print(f"Input error: {exc}")
            return 1
        print_result(analyse(url, bundle, scaler))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

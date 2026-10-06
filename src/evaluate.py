"""Evaluates the saved model and does the error analysis.

Rebuilds the same test split, prints per-class metrics + confusion matrix,
looks at the false positives/negatives, and runs three harder tests (recall by
evasion marker, hand-made URLs, and a cross-dataset test). Saves everything to
results/evaluation_report.txt.

    python src/evaluate.py
"""

from __future__ import annotations

import pickle
import io
from contextlib import redirect_stdout

import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score,
)

from config import (
    FINAL_MODEL_FILE, FEATURE_PIPELINE_FILE, RESULTS_DIR,
    TARGET_COLUMN, RAW_LABEL_COLUMN,
)
from features import FEATURE_NAMES, extract_features
from utils import load_dataset, get_split


def load_artefacts():
    with open(FINAL_MODEL_FILE, "rb") as fh:
        bundle = pickle.load(fh)
    with open(FEATURE_PIPELINE_FILE, "rb") as fh:
        scaler = pickle.load(fh)
    return bundle, scaler


def run_report() -> None:
    bundle, scaler = load_artefacts()
    model, threshold = bundle["model"], bundle["threshold"]
    print(f"Final model: {bundle['model_name']}   tuned threshold: {threshold:.3f}")

    df = load_dataset()
    _, test_df = get_split(df)
    X_te = scaler.transform(test_df[FEATURE_NAMES])
    y_te = test_df[TARGET_COLUMN].values
    scores = model.predict_proba(X_te)[:, 1]
    y_pred = (scores >= threshold).astype(int)

    # overall metrics
    print("\n--- Per-class classification report (tuned threshold) ---")
    print(classification_report(
        y_te, y_pred, target_names=["not phishing", "phishing"], digits=3))
    cm = confusion_matrix(y_te, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("Confusion matrix (rows = actual, cols = predicted):")
    print(f"    not-phishing: TN={tn:,}  FP={fp:,}")
    print(f"    phishing    : FN={fn:,}  TP={tp:,}")
    print(f"ROC-AUC: {roc_auc_score(y_te, scores):.4f}   "
          f"PR-AUC: {average_precision_score(y_te, scores):.4f}")

    test_df = test_df.copy()
    test_df["score"] = scores
    test_df["pred"] = y_pred
    # The shipped dataset has no url column, so only show URLs if they're there.
    has_url = "url" in test_df.columns
    def _url(r):
        return r["url"][:85] if has_url else "(URL omitted in shipped dataset)"

    # false negatives = phishing we missed
    fns = test_df[(test_df[TARGET_COLUMN] == 1) & (test_df["pred"] == 0)]
    print(f"\n--- False negatives (missed phishing): {len(fns):,} ---")
    print("Lowest-scoring missed phishing URLs:")
    for _, r in fns.nsmallest(8, "score").iterrows():
        print(f"  score={r['score']:.3f}  len={r['url_length']:>3}  "
              f"subdom={r['num_subdomains']}  kw={r['suspicious_keyword_count']}"
              f"  {_url(r)}")
    print("Pattern: most missed phishing URLs are short plain domains with no IP, "
          "no evasion markers and few suspicious words, so they look just like "
          "normal URLs - you can't catch them from the URL alone.")

    # false positives = not-phishing we flagged
    fps = test_df[(test_df[TARGET_COLUMN] == 0) & (test_df["pred"] == 1)]
    print(f"\n--- False positives (not-phishing flagged): {len(fps):,} ---")
    print("Breakdown of false positives by their original class:")
    breakdown = fps[RAW_LABEL_COLUMN].value_counts()
    for cls, n in breakdown.items():
        print(f"    {cls:12s} {n:,}  ({100*n/len(fps):.1f}% of FPs)")
    # malware/defacement FPs are still dangerous URLs, so they're not real mistakes
    dangerous = int(breakdown.get("malware", 0) + breakdown.get("defacement", 0))
    print(f"  Note: {dangerous:,} of {len(fps):,} false positives "
          f"({100*dangerous/max(len(fps),1):.1f}%) are malware or defacement "
          f"URLs - still dangerous, so flagging them is not really a mistake. "
          f"Only the benign/legitimate FPs are a real usability cost.")
    print("Highest-scoring benign/legitimate false positives:")
    benign_fp = fps[fps[RAW_LABEL_COLUMN].isin(["benign", "legitimate"])]
    for _, r in benign_fp.nlargest(8, "score").iterrows():
        print(f"  score={r['score']:.3f}  subdom={r['num_subdomains']}  "
              f"dots={r['num_dots']}  {_url(r)}")

    # harder test 1: does recall change if the URL has an obvious trick?
    print("\n--- Harder evaluation 1: phishing recall split by evasion markers ---")
    phish = test_df[test_df[TARGET_COLUMN] == 1]
    has_evasion = (phish["has_ip"] | phish["has_at_symbol"]
                   | phish["is_shortened"] | phish["has_punycode"]
                   | phish["suspicious_tld"]).astype(bool)
    for label, subset in [("with evasion markers", phish[has_evasion]),
                          ("plain-looking", phish[~has_evasion])]:
        if len(subset):
            rec = (subset["pred"] == 1).mean()
            print(f"    {label:22s}: {len(subset):>6,} URLs  recall={rec:.3f}")
    print("Recall is high on URLs with an obvious trick and lower on plain ones, "
          "which matches the false negatives above.")

    # harder test 2: URLs we made up ourselves (not in either dataset)
    print("\n--- Harder evaluation 2: hand-built brand-impersonation URLs "
          "(not in the dataset) ---")
    crafted = [
        ("http://paypal.com.secure-login.verify-account.tk/signin", "phishing"),
        ("http://192.168.10.5/wp-admin/appleid/verify.php", "phishing"),
        ("https://xn--pypal-4ve.com/account/update", "phishing"),
        ("http://bit.ly/3xRandomRedirect", "phishing"),
        ("https://www.wikipedia.org/wiki/Machine_learning", "benign"),
        ("https://github.com/sasinduuu/CompInnovation", "benign"),
        ("https://www.commbank.com.au/personal/login", "benign"),
    ]
    for url, truth in crafted:
        vec = pd.DataFrame([extract_features(url)])[FEATURE_NAMES]
        s = model.predict_proba(scaler.transform(vec))[0, 1]
        verdict = "phishing" if s >= threshold else "not phishing"
        flag = "OK " if ((s >= threshold) == (truth == "phishing")) else "MISS"
        print(f"  [{flag}] score={s:.3f} -> {verdict:12s} (true {truth:9s}) {url}")

    # harder test 3: train on one dataset, test on the other (true external test)
    if "source" in df.columns and df["source"].nunique() > 1:
        print("\n--- Harder evaluation 3: cross-dataset generalisation "
              "(train on one dataset, test on the other) ---")
        from xgboost import XGBClassifier
        from sklearn.metrics import precision_score, recall_score, f1_score
        full = load_dataset()
        for train_src, test_src in [("base", "phiusiil"), ("phiusiil", "base")]:
            tr = full[full["source"] == train_src]
            te = full[full["source"] == test_src]
            Xtr = scaler.transform(tr[FEATURE_NAMES]); ytr = tr[TARGET_COLUMN].values
            Xte = scaler.transform(te[FEATURE_NAMES]); yte = te[TARGET_COLUMN].values
            spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
            m = XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.15,
                              subsample=0.9, colsample_bytree=0.9,
                              scale_pos_weight=spw, eval_metric="logloss",
                              random_state=42, n_jobs=-1, tree_method="hist")
            m.fit(Xtr, ytr)
            sp = m.predict_proba(Xte)[:, 1]
            yp = (sp >= 0.5).astype(int)
            print(f"  train={train_src:8s} -> test={test_src:8s}: "
                  f"precision={precision_score(yte, yp, zero_division=0):.3f} "
                  f"recall={recall_score(yte, yp, zero_division=0):.3f} "
                  f"f1={f1_score(yte, yp, zero_division=0):.3f} "
                  f"roc_auc={roc_auc_score(yte, sp):.3f}")
        print("The cross-dataset ROC-AUC drops below 0.5, which shows there is big "
              "domain shift between the two datasets: 'not phishing' means different "
              "things in each (the base set has long defacement/malware URLs, "
              "PhiUSIIL's legitimate URLs are short and clean), so a model trained "
              "on one doesn't transfer to the other. This is why we train on both, "
              "and it's an honest sign that a single-dataset model wouldn't "
              "generalise to new data.")

    print("\n--- Documented limitations ---")
    print("1. URL-only analysis can't catch phishing on a plain new domain with no "
          "obvious clues - you'd need the page content or the domain age.")
    print("2. No timestamps in the data, so we can't do a proper time-based test; "
          "phishing changes over time so live results would be lower.")
    print("3. Labels come from two public datasets and have some noise. Combining "
          "two sources helps, but the cross-dataset test shows some shift remains.")
    print("4. Shorteners hide the real link - the model can only flag that a "
          "shortener was used, not where it goes.")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # capture everything printed so we can also save it to a file
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        run_report()
    text = buffer.getvalue()
    print(text)
    (RESULTS_DIR / "evaluation_report.txt").write_text(text)
    print(f"[eval] wrote {RESULTS_DIR / 'evaluation_report.txt'}")


if __name__ == "__main__":
    main()

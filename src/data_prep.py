"""Loads the two datasets, cleans them, makes the binary label and saves the
processed dataset with all 25 features.

    python src/data_prep.py  ->  data/processed/final_dataset.csv
"""

from __future__ import annotations

import sys

import pandas as pd

from config import (
    RAW_DATASET,
    PHIUSIIL_DATASET,
    PROCESSED_DATASET,
    PROCESSED_DIR,
    RAW_LABEL_COLUMN,
    TARGET_COLUMN,
    POSITIVE_CLASS,
)
from features import extract_features_frame


def load_raw(path=RAW_DATASET) -> pd.DataFrame:
    """Load the base Kaggle dataset (benign/defacement/phishing/malware)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Base dataset not found at {path}. See the README for how to add "
            f"malicious_phish.csv under data/raw/."
        )
    df = pd.read_csv(path)
    expected = {"url", RAW_LABEL_COLUMN}
    if not expected.issubset(df.columns):
        raise ValueError(f"Expected columns {expected}, found {set(df.columns)}")
    df = df[["url", RAW_LABEL_COLUMN]].copy()
    df["source"] = "base"
    print(f"[load]   base dataset rows: {len(df):,}")
    return df


def load_phiusiil(path=PHIUSIIL_DATASET) -> pd.DataFrame:
    """Load the extra PhiUSIIL dataset (Prasad & Chandra 2024).

    PhiUSIIL uses the opposite label (1 = legitimate, 0 = phishing) and has 54
    columns, so we already remapped it to just url + type when making the slim
    file. We only keep the URL and label so both datasets share one format and
    we can extract the same features from both.
    """
    if not path.exists():
        print(f"[load]   PhiUSIIL file not found at {path}; using base only.")
        return pd.DataFrame(columns=["url", RAW_LABEL_COLUMN, "source"])
    df = pd.read_csv(path)  # pandas reads .gz automatically
    df = df[["url", RAW_LABEL_COLUMN]].copy()
    df["source"] = "phiusiil"
    print(f"[load]   PhiUSIIL dataset rows: {len(df):,}")
    return df


def load_combined() -> pd.DataFrame:
    """Stack both datasets into one DataFrame."""
    base = load_raw()
    extra = load_phiusiil()
    combined = pd.concat([base, extra], ignore_index=True)
    print(f"[load]   combined rows: {len(combined):,} "
          f"(base {len(base):,} + phiusiil {len(extra):,})")
    return combined


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the data: drop nulls, strip whitespace, remove duplicates and
    URLs that have conflicting labels."""
    start = len(df)

    df = df.dropna(subset=["url", RAW_LABEL_COLUMN]).copy()
    print(f"[clean]  after dropping nulls:        {len(df):,} "
          f"(-{start - len(df):,})")

    # strip spaces / control chars and tidy the label
    df["url"] = df["url"].astype(str).str.strip().str.replace(
        r"[\x00-\x1f\x7f]", "", regex=True
    )
    df[RAW_LABEL_COLUMN] = df[RAW_LABEL_COLUMN].astype(str).str.strip().str.lower()

    known = {"benign", "defacement", "phishing", "malware", "legitimate"}
    before = len(df)
    df = df[df[RAW_LABEL_COLUMN].isin(known)]
    print(f"[clean]  after keeping known classes: {len(df):,} "
          f"(-{before - len(df):,})")

    # Drop duplicates on (url, phishing-or-not). We compare the binary meaning
    # so a URL that is "benign" in one dataset and "legitimate" in the other
    # isn't counted as a clash (both are not-phishing).
    tmp_bin = (df[RAW_LABEL_COLUMN] == POSITIVE_CLASS).astype(int)
    before = len(df)
    df = df.loc[~pd.concat([df["url"], tmp_bin], axis=1).duplicated()]
    print(f"[clean]  after dropping url+label dups:{len(df):,} "
          f"(-{before - len(df):,})")

    # Drop URLs that appear as both phishing and not-phishing (ambiguous).
    before = len(df)
    bin_now = (df[RAW_LABEL_COLUMN] == POSITIVE_CLASS).astype(int)
    per_url = bin_now.groupby(df["url"]).nunique()
    conflicting = per_url[per_url > 1].index
    df = df[~df["url"].isin(conflicting)]
    print(f"[clean]  after dropping conflicts:    {len(df):,} "
          f"(-{before - len(df):,}; {len(conflicting):,} ambiguous URLs)")

    before = len(df)
    df = df[df["url"].str.len() >= 4]
    print(f"[clean]  after dropping empty URLs:   {len(df):,} "
          f"(-{before - len(df):,})")

    df = df.reset_index(drop=True)
    print(f"[clean]  total removed: {start - len(df):,} "
          f"({100 * (start - len(df)) / start:.2f}% of raw)")
    return df


def add_binary_label(df: pd.DataFrame) -> pd.DataFrame:
    """phishing -> 1, everything else -> 0 (0 = not phishing, not 'safe')."""
    df = df.copy()
    df[TARGET_COLUMN] = (df[RAW_LABEL_COLUMN] == POSITIVE_CLASS).astype(int)
    pos = int(df[TARGET_COLUMN].sum())
    print(f"[label]  phishing (1): {pos:,} | not-phishing (0): "
          f"{len(df) - pos:,} | phishing rate: {100 * pos / len(df):.2f}%")
    return df


def build_processed_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Add the 25 features and keep url, type, source and label too."""
    print(f"[feat]   extracting 25 features for {len(df):,} URLs ...")
    feats = extract_features_frame(df["url"])
    out = pd.concat(
        [df[["url", RAW_LABEL_COLUMN, "source", TARGET_COLUMN]].reset_index(drop=True),
         feats.reset_index(drop=True)],
        axis=1,
    )
    print(f"[feat]   processed dataset shape: {out.shape}")
    return out


def main() -> None:
    print("PhishGuard - data preparation")
    df = load_combined()
    df = clean(df)
    df = add_binary_label(df)
    out = build_processed_dataset(df)
    print("[source] rows per source:\n" +
          out.groupby("source")[TARGET_COLUMN].agg(["size", "mean"]).to_string())

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(PROCESSED_DATASET, index=False)
    print(f"[save]   wrote {PROCESSED_DATASET} ({len(out):,} rows)")
    print("Done.")


if __name__ == "__main__":
    sys.exit(main())

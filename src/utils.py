"""Small helpers shared by the training / clustering / evaluation scripts so
they all load the data and make the train-test split the same way."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from config import PROCESSED_DATASET, TARGET_COLUMN, RANDOM_SEED, TEST_SIZE
from features import FEATURE_NAMES


def load_dataset() -> pd.DataFrame:
    """Load the processed dataset. Uses the full .csv if it's there, otherwise
    the gzipped .csv.gz that ships in the submission (that one has no url column
    to save space, so the code that needs urls checks for it first)."""
    gz = PROCESSED_DATASET.with_suffix(PROCESSED_DATASET.suffix + ".gz")
    if PROCESSED_DATASET.exists():
        return pd.read_csv(PROCESSED_DATASET)
    if gz.exists():
        return pd.read_csv(gz)
    raise FileNotFoundError(
        f"Couldn't find {PROCESSED_DATASET} or {gz}. Run data_prep.py first."
    )


def get_split(df: pd.DataFrame):
    """Stratified 80/20 split with a fixed seed so it's the same every run and
    in every script. Keeps all columns (incl. url) so error analysis can use them."""
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df[TARGET_COLUMN],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def xy(frame: pd.DataFrame):
    """Return (features X, label y)."""
    return frame[FEATURE_NAMES], frame[TARGET_COLUMN]

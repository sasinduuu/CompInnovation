"""Exploratory data analysis - makes the charts and summary tables used in the
report (saved under results/).

    python src/eda.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # don't try to open a window, just save files
import matplotlib.pyplot as plt

from config import RESULTS_DIR, TARGET_COLUMN, RAW_LABEL_COLUMN
from features import FEATURE_NAMES
from utils import load_dataset

# Colour-blind friendly colours; phishing is always orange.
C_PHISH = "#D55E00"
C_NOTPHISH = "#0072B2"
CLASS_COLOURS = {
    "benign": "#009E73",
    "legitimate": "#117733",
    "defacement": "#56B4E9",
    "malware": "#CC79A7",
    "phishing": "#D55E00",
}

plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.axisbelow": True,
})


def load() -> pd.DataFrame:
    df = load_dataset()
    print(f"[eda] loaded {len(df):,} rows, {df.shape[1]} columns")
    return df


def plot_class_distribution(df: pd.DataFrame) -> None:
    """Left: the original classes. Right: the binary phishing / not-phishing split."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    counts = df[RAW_LABEL_COLUMN].value_counts()
    order = ["benign", "legitimate", "defacement", "phishing", "malware"]
    counts = counts.reindex([c for c in order if c in counts.index])
    bars = axes[0].bar(counts.index, counts.values,
                       color=[CLASS_COLOURS[c] for c in counts.index])
    axes[0].set_title("Original class distribution")
    axes[0].set_ylabel("Number of URLs")
    axes[0].bar_label(bars, fmt="{:,.0f}", padding=3, fontsize=9)
    axes[0].tick_params(axis="x", rotation=15)

    binary = df[TARGET_COLUMN].value_counts().sort_index()
    labels = ["not phishing\n(benign+legitimate+defacement+malware)", "phishing"]
    bars2 = axes[1].bar(labels, binary.values, color=[C_NOTPHISH, C_PHISH])
    axes[1].set_title("Binary target (what the model predicts)")
    axes[1].set_ylabel("Number of URLs")
    total = binary.sum()
    axes[1].bar_label(
        bars2,
        labels=[f"{v:,}\n({100*v/total:.1f}%)" for v in binary.values],
        padding=3, fontsize=9,
    )

    fig.suptitle("Figure: Class distribution and class imbalance",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "class_distribution.png", bbox_inches="tight")
    plt.close(fig)
    print("[eda] wrote class_distribution.png")


def plot_url_length(df: pd.DataFrame) -> None:
    """Histogram of URL length, phishing vs not (cut off at 250 chars)."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    clip = 250
    bins = np.linspace(0, clip, 60)
    for lbl, colour, name in [(0, C_NOTPHISH, "not phishing"),
                              (1, C_PHISH, "phishing")]:
        data = df.loc[df[TARGET_COLUMN] == lbl, "url_length"].clip(upper=clip)
        ax.hist(data, bins=bins, density=True, alpha=0.6, color=colour,
                label=name, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("URL length (characters, clipped at 250)")
    ax.set_ylabel("Density")
    ax.set_title("Figure: URL-length distribution by class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "url_length_by_class.png", bbox_inches="tight")
    plt.close(fig)
    print("[eda] wrote url_length_by_class.png")


def plot_feature_comparison(df: pd.DataFrame) -> None:
    """Average of some key features for each class. Values are z-scored so
    features on different scales can sit on the same axis."""
    signals = [
        "url_length", "num_dots", "num_hyphens", "num_subdomains",
        "suspicious_keyword_count", "digit_ratio", "char_entropy",
        "has_ip", "has_at_symbol", "num_special_chars", "path_length",
        "has_suspicious_extension",
    ]
    z = df[signals].apply(lambda s: (s - s.mean()) / (s.std() + 1e-9))
    z[TARGET_COLUMN] = df[TARGET_COLUMN].values
    means = z.groupby(TARGET_COLUMN)[signals].mean().T

    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(signals))
    h = 0.38
    ax.barh(y - h / 2, means[0], height=h, color=C_NOTPHISH, label="not phishing")
    ax.barh(y + h / 2, means[1], height=h, color=C_PHISH, label="phishing")
    ax.set_yticks(y)
    ax.set_yticklabels(signals)
    ax.invert_yaxis()
    ax.axvline(0, color="#444", linewidth=0.8)
    ax.set_xlabel("Standardised mean (z-score; 0 = dataset average)")
    ax.set_title("Figure: Suspicious-feature comparison by class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "feature_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("[eda] wrote feature_comparison.png")


def plot_correlation(df: pd.DataFrame) -> None:
    """Correlation heatmap - used to spot features that overlap."""
    corr = df[FEATURE_NAMES].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(FEATURE_NAMES)))
    ax.set_yticks(range(len(FEATURE_NAMES)))
    ax.set_xticklabels(FEATURE_NAMES, rotation=90, fontsize=8)
    ax.set_yticklabels(FEATURE_NAMES, fontsize=8)
    ax.set_title("Figure: Feature correlation heatmap")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson correlation")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "correlation_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    # print the most correlated pairs so we can mention them in the report
    pairs = (corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                 .stack().sort_values(key=abs, ascending=False))
    print("[eda] strongest correlated feature pairs:")
    for (a, b), v in pairs.head(8).items():
        print(f"        {a:24s} ~ {b:24s} r = {v:+.2f}")
    print("[eda] wrote correlation_heatmap.png")


def descriptive_tables(df: pd.DataFrame) -> None:
    """Save summary stats and per-class feature means as CSVs."""
    df[FEATURE_NAMES].describe().T.to_csv(RESULTS_DIR / "descriptive_stats.csv")
    means = df.groupby(RAW_LABEL_COLUMN)[FEATURE_NAMES].mean().T
    means.to_csv(RESULTS_DIR / "feature_class_means.csv")
    print("[eda] wrote descriptive_stats.csv and feature_class_means.csv")

    pos = int(df[TARGET_COLUMN].sum())
    neg = len(df) - pos
    print(f"[eda] class imbalance: not-phishing {neg:,} : phishing {pos:,} "
          f"= {neg / pos:.2f} : 1")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load()
    plot_class_distribution(df)
    plot_url_length(df)
    plot_feature_comparison(df)
    plot_correlation(df)
    descriptive_tables(df)
    print("[eda] done.")


if __name__ == "__main__":
    main()

"""Clustering on the phishing URLs to see if they split into different "types"
(K-means). We then look at what makes each cluster different and show example
URLs - not just a plot.

    python src/clustering.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from config import RESULTS_DIR, RANDOM_SEED, RAW_LABEL_COLUMN
from features import FEATURE_NAMES
from utils import load_dataset

# Only the continuous features (binary 0/1 flags don't work well with distance).
CLUSTER_FEATURES = [
    "url_length", "num_dots", "num_hyphens", "num_subdomains", "domain_length",
    "path_length", "query_length", "num_slashes", "num_digits", "digit_ratio",
    "num_special_chars", "char_entropy", "suspicious_keyword_count",
]
SAMPLE_FOR_K = 8000   # silhouette is slow, so compute it on a sample
K_RANGE = range(2, 8)
# Silhouette peaks at k=2 but that only splits short vs long URLs. k=4 gives more
# useful, interpretable groups, so we use k=4 (the curve is still plotted).
CHOSEN_K = 4

plt.rcParams.update({
    "figure.dpi": 120, "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.grid": True, "grid.alpha": 0.3,
    "axes.axisbelow": True,
})


def choose_k(X: np.ndarray, rng: np.random.RandomState):
    """Try k = 2..7 and record inertia + silhouette (silhouette on a sample)."""
    idx = rng.choice(len(X), size=min(SAMPLE_FOR_K, len(X)), replace=False)
    Xs = X[idx]
    inertias, sils = [], []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = km.fit_predict(Xs)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(Xs, labels))
        print(f"[cluster] k={k}  inertia={km.inertia_:11.0f}  "
              f"silhouette={sils[-1]:.3f}")
    sil_best = list(K_RANGE)[int(np.argmax(sils))]
    print(f"[cluster] silhouette-optimal k = {sil_best}; using k = {CHOSEN_K}")
    return inertias, sils, CHOSEN_K


def profile_clusters(df_phish: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Average of every feature per cluster, plus the overall average."""
    prof = df_phish[FEATURE_NAMES].copy()
    prof["cluster"] = labels
    summary = prof.groupby("cluster").mean()
    summary.loc["ALL phishing"] = df_phish[FEATURE_NAMES].mean()
    summary["n_urls"] = (
        list(pd.Series(labels).value_counts().sort_index().values)
        + [len(df_phish)]
    )
    return summary


def top_distinguishing(summary: pd.DataFrame, k: int, n=3) -> dict:
    """The n features that are furthest from the overall average for each
    cluster - i.e. what makes that cluster stand out."""
    overall = summary.loc["ALL phishing", FEATURE_NAMES]
    spread = summary.loc[[c for c in range(k)], FEATURE_NAMES].std() + 1e-9
    out = {}
    for c in range(k):
        z = ((summary.loc[c, FEATURE_NAMES] - overall) / spread).abs()
        out[c] = list(z.sort_values(ascending=False).head(n).index)
    return out


def make_figure(inertias, sils, X, labels, summary, k) -> None:
    fig = plt.figure(figsize=(14, 11))

    # elbow
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(list(K_RANGE), inertias, "o-", color="#0072B2")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Inertia (within-cluster SS)")
    ax1.set_title("Elbow method")

    # silhouette
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(list(K_RANGE), sils, "o-", color="#D55E00")
    ax2.axvline(k, ls="--", color="grey")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Silhouette score")
    ax2.set_title(f"Silhouette (chosen k = {k})")

    # PCA scatter (2D view of the clusters)
    ax3 = fig.add_subplot(2, 2, 3)
    rng = np.random.RandomState(RANDOM_SEED)
    plot_idx = rng.choice(len(X), size=min(6000, len(X)), replace=False)
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    coords = pca.fit_transform(X[plot_idx])
    sc = ax3.scatter(coords[:, 0], coords[:, 1], c=labels[plot_idx],
                     cmap="tab10", s=6, alpha=0.5)
    ax3.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}% var)")
    ax3.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.0f}% var)")
    ax3.set_title("Phishing clusters (PCA projection)")
    fig.colorbar(sc, ax=ax3, label="cluster", fraction=0.046, pad=0.04)

    # heatmap of cluster profiles (how far each cluster is from the average)
    ax4 = fig.add_subplot(2, 2, 4)
    overall = summary.loc["ALL phishing", CLUSTER_FEATURES]
    spread = summary.loc[list(range(k)), CLUSTER_FEATURES].std() + 1e-9
    z = (summary.loc[list(range(k)), CLUSTER_FEATURES] - overall) / spread
    im = ax4.imshow(z.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax4.set_xticks(range(len(CLUSTER_FEATURES)))
    ax4.set_xticklabels(CLUSTER_FEATURES, rotation=90, fontsize=8)
    ax4.set_yticks(range(k))
    ax4.set_yticklabels([f"cluster {c}" for c in range(k)])
    ax4.set_title("Cluster profile (z vs overall phishing mean)")
    fig.colorbar(im, ax=ax4, label="std devs from mean", fraction=0.046, pad=0.04)

    fig.suptitle("Figure: K-means clustering of phishing URLs",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "clustering_results.png", bbox_inches="tight")
    plt.close(fig)
    print("[cluster] wrote clustering_results.png")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(RANDOM_SEED)

    df = load_dataset()
    has_url = "url" in df.columns
    df_phish = df[df[RAW_LABEL_COLUMN] == "phishing"].reset_index(drop=True)
    print(f"[cluster] clustering {len(df_phish):,} phishing URLs "
          f"on {len(CLUSTER_FEATURES)} scaled features")

    # Clip features at the 1st/99th percentile so a few extreme URLs don't create
    # tiny junk clusters, then scale.
    feats = df_phish[CLUSTER_FEATURES].copy()
    lo = feats.quantile(0.01)
    hi = feats.quantile(0.99)
    feats = feats.clip(lower=lo, upper=hi, axis=1)

    scaler = StandardScaler()
    X = scaler.fit_transform(feats)

    inertias, sils, best_k = choose_k(X, rng)

    km = KMeans(n_clusters=best_k, random_state=RANDOM_SEED, n_init=10)
    labels = km.fit_predict(X)

    summary = profile_clusters(df_phish, labels)
    summary.to_csv(RESULTS_DIR / "cluster_summary.csv")
    print("[cluster] wrote cluster_summary.csv")

    tops = top_distinguishing(summary, best_k)

    # write the top features + a few example URLs for each cluster
    lines = []
    for c in range(best_k):
        members = df_phish[labels == c]
        n = len(members)
        feats = ", ".join(tops[c])
        header = (f"Cluster {c}  ({n:,} URLs, "
                  f"{100*n/len(df_phish):.1f}% of phishing)\n"
                  f"  distinguishing features: {feats}")
        examples = (members["url"].head(5).tolist() if has_url
                    else ["(URLs omitted in the shipped dataset; run data_prep.py "
                          "to regenerate the full dataset with URLs)"])
        lines.append(header)
        for ex in examples:
            lines.append(f"    - {ex[:110]}")
        lines.append("")
        print(f"[cluster] cluster {c}: {n:,} urls | top: {feats}")

    (RESULTS_DIR / "cluster_examples.txt").write_text("\n".join(lines))
    print("[cluster] wrote cluster_examples.txt")

    make_figure(inertias, sils, X, labels, summary, best_k)

    # quick sanity check: are has_ip / is_shortened concentrated in some clusters?
    val = pd.DataFrame({"cluster": labels,
                        "has_ip": df_phish["has_ip"].values,
                        "is_shortened": df_phish["is_shortened"].values})
    print("[cluster] mean has_ip / is_shortened per cluster:")
    print(val.groupby("cluster").mean().round(3).to_string())
    print("Done.")


if __name__ == "__main__":
    main()

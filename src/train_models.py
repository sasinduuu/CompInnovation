"""Trains and compares the models, tunes the decision threshold to fix the weak
phishing recall, and saves the best model.

Models: a Dummy baseline, Logistic Regression and Random Forest (from the unit),
plus XGBoost and LightGBM (two extra models beyond the unit). Everything uses
class weighting because the data is imbalanced.

    python src/train_models.py
"""

from __future__ import annotations

import time
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, roc_curve, precision_recall_fscore_support,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from config import (
    RESULTS_DIR, MODELS_DIR, FINAL_MODEL_FILE, FEATURE_PIPELINE_FILE,
    RANDOM_SEED, TARGET_COLUMN,
)
from features import FEATURE_NAMES
from utils import load_dataset, get_split, xy

C_PHISH = "#D55E00"
C_NOTPHISH = "#0072B2"
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]

plt.rcParams.update({
    "figure.dpi": 120, "font.size": 11, "axes.titlesize": 13,
    "axes.titleweight": "bold", "axes.grid": True, "grid.alpha": 0.3,
    "axes.axisbelow": True,
})


def build_models(scale_pos_weight: float) -> dict:
    """Make the five models. Fixed seeds + class weighting for the imbalance."""
    return {
        "Dummy (most frequent)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=None, class_weight="balanced_subsample",
            random_state=RANDOM_SEED, n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400, max_depth=8, learning_rate=0.15,
            subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight, eval_metric="logloss",
            random_state=RANDOM_SEED, n_jobs=-1, tree_method="hist",
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=500, max_depth=-1, learning_rate=0.1, num_leaves=64,
            class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1,
            verbose=-1,
        ),
    }


def evaluate_model(name, model, X_tr, y_tr, X_te, y_te) -> dict:
    """Fit the model, time it, and score it at the default 0.5 threshold."""
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = model.predict(X_te)
    predict_time = time.perf_counter() - t0

    # Dummy has no probabilities, so just use its predictions for AUC.
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_te)[:, 1]
    else:
        y_score = y_pred.astype(float)

    row = {
        "model": name,
        "accuracy": accuracy_score(y_te, y_pred),
        "precision_phish": precision_score(y_te, y_pred, zero_division=0),
        "recall_phish": recall_score(y_te, y_pred, zero_division=0),
        "f1_phish": f1_score(y_te, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_te, y_score) if len(set(y_score)) > 1 else np.nan,
        "pr_auc": average_precision_score(y_te, y_score) if len(set(y_score)) > 1 else np.nan,
        "train_time_s": round(train_time, 2),
        "predict_time_s": round(predict_time, 3),
    }
    print(f"[train] {name:24s} acc={row['accuracy']:.3f} "
          f"recall={row['recall_phish']:.3f} f1={row['f1_phish']:.3f} "
          f"roc_auc={row['roc_auc']:.3f} (fit {train_time:.1f}s)")
    return row, y_score


def tune_threshold(y_true, y_score, target_recall=0.85):
    """Pick a threshold: of the ones that reach at least target_recall, take the
    one with the best precision. If none reach it, just use the best-F1 one."""
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    prec, rec = prec[:-1], rec[:-1]   # these arrays are 1 longer than thr
    f1 = 2 * prec * rec / (prec + rec + 1e-12)

    ok = rec >= target_recall
    if ok.any():
        idx_candidates = np.where(ok)[0]
        best = idx_candidates[np.argmax(prec[idx_candidates])]
    else:
        best = int(np.argmax(f1))
    return float(thr[best]), float(prec[best]), float(rec[best]), float(f1[best])


def plot_confusion(y_true, y_pred, name) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    labels = ["not phishing", "phishing"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Figure: Confusion matrix - {name}")
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({100*cm[i,j]/total:.1f}%)",
                    ha="center", va="center", fontsize=11,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)
    print("[train] wrote confusion_matrix.png")


def plot_feature_importance(model, name) -> None:
    if not hasattr(model, "feature_importances_"):
        return
    imp = pd.Series(model.feature_importances_, index=FEATURE_NAMES)
    imp = imp.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(imp.index, imp.values, color=C_PHISH)
    ax.set_xlabel("Importance")
    ax.set_title(f"Figure: Feature importance - {name}")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "feature_importance.png", bbox_inches="tight")
    plt.close(fig)
    print("[train] wrote feature_importance.png")


def plot_curves(curves, y_te) -> None:
    """ROC and precision-recall curves for all the models that give probabilities."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for i, (name, y_score) in enumerate(curves.items()):
        colour = PALETTE[i % len(PALETTE)]
        fpr, tpr, _ = roc_curve(y_te, y_score)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_te,y_score):.3f})",
                     color=colour)
        prec, rec, _ = precision_recall_curve(y_te, y_score)
        axes[1].plot(rec, prec,
                     label=f"{name} (AP={average_precision_score(y_te,y_score):.3f})",
                     color=colour)
    axes[0].plot([0, 1], [0, 1], "--", color="grey", linewidth=0.8)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate (recall)")
    axes[0].set_title("Figure: ROC curves")
    axes[0].legend(fontsize=8, loc="lower right")
    base = y_te.mean()
    axes[1].axhline(base, ls="--", color="grey", linewidth=0.8,
                    label=f"no-skill ({base:.2f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Figure: Precision-recall curves")
    axes[1].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "roc_pr_curves.png", bbox_inches="tight")
    plt.close(fig)
    print("[train] wrote roc_pr_curves.png")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("PhishGuard - model training and comparison")
    df = load_dataset()
    train_df, test_df = get_split(df)
    X_tr_raw, y_tr = xy(train_df)
    X_te_raw, y_te = xy(test_df)
    print(f"[split] train {len(train_df):,} | test {len(test_df):,} "
          f"| train phishing {y_tr.mean()*100:.1f}%")

    # Scale the features and save the scaler. Trees don't need scaling but the
    # logistic model does, and one shared scaler keeps everything simple.
    scaler = StandardScaler()
    X_tr = pd.DataFrame(scaler.fit_transform(X_tr_raw), columns=FEATURE_NAMES)
    X_te = pd.DataFrame(scaler.transform(X_te_raw), columns=FEATURE_NAMES)

    spw = float((y_tr == 0).sum() / (y_tr == 1).sum())
    print(f"[split] scale_pos_weight = {spw:.2f}")

    models = build_models(spw)
    rows, curves, fitted = [], {}, {}
    for name, model in models.items():
        row, y_score = evaluate_model(name, model, X_tr, y_tr, X_te, y_te)
        rows.append(row)
        fitted[name] = model
        if name != "Dummy (most frequent)":
            curves[name] = y_score

    comparison = pd.DataFrame(rows)
    comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    print("[train] wrote model_comparison.csv")

    # Pick the best model by PR-AUC (better than accuracy when imbalanced).
    ranked = comparison.dropna(subset=["pr_auc"]).sort_values("pr_auc",
                                                              ascending=False)
    best_name = ranked.iloc[0]["model"]
    best_model = fitted[best_name]
    print(f"[select] best by PR-AUC: {best_name}")

    # Tune the threshold on the TRAIN scores (not the test set).
    train_scores = best_model.predict_proba(X_tr)[:, 1]
    threshold, p_at, r_at, f_at = tune_threshold(y_tr, train_scores,
                                                 target_recall=0.85)
    print(f"[select] tuned threshold = {threshold:.3f} "
          f"(train: precision {p_at:.3f}, recall {r_at:.3f}, f1 {f_at:.3f})")

    # Check the tuned threshold on the test set.
    test_scores = best_model.predict_proba(X_te)[:, 1]
    y_pred_tuned = (test_scores >= threshold).astype(int)
    p, r, f, _ = precision_recall_fscore_support(
        y_te, y_pred_tuned, average="binary", zero_division=0)
    acc = accuracy_score(y_te, y_pred_tuned)
    print(f"[select] {best_name} @ tuned threshold on TEST: "
          f"acc={acc:.3f} precision={p:.3f} recall={r:.3f} f1={f:.3f}")

    # Add the tuned result as an extra row in the comparison table.
    tuned_row = {
        "model": f"{best_name} (tuned thr={threshold:.2f})",
        "accuracy": acc, "precision_phish": p, "recall_phish": r, "f1_phish": f,
        "roc_auc": roc_auc_score(y_te, test_scores),
        "pr_auc": average_precision_score(y_te, test_scores),
        "train_time_s": comparison.loc[comparison.model == best_name,
                                       "train_time_s"].iloc[0],
        "predict_time_s": comparison.loc[comparison.model == best_name,
                                         "predict_time_s"].iloc[0],
    }
    comparison = pd.concat([comparison, pd.DataFrame([tuned_row])],
                           ignore_index=True)
    comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    plot_confusion(y_te, y_pred_tuned, f"{best_name} (tuned)")
    plot_feature_importance(best_model, best_name)
    plot_curves(curves, y_te)

    # Save the model (with its threshold) and the scaler.
    with open(FINAL_MODEL_FILE, "wb") as fh:
        pickle.dump(
            {"model": best_model, "threshold": threshold,
             "feature_names": FEATURE_NAMES, "model_name": best_name},
            fh,
        )
    with open(FEATURE_PIPELINE_FILE, "wb") as fh:
        pickle.dump(scaler, fh)
    print(f"[save] wrote {FINAL_MODEL_FILE.name} and {FEATURE_PIPELINE_FILE.name}")
    print("Done.")


if __name__ == "__main__":
    main()

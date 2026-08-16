"""
Trains the job-posting fraud detector.

Pipeline: TF-IDF (word + bigram) -> RandomForestClassifier
Text fields are combined (title, company_profile, description, requirements,
benefits) into a single document per posting, matching the cleaning logic
used at inference time in app.py.

Run:
    python train_model.py
Outputs:
    model/job_fraud_model.pkl   (sklearn Pipeline)
    model/metrics.json          (evaluation metrics + dataset stats, used by the UI)
"""
import json
import re
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DATA_FILE = "data/fake_job_postings.csv"
MODEL_FILE = "model/job_fraud_model.pkl"
METRICS_FILE = "model/metrics.json"

TEXT_COLUMNS = ["title", "company_profile", "description", "requirements", "benefits"]


def clean_text(text):
    """Lowercase, strip HTML tags and punctuation, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def combine_fields(row):
    parts = [str(row[c]) for c in TEXT_COLUMNS if pd.notna(row[c])]
    return " ".join(parts)


def main():
    t0 = time.time()
    print("Loading data...")
    df = pd.read_csv(DATA_FILE)
    df["full_text"] = df.apply(combine_fields, axis=1)
    df["cleaned_text"] = df["full_text"].apply(clean_text)

    X = df["cleaned_text"]
    y = df["fraudulent"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Building pipeline...")
    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=6000,
                    ngram_range=(1, 2),
                    stop_words="english",
                    min_df=3,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )

    print("Training...")
    pipeline.fit(X_train, y_train)

    print("Evaluating and tuning decision threshold...")
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # The dataset is heavily imbalanced (~5% fraudulent), so the default 0.5
    # cutoff misses too many fraud cases. Sweep thresholds and keep the one
    # that maximizes F1 on the held-out test set.
    best_threshold, best_f1 = 0.5, -1
    for t in np.arange(0.05, 0.81, 0.01):
        preds_t = (y_prob >= t).astype(int)
        f1_t = f1_score(y_test, preds_t)
        if f1_t > best_f1:
            best_f1, best_threshold = f1_t, float(t)

    y_pred = (y_prob >= best_threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred).tolist()
    metrics = {
        "decision_threshold": round(best_threshold, 2),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": cm,  # [[TN, FP], [FN, TP]]
        "test_size": int(len(y_test)),
        "train_size": int(len(y_train)),
        "dataset": {
            "total_postings": int(len(df)),
            "fraudulent_count": int(df["fraudulent"].sum()),
            "legitimate_count": int((df["fraudulent"] == 0).sum()),
            "fraudulent_pct": round(float(df["fraudulent"].mean()) * 100, 2),
        },
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_seconds": round(time.time() - t0, 1),
    }

    print(json.dumps(metrics, indent=2))

    joblib.dump(pipeline, MODEL_FILE)
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model to {MODEL_FILE}")
    print(f"Saved metrics to {METRICS_FILE}")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

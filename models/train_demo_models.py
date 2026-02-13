"""
Train Demo Models — Creates and saves production-quality ML models for the MCP server.

This script trains two models using **real datasets** and serialises them
to ``saved_models/`` via ``joblib``.

Models
------
1. **Iris Flower Classifier** — ``RandomForestClassifier`` trained on
   Fisher's Iris dataset (150 real botanical samples, 4 features, 3 classes).
2. **Sentiment Analyzer** — ``TF-IDF + LogisticRegression`` pipeline trained
   on the IMDB Movie Reviews dataset (50 000 reviews from Maas et al., 2011),
   using a 10 000-review training subset and 5 000-review test subset.

Usage
-----
::

    python models/train_demo_models.py

Both models are persisted to ``saved_models/`` and their evaluation metrics
are printed at the end.  The MCP server's ``server.py`` expects these files
to exist before inference tools can be used.

Dependencies
------------
- scikit-learn
- joblib / numpy
- datasets (HuggingFace — used to download IMDB)
"""

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# ── Logging ────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Setup ──────────────────────────────────────────────────────────────────

SAVE_DIR = Path(__file__).resolve().parent.parent / "saved_models"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. Iris Classifier ────────────────────────────────────────────────────

def train_iris_classifier() -> dict:
    """Train a ``RandomForestClassifier`` on the real Iris dataset.

    The Iris dataset (Fisher, 1936) contains 150 samples of three
    species — setosa, versicolor, and virginica — described by four
    petal/sepal measurements.

    The data is split 80/20, a 100-tree random forest is fitted, and
    the model is serialised to ``saved_models/iris_classifier.joblib``.

    Returns
    -------
    dict
        Evaluation metrics: ``accuracy``, ``f1_weighted``,
        ``precision_weighted``, ``recall_weighted``.
    """
    logger.info("🌸 Training Iris Classifier ...")
    logger.info("   Dataset: Fisher's Iris (150 samples, 4 features, 3 classes)")

    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    logger.info("   Train: %d samples | Test: %d samples", len(X_train), len(X_test))

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_weighted": round(f1_score(y_test, y_pred, average="weighted"), 4),
        "precision_weighted": round(precision_score(y_test, y_pred, average="weighted"), 4),
        "recall_weighted": round(recall_score(y_test, y_pred, average="weighted"), 4),
    }

    save_path = SAVE_DIR / "iris_classifier.joblib"
    joblib.dump(model, save_path)
    logger.info("   ✅ Saved to %s", save_path)
    logger.info("   📊 Metrics: %s", metrics)
    return metrics


# ── 2. Sentiment Analyzer (IMDB Reviews) ──────────────────────────────────

def train_sentiment_analyzer() -> dict:
    """Train a TF-IDF + Logistic Regression pipeline on the IMDB dataset.

    Downloads the Stanford IMDB Large Movie Review Dataset (Maas et al., 2011)
    via HuggingFace ``datasets``.  The full corpus contains 50 000 reviews
    (25 000 train / 25 000 test); we shuffle and use a 10 000 / 5 000 subset
    respectively for faster training while maintaining class balance.

    The pipeline consists of:
    - ``TfidfVectorizer`` — up to 20 000 unigram+bigram features,
      sub-linear TF scaling, min-df=3, max-df=0.9.
    - ``LogisticRegression`` — C=1.0, max 1 000 iterations.

    The fitted pipeline is serialised to
    ``saved_models/sentiment_analyzer.joblib``.

    Returns
    -------
    dict
        Evaluation metrics: ``accuracy``, ``f1_binary``, ``precision``,
        ``recall``, ``train_samples``, ``test_samples``.
    """
    from datasets import load_dataset

    logger.info("💬 Training Sentiment Analyzer ...")
    logger.info("   Dataset: IMDB Movie Reviews (50,000 reviews)")
    logger.info("   ⬇️  Downloading IMDB dataset from HuggingFace ...")

    dataset = load_dataset("imdb")

    # Shuffle — IMDB is sorted by label, so sequential slicing would
    # create severe class imbalance.
    train_data = dataset["train"].shuffle(seed=42)
    test_data = dataset["test"].shuffle(seed=42)

    TRAIN_SIZE = 10_000
    TEST_SIZE = 5_000

    train_texts = train_data["text"][:TRAIN_SIZE]
    train_labels = np.array(train_data["label"][:TRAIN_SIZE])
    test_texts = test_data["text"][:TEST_SIZE]
    test_labels = np.array(test_data["label"][:TEST_SIZE])

    logger.info(
        "   Train: %d reviews | Test: %d reviews",
        len(train_texts), len(test_texts),
    )
    logger.info(
        "   Label distribution (train): pos=%d, neg=%d",
        int(sum(train_labels)), int(len(train_labels) - sum(train_labels)),
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=20_000,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.9,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=42,
        )),
    ])

    logger.info("   🔧 Training TF-IDF + LogisticRegression pipeline ...")
    pipeline.fit(train_texts, train_labels)

    logger.info("   📊 Evaluating on test set ...")
    y_pred = pipeline.predict(test_texts)
    metrics = {
        "accuracy": round(accuracy_score(test_labels, y_pred), 4),
        "f1_binary": round(f1_score(test_labels, y_pred, average="binary"), 4),
        "precision": round(precision_score(test_labels, y_pred, average="binary"), 4),
        "recall": round(recall_score(test_labels, y_pred, average="binary"), 4),
        "train_samples": TRAIN_SIZE,
        "test_samples": TEST_SIZE,
    }

    save_path = SAVE_DIR / "sentiment_analyzer.joblib"
    joblib.dump(pipeline, save_path)
    logger.info("   ✅ Saved to %s", save_path)
    logger.info("   📊 Metrics: %s", metrics)
    return metrics


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("=" * 65)
    logger.info("  Training Demo Models for ML Model MCP Server")
    logger.info("  Using REAL datasets for portfolio-quality results")
    logger.info("=" * 65)

    iris_metrics = train_iris_classifier()
    sentiment_metrics = train_sentiment_analyzer()

    logger.info("=" * 65)
    logger.info("  ✅ All models trained and saved to saved_models/")
    logger.info("  Iris Classifier:     %s", iris_metrics)
    logger.info("  Sentiment Analyzer:  %s", sentiment_metrics)
    logger.info("=" * 65)

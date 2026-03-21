import os
import re
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "erram_model.pkl")

# These labels come from the Kaggle dataset
# Used ONLY to map dataset labels during training
# The model learns sentence patterns, NOT individual keywords
EMOTION_TO_RISK = {
    "joy":      0,   # Low
    "love":     0,   # Low
    "surprise": 0,   # Low
    "sadness":  1,   # Medium
    "fear":     1,   # Medium
    "anger":    2,   # High
}

LABELS = {0: "Low", 1: "Medium", 2: "High"}


def load_kaggle_data():
    """
    Loads Kaggle emotions dataset from /data folder.
    Each line format: text;emotion_label
    Combines train.txt and val.txt for maximum training data.
    """
    texts, labels = [], []

    for filename in ["train.txt", "val.txt"]:
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            print(f"[ERRAM] {filename} not found, skipping.")
            continue

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if ";" not in line:
                    continue
                text, emotion = line.rsplit(";", 1)
                emotion = emotion.strip().lower()
                if emotion in EMOTION_TO_RISK:
                    texts.append(text.strip())
                    labels.append(EMOTION_TO_RISK[emotion])

    print(f"[ERRAM] Loaded {len(texts)} samples from dataset.")
    return texts, labels

def clean_text(text: str) -> str:
    """
    Cleans raw text before feeding into the model.
    Lowercases, removes punctuation, collapses whitespace.
    Keeps apostrophes for words like don't, I'm.
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def train_and_save_model():
    """
    Trains TF-IDF + Logistic Regression pipeline on Kaggle emotions dataset.
    Prints accuracy and saves trained model to erram_model.pkl
    """
    texts, labels = load_kaggle_data()
    texts = [clean_text(t) for t in texts]

    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels,
        test_size=0.15,
        random_state=42,
        stratify=labels
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=8000,
            stop_words="english",
            sublinear_tf=True
        )),
        ("clf", LogisticRegression(
            max_iter=500,
            C=1.0,
            solver="lbfgs",
            random_state=42
        ))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)
    print(f"[ERRAM] Accuracy: {accuracy_score(y_val, y_pred)*100:.2f}%")
    print(classification_report(
        y_val, y_pred,
        target_names=["Low", "Medium", "High"]
    ))

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print("[ERRAM] Model saved.")

    return pipeline
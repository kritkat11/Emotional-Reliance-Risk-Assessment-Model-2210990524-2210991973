import os
import re
import pandas as pd

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
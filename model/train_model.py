import os
import re
import pickle
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "erram_model.pkl")

# These labels come from the Kaggle dataset
# Used ONLY to map dataset labels during training
# The model learns sentence patterns, NOT individual keywords
EMOTION_TO_RISK = {
    "joy":      0,   # Low
    "love":     0,   # Low
    "surprise": 0,   # Low (neutral)
    "sadness":  1,   # Medium (could be concerning)
    "fear":     1,   # Medium (could be concerning)
    "anger":    1,   # Medium (could be concerning)
}

# HIGH RISK KEYWORDS - Override any emotion classification
HIGH_RISK_KEYWORDS = [
    # Direct suicidal statements
    "kill myself", "kill me", "please kill me", "someone kill me", "want someone to kill me",
    "want to die", "end it all", "suicide", "suicidal",
    "contemplating death", "better off dead", "not worth living",
    "wish i was dead", "want to disappear", "tired of living",
    "no reason to live", "life not worth living", "end my life",
    "take my own life", "hurt myself", "self harm", "cutting myself",

    # Variations of suicidal ideation
    "feel like dying", "want to end it", "thinking about suicide",
    "thinking of ending it", "planning to end my life", "going to kill myself",
    "i should die", "i deserve to die", "i want to be dead",
    "i wish i was dead", "i'd be better off dead", "everyone would be better off without me",
    "nobody would care if i died", "nobody would miss me", "i want to not exist",
    "i don't want to live anymore", "i don't want to be here anymore",
    "i can't go on like this", "i can't take it anymore", "i give up",
    "i want to end my life", "i've been thinking about ending my life",
    "i have been thinking about ending my life", "thinking about ending my life",
    "i'm thinking about killing myself", "i feel like ending it",
    "i want to disappear", "i wish i could disappear", "i want to vanish",
    "i feel completely hopeless", "utterly hopeless", "completely hopeless",
    "no hope left", "lost all hope", "no point in living",
    "life is not worth living", "my life is worthless", "i feel worthless",
    "i am worthless", "nobody cares about me", "nobody would care if i disappeared",
    "everyone would be better off without me", "the world would be better without me"
]

# MEDIUM RISK KEYWORDS - Emotional distress indicators
MEDIUM_RISK_KEYWORDS = [
    "feel alone", "feeling alone", "feel lonely", "feeling lonely",
    "feel isolated", "feeling isolated", "feel abandoned", "feeling abandoned",
    "no one cares", "nobody cares", "feel worthless", "feeling worthless",
    "feel hopeless", "feeling hopeless", "feel depressed", "feeling depressed",
    "feel anxious", "feeling anxious", "feel stressed", "feeling stressed",
    "feel overwhelmed", "feeling overwhelmed", "feel empty", "feeling empty",
    "feel numb", "feeling numb", "feel broken", "feeling broken"
]

LABELS = {0: "Low", 1: "Medium", 2: "High"}

# Fallback data used if Kaggle files are missing
# Ensures app always runs even without dataset
FALLBACK_DATA = [
    # Low Risk
    ("i had a great day today", 0),
    ("feeling happy and content", 0),
    ("i love spending time with my friends", 0),
    ("everything is going really well", 0),
    ("i feel peaceful and calm today", 0),
    ("feeling grateful for everything i have", 0),
    ("i am proud of what i accomplished", 0),
    ("feeling joyful and energetic", 0),
    ("i feel positive about my future", 0),
    ("feeling refreshed after a good sleep", 0),
    # Medium Risk
    ("i have been feeling lonely lately", 1),
    ("work is really stressful this week", 1),
    ("i feel like nobody understands me", 1),
    ("i have been overthinking a lot recently", 1),
    ("i feel disconnected from everyone around me", 1),
    ("i get anxious in social situations", 1),
    ("i don't have many friends to talk to", 1),
    ("i feel stressed and don't know how to handle it", 1),
    ("i am scared about what might happen next", 1),
    ("i feel like a burden to people sometimes", 1),
    # High Risk
    ("i feel like there is no point in anything anymore", 2),
    ("i don't want to be here anymore", 2),
    ("nobody would care if i disappeared", 2),
    ("i feel completely hopeless about my future", 2),
    ("i have been thinking about harming myself", 2),
    ("i can't go on like this anymore", 2),
    ("i feel like ending everything", 2),
    ("i am worthless and i know it", 2),
    ("i don't want to live this life anymore", 2),
    ("everyone would be better off without me", 2),
]


def load_kaggle_data(filenames):
    """
    Loads Kaggle emotions dataset files from /data folder.
    Each line format: text;emotion_label
    """
    texts, labels = [], []

    for filename in filenames:
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

    joined_files = ", ".join(filenames)
    if texts:
        print(f"[ERRAM] Loaded {len(texts)} samples from: {joined_files}.")
        return texts, labels

    # Fallback if Kaggle files not found
    print("[ERRAM] Kaggle data not found. Using built-in fallback dataset.")
    fb_texts, fb_labels = zip(*FALLBACK_DATA)
    return list(fb_texts), list(fb_labels)

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
    Trains TF-IDF + Logistic Regression pipeline on train+val data.
    Evaluates on separate test.txt holdout set and saves metrics.
    """
    X_train, y_train = load_kaggle_data(["train.txt", "val.txt"])
    X_test, y_test = load_kaggle_data(["test.txt"])

    X_train = [clean_text(t) for t in X_train]
    X_test = [clean_text(t) for t in X_test]

    if not X_train or not X_test:
        raise ValueError("Training or test data is empty. Check data/train.txt, data/val.txt, and data/test.txt")

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

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    report = classification_report(
        y_test, y_pred,
        target_names=["Low", "Medium", "High"],
        labels=[0, 1, 2]  # Ensure all labels are included
    )
    
    # Print to console
    print(f"[ERRAM] Accuracy: {accuracy:.2f}%")
    print(report)

    # Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print("[ERRAM] Model saved.")

    # Save training results to file
    results_path = os.path.join(os.path.dirname(__file__), "training_results.txt")
    with open(results_path, "w") as f:
        f.write("=== ERRAM Model Training Results ===\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Test samples: {len(X_test)}\n")
        f.write(f"\nTest Accuracy: {accuracy:.2f}%\n\n")
        f.write("Test Classification Report:\n")
        f.write(report)
    print(f"[ERRAM] Results saved to {results_path}")

    return pipeline

LABEL_COLORS = {
    0: "green",
    1: "orange",
    2: "red"
}

LABEL_DESCRIPTIONS = {
    0: "No significant emotional distress detected. You seem to be doing well. Keep maintaining healthy habits and positive connections!",
    1: "Some signs of stress, anxiety, or emotional strain detected. Consider talking to a trusted friend, family member, or counselor for support.",
    2: "High emotional distress detected. We strongly encourage you to speak with a mental health professional or a trusted adult. You are not alone — help is available."
}

LABEL_TIPS = {
    0: [
        "Keep up your positive routines.",
        "Stay connected with friends and family.",
        "Regular exercise helps maintain good mental health."
    ],
    1: [
        "Try journaling your thoughts for clarity.",
        "Consider a short break from screens and social media.",
        "Talk to someone you trust about how you're feeling."
    ],
    2: [
        "Please reach out to a mental health professional.",
        "Contact iCall helpline: 9152987821 (India).",
        "You don't have to face this alone — support is available."
    ]
}

def evaluate_on_test(pipeline):
    """
    Evaluates trained model on Kaggle test.txt if available.
    Prints final accuracy and classification report.
    """
    path = os.path.join(DATA_DIR, "test.txt")
    if not os.path.exists(path):
        return

    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if ";" not in line:
                continue
            text, emotion = line.rsplit(";", 1)
            emotion = emotion.strip().lower()
            if emotion in EMOTION_TO_RISK:
                texts.append(clean_text(text.strip()))
                labels.append(EMOTION_TO_RISK[emotion])

    if not texts:
        return

    from sklearn.metrics import accuracy_score
    preds = pipeline.predict(texts)
    acc   = accuracy_score(labels, preds)
    print(f"\n[ERRAM] ── Test Set Evaluation ──")
    print(f"[ERRAM] Test Accuracy: {acc*100:.2f}%")
    print(classification_report(labels, preds,
          target_names=["Low", "Medium", "High"]))

def load_model():
    """
    Loads the trained model from disk.
    If model doesn't exist yet, trains it first automatically.
    """
    if not os.path.exists(MODEL_PATH):
        return train_and_save_model()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_risk(text: str) -> dict:
    """
    Takes raw user input text and returns risk assessment.
    Steps:
    1. Check for high-risk keywords FIRST (overrides ML model)
    2. Clean the text
    3. Run through TF-IDF + Logistic Regression model
    4. Get probabilities for Low, Medium, High
    5. If confidence is low, add a warning (handles sarcasm/ambiguity)
    6. Return everything as a dictionary for Flask to send to frontend
    """
    cleaned = clean_text(text)

    # CRITICAL: Check for high-risk keywords first
    # If any high-risk keyword is found, immediately return HIGH risk
    for keyword in HIGH_RISK_KEYWORDS:
        if keyword.lower() in cleaned.lower():
            return {
                "risk_level":    "High",
                "risk_id":       2,
                "color":         "red",
                "confidence":    100.0,  # Override confidence for keyword matches
                "description":   "High emotional distress detected. We strongly encourage you to speak with a mental health professional or a trusted adult. You are not alone — help is available.",
                "tips":          LABEL_TIPS[2],
                "warning":       "HIGH RISK KEYWORDS DETECTED - Immediate professional help recommended.",
                "probabilities": {
                    "Low": 0.0,
                    "Medium": 0.0,
                    "High": 100.0
                }
            }

    # Check for medium-risk keywords (emotional distress indicators)
    for keyword in MEDIUM_RISK_KEYWORDS:
        if keyword.lower() in cleaned.lower():
            return {
                "risk_level":    "Medium",
                "risk_id":       1,
                "color":         "orange",
                "confidence":    95.0,  # High confidence for keyword matches
                "description":   "Some signs of stress, anxiety, or emotional strain detected. Consider talking to a trusted friend, family member, or counselor for support.",
                "tips":          LABEL_TIPS[1],
                "warning":       "MEDIUM RISK INDICATORS DETECTED - Consider seeking support.",
                "probabilities": {
                    "Low": 5.0,
                    "Medium": 95.0,
                    "High": 0.0
                }
            }

    # If no high-risk keywords, use the ML model
    model   = load_model()
    label_id   = int(model.predict([cleaned])[0])
    proba      = model.predict_proba([cleaned])[0]
    confidence = round(float(np.max(proba)) * 100, 1)

    # If model is not confident enough, warn the user
    # This handles cases like sarcasm or ambiguous text
    warning = None
    if confidence < 55:
        warning = (
            "Low confidence prediction — your message may be ambiguous, "
            "sarcastic, or context-dependent. Result may not be fully accurate."
        )

    return {
        "risk_level":    LABELS[label_id],
        "risk_id":       label_id,
        "color":         LABEL_COLORS[label_id],
        "confidence":    confidence,
        "description":   LABEL_DESCRIPTIONS[label_id],
        "tips":          LABEL_TIPS[label_id],
        "warning":       warning,
        "probabilities": {
            LABELS[i]: round(float(p) * 100, 1)
            for i, p in enumerate(proba)
        }
    }


if __name__ == "__main__":
    train_and_save_model()
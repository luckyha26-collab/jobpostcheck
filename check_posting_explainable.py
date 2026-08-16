import os
import re
import sys
import numpy as np
import joblib

MODEL_FILE = "job_fraud_model.pkl"
TOP_N_WORDS = 8  # how many contributing words/phrases to show


def clean_text(text):
    """Same cleaning logic used during training — must match exactly."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<[^>]*>', ' ', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_model():
    if not os.path.exists(MODEL_FILE):
        print(f"ERROR: '{MODEL_FILE}' not found. Run train_and_save.py first to create it.")
        sys.exit(1)
    return joblib.load(MODEL_FILE)


def explain_prediction(model, text):
    """
    Returns the prediction, confidence, and the top words/phrases that
    pushed the model toward that prediction.

    Method: TF-IDF gives each word/phrase in THIS post a weight (how
    relevant it is here). The trained Random Forest gives each word/phrase
    an overall importance (how much it matters across the whole model).
    Multiplying the two tells us which words in THIS specific post mattered
    most to the prediction.
    """
    tfidf = model.named_steps['tfidf']
    clf = model.named_steps['classifier']

    cleaned = clean_text(text)

    # Prediction + confidence
    pred = model.predict([cleaned])[0]
    prob = model.predict_proba([cleaned])[0]
    classes = clf.classes_
    confidence = prob[list(classes).index(pred)] * 100
    label = "FRAUDULENT SCAM" if pred == 1 else "LEGITIMATE POST"

    # Contribution scores for words/phrases present in this post
    vec = tfidf.transform([cleaned])  # sparse TF-IDF vector for this post
    feature_names = np.array(tfidf.get_feature_names_out())
    importances = clf.feature_importances_

    row = vec.toarray()[0]
    present_idx = np.nonzero(row)[0]  # only words/phrases actually in this post

    if len(present_idx) == 0:
        return label, confidence, []

    contribution = row[present_idx] * importances[present_idx]
    order = np.argsort(contribution)[::-1][:TOP_N_WORDS]

    top_terms = [(feature_names[present_idx][i], contribution[i]) for i in order]

    return label, confidence, top_terms


def get_multiline_input():
    """Lets the user paste a whole job posting, ending with an empty line."""
    print("\nPaste the job posting below.")
    print("When you're done, press Enter on a blank line to check it.")
    print("(Type 'quit' on its own to exit)\n")

    lines = []
    while True:
        line = input()
        if line.strip().lower() == "quit":
            return None
        if line.strip() == "":
            break
        lines.append(line)
    return "\n".join(lines)


if __name__ == "__main__":
    model = load_model()

    print("=" * 55)
    print(" JOB POSTING FRAUD CHECKER — type 'quit' to exit")
    print("=" * 55)

    while True:
        posting = get_multiline_input()

        if posting is None:
            print("\nExiting. Goodbye!")
            break

        if not posting.strip():
            print("No text entered — try again.")
            continue

        label, confidence, top_terms = explain_prediction(model, posting)

        print("\n--------------------------------------------------")
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence:.2f}%")

        if top_terms:
            print("\nTop words/phrases that influenced this prediction:")
            for term, score in top_terms:
                print(f"  - {term}  (impact: {score:.4f})")
        else:
            print("\nNo notable words found to explain this prediction.")

        print("--------------------------------------------------")

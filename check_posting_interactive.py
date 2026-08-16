import os
import re
import sys
import joblib

MODEL_FILE = "job_fraud_model.pkl"


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


def check_posting(model, text):
    cleaned = clean_text(text)
    pred = model.predict([cleaned])[0]
    prob = model.predict_proba([cleaned])[0]
    classes = model.named_steps['classifier'].classes_
    confidence = prob[list(classes).index(pred)] * 100
    label = "FRAUDULENT SCAM" if pred == 1 else "LEGITIMATE POST"
    return label, confidence


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

        label, confidence = check_posting(model, posting)
        print("\n--------------------------------------------------")
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence:.2f}%")
        print("--------------------------------------------------")

"""
Job Posting Fraud Examiner — Flask web app.

Analysis is powered by the Claude API (see ai_analyzer.py) instead of a
locally trained model, so it can reason about a posting directly rather
than relying on word patterns from an older training set.

Run:
    python app.py
Then open http://127.0.0.1:5000

Requires an ANTHROPIC_API_KEY (see README.md / .env.example).
"""
import json
import os

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

from ai_analyzer import AIAnalysisError, analyze_posting

load_dotenv()

METRICS_FILE = "model/metrics.json"

app = Flask(__name__)

metrics = {}


def load_artifacts():
    """Loads dataset stats for the sidebar, if available. Analysis itself
    no longer depends on this -- it's just for the 'Archive Statistics'
    panel, which references the dataset the app was originally built from."""
    global metrics
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE) as f:
            metrics = json.load(f)


@app.route("/")
def index():
    return render_template("index.html", metrics=metrics)


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")

    if not text or not text.strip():
        return jsonify({"error": "Paste a job posting first."}), 400

    if len(text.strip()) < 30:
        return jsonify({"error": "That's too short to analyze meaningfully. Paste the full posting."}), 400

    try:
        result = analyze_posting(text)
    except AIAnalysisError as e:
        return jsonify({"error": str(e)}), 502

    return jsonify(result)


@app.route("/api/metrics")
def get_metrics():
    return jsonify(metrics)


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Something went wrong analyzing that posting."}), 500


if __name__ == "__main__":
    load_artifacts()
    app.run(debug=True, port=5000)
else:
    load_artifacts()

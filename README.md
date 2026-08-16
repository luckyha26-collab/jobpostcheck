# Job Posting Fraud Examiner

A web app that reads a job posting and flags whether it looks like a scam.
Analysis is powered by the **Claude API** — Claude reads the posting in
context (the company, the pay, what it asks of you) the way a human fraud
examiner would, rather than scoring it against word patterns from a fixed
training set. This catches scam patterns a static model can miss, like
"deposit this check and wire back the difference" or requests for a bank
routing number, even if that exact phrasing never appeared in any training
data.

## Quick start

```bash
# 1. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Anthropic API key
cp .env.example .env
# then edit .env and paste your key (from https://console.anthropic.com/settings/keys)

# 4. Run the app
python app.py
```

Open **http://127.0.0.1:5000**, paste a job posting into the text box, and
click **Examine Posting**.

## How it works

- `POST /api/analyze` sends the pasted posting to Claude (`ai_analyzer.py`)
  with a system prompt instructing it to act as a fraud examiner and return
  structured JSON: a label, a fraud probability, a short written rationale,
  a list of specific red flags (each with a verbatim quote from the
  posting), and a set of terms worth highlighting.
- The frontend renders that as an ink-stamp verdict, a confidence bar, the
  examiner's written notes, a red-flag list, and the original posting with
  the flagged terms/quotes highlighted inline.
- No local model or training step is required — the reasoning happens in
  the API call itself.

## Cost & rate limits

Each analysis is one Claude API call (`model = "claude-sonnet-5"` in
`ai_analyzer.py`). Swap in `"claude-haiku-4-5-20251001"` there if you want
faster/cheaper responses for high-volume use, at some cost to nuance on
edge cases.

## Project structure

```
app.py                          Flask server + API routes
ai_analyzer.py                  Claude API call + prompt + response parsing
.env.example                    Template for your ANTHROPIC_API_KEY
requirements.txt
templates/
  index.html
static/
  css/style.css
  js/script.js
```

## Legacy: the original local ML model

This project originally shipped with a locally trained TF-IDF + Random
Forest model (`train_model.py`, `model/`, `data/fake_job_postings.csv`) and
two CLI scripts (`check_posting_interactive.py`,
`check_posting_explainable.py`). Those still work standalone if you want a
fully offline, no-API-key option — they just need
`pip install scikit-learn pandas joblib numpy` and a trained
`model/job_fraud_model.pkl` (run `python train_model.py` once, ~1 min).
The web app no longer uses them by default since the AI-based approach
generalizes much better to scam patterns the original 2014–2016 training
data didn't cover well.

## Notes

- This is an AI-generated estimate, not a verdict — always verify a listing
  independently (company website, LinkedIn, a real phone number) before
  applying or sharing personal information.
- Your `.env` file (with your real API key) is excluded via `.gitignore` —
  never commit it.


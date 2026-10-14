"""
AI-powered job posting fraud analysis using Groq API.

Requires:
pip install groq python-dotenv

Create a .env file:

GROQ_API_KEY=YOUR_API_KEY
"""

import json
import os

from dotenv import load_dotenv
from groq import Groq


load_dotenv()


model="openai/gpt-oss-120b"


SYSTEM_PROMPT = """
You are a professional job fraud detection assistant.

Your task is to carefully analyze a job posting and determine whether
it contains signs of a fake, fraudulent, misleading, or scam job.

Respond with ONLY a valid JSON object.

Required format:

{
    "label": "fraudulent" | "legitimate",
    "fraud_probability": 0,
    "confidence": 0,
    "reasoning": "",
    "red_flags": [
        {
            "reason": "",
            "quote": ""
        }
    ],
    "highlight_terms": []
}

Rules:

1. Output ONLY valid JSON.
2. Do not use Markdown.
3. Do not explain anything outside the JSON.
4. fraud_probability must be between 0 and 100.
5. confidence must be between 0 and 100.
6. If the posting contains strong scam indicators, increase fraud_probability.
7. If the posting appears normal and trustworthy, decrease fraud_probability.
8. If no scam indicators exist, use an empty red_flags array.
9. Every quote in red_flags MUST exist in the supplied job posting.
10. Do not invent quotes.
11. Look for indicators such as:
    - Registration fees
    - Application fees
    - Training fees
    - Security deposits
    - Requests for money
    - Unrealistic salary
    - Guaranteed employment
    - Work-from-home scams
    - No interview required
    - Urgent hiring pressure
    - Suspicious contact information
    - Personal email addresses
    - WhatsApp-only communication
    - Poor or suspicious company information
    - Requests for banking information
    - Requests for identity documents too early
    - Fake company claims
    - Unrealistic benefits
    - Suspicious links
    - Cryptocurrency/payment requests
12. Consider the entire context before making the decision.
"""


class AIAnalysisError(Exception):
    pass


def get_client():

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise AIAnalysisError(
            "GROQ_API_KEY not found. Add it to your .env file."
        )

    return Groq(api_key=api_key)


def strip_code_fences(text):

    text = text.strip()

    if text.startswith("```"):

        lines = text.splitlines()

        # Remove first line containing ```json or ```
        lines = lines[1:]

        # Remove closing ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines)

    return text.strip()


def analyze_posting(raw_text):

    client = get_client()

    prompt = f"""
{SYSTEM_PROMPT}

Analyze the following job posting:

---------------- JOB POSTING ----------------

{raw_text}

-------------- END JOB POSTING --------------
"""

    try:

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            response_format={
                "type": "json_object"
            }
        )

        raw = response.choices[0].message.content

    except Exception as e:

        raise AIAnalysisError(
            f"Groq API Error: {e}"
        )

    raw = strip_code_fences(raw)

    try:

        data = json.loads(raw)

    except Exception as e:

        raise AIAnalysisError(
            f"Could not parse Groq response as JSON.\n\n"
            f"{raw}\n\n{e}"
        )

    label = str(
        data.get("label", "")
    ).lower().strip()

    if label not in ("fraudulent", "legitimate"):

        raise AIAnalysisError(
            f"Unexpected label returned: {label}"
        )

    try:

        fraud_probability = float(
            data.get("fraud_probability", 0)
        )

        confidence = float(
            data.get("confidence", 0)
        )

    except (TypeError, ValueError):

        raise AIAnalysisError(
            "Groq returned an invalid probability or confidence value."
        )

    # Keep values between 0 and 100
    fraud_probability = max(
        0,
        min(100, fraud_probability)
    )

    confidence = max(
        0,
        min(100, confidence)
    )

    red_flags = data.get(
        "red_flags",
        []
    )

    highlight_terms = data.get(
        "highlight_terms",
        []
    )

    return {

        "label": label,

        "fraud_probability": round(
            fraud_probability,
            1
        ),

        "confidence": round(
            confidence,
            1
        ),

        "reasoning": data.get(
            "reasoning",
            ""
        ),

        "red_flags": red_flags,

        "top_terms": [
            {
                "term": str(term)
            }
            for term in highlight_terms
        ]
    }

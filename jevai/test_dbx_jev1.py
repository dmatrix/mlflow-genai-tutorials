"""Smallest possible check that the Jev gateway endpoint answers.

Raw HTTP on purpose — no ``jev_client`` import — so that when the tutorial
breaks you can tell a credential problem from a client-code problem.

    uv run python jevai/test_dbx_jev1.py
"""

import os
from pathlib import Path

import requests
from dotenv import load_dotenv

# Only jevai/.env, to match the tutorial notebook.
load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

DATABRICKS_HOST = os.environ["DATABRICKS_HOST"].rstrip("/")
DATABRICKS_TOKEN = os.environ["DATABRICKS_TOKEN"]
JEV_MODEL = os.environ["JEV_MODEL"]

response = requests.post(
  f"{DATABRICKS_HOST}/ai-gateway/typesafe/v1/systemone",
  headers={
    "Authorization": f"Bearer {DATABRICKS_TOKEN}",
    "Content-Type": "application/json"
  },
  json={
    "model": JEV_MODEL,
    "state": {
      "conversation": "I was charged twice for my subscription this month and need a refund today. Otherwise I am going to dispute this with my bank."
    },
    "questions": {
      "intent": {
        "type": "choice",
        "instructions": "Classify the primary intent.",
        "criteria": {
          "billing": "General billing question",
          "refund": "Refund request",
          "cancellation": "Wants to cancel",
          "other": "Other"
        }
      },
      "sentiment": {
        "type": "choice",
        "instructions": "Classify the customer sentiment.",
        "criteria": {
          "positive": "Happy",
          "neutral": "Calm",
          "negative": "Upset or frustrated"
        }
      },
      "urgency": {
        "type": "choice",
        "instructions": "How urgent is this request?",
        "criteria": {
          "low": "No time pressure, can wait in the normal queue",
          "medium": "Wants a resolution soon",
          "high": "Needs action or escalation immediately"
        }
      }
    }
  },
)
if not response.ok:
    # The gateway says *why* in the body; raise_for_status() would hide it.
    raise SystemExit(f"{response.status_code} from the gateway: {response.text}")

print(response.json())

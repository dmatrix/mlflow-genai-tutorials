"""Lesson 6: a support-ticket router built on Jev.

The shape worth remembering:

    ticket text  ->  Jev answers three questions  ->  ordinary code decides

Jev never routes anything. It returns a label, a level, and a probability.
The ``if`` statements below are the routing policy, and they are yours.

Two functions, on purpose:

* ``decide(answers)`` is the policy -- ``if``s on numbers, nothing else. It
  takes answers rather than a ticket, so it runs with no network and no key.
* ``route(ticket)`` is the thin wrapper: ask Jev, then call ``decide``.

Needs ``DATABRICKS_HOST``, ``DATABRICKS_TOKEN``, and ``JEV_MODEL`` in the
environment -- Jev is reached through the workspace's Unity AI Gateway (see
``jev_client.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .jev_client import Choice, JevClient, Noul, Score

# ---------------------------------------------------------------- judgment
# What we ask Jev. Nothing here mentions queues, priorities, or policy.

DEPARTMENTS = {
    "billing": "Charges, invoices, refunds, duplicate payments",
    "shipping": "Delivery, tracking, lost or late packages",
    "technical": "Errors, crashes, logins, broken integrations",
    "account": "Passwords, email changes, closing an account",
}

URGENCY = [
    "No rush, a general question",
    "Normal, wants an answer this week",
    "Needs an answer soon, is blocked",
    "Needs an answer today, losing money or access",
]

# One call carries all three questions. Ask them together, not one per request.
QUESTIONS = {
    "department": Choice(
        instructions="Which team should handle this ticket",
        criteria=DEPARTMENTS,
    ),
    "urgency": Score(
        instructions="How soon this customer needs an answer",
        criteria=URGENCY,
    ),
    "needs_human": Noul(
        instructions="This ticket needs a human agent rather than an automated reply",
    ),
}

# ------------------------------------------------------------------ policy
# Plain numbers in plain code, reviewable without reading a prompt. Start
# conservative, then tune them against your own labelled tickets.

CONFIDENT_ENOUGH = 0.60  # below this, we do not trust the department label
HUMAN_ABOVE = 0.75  # p(needs a human) above this goes to a person
PRIORITY_AT_OR_ABOVE = 2.5  # urgency at or above this jumps the queue


@dataclass
class Routing:
    """What the application decided, and why."""

    queue: str
    priority: bool
    why: str

    def __str__(self) -> str:
        flag = " · PRIORITY" if self.priority else ""
        return f"{self.queue}{flag}  ({self.why})"


def decide(answers: dict[str, Any]) -> Routing:
    """The policy. Jev's answers in, one routing decision out.

    No model and no network in here -- just ``if``s on the numbers, which is
    what makes it testable with answers you made up yourself.
    """
    department = answers["department"]
    urgency = answers["urgency"]
    needs_human = answers["needs_human"]

    priority = urgency.score >= PRIORITY_AT_OR_ABOVE

    # Order matters, and it is visible. Uncertainty is handled first.
    if department.confidence < CONFIDENT_ENOUGH:
        return Routing("triage", priority,
                       f"department unclear, confidence {department.confidence:.2f}")
    if needs_human.noul > HUMAN_ABOVE:
        return Routing(f"{department.choice}-human", priority,
                       f"needs a human, p={needs_human.noul:.2f}")
    return Routing(f"{department.choice}-auto", priority,
                   f"confident ({department.confidence:.2f}) and automatable")


def route(ticket: str, client: JevClient | None = None) -> tuple[Routing, dict[str, Any]]:
    """Ask Jev about one ticket, then apply the policy.

    Returns the decision and the answers it was made from.
    """
    client = client or JevClient()
    answers = client.system_one(state=ticket, questions=QUESTIONS).answers
    return decide(answers), answers

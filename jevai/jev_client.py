"""A small client for Jev served through the Databricks Unity AI Gateway.

Jev is TypeSafe's System One model. TypeSafe ships its own ``typesafe_sdk``
package, but in this repo Jev is reached through the workspace's Unity AI
Gateway instead — so the only credentials you need are the
``DATABRICKS_HOST``/``DATABRICKS_TOKEN`` pair every other notebook here
already uses, with no second vendor account and no extra install.

The gateway speaks one endpoint::

    POST {DATABRICKS_HOST}/ai-gateway/typesafe/v1/systemone

and its wire format is a near-exact match for the SDK's object model, so this
module keeps the SDK's shape:

    state  +  questions   ->   client.system_one()   ->   typed answers

Three things the gateway does differently, all smoothed over here:

* ``request_id`` comes back as the ``x-typesafe-request-id`` response header
  rather than a field in the body.
* ``score`` answers arrive with *string* keys (``"0"``, ``"1"``, ...) in
  ``legend`` and ``probabilities``; we coerce them to ``int`` so that
  ``legend[round(answer.score)]`` works and distributions sort numerically.
* Answers are not grouped by type, so ``.nouls`` / ``.choices`` / ``.scores``
  are derived on this side.

Needs ``DATABRICKS_HOST``, ``DATABRICKS_TOKEN``, and ``JEV_MODEL`` (the Unity
Catalog endpoint path, e.g. ``my_catalog.my_schema.my_jev_endpoint``) in the
environment. ``JEV_GATEWAY_URL`` overrides the derived URL if you need it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

# The gateway's single System One route, appended to DATABRICKS_HOST.
SYSTEM_ONE_PATH = "/ai-gateway/typesafe/v1/systemone"

# This tutorial's credentials live next to this module, and this module is the
# one that loads them. Doing it here rather than in the notebook means a stale
# kernel, a notebook left open in another tab, or a cell run out of order can't
# leave the client holding someone else's token.
ENV_FILE = Path(__file__).resolve().parent / ".env"


# ------------------------------------------------------------------ questions
# The three primitives. Each one knows how to render itself onto the wire;
# nothing else in this module needs to know the JSON shape.


@dataclass(frozen=True)
class Noul:
    """Does this proposition hold? Returns one probability."""

    instructions: str

    def wire(self) -> dict[str, Any]:
        return {"type": "noul", "instructions": self.instructions}


@dataclass(frozen=True)
class Choice:
    """Which of these named options fits? ``criteria`` maps label -> meaning."""

    instructions: str
    criteria: dict[str, str]

    def wire(self) -> dict[str, Any]:
        return {
            "type": "choice",
            "instructions": self.instructions,
            "criteria": dict(self.criteria),
        }


@dataclass(frozen=True)
class Score:
    """Where on this rubric? ``criteria`` is ordered lowest level to highest."""

    instructions: str
    criteria: list[str]

    def wire(self) -> dict[str, Any]:
        return {
            "type": "score",
            "instructions": self.instructions,
            "criteria": list(self.criteria),
        }


# -------------------------------------------------------------------- answers
# Field names match the gateway's JSON exactly, so the notebooks can read
# answer.choice / answer.confidence / answer.noul and so on. ``type`` comes
# last because it is the only field with a default.


@dataclass(frozen=True)
class NoulAnswer:
    """No ``confidence`` field: for a Noul the probability *is* the uncertainty."""

    noul: float
    type: str = "noul"


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    confidence: float
    probabilities: dict[str, float]
    type: str = "choice"


@dataclass(frozen=True)
class ScoreAnswer:
    score: float
    confidence: float
    legend: dict[int, str]
    probabilities: dict[int, float]
    type: str = "score"


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int


Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer


def _parse_answer(raw: dict[str, Any]) -> Answer:
    kind = raw.get("type")
    if kind == "noul":
        return NoulAnswer(noul=raw["noul"])
    if kind == "choice":
        return ChoiceAnswer(
            choice=raw["choice"],
            confidence=raw["confidence"],
            probabilities=dict(raw["probabilities"]),
        )
    if kind == "score":
        # Levels are rubric positions, so int keys: `legend[round(score)]`.
        return ScoreAnswer(
            score=raw["score"],
            confidence=raw["confidence"],
            legend={int(k): v for k, v in raw["legend"].items()},
            probabilities={int(k): v for k, v in raw["probabilities"].items()},
        )
    raise ValueError(f"Unknown answer type from the gateway: {kind!r}")


@dataclass(frozen=True)
class SystemOneResponse:
    """One gateway round trip. ``answers`` is keyed by your question names."""

    model: str
    usage: Usage
    answers: dict[str, Answer]
    request_id: str | None = None

    @property
    def nouls(self) -> dict[str, float]:
        """Noul answers as bare probabilities."""
        return {k: a.noul for k, a in self.answers.items() if a.type == "noul"}

    @property
    def choices(self) -> dict[str, ChoiceAnswer]:
        return {k: a for k, a in self.answers.items() if a.type == "choice"}

    @property
    def scores(self) -> dict[str, ScoreAnswer]:
        return {k: a for k, a in self.answers.items() if a.type == "score"}


class GatewayError(RuntimeError):
    """A non-2xx from the gateway, carrying the message it actually sent."""


# --------------------------------------------------------------------- client


class JevClient:
    """Calls Jev through the Unity AI Gateway.

    Every argument falls back to an environment variable, so ``JevClient()``
    with no arguments works once ``.env`` is loaded.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        gateway_url: str | None = None,
        token: str | None = None,
        host: str | None = None,
        timeout: float = 120,
    ) -> None:
        if ENV_FILE.exists():
            load_dotenv(ENV_FILE, override=True)

        host = (host or os.environ.get("DATABRICKS_HOST") or "").rstrip("/")
        self.url = gateway_url or os.environ.get("JEV_GATEWAY_URL") or (
            f"{host}{SYSTEM_ONE_PATH}" if host else ""
        )
        self.model = model or os.environ.get("JEV_MODEL") or ""
        self.timeout = timeout
        self._token = token or os.environ.get("DATABRICKS_TOKEN") or ""

        # Fail here with the variable's name rather than at the first 401.
        if not self.url:
            raise RuntimeError("Set DATABRICKS_HOST (or pass gateway_url=/host=).")
        if not self.model:
            raise RuntimeError(
                "Set JEV_MODEL to the Unity Catalog endpoint path "
                "(catalog.schema.endpoint), or pass model=."
            )
        if not self._token:
            raise RuntimeError("Set DATABRICKS_TOKEN (or pass token=).")

        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    def system_one(
        self,
        *,
        state: Any,
        questions: dict[str, Noul | Choice | Score],
        model: str | None = None,
    ) -> SystemOneResponse:
        """Ask every question in ``questions`` about ``state``, in one call.

        ``state`` is any JSON-serializable value — a string, or the nested dict
        of context a human agent would need.
        """
        response = self._session.post(
            self.url,
            headers={"Authorization": f"Bearer {self._token}"},
            json={
                "model": model or self.model,
                "state": state,
                "questions": {name: q.wire() for name, q in questions.items()},
            },
            timeout=self.timeout,
        )

        if not response.ok:
            # A bare raise_for_status() would hide "Invalid access token"
            # behind a generic 401, which is the error readers hit most.
            try:
                detail = response.json().get("message") or response.text
            except ValueError:
                detail = response.text
            raise GatewayError(f"{response.status_code} from {self.url}: {detail}")

        body = response.json()
        return SystemOneResponse(
            model=body["model"],
            usage=Usage(**body["usage"]),
            answers={k: _parse_answer(v) for k, v in body["answers"].items()},
            request_id=response.headers.get("x-typesafe-request-id"),
        )

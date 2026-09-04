"""Simulate coding agents sending requests through the open-source MLflow AI Gateway.

Every agent talks to the gateway with the OpenAI SDK pointed at
`{tracking_uri}/gateway/mlflow/v1`; the request's `model` field is the name of a
gateway *endpoint* (e.g. `openai-chat`), which is the unit of governance — its
guardrails and any budget policy apply to whatever routes through it.

Using the OpenAI SDK (rather than raw HTTP) matters for observability: with
`mlflow.openai.autolog()` enabled, each call is captured as a span carrying the
provider's token usage, so every request becomes a trace whose token counts are
visible in the MLflow UI. The outer `@mlflow.trace` span adds the agent /
provider / endpoint tags that make per-agent and per-provider attribution work.

Governance verdicts in the open-source gateway:
  * A guardrail **Block** comes back as HTTP 400 (`openai.BadRequestError`),
    with the judge's rationale in the body.
  * A **budget policy** rejection comes back as HTTP 429
    (`openai.RateLimitError`) once cumulative spend crosses the cap.
"""

import time
from dataclasses import dataclass

import mlflow
import openai
from openai import OpenAI

MAX_RETRIES = 5
INITIAL_BACKOFF = 2


@dataclass
class SimulatedAgent:
    name: str
    display_name: str
    system_prompt: str
    endpoint: str  # gateway endpoint name, sent as the `model` field
    provider: str  # display label, e.g. "openai" / "anthropic"


def create_gateway_client(tracking_uri: str = "http://localhost:5000") -> OpenAI:
    """Create an OpenAI client pointing at the MLflow AI Gateway.

    The gateway configures provider credentials server-side, so no API key is
    needed here. The endpoint is selected per request by the `model` field in
    `send_request` / `send_burst_request`.
    """
    base_url = f"{tracking_uri.rstrip('/')}/gateway/mlflow/v1"
    # api_key is unused by the gateway but the SDK requires a non-empty string.
    return OpenAI(base_url=base_url, api_key="not-needed")


def _extract_error_detail(err: Exception) -> str:
    """Pull a human-readable reason out of an OpenAI SDK error."""
    body = getattr(err, "body", None)
    if isinstance(body, dict):
        detail = body.get("detail") or body.get("message") or body.get("error")
        if isinstance(detail, dict):
            return str(detail.get("message") or detail)
        if detail:
            return str(detail)
    resp = getattr(err, "response", None)
    if resp is not None:
        try:
            return str(resp.json())[:500]
        except Exception:
            return str(getattr(resp, "text", err))[:500]
    return str(err)[:500]


def _usage_dict(response) -> dict:
    """Flatten an OpenAI usage object into input/output/total token counts."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"input": 0, "output": 0, "total": 0}
    return {
        "input": getattr(usage, "prompt_tokens", 0) or 0,
        "output": getattr(usage, "completion_tokens", 0) or 0,
        "total": getattr(usage, "total_tokens", 0) or 0,
    }


@mlflow.trace(span_type="CHAT_MODEL", name="gateway_request")
def send_request(
    client: OpenAI,
    agent: SimulatedAgent,
    messages: list[dict],
    max_tokens: int = 1024,
) -> dict:
    """Send a chat completion through the gateway and return a result dict.

    Retries transient failures (429/5xx/connection). A guardrail Block (HTTP 400)
    is a real verdict, not a transient error, so it is never retried — it is
    reported as `blocked` with the guardrail's reason.
    """
    full_messages = [{"role": "system", "content": agent.system_prompt}] + messages

    mlflow.update_current_trace(
        tags={
            "agent": agent.name,
            "provider": agent.provider,
            "endpoint": agent.endpoint,
        }
    )

    def base_result(**overrides) -> dict:
        result = {
            "agent": agent.display_name,
            "provider": agent.provider,
            "endpoint": agent.endpoint,
            "status": 200,
            "content": None,
            "tokens": None,
            "blocked": False,
            "reason": None,
            "latency_s": None,
            "error": None,
        }
        result.update(overrides)
        return result

    backoff = INITIAL_BACKOFF
    for attempt in range(MAX_RETRIES):
        attempt_start = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=agent.endpoint,
                messages=full_messages,
                max_tokens=max_tokens,
            )
            latency_s = round(time.perf_counter() - attempt_start, 2)
            return base_result(
                status=200,
                content=response.choices[0].message.content,
                tokens=_usage_dict(response),
                latency_s=latency_s,
            )
        except openai.BadRequestError as e:
            # HTTP 400 = a guardrail blocked the request. A real verdict, not
            # transient — report it rather than retrying.
            latency_s = round(time.perf_counter() - attempt_start, 2)
            return base_result(
                status=400,
                blocked=True,
                reason=_extract_error_detail(e),
                latency_s=latency_s,
            )
        except (
            openai.RateLimitError,
            openai.InternalServerError,
            openai.APITimeoutError,
            openai.APIConnectionError,
        ) as e:
            # Transient in the normal path (a budget policy is only enabled for
            # the budget act). Back off and retry.
            if attempt < MAX_RETRIES - 1:
                time.sleep(backoff)
                backoff *= 2
                continue
            status = getattr(getattr(e, "response", None), "status_code", 503)
            return base_result(status=status, error=_extract_error_detail(e))
        except Exception as e:
            return base_result(status=500, error=str(e)[:500])


def run_scenario(client: OpenAI, agent: SimulatedAgent, scenario: dict) -> dict:
    """Run a single scenario and attach its metadata + a pass/fail verdict."""
    result = send_request(client, agent, scenario["messages"])
    result["scenario"] = scenario["name"]
    result["description"] = scenario["description"]
    result["expected_outcome"] = scenario["expected_outcome"]
    result["guardrail_type"] = scenario["guardrail_type"]
    result["agent_key"] = scenario["agent"]

    # A guardrail block is HTTP 400 (result["blocked"]); anything non-200 also
    # counts as "not allowed through".
    blocked = result["blocked"] or result["status"] != 200
    result["actual_outcome"] = "blocked" if blocked else "allowed"
    result["pass"] = result["actual_outcome"] == scenario["expected_outcome"]
    return result


def send_burst_request(
    client: OpenAI,
    agent: SimulatedAgent,
    messages: list[dict],
    max_tokens: int = 512,
) -> dict:
    """Send a single request, retrying only transient 5xx failures.

    Deliberately does NOT retry HTTP 429 — surfacing the budget-policy rejection
    is the whole point of the burst. Transient 5xx are still retried briefly so
    they don't show up as spurious errors.
    """
    full_messages = [{"role": "system", "content": agent.system_prompt}] + messages
    backoff = INITIAL_BACKOFF
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=agent.endpoint,
                messages=full_messages,
                max_tokens=max_tokens,
            )
            return {
                "status": 200,
                "outcome": "allowed",
                "content": (response.choices[0].message.content or "")[:200],
                "total_tokens": _usage_dict(response)["total"],
                "reason": None,
            }
        except openai.RateLimitError as e:
            # HTTP 429 = budget exceeded. This is the signal we want — return it.
            return {
                "status": 429,
                "outcome": "budget_rejected",
                "content": None,
                "total_tokens": 0,
                "reason": _extract_error_detail(e),
            }
        except openai.BadRequestError as e:
            return {
                "status": 400,
                "outcome": "blocked",
                "content": None,
                "total_tokens": 0,
                "reason": _extract_error_detail(e),
            }
        except (
            openai.InternalServerError,
            openai.APITimeoutError,
            openai.APIConnectionError,
        ) as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(backoff)
                backoff *= 2
                continue
            return {"status": 503, "outcome": "error", "content": str(e)[:200],
                    "total_tokens": 0, "reason": _extract_error_detail(e)}
        except Exception as e:
            return {"status": 500, "outcome": "error", "content": str(e)[:200],
                    "total_tokens": 0, "reason": str(e)[:200]}


def run_budget_burst(
    client: OpenAI,
    agent: SimulatedAgent,
    scenario: dict,
    n_requests: int = 30,
) -> list[dict]:
    """Fire n_requests sequential requests and return all results.

    Runs until the budget policy starts rejecting (HTTP 429) or the count is
    reached. Because a budget is USD-based, the number needed to trip it depends
    on the cap and the model's price — set a low cap (a few cents) in the UI.
    """
    results = []
    for i in range(n_requests):
        result = send_burst_request(client, agent, scenario["messages"])
        result["request_num"] = i + 1
        results.append(result)
    return results


# --------------------------------------------------------------------------- #
# Printing helpers
# --------------------------------------------------------------------------- #

def print_result(result: dict) -> None:
    """Pretty-print a single scenario result."""
    passed = result["pass"]
    status_icon = "PASS" if passed else "FAIL"
    outcome_icon = "BLOCKED" if result["actual_outcome"] == "blocked" else "ALLOWED"

    # PASS/FAIL is the assertion (actual == expected), not the gateway verdict —
    # so a correctly blocked PII request is a PASS. Print the expected outcome
    # alongside it so the two axes don't read as contradictory.
    print(f"  [{status_icon}] {result['description']}")
    print(f"    Agent:    {result['agent']}")
    print(f"    Provider: {result.get('provider', '')}")
    print(f"    Endpoint: {result['endpoint']}")
    print(f"    Expected: {result['expected_outcome'].upper()}")
    print(f"    Status:   {result['status']} ({outcome_icon})")

    if result.get("blocked") and result.get("reason"):
        print(f"    Guardrail: HTTP 400 — blocked")
        print(f"    Reason:   {result['reason'][:300]}")

    if result["actual_outcome"] == "allowed" and result.get("tokens"):
        t = result["tokens"]
        print(f"    Tokens:   {t['total']} (in: {t['input']}, out: {t['output']})")

    # Only show a model response for an allowed request; a blocked one's "content"
    # is None and its reason is already reported above.
    if result.get("content") and not result.get("blocked"):
        print("-------------------------------- RESPONSE --------------------------------")
        preview = result["content"][:750]
        if len(result["content"]) > 750:
            preview += "..."
        print(f"    {preview}")
        print("-------------------------------- RESPONSE --------------------------------")

    if result.get("error"):
        print(f"    Message:  {result['error'][:250]}")
    print()


def print_progress(result: dict, index: int, total: int) -> None:
    """Print one compact line for a result (for high-volume runs)."""
    verdict = "ok  " if result["pass"] else "FAIL"
    tokens = (result.get("tokens") or {}).get("total") or 0
    latency = result.get("latency_s") or 0.0

    note = ""
    if result["status"] == 429:
        note = "  [budget-rejected]"
    elif result.get("blocked"):
        note = f"  [{result['guardrail_type']}]"
    elif result["status"] != 200:
        note = f"  [HTTP {result['status']}]"

    print(
        f"  [{index:>3}/{total}] {verdict} {result['agent']:<12} {result['provider']:<10} "
        f"{result['actual_outcome']:<8} {tokens:>5} tok {latency:>6.1f}s  "
        f"{result['description'][:52]}{note}"
    )


def print_burst_summary(results: list[dict]) -> None:
    """Print per-request outcomes then a pass/fail summary for the budget burst."""
    for r in results:
        icon = "+" if r["outcome"] == "allowed" else "x"
        line = f"  [{icon}] Request {r['request_num']:>2}  HTTP {r['status']}  {r['outcome']}"
        if r["outcome"] == "budget_rejected" and r.get("reason"):
            line += f"  — {r['reason'][:80]}"
        elif r["outcome"] == "error":
            line += f"  — {r['content']}"
        print(line)
    print()
    allowed = sum(1 for r in results if r["outcome"] == "allowed")
    rejected = sum(1 for r in results if r["outcome"] == "budget_rejected")
    errors = sum(1 for r in results if r["outcome"] == "error")
    blocked = sum(1 for r in results if r["outcome"] == "blocked")
    total_tokens = sum(r["total_tokens"] for r in results)
    print(f"  Allowed:          {allowed}/{len(results)}  ({total_tokens} tokens spent)")
    print(f"  Budget-rejected:  {rejected}/{len(results)}")
    if blocked:
        print(f"  Guardrail-blocked: {blocked}/{len(results)}")
    if errors:
        print(f"  Errors:           {errors}/{len(results)}")

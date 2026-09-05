# Governing Coding-Agent Sprawl with the Open-Source MLflow AI Gateway

![AI Gateway Architecture](./images/ai_gateway_architecture.png)

Five coding agents (Cursor, Claude Code, Codex CLI, Gemini CLI, Pi) call LLMs through **one MLflow AI Gateway** instead of each holding its own API key. The gateway enforces guardrails, caps spend with budget policies, and traces every request — one place for policy, cost control, and an audit trail.

The gateway ships inside `mlflow server` (MLflow 3.15.1) and runs locally. This build routes to Databricks-hosted models via a Databricks connection; any provider connection (OpenAI, Anthropic, Google) works the same. It's the open-source counterpart of the Databricks "Unity AI Gateway" demo.

## 1. Prerequisites

- MLflow 3.15.1 — `uv sync` at the repo root (or `pip install 'mlflow[genai]==3.15.1'`).
- A Databricks connection: `DATABRICKS_HOST` and `DATABRICKS_TOKEN` in the repo's root `.env`.

## 2. Configure the gateway (MLflow UI, once)

The notebook only *uses* endpoints — create them first. See the [AI Gateway quickstart](https://mlflow.org/docs/latest/genai/governance/ai-gateway/quickstart/).

1. Start the server from the repo root, then open `http://localhost:5000`:

   ```bash
   uv run mlflow server --port 5000
   ```

2. **Settings → LLM Connections**: add a **Databricks** connection (`DATABRICKS_HOST` / `DATABRICKS_TOKEN`). It backs all three endpoints. (For OpenAI/Anthropic/Google, add those connections instead.)

3. **Gateway → Create Endpoint**: create three, using chat-completions-capable models:

   | Endpoint | Model | Agents |
   |----------|-------|--------|
   | `dbx-codex-endpoint` | `databricks-gpt-5-5` | Codex CLI |
   | `dbx-claude-endpoint` | `databricks-claude-haiku-4-5` | Cursor, Claude Code |
   | `dbx-gemini-endpoint` | `databricks-gemini-3-5-flash` | Gemini CLI, Pi |

   The endpoint name is the `model` value the notebook sends. If you rename them, update `ENDPOINTS` in the notebook's setup cell.

4. Add two guardrails per endpoint: **PII detection** (Pre-LLM, Block) and **Safety** (Post-LLM, Block). Point the judge at one of your endpoints. (Act 3)

5. Enable **usage tracking** per endpoint — it feeds the Usage Dashboard.

6. For Act 4 only: **AI Gateway → Budgets** → a Reject policy, ~**$0.05/day**. Restart with a short refresh so it triggers quickly:

   ```bash
   MLFLOW_GATEWAY_BUDGET_REFRESH_INTERVAL=30 uv run mlflow server --port 5000
   ```

   That cap will also reject Acts 2–3, so enable it only for Act 4.

## 3. Run

```bash
uv run mlflow server --port 5000        # after the UI setup above
uv run jupyter notebook ai_gateway_governance/mlflow_ai_gateway_governance.ipynb
```

Run the cells top to bottom. Do Acts 1–3 and 5 with no budget (or a high one); enable the Reject budget just before Act 4.

## What the notebook does — 5 acts

| Act | Shows |
|-----|-------|
| 1. Verify | Pings each endpoint; fails fast if one is missing. |
| 2. Agent swarm | Five agents send clean coding requests, round-robin across the three endpoints. |
| 3. Guardrails | PII / jailbreak / unsafe requests are blocked — **HTTP 400** with the judge's reason. |
| 4. Budget policy | A Reject budget caps spend; requests over the cap get **HTTP 429**. |
| 5. Tracing | Every request is a trace tagged `agent` / `provider` / `endpoint`, with token usage per trace. |

Routing: Cursor + Claude Code → `dbx-claude-endpoint`, Codex CLI → `dbx-codex-endpoint`, Gemini CLI + Pi → `dbx-gemini-endpoint`.

## Reference

**Calling the gateway** — it's OpenAI-compatible: point `base_url` at it and use the endpoint name as the model.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:5000/gateway/mlflow/v1", api_key="not-needed")
client.chat.completions.create(
    model="dbx-codex-endpoint",
    messages=[{"role": "user", "content": "What is MLflow?"}],
    max_tokens=1024,
)
```

```bash
curl -X POST http://localhost:5000/gateway/dbx-codex-endpoint/mlflow/invocations \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is MLflow?"}]}'
```

**Response codes**

- **200** — allowed.
- **400** (`openai.BadRequestError`) — guardrail block; the reason is in `e.body`. `send_request` in `gateway_agents.py` catches it and sets `blocked=True`.
- **429** (`openai.RateLimitError`) — budget exceeded.

A Post-LLM guardrail can block a *clean* prompt for what the model wrote back (e.g. a generated config containing an email). Ask for artifacts without those fields, or run PII Pre-LLM only.

**Files**

```
ai_gateway_governance/
├── mlflow_ai_gateway_governance.ipynb   # the demo (acts 1–5)
├── gateway_agents.py                    # gateway client, send_request, printers
├── scenarios.py                         # PII / injection / unsafe / clean scenarios + budget burst
├── prompts.py                           # agent persona prompts
└── images/ai_gateway_architecture.png   # + editable .svg
```

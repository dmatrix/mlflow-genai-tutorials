# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A tutorial series teaching MLflow's GenAI platform (tracing, experiment tracking, prompt management, evaluation, RAG). Nine sequential Jupyter notebooks plus a bonus multi-agent notebook. MLflow version: **3.10.0rc0** (pinned in `pyproject.toml`). Uses the **MLflow 3.x API** — specifically `mlflow.genai.evaluate()` not the old `mlflow.evaluate()`.

## Environment Setup

This project uses [UV](https://docs.astral.sh/uv/) for dependency management:

```bash
uv sync                              # Install all dependencies
cp env_template .env                 # Create .env from template, then fill in values
uv run jupyter notebook              # Start Jupyter
uv run mlflow ui --port 5000         # Start MLflow UI (separate terminal)
```

Required `.env` variables:
- `OPENAI_API_KEY` — for notebooks using OpenAI directly
- `MLFLOW_TRACKING_URI=http://localhost:5000` — MLflow server location
- `DATABRICKS_HOST`, `DATABRICKS_TOKEN` — only if using Databricks AI Gateway

## Running Tests / Scripts

There are no automated unit tests. The `test/` directory contains a manual endpoint smoke test:

```bash
uv run python test/test_ai_gateway_endpoints.py
```

To run `utils/clnt_utils.py` directly (tests LangChain client connectivity):

```bash
uv run python utils/clnt_utils.py
```

Both scripts load `.env` from the project root using `python-dotenv`.

## Architecture

### Notebooks (sequential, numbered)

| Notebook | Focus |
|----------|-------|
| `01` | MLflow setup, first tracked run |
| `02` | Experiment tracking, cost tracking, parent-child runs |
| `03` | Auto-tracing with `mlflow.openai.autolog()` |
| `04` | Manual tracing: `@mlflow.trace`, `mlflow.start_span()` |
| `05` | Prompt Registry: create, version, link to experiments |
| `06` | Framework integrations: OpenAI, LangChain, LlamaIndex |
| `07` | Evaluation: built-in scorers, custom `@scorer`, DeepEval |
| `08` | Prompt optimization with GEPA algorithm |
| `09` | Complete RAG app with RAGAS evaluation |
| `multiagent_orchestration` | AutoGen v0.4 multi-agent + Agent-as-a-Judge |

### Key MLflow 3.x Patterns Used

**Tracing:**
```python
mlflow.openai.autolog()              # Auto-trace OpenAI calls
@mlflow.trace                        # Trace a function
with mlflow.start_span("name"):      # Manual span
```

**Evaluation (MLflow 3.x API):**
```python
from mlflow.genai.scorers import Correctness, RelevanceToQuery, Safety, Guidelines
results = mlflow.genai.evaluate(data=dataset, scorers=[...])
results = mlflow.genai.evaluate(data=dataset, predict_fn=my_fn, scorers=[...])
```

**Agent-as-a-Judge** (multi-agent notebook):
```python
from mlflow.genai.judges import make_judge
judge = make_judge(name="...", instructions="...{{ trace }}...", model="databricks/...")
feedback = judge(trace=mlflow.get_trace(trace_id))
```

### `utils/clnt_utils.py`

Shared client factory used across notebooks. Supports three providers controlled by env vars:
- **OpenAI** (`USE_DATABRICKS_CLIENT=False`, `USE_DATABRICKS_AI_GATEWAY=False`)
- **Databricks workspace** (`USE_DATABRICKS_CLIENT=True`)
- **Databricks AI Gateway** (`USE_DATABRICKS_AI_GATEWAY=True`)

### MLflow Tracking

Local SQLite backend (`mlflow.db`) is used by default. The `mlartifacts/` directory stores run artifacts. Both are gitignored.

## Skills (Claude Code Slash Commands)

See `SKILLS.md` for the full catalog. Key ones:

| Skill | When to use |
|-------|-------------|
| `/mlflow-onboarding` | First-time MLflow setup |
| `/instrumenting-with-mlflow-tracing` | Adding tracing to code |
| `/agent-evaluation` | Setting up/running evaluation |
| `/analyze-mlflow-trace` | Debugging a specific trace by ID |
| `/searching-mlflow-docs` | Looking up MLflow APIs (fetches live docs) |

## Common Pitfalls

- **`Correctness` scorer requires `expected_facts`** in dataset `expectations` field
- **RAG judges** (`RetrievalGroundedness`, `RetrievalRelevance`) require `predict_fn` with tracing enabled — they read from the trace, not from `outputs`
- **Old vs new API**: Use `mlflow.genai.evaluate()` (MLflow 3.x), not `mlflow.evaluate()` with `model_type="databricks-agent"` (MLflow 2.x)
- **Judge model format for LiteLLM**: `"databricks/model-name"`, `"anthropic/claude-..."`, `"azure:/gpt-4o"` — not raw model names

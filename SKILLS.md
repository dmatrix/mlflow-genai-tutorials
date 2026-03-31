# MLflow GenAI Tutorials — Skills Reference

## What Are Skills?

Skills are **slash commands** you can invoke inside [Claude Code](https://claude.ai/claude-code) (the AI coding assistant CLI) while working through these tutorials. Each skill activates a specialized AI agent that knows how to help with a specific MLflow or GenAI task — without you needing to explain the context from scratch.

Think of skills as expert assistants you can summon on demand:

```
/skill-name
```

Type the command in the Claude Code chat, and the agent takes over — reading your code, checking your environment, fetching the relevant docs, and guiding you step by step.

---

## How to Use a Skill

1. **Open Claude Code** in your terminal from the project root:
   ```bash
   claude
   ```

2. **Invoke a skill** by typing its slash command:
   ```
   /agent-evaluation
   ```

3. **The agent will ask clarifying questions** if it needs more context (e.g., which notebook you're working in, what your agent does).

4. **Follow the guided workflow** — the agent reads your files, checks your environment, and produces runnable code or commands for you.

---

## Skills Catalog

### Getting Started

| Skill | When to Use |
|-------|-------------|
| `/mlflow-onboarding` | First time setting up MLflow in this project |
| `/searching-mlflow-docs` | Looking up any MLflow feature, API, or integration |

---

### `/mlflow-onboarding`

**Use this when:** You're starting fresh and want guided setup of MLflow tracking, tracing, or experiment configuration.

**What it does:**
- Detects whether your use case is GenAI agents/apps or traditional ML
- Walks you through the relevant quickstart
- Configures `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_ID`, and other environment variables
- Points you to the right notebook in this series

**Example trigger phrases:**
> "Get me started with MLflow"
> "Set up MLflow tracking for my project"
> "How do I use MLflow?"

**Relevant notebooks:** [01_setup_and_introduction.ipynb](01_setup_and_introduction.ipynb), [02_experiment_tracking.ipynb](02_experiment_tracking.ipynb)

---

### `/searching-mlflow-docs`

**Use this when:** You need accurate, up-to-date MLflow documentation for any feature — tracing, evaluation, prompt registry, integrations, etc.

**What it does:**
- Fetches live documentation from `mlflow.org`
- Returns targeted answers with code examples
- Covers integrations: LangChain, LangGraph, OpenAI, DSPy, CrewAI, AutoGen

**Example trigger phrases:**
> "How do I use MLflow with LangChain?"
> "Find MLflow docs for the Prompt Registry"
> "What's the MLflow API for custom scorers?"

---

### Observability & Tracing

| Skill | When to Use |
|-------|-------------|
| `/instrumenting-with-mlflow-tracing` | Adding tracing to your agent or LLM app |
| `/retrieving-mlflow-traces` | Fetching or filtering traces by ID, status, or metadata |
| `/analyze-mlflow-trace` | Debugging a single trace — why did it fail or behave unexpectedly? |
| `/analyze-mlflow-chat-session` | Debugging a multi-turn conversation session |
| `/querying-mlflow-metrics` | Viewing aggregated token usage, latency, and costs |

---

### `/instrumenting-with-mlflow-tracing`

**Use this when:** You want to add MLflow tracing to your Python or TypeScript code, or you're not sure how to instrument a specific framework.

**What it does:**
- Adds `mlflow.<framework>.autolog()` calls to your code
- Inserts `@mlflow.trace` decorators on key functions
- Sets up session tracking for multi-turn conversations
- Verifies the tracing is working end-to-end

**Example trigger phrases:**
> "How do I add tracing to my agent?"
> "Instrument my LangChain app with MLflow"
> "Getting started with MLflow tracing"
> "Trace my TypeScript app"

**Relevant notebooks:** [03_introduction_to_tracing.ipynb](03_introduction_to_tracing.ipynb), [04_manual_tracing_advanced.ipynb](04_manual_tracing_advanced.ipynb)

---

### `/retrieving-mlflow-traces`

**Use this when:** You need to fetch specific traces or search across many traces using filters.

**What it does:**
- Gets a trace by ID using CLI or Python API
- Filters traces by status (error, success), tags, metadata, or execution time
- Shows you how to query traces slower than a threshold, or failed traces only

**Example trigger phrases:**
> "Get trace abc123"
> "Find all failed traces from today"
> "Filter traces slower than 5 seconds"
> "Query MLflow traces from the last hour"

---

### `/analyze-mlflow-trace`

**Use this when:** You have a trace ID and want to understand why a single agent call went wrong, was slow, or produced a bad response.

**What it does:**
- Fetches the full trace from your MLflow server
- Analyzes LLM prompts, tool calls, span timings, and errors
- Provides a root-cause diagnosis and actionable recommendations

**Example trigger phrases:**
> "Analyze this trace: tr-abc123"
> "What went wrong with trace tr-xyz?"
> "Debug trace tr-000, it returned the wrong answer"
> "Root cause this trace failure"

---

### `/analyze-mlflow-chat-session`

**Use this when:** You have a multi-turn chat session and want to understand where the conversation went wrong or degraded in quality.

**What it does:**
- Analyzes a sequence of traces that form a conversation session
- Identifies which turn introduced an error or quality issue
- Surfaces patterns across turns (context loss, hallucination drift, tool misuse)

**Example trigger phrases:**
> "Analyze this chat session"
> "Where did this conversation go wrong?"
> "Review the session traces for session-id-456"
> "Debug my multi-turn chat history"

---

### `/querying-mlflow-metrics`

**Use this when:** You want aggregated statistics across many traces — not individual trace debugging.

**What it does:**
- Fetches token usage, latency distributions, trace counts
- Shows cost trends and quality evaluation summaries
- Supports querying by experiment, time window, or model

**Example trigger phrases:**
> "Show me token usage for this experiment"
> "What's my average latency this week?"
> "How many traces failed today?"
> "Analyze LLM costs across all runs"

**Relevant notebook:** [02_experiment_tracking.ipynb](02_experiment_tracking.ipynb)

---

### Evaluation

| Skill | When to Use |
|-------|-------------|
| `/agent-evaluation` | Systematically improve an agent's quality using MLflow evaluation |

---

### `/agent-evaluation`

**Use this when:** You want to measure and improve your agent's response quality, tool selection accuracy, or cost efficiency using a structured evaluation workflow.

**What it does — 4-step workflow:**

1. **Understand** — Runs your agent with sample inputs, inspects traces, confirms the agent's purpose
2. **Define Scorers** — Selects built-in LLM judges (Correctness, RelevanceToQuery, Guidelines, Safety, etc.) or helps you write custom scorers
3. **Prepare Dataset** — Discovers existing evaluation datasets in your MLflow experiment, or helps create a new one
4. **Run Evaluation** — Executes `mlflow.genai.evaluate()` on the dataset, scores all traces, produces a results report with pass rates and recommendations

**Example trigger phrases:**
> "Evaluate my agent's performance"
> "My agent is giving wrong answers, help me fix it"
> "Set up evaluation for my RAG pipeline"
> "I want to measure tool selection accuracy"

**Relevant notebooks:** [07_evaluating_agents.ipynb](07_evaluating_agents.ipynb), [08_prompt_optimization.ipynb](08_prompt_optimization.ipynb), [09_complete_rag_application.ipynb](09_complete_rag_application.ipynb)

**Also see:** [MLflow_Offline_Evaluation_SKILL.md](MLflow_Offline_Evaluation_SKILL.md) for a standalone offline evaluation reference.

---

### Code Quality

| Skill | When to Use |
|-------|-------------|
| `/simplify` | Review recently changed code for quality, reuse, and efficiency |

---

### `/simplify`

**Use this when:** You've written new code and want a second pass to identify redundancy, inefficiency, or missed reuse opportunities.

**What it does:**
- Reviews changed code against existing patterns in the repo
- Identifies over-engineering, duplication, or unnecessary complexity
- Applies targeted fixes — does not refactor code beyond what was changed

**Example trigger phrases:**
> "Simplify my new scorer function"
> "Review this for quality"
> "Is there a cleaner way to write this?"

---

## Skills by Tutorial Stage

Use this as a quick map of which skills are most useful at each point in the tutorial series:

| Tutorial Stage | Recommended Skill |
|----------------|-------------------|
| **First time setup** (Notebook 1.1–1.2) | `/mlflow-onboarding` |
| **Adding tracing** (Notebook 1.3–1.4) | `/instrumenting-with-mlflow-tracing` |
| **Debugging a specific trace** (any stage) | `/analyze-mlflow-trace` |
| **Debugging a chat session** (Notebook 1.4+) | `/analyze-mlflow-chat-session` |
| **Looking up docs** (any stage) | `/searching-mlflow-docs` |
| **Evaluating responses** (Notebook 1.7–1.9) | `/agent-evaluation` |
| **Checking usage/costs** (Notebook 1.2, 1.9) | `/querying-mlflow-metrics` |
| **After writing new code** (any stage) | `/simplify` |
---

## Environment Prerequisites

All skills assume the following are in place:

```bash
# 1. MLflow is installed and >=3.8.0
uv run mlflow --version

# 2. Environment variables are set
export MLFLOW_TRACKING_URI=http://localhost:5000
export OPENAI_API_KEY=sk-...

# 3. MLflow UI is running (in a separate terminal)
uv run mlflow ui --port 5000
```

See [01_setup_and_introduction.ipynb](01_setup_and_introduction.ipynb) and the `env_template` file for full setup instructions.

---

## Tips

- **You don't need to explain the whole project.** Skills read your files and environment automatically.
- **Skills work best when tracing is already set up.** Many skills (evaluation, trace analysis) need active MLflow traces to work with.
- **You can chain skills.** For example: use `/instrumenting-with-mlflow-tracing` to add tracing, then `/agent-evaluation` to run evaluation on the resulting traces.
- **For general MLflow API questions**, always try `/searching-mlflow-docs` first — it fetches live docs rather than relying on potentially outdated training data.

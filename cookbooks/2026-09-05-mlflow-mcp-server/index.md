---
title: "Observe and Act on MLflow with the MCP Server"
slug: mlflow-mcp-server
description: "Expose MLflow as MCP tools, then use them to debug failures, analyze latency, log feedback, and wire the server into Claude, VS Code, and Cursor."
tags: [tracing, mcp, tools, observability]
---

Turn MLflow into a set of tools an AI assistant can call. Start the MLflow MCP Server, connect a client over stdio, and use the built-in tools to debug failing traces, analyze latency, log quality feedback, and clean up -- then register the server so Claude, VS Code, or Cursor can do the same from natural language.

<!-- truncate -->

![Clients (Claude, VS Code, Cursor, or a programmatic fastmcp client) call MLflow operations as MCP tools over stdio, served by `mlflow mcp run` and scoped by MLFLOW_MCP_TOOLS.](./mlflow_mcp_server.svg)

The [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) is an open standard that lets AI assistants call external *tools* through a uniform interface. The MLflow MCP Server -- `mlflow mcp run` -- exposes MLflow operations (searching traces, reading a trace, logging feedback, managing experiments and runs) as MCP tools. Point an assistant at it and it can observe and act on your MLflow data for you: "find the failed traces from the last hour," "show me the slowest ones," "log a relevance score on this trace."

:::tip[Prerequisites]

```bash
pip install "mlflow[mcp]>=3.5.1"
```

Start a tracking server in a separate terminal and leave it running:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```

:::

:::note
The MLflow MCP Server is experimental (requires MLflow >= 3.5.1) and its API may evolve. The `mlflow[mcp]` extra pulls in `fastmcp`, used below to drive the server programmatically.
:::

## What You'll Build

Point MLflow at your running server, pick an experiment, and confirm the version supports the MCP Server.

```python
import os
import sys
import mlflow
from packaging.version import Version

TRACKING_URI = "http://localhost:5000"
os.environ["MLFLOW_TRACKING_URI"] = TRACKING_URI  # the MCP server subprocess reads this
mlflow.set_tracking_uri(TRACKING_URI)

experiment = mlflow.set_experiment("mcp-server-demo")

assert Version(mlflow.__version__) >= Version("3.5.1"), (
    f"The MLflow MCP Server needs MLflow >= 3.5.1 (found {mlflow.__version__})."
)
print(f"Experiment: {experiment.name} (id={experiment.experiment_id})")
# Example: Experiment: mcp-server-demo (id=1)
```

### Seed some traces to work with

So the tools have realistic data to act on, generate a handful of traces an assistant would triage. These use plain `@mlflow.trace` functions -- no LLM or API key required -- each tagged with a `span_type` so it is labeled by the kind of work it represents. The set is chosen to exercise the whole tracing surface: a healthy `LLM` trace, a slow `TASK`, a failing `TOOL`, a multi-span RAG `CHAIN`, a tagged production trace, and a chat trace that records token usage.

```python
import time
from mlflow.entities import SpanType


@mlflow.trace(span_type=SpanType.LLM)
def answer(question: str) -> str:
    """A healthy (OK) trace tagged as an LLM span. We attach feedback to one later."""
    return f"Here is a helpful answer to: {question}"


@mlflow.trace(span_type=SpanType.TASK)
def slow_report() -> str:
    """An OK but deliberately slow (~1.3s) trace for the latency workflow."""
    time.sleep(1.3)
    return "generated a large report"


@mlflow.trace(span_type=SpanType.TOOL)
def flaky_tool() -> str:
    """Raises on purpose so MLflow records the trace with ERROR status."""
    raise RuntimeError("simulated downstream failure")


@mlflow.trace(span_type=SpanType.RETRIEVER)
def retrieve(question: str) -> list[str]:
    """A RETRIEVER child of the RAG trace below."""
    return ["doc: MLflow Tracing records spans", "doc: spans nest into a trace"]


@mlflow.trace(span_type=SpanType.LLM)
def generate(question: str, docs: list[str]) -> str:
    """An LLM child of the RAG trace."""
    return f"Based on {len(docs)} documents: here is the answer to '{question}'."


@mlflow.trace(span_type=SpanType.CHAIN)
def rag_answer(question: str) -> str:
    """A small RAG pipeline: a CHAIN parent with RETRIEVER and LLM children."""
    docs = retrieve(question)
    return generate(question, docs)


@mlflow.trace(span_type=SpanType.LLM)
def production_query(question: str) -> str:
    """Attaches custom tags so we can filter by deployment context."""
    mlflow.update_current_trace(tags={"environment": "production", "user_tier": "premium"})
    return f"Production answer to: {question}"


@mlflow.trace(span_type=SpanType.CHAT_MODEL)
def chat_with_usage(question: str) -> str:
    """Records token usage on its span; rolls up to trace.info.token_usage."""
    span = mlflow.get_current_active_span()
    span.set_attribute(
        "mlflow.chat.tokenUsage",
        {"input_tokens": 42, "output_tokens": 58, "total_tokens": 100},
    )
    return f"Chat answer to: {question}"
```

Run them, flushing the async exporter so the traces are immediately queryable, and keep a few trace ids for later.

```python
# A healthy trace we'll attach feedback to later.
answer("What is MLflow Tracing?")
mlflow.flush_trace_async_logging()
ok_trace_id = mlflow.get_last_active_trace_id()

# More OK / slow / error traces.
answer("How do I evaluate an agent?")
slow_report()
try:
    flaky_tool()
except RuntimeError:
    pass  # the failure is captured in the trace as an ERROR

# A nested RAG trace -- capture its id to inspect the span hierarchy later.
rag_answer("How does MLflow tracing work?")
mlflow.flush_trace_async_logging()
rag_trace_id = mlflow.get_last_active_trace_id()

# A tagged production trace and a chat trace with token usage.
production_query("What is my order status?")
chat_with_usage("Summarize the MCP docs.")
mlflow.flush_trace_async_logging()
chat_trace_id = mlflow.get_last_active_trace_id()

print(f"Seeded 7 traces in experiment {experiment.experiment_id}")
# Example: Seeded 7 traces in experiment 1
```

### Connect an MCP client and discover the tools

`mlflow mcp run` speaks MCP over **stdio**: the client launches it as a subprocess and the two exchange JSON-RPC messages over the child's standard I/O pipes -- plain pipe-based IPC, no network socket, so the server lives only as long as the client.

The `MLFLOW_MCP_TOOLS` environment variable controls which tool *categories* are exposed. Smaller tool lists mean less token overhead for an assistant.

| `MLFLOW_MCP_TOOLS` | Categories | Tool count |
|--------------------|-----------|-----------|
| `genai` *(default)* | traces, scorers, experiments, runs | 26 |
| `ml` | experiments, runs, models, deployments | 32 |
| `all` | everything above | 45 |
| comma-separated | a custom subset, e.g. `"traces,scorers"` | varies |

Launch the server with the default `genai` toolset and list what it exposes. Define a small `mcp(...)` helper that opens a client, calls one tool, and returns its text -- reused throughout.

```python
import asyncio
import nest_asyncio
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

# In a notebook or REPL an event loop is already running; nest_asyncio lets us
# call asyncio.run() inside it. Omit this line (and the import) in a plain .py script.
nest_asyncio.apply()


def mcp_transport(tools: str = "genai") -> StdioTransport:
    """Launch `mlflow mcp run` over stdio with a chosen MLFLOW_MCP_TOOLS set."""
    return StdioTransport(
        command=sys.executable,  # `python -m mlflow`; no dependency on `mlflow` being on PATH
        args=["-m", "mlflow", "mcp", "run"],
        env={**os.environ, "MLFLOW_MCP_TOOLS": tools},
    )


def mcp(tool: str, arguments: dict | None = None, tools: str = "genai") -> str:
    """Call one MLflow MCP tool and return its text output."""
    async def _call():
        async with Client(mcp_transport(tools)) as client:
            result = await client.call_tool(tool, arguments or {})
            return result.content[0].text

    return asyncio.run(_call())


async def _list_tools():
    async with Client(mcp_transport("genai")) as client:
        return await client.list_tools()


tools = asyncio.run(_list_tools())
print(f"{len(tools)} tools exposed with MLFLOW_MCP_TOOLS=genai")
# Example: 26 tools exposed with MLFLOW_MCP_TOOLS=genai
# search_traces, get_trace, delete_traces, set_trace_tag, log_trace_feedback,
# get_trace_assessment, evaluate_traces, list_scorers, create_experiment,
# search_experiments, list_runs, create_run, link_traces_to_run, ...
```

Each workflow below is a single real tool call. In practice an assistant issues these for you from natural language -- the prompt it would receive is shown above each call.

### Debug failures

> *"Find the failed traces in my experiment."*

`search_traces` accepts a `filter_string` (the same grammar as the MLflow UI) and an optional `extract_fields` to return only the columns you care about, keeping responses small. Its output is a compact table.

```python
print(mcp("search_traces", {
    "experiment_id": experiment.experiment_id,
    "filter_string": "status = 'ERROR'",
    "extract_fields": "info.trace_id,info.state,info.request_preview",
}))
# Example:
# info.trace_id                        info.state    info.request_preview
# -----------------------------------  ------------  --------------------
# tr-f3912f13ffa051dd0cab558a42b32989  ERROR         {}
```

### Analyze latency

> *"Show me the traces that took longer than a second."*

Filter on `execution_time_ms` and pull just the duration field.

```python
print(mcp("search_traces", {
    "experiment_id": experiment.experiment_id,
    "filter_string": "execution_time_ms > 1000",
    "extract_fields": "info.trace_id,info.execution_duration_ms",
}))
# Example:
# info.trace_id                        info.execution_duration_ms
# -----------------------------------  --------------------------
# tr-0cc558dc15db559253da5d9b4917f8ce  1.3s
```

### Log quality feedback

> *"Log a relevance score of 0.9 on this trace, with a rationale."*

`log_trace_feedback` records an assessment on a trace -- ideal for human review or LLM-judge scores. The `value` is passed as a JSON-encoded string. Read it back with `get_trace`, selecting only the assessments; `get_trace` returns JSON.

```python
print(mcp("log_trace_feedback", {
    "trace_id": ok_trace_id,
    "name": "relevance",
    "value": "0.9",              # JSON-encoded value
    "source_type": "HUMAN",
    "source_id": "reviewer@example.com",
    "rationale": "Answer is on-topic and accurate.",
}))
# Example: Logged feedback 'relevance' to trace tr-a3a3... Assessment ID: a-f2f8...

print(mcp("get_trace", {
    "trace_id": ok_trace_id,
    "extract_fields": "info.assessments.*",
}))
# Example (abridged):
# {"info": {"assessments": [{"assessment_name": "relevance",
#   "source": {"source_type": "HUMAN", "source_id": "reviewer@example.com"},
#   "feedback": {"value": 0.9}, "rationale": "Answer is on-topic and accurate."}]}}
```

### Experiments, runs, and richer traces

The `genai` toolset also manages experiments and runs (the `ml` set adds models and deployments). List the experiments on the server:

```python
print(mcp("search_experiments", {"max_results": 5}))
```

The three richer traces seeded earlier show off more of what tracing captures -- each one tool call away. **Filter by tag** to find traces that came from production:

```python
print(mcp("search_traces", {
    "experiment_id": experiment.experiment_id,
    "filter_string": "tags.environment = 'production'",
    "extract_fields": "info.trace_id,info.tags.environment,info.tags.user_tier",
}))
# Example:
# info.trace_id                        info.tags.environment    info.tags.user_tier
# -----------------------------------  -----------------------  -------------------
# tr-e7415d0d5886aeab90c82dc0b13558d7  production               premium
```

**Reveal the span hierarchy** of the RAG trace -- `parent_span_id` links the RETRIEVER and LLM children to their `CHAIN` parent (`rag_answer`, the root, has no parent):

```python
print(mcp("get_trace", {
    "trace_id": rag_trace_id,
    "extract_fields": "data.spans.*.name,data.spans.*.span_id,data.spans.*.parent_span_id",
}))
# Example (abridged): rag_answer (root) -> retrieve, generate (both parented to rag_answer)
```

**Read token usage** -- the chat span records token counts under the standard `mlflow.chat.tokenUsage` attribute. The field selector can return the whole `attributes` map but can't isolate the dotted key, and attribute values come back JSON-encoded, so grab `data.spans.*.attributes` and decode the one you want.

```python
import json

detail = mcp("get_trace", {
    "trace_id": chat_trace_id,
    "extract_fields": "data.spans.*.attributes",
})

for span in json.loads(detail)["data"]["spans"]:
    raw_usage = span.get("attributes", {}).get("mlflow.chat.tokenUsage")
    if raw_usage:
        print(json.loads(raw_usage))
        # Example: {'input_tokens': 42, 'output_tokens': 58, 'total_tokens': 100}
```

:::note
If you are already in Python, the SDK exposes a tidier `mlflow.get_trace(id).info.token_usage` accessor. The `get_trace` tool exists so an *assistant* -- which only has the MCP tools -- can reach the same data.
:::

### Wire the server into your assistant

You normally don't hand-write a client. Register the server once and its tools become available to your assistant. Set `MLFLOW_TRACKING_URI` to your server (`http://localhost:5000`, a remote URL, or `databricks` with `DATABRICKS_HOST` / `DATABRICKS_TOKEN`).

**Claude Code / Claude Desktop** (CLI):

```bash
claude mcp add mlflow-mcp -e MLFLOW_TRACKING_URI=http://localhost:5000 \
  -- uv run --with "mlflow[mcp]>=3.5.1" mlflow mcp run
```

**Claude -- project-level `.mcp.json`** (VS Code uses the same shape under a `servers` key; Cursor under `.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "mlflow-mcp": {
      "command": "uv",
      "args": ["run", "--with", "mlflow[mcp]>=3.5.1", "mlflow", "mcp", "run"],
      "env": { "MLFLOW_TRACKING_URI": "http://localhost:5000", "MLFLOW_MCP_TOOLS": "genai" }
    }
  }
}
```

Then just ask, in natural language:

- *"Find all failed traces in experiment 1 from the last hour."*
- *"Show me the slowest traces with execution time over 1 second."*
- *"Log a relevance score of 0.85 for trace tr-… with a rationale about accuracy."*

### Clean up

`delete_traces` accepts trace ids or a max timestamp. Delete this experiment's traces so the recipe stays re-runnable.

```python
import time

print(mcp("delete_traces", {
    "experiment_id": experiment.experiment_id,
    "max_timestamp_millis": int(time.time() * 1000),
}))
# Example: Deleted 7 trace(s) from experiment 1.
```

## Results

After completing this cookbook, you have:

1. **A running MLflow MCP Server** -- started `mlflow mcp run` and driven it from a programmatic `fastmcp` client over stdio.
2. **Scoped tools** -- controlled which tools are exposed with `MLFLOW_MCP_TOOLS` (`genai` / `ml` / `all` / a custom subset) to trim an assistant's token overhead.
3. **Four real workflows** -- debugged failures (`search_traces` on `status='ERROR'`), analyzed latency (`execution_time_ms`), logged quality feedback (`log_trace_feedback` + `get_trace`), and cleaned up (`delete_traces`), using `extract_fields` to keep responses small.
4. **Richer trace access** -- filtered by tag, revealed a span hierarchy, and read token usage through the same `get_trace` tool an assistant would use.
5. **Assistant integration** -- registered the server with Claude, VS Code, and Cursor so the same operations run from natural language.

## Next Steps

- [MLflow MCP Server docs](https://mlflow.org/docs/latest/genai/mcp/) -- Full tool reference and configuration options.
- [Search Traces Reference](https://mlflow.org/docs/latest/genai/tracing/search-traces) -- The complete `filter_string` grammar used above.
- [Production Observability with MLflow Tracing](/cookbook/production-observability) -- Turn these queries into latency dashboards, error monitors, and alerts.

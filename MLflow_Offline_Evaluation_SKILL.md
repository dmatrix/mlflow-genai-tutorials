# MLflow Offline Evaluation with Built-in LLM Judges

## Overview

This skill guides the implementation of **offline evaluations** for GenAI applications using MLflow's built-in LLM-as-a-Judge scorers. Offline evaluation happens during development and testing phases, using curated datasets to benchmark performance before deployment.

## When to Use This Skill

Use this skill when:
- Setting up pre-deployment evaluation for LLM/GenAI applications
- Evaluating RAG (Retrieval-Augmented Generation) pipelines
- Assessing agent response quality, safety, or correctness
- Creating regression test suites for GenAI apps
- Comparing prompt variations or model versions

Do NOT use for:
- Real-time production monitoring (use online evaluation instead)
- Traditional ML model evaluation (use `mlflow.evaluate()` with `model_type`)
- Custom metric development without LLM judges

---

## Quick Reference

| Task | Code Pattern |
|------|--------------|
| Basic evaluation | `mlflow.genai.evaluate(data=dataset, scorers=[...])` |
| With predict function | `mlflow.genai.evaluate(data=dataset, predict_fn=my_fn, scorers=[...])` |
| Change judge model | `Correctness(model="openai:/gpt-4o")` |
| Custom guidelines | `Guidelines(name="tone", guidelines="Response must be professional")` |

---

## Installation & Setup

```bash
# Install MLflow 3.x with GenAI support
pip install --upgrade "mlflow[databricks]>=3.1.0"

# For OpenAI-hosted judges
pip install openai
export OPENAI_API_KEY="your-key"

# For Anthropic-hosted judges
pip install anthropic
export ANTHROPIC_API_KEY="your-key"
```

### Set Up Tracking

```python
import mlflow

# Local tracking
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("my-genai-eval")

# Or Databricks tracking
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Users/me/my-eval-experiment")
```

---

## Built-in Judges Reference

### Response Quality Judges

| Judge | Purpose | Requires Ground Truth? |
|-------|---------|------------------------|
| `RelevanceToQuery` | Does response address the user's input? | No |
| `Correctness` | Are expected facts in the response? | Yes (`expected_facts`) |
| `Safety` | Is content free of harmful/toxic material? | No |
| `Guidelines` | Does response follow custom rules? | Yes (guidelines) |
| `ExpectationsGuidelines` | Per-row custom expectations | Yes |
| `Fluency` | Is response grammatically correct? | No |
| `Equivalence` | Is response equivalent to expected output? | Yes |

### RAG-Specific Judges (Require Traces)

| Judge | Purpose | Requires Ground Truth? |
|-------|---------|------------------------|
| `RetrievalRelevance` | Are retrieved docs relevant to query? | No |
| `RetrievalGroundedness` | Is response grounded in retrieved context? | No |
| `RetrievalSufficiency` | Do retrieved docs contain enough info? | Yes |

### Tool Call Judges (Require Traces)

| Judge | Purpose |
|-------|---------|
| `ToolCallCorrectness` | Are tool calls and arguments correct? |
| `ToolCallEfficiency` | Are tool calls efficient without redundancy? |

---

## Dataset Structure

### Basic Format (No Ground Truth)

```python
eval_dataset = [
    {
        "inputs": {"query": "What is MLflow?"},
        "outputs": "MLflow is an open-source platform for ML lifecycle management."
    },
    {
        "inputs": {"query": "How do I track experiments?"},
        "outputs": "Use mlflow.start_run() to begin tracking."
    }
]
```

### With Ground Truth (Required for Correctness)

```python
eval_dataset = [
    {
        "inputs": {"query": "What is the capital of France?"},
        "outputs": "Paris is the capital of France.",
        "expectations": {
            "expected_facts": ["Paris is the capital of France."]
        }
    }
]
```

### With Retrieved Context (For RAG Evaluation)

```python
eval_dataset = [
    {
        "inputs": {"query": "What are MLflow's components?"},
        "outputs": "MLflow has four main components: Tracking, Projects, Models, and Registry.",
        "outputs": {
            "response": "MLflow has four main components...",
            "context": "MLflow Documentation: MLflow consists of Tracking, Projects, Models, and Model Registry..."
        }
    }
]
```

---

## Code Patterns

### Pattern 1: Simple Evaluation with Built-in Judges

```python
import mlflow
from mlflow.genai.scorers import Correctness, RelevanceToQuery, Safety

eval_dataset = [
    {
        "inputs": {"query": "What is machine learning?"},
        "outputs": "Machine learning is a subset of AI that enables systems to learn from data.",
        "expectations": {
            "expected_facts": ["Machine learning is a subset of AI."]
        }
    }
]

results = mlflow.genai.evaluate(
    data=eval_dataset,
    scorers=[
        Correctness(),
        RelevanceToQuery(),
        Safety()
    ]
)

# View results
print(results.metrics)
```

### Pattern 2: Evaluation with Predict Function

Use when you want to generate outputs during evaluation:

```python
import mlflow
import openai
from mlflow.genai.scorers import RelevanceToQuery, Guidelines

client = openai.OpenAI()

def predict_fn(inputs: dict) -> str:
    """Generate response for evaluation."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": inputs["query"]}]
    )
    return response.choices[0].message.content

# Dataset only needs inputs when using predict_fn
eval_dataset = [
    {"inputs": {"query": "Explain gradient descent in simple terms."}},
    {"inputs": {"query": "What is overfitting?"}},
]

results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=predict_fn,
    scorers=[
        RelevanceToQuery(),
        Guidelines(
            name="educational_tone",
            guidelines="Response should be educational and accessible to beginners."
        )
    ]
)
```

### Pattern 3: RAG Evaluation with Tracing

For RAG apps, enable tracing to capture retrieved context:

```python
import mlflow
from mlflow.genai.scorers import RetrievalGroundedness, RetrievalRelevance

# Enable autologging to capture traces
mlflow.openai.autolog()

@mlflow.trace
def rag_app(query: str) -> dict:
    """RAG application with tracing."""
    # Retrieval step (will be captured in trace)
    with mlflow.start_span("retrieval") as span:
        context = retrieve_documents(query)  # Your retrieval logic
        span.set_outputs({"retrieved_docs": context})
    
    # Generation step
    response = generate_answer(query, context)
    return {"response": response, "context": context}

eval_dataset = [
    {"inputs": {"query": "What are MLflow's main features?"}},
]

results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=rag_app,
    scorers=[
        RetrievalGroundedness(),  # Requires trace with retrieval span
        RetrievalRelevance(),     # Requires trace with retrieval span
    ]
)
```

### Pattern 4: Custom Guidelines Judge

```python
from mlflow.genai.scorers import Guidelines

# Single guideline
tone_checker = Guidelines(
    name="professional_tone",
    guidelines="The response must be professional and avoid casual language."
)

# Multiple guidelines
comprehensive_checker = Guidelines(
    name="response_quality",
    guidelines=[
        "Response must be factually accurate.",
        "Response must not include personal opinions.",
        "Response must cite sources when making claims.",
        "Response must be under 200 words."
    ]
)

results = mlflow.genai.evaluate(
    data=eval_dataset,
    scorers=[tone_checker, comprehensive_checker]
)
```

### Pattern 5: Changing the Judge Model

```python
from mlflow.genai.scorers import Correctness, RelevanceToQuery

# Use different models for different judges
results = mlflow.genai.evaluate(
    data=eval_dataset,
    scorers=[
        # OpenAI models
        Correctness(model="openai:/gpt-4o"),
        Correctness(model="openai:/gpt-4o-mini"),
        
        # Anthropic models
        RelevanceToQuery(model="anthropic:/claude-sonnet-4-20250514"),
        
        # Google models (requires litellm)
        RelevanceToQuery(model="google:/gemini-2.0-flash"),
        
        # Databricks-hosted models
        Correctness(model="databricks:/databricks-claude-sonnet-4"),
    ]
)
```

---

## Common Pitfalls

### 1. Missing `expected_facts` for Correctness Judge

```python
# ❌ WRONG - Correctness requires expected_facts
eval_dataset = [
    {
        "inputs": {"query": "What is Python?"},
        "outputs": "Python is a programming language."
    }
]
results = mlflow.genai.evaluate(data=eval_dataset, scorers=[Correctness()])
# Will fail or return errors

# ✅ CORRECT - Include expected_facts
eval_dataset = [
    {
        "inputs": {"query": "What is Python?"},
        "outputs": "Python is a programming language.",
        "expectations": {
            "expected_facts": ["Python is a programming language."]
        }
    }
]
```

### 2. Using RAG Judges Without Traces

```python
# ❌ WRONG - RetrievalGroundedness needs trace data
results = mlflow.genai.evaluate(
    data=[{"inputs": {...}, "outputs": {...}}],
    scorers=[RetrievalGroundedness()]
)
# Judge cannot find retrieval context

# ✅ CORRECT - Use predict_fn with tracing enabled
mlflow.openai.autolog()  # Enable tracing

@mlflow.trace
def my_rag_app(query):
    with mlflow.start_span("retrieval"):
        context = get_context(query)
    return generate(query, context)

results = mlflow.genai.evaluate(
    data=[{"inputs": {"query": "..."}}],
    predict_fn=my_rag_app,
    scorers=[RetrievalGroundedness()]
)
```

### 3. Forgetting API Keys

```python
# ❌ WRONG - No API key set
from mlflow.genai.scorers import Correctness
Correctness(model="openai:/gpt-4o")  # Will fail at runtime

# ✅ CORRECT - Set API key before evaluation
import os
os.environ["OPENAI_API_KEY"] = "sk-..."
# Or use environment variables in your shell
```

### 4. Incorrect Dataset Key Names

```python
# ❌ WRONG - Using 'question' instead of 'query'
eval_dataset = [{"inputs": {"question": "What is AI?"}}]

# ✅ CORRECT - Use the expected key names
eval_dataset = [{"inputs": {"query": "What is AI?"}}]

# Note: Key names depend on your predict_fn signature
# The inputs dict keys should match your function parameters
```

### 5. Mixing Old and New API

```python
# ❌ WRONG - Using old mlflow.evaluate() with model_type for GenAI
results = mlflow.evaluate(
    data=eval_data,
    model=my_agent,
    model_type="databricks-agent"  # Old MLflow 2.x pattern
)

# ✅ CORRECT - Use mlflow.genai.evaluate() for MLflow 3.x
from mlflow.genai.scorers import Correctness
results = mlflow.genai.evaluate(
    data=eval_data,
    predict_fn=my_agent,
    scorers=[Correctness()]
)
```

---

## Validation Checklist

Before running evaluation, verify:

- [ ] MLflow 3.1.0+ is installed (`pip show mlflow`)
- [ ] Tracking URI is set (`mlflow.get_tracking_uri()`)
- [ ] Experiment exists (`mlflow.set_experiment()`)
- [ ] API keys are configured for judge models
- [ ] Dataset has correct structure (`inputs`, `outputs`, `expectations`)
- [ ] Ground truth provided for judges that require it
- [ ] Tracing enabled for RAG judges (`mlflow.openai.autolog()`)

After evaluation, check:

- [ ] Review results in MLflow UI (Experiments → Evaluation tab)
- [ ] Check `results.metrics` for aggregate scores
- [ ] Examine `results.data` for per-row feedback
- [ ] Look for low-scoring examples to identify failure modes

---

## Integration with CI/CD

```python
# Example: Fail CI if evaluation score drops below threshold
import mlflow
from mlflow.genai.scorers import Correctness, RelevanceToQuery

results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=my_app,
    scorers=[Correctness(), RelevanceToQuery()]
)

# Extract pass rates
correctness_rate = results.metrics.get("Correctness/pass_rate", 0)
relevance_rate = results.metrics.get("RelevanceToQuery/pass_rate", 0)

# Assert minimum quality thresholds
assert correctness_rate >= 0.9, f"Correctness pass rate {correctness_rate} below 90%"
assert relevance_rate >= 0.85, f"Relevance pass rate {relevance_rate} below 85%"

print("✅ All evaluation thresholds passed!")
```

---

## Dependencies

- **mlflow>=3.1.0**: Core evaluation framework
- **openai**: For OpenAI-hosted judge models
- **anthropic**: For Anthropic-hosted judge models  
- **litellm** (optional): For Google Gemini and other providers

---

## Related Resources

- [MLflow GenAI Evaluation Docs](https://mlflow.org/docs/latest/genai/eval-monitor/)
- [Built-in Judges Reference](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/predefined)
- [Custom Judges Guide](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/custom-judges/)
- [Online Monitoring](https://mlflow.org/docs/latest/genai/eval-monitor/automatic-evaluations/)

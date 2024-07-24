# grouse

Evaluate Grounded Question Answering models and Grounded Question Answering evaluator models.

## Quickstart

```python
from grouse import EvaluationSample, evaluate

sample = EvaluationSample(
    input="What is the capital of France?",
    # Replace this with the actual output from your LLM application
    actual_output="The capital of France is Marseille.",
    expected_output="The capital of France is Paris.",
    retrieval_context=["Paris is the capital of France."]
)
evaluate([sample])
```

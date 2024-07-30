# grouse

Evaluate Grounded Question Answering models and Grounded Question Answering evaluator models.

## Install

```bash
pip install -e .
```

## CLI Usage

### Evaluation of Generations

```bash
grouse evaluate {PATH_TO_GENERATIONS} outputs/gpt-4o
```

### Unit Testing of Evaluators with GroUSE

```bash
grouse meta-evaluate gpt-4o meta-outputs/gpt-4o
```

## Library Usage

```python
from grouse import EvaluationSample, GroundedQAEvaluator

sample = EvaluationSample(
    input="What is the capital of France?",
    # Replace this with the actual output from your LLM application
    actual_output="The capital of France is Marseille.",
    expected_output="The capital of France is Paris.",
    references=["Paris is the capital of France."]
)
evaluator = GroundedQAEvaluator()
evaluator.evaluate([sample])
```

# GroUSE

Evaluate Grounded Question Answering (GQA) models and GQA evaluator models. We implement the evaluation methods described in GroUSE: A Benchmark to Evaluate Evaluators in Grounded Question Answering.

- [Install](#install)
- [Command Line Usage](#command-line-usage)
  - [Evaluation of the Grounded Question Answering task](#evaluation-of-the-grounded-question-answering-task)
  - [Unit Testing of Evaluators with GroUSE](#unit-testing-of-evaluators-with-grouse)
  - [Plot Matrices of unit tests success](#plot-matrices-of-unit-tests-success)
- [Python Usage](#python-usage)
- [Links](#links)
- [Citation](#citation)

## Install

```bash
pip install grouse
```

Then, setup your OpenAI credentials by creating an `.env` file by copying the `.env.dist` file, filling in your OpenAI API key and organization id and exporting the environment variables `export $(cat .env | xargs)`.

## Command Line Usage

### Evaluation of the Grounded Question Answering task

You can build a dataset in a `jsonl` file with the following format per line:

```json
{
    "references": ["", ...], // List of references
    "input": "", // Query
    "actual_output": "", // Predicted answer generated by the model we want to evaluate
    "expected_output": "" // Ground truth answer to the input
}
```

You can also check this example `example_data/grounded_qa.jsonl`.

Then, run this command:

```bash
grouse evaluate {PATH_TO_DATASET_WITH_GENERATIONS} outputs/gpt-4o
```

We recommend using GPT-4 as an evaluator model as we optimised prompts for this model, but you can change the model and prompts using the otional arguments : 
- `--evaluator_model_name`: Name of the evaluator model. It can be any LiteLLM model. The default model is GPT-4.
- `--prompts_path`: Path to the folder containing the prompts of the evaluator. By default, the prompts are those optimized for GPT-4.

### Unit Testing of Evaluators with GroUSE

Meta-Evaluation consists in evaluating GQA evaluators with the GroUSE unit tests.

```bash
grouse meta-evaluate gpt-4o meta-outputs/gpt-4o
```

Optional arguments : 
- `--prompts_path`: Path to the folder containing the prompts of the evaluator. By default, the prompts are those optimized for GPT-4.
- `--train_set`: Optional flag to meta-evaluate on the train set (16 tests) instead of the test set (144 tests). The train set is meant to be used during the prompt engineering phase.

### Plot Matrices of unit tests success

You can plot the results of unit tests in the shape of matrices:

```bash
grouse plot meta-outputs/gpt-4o
```

The resulting matrices look like this:

![result_matrices_plot](assets/result_matrices_plot.png)

## Python Usage

```python
from grouse import EvaluationSample, GroundedQAEvaluator

sample = EvaluationSample(
    input="What is the capital of France?",
    # Replace this with the actual output from your LLM application
    actual_output="The capital of France is Marseille.[1]",
    expected_output="The capital of France is Paris.[1]",
    references=["Paris is the capital of France."]
)
evaluator = GroundedQAEvaluator()
evaluator.evaluate([sample])
```

## Links

- [Paper](https://arxiv.org/abs/2409.06595)
- [Unit Tests](https://huggingface.co/datasets/illuin/grouse)
- [Finetuned model](https://huggingface.co/illuin/llama-3-grouse)

## Citation

```latex
@misc{muller2024grousebenchmarkevaluateevaluators,
      title={GroUSE: A Benchmark to Evaluate Evaluators in Grounded Question Answering}, 
      author={Sacha Muller and António Loison and Bilel Omrani and Gautier Viaud},
      year={2024},
      eprint={2409.06595},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.06595}, 
}
```

import json
import os
from typing import Optional

import click
import jsonlines

from grouse.dtos import EvaluationSample, MetaTestCase, MetaTestCaseResult
from grouse.grounded_qa_evaluator import GroundedQAEvaluator
from grouse.meta_evaluator import MetaEvaluator
from grouse.plot import plot_matrices
from grouse.register_models import register_models
from grouse.utils import NanConverter, load_unit_tests

register_models()


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.argument("dataset_path", type=str)
@click.argument("output_dir_path", type=str)
@click.option(
    "--evaluator_model_name",
    type=str,
    help=(
        "Name of the evaluator model. It can be any LiteLLM model. "
        "The default model is GPT-4."
    ),
    default="gpt-4",
)
@click.option(
    "--prompts_path",
    type=str,
    help=(
        "Path to the folder containing the prompts of the evaluator. "
        "By default, the prompts are those optimized for GPT-4."
    ),
    default=None,
)
def evaluate(
    dataset_path: str,
    output_dir_path: str,
    evaluator_model_name: Optional[str] = None,
    prompts_path: Optional[str] = None,
) -> None:
    """Evaluate models on grounded question answering using any LiteLLM model.
    The default model is GPT-4.

    Args:
        DATASET_PATH (str): Path to jsonlines file with references, input,
        actual_output (generation from the model to evaluate) and expected_output.
        OUTPUT_DIR_PATH (str): Path to directory where results report and
        evaluations are saved.
    """
    evaluator = GroundedQAEvaluator(
        model_name=evaluator_model_name, prompts_path=prompts_path
    )
    eval_samples = []
    with jsonlines.open(dataset_path) as reader:
        for obj in reader:
            eval_samples.append(EvaluationSample(**obj))

    results = evaluator.evaluate(eval_samples)

    os.makedirs(output_dir_path, exist_ok=True)
    with open(
        os.path.join(output_dir_path, "report.json"), "w", encoding="utf-8"
    ) as file:
        json.dump(results.report.model_dump(mode="json"), file, cls=NanConverter)

    with jsonlines.open(
        os.path.join(output_dir_path, "evaluations.jsonl"), "w"
    ) as writer:
        for evaluation in results.evaluations:
            writer.write(evaluation.model_dump(mode="json"))


@cli.command()
@click.argument("model_name", type=str)
@click.argument("output_dir_path", type=str)
@click.option(
    "--prompts_path",
    type=str,
    help=(
        "Path to the folder containing the prompts of the evaluator. "
        "By default, the prompts are those optimized for GPT-4."
    ),
    default=None,
)
def meta_evaluate(
    model_name: str, output_dir_path: str, prompts_path: Optional[str] = None
) -> None:
    """Evaluate evaluators on GroUSE unit tests.

    Args:
        MODEL_NAME (str): Name of model available through LiteLLM.
        OUTPUT_DIR_PATH (str): Path to directory where results report and
        unit test results are saved.
    """
    evaluation_samples, conditions = load_unit_tests()

    evaluator = GroundedQAEvaluator(model_name, prompts_path=prompts_path)
    evaluations = evaluator.evaluate_multiple_samples(evaluation_samples)

    meta_evaluator = MetaEvaluator()

    meta_test_cases = []
    for sample, evaluation, condition in zip(
        evaluation_samples, evaluations, conditions
    ):
        meta_test_cases.append(
            MetaTestCase(
                evaluation_sample=sample,
                actual_evaluation=evaluation,
                expected_evaluation=condition,
            )
        )

    meta_evaluations = meta_evaluator.evaluate(meta_test_cases)

    os.makedirs(output_dir_path, exist_ok=True)
    with open(
        os.path.join(output_dir_path, "report.json"), "w", encoding="utf-8"
    ) as file:
        json.dump(
            meta_evaluations.report.model_dump(mode="json"), file, cls=NanConverter
        )

    with jsonlines.open(
        os.path.join(output_dir_path, "meta_evaluations.jsonl"), "w"
    ) as writer:
        for evaluation in meta_evaluations.evaluations:
            writer.write(evaluation.model_dump(mode="json"))


@cli.command()
@click.argument("meta_test_results_path", type=str)
def plot(meta_test_results_path: str) -> None:
    """Create matrix plots for the four main metrics

    Args:
        META_TEST_RESULTS_PATH (str): Path to meta evaluation results in
        jsonlines format.
    """
    evaluation_samples, _ = load_unit_tests()

    results = []
    with jsonlines.open(meta_test_results_path, "r") as reader:
        for obj in reader:
            results.append(MetaTestCaseResult(**obj))
    plot_matrices(evaluation_samples, results)

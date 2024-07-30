import json
import os

import click
import jsonlines

from grouse.dtos import EvaluationSample, MetaTestCase
from grouse.grounded_qa_evaluator import GroundedQAEvaluator
from grouse.meta_evaluator import MetaEvaluator
from grouse.utils import NanConverter, load_unit_tests


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.argument("dataset_path", type=str)
@click.argument("output_dir_path", type=str)
def evaluate(dataset_path: str, output_dir_path: str) -> None:
    evaluator = GroundedQAEvaluator()
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
def meta_evaluate(model_name: str, output_dir_path: str) -> None:
    evaluation_samples, conditions = load_unit_tests()

    evaluator = GroundedQAEvaluator(model_name)
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

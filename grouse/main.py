import click
import jsonlines
import json
from json import JSONEncoder
import math
import os
from typing import List, Tuple
from datasets import load_dataset

from grouse.dtos import EvaluationSample, MetaTestCase, ExpectedGroundedQAEvaluation
from grouse.grounded_qa_evaluator import GroundedQAEvaluator
from grouse.meta_evaluator import MetaEvaluator


def nan_to_none(obj):
    if isinstance(obj, dict):
        return {k: nan_to_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [nan_to_none(v) for v in obj]
    elif isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


class NanConverter(JSONEncoder):
    def encode(self, obj, *args, **kwargs):
        return super().encode(nan_to_none(obj), *args, **kwargs)

    def iterencode(self, obj, *args, **kwargs):
        return super().iterencode(nan_to_none(obj), *args, **kwargs)


def load_unit_tests() -> Tuple[List[EvaluationSample], List[ExpectedGroundedQAEvaluation]]:
    unit_tests = load_dataset("illuin/grouse")["test"]
    evaluation_samples = []
    conditions = []

    for unit_test in unit_tests:
        evaluation_samples.append(
            EvaluationSample(
                input=unit_test["input"],
                actual_output=unit_test["actual_output"],
                expected_output=unit_test["expected_output"],
                references=unit_test["references"],
            )
        )
        conditions.append(ExpectedGroundedQAEvaluation(**unit_test["conditions"]))
    return evaluation_samples, conditions


@click.group()
def cli():
    pass


@cli.command()
@click.argument("dataset_path", type=str)
@click.argument("output_dir_path", type=str)
def evaluate(dataset_path: str, output_dir_path: str):
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
def meta_evaluate(model_name: str, output_dir_path: str):
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


if __name__ == "__main__":
    cli()

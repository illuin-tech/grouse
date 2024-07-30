import click
import jsonlines
import json
from json import JSONEncoder
import math
import os
from typing import List, Tuple

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


def load_unit_tests(
    dataset_path: str,
) -> Tuple[List[EvaluationSample], List[ExpectedGroundedQAEvaluation]]:
    evaluation_samples = []
    conditions = []
    with jsonlines.open(dataset_path) as reader:
        for obj in reader:
            evaluation_samples.append(
                EvaluationSample(
                    input=obj["input"],
                    actual_output=obj["actual_output"],
                    expected_output=obj["expected_output"],
                    references=obj["references"],
                )
            )
            conditions.append(ExpectedGroundedQAEvaluation(**obj["conditions"]))
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
@click.argument("dataset_path", type=str)
@click.argument("model_name", type=str)
@click.argument("output_dir_path", type=str)
def meta_evaluate(dataset_path: str, model_name: str, output_dir_path: str):
    evaluation_samples, conditions = load_unit_tests(dataset_path)

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

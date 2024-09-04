import math
from json import JSONEncoder
from typing import Any, List, Optional, Tuple

from datasets import load_dataset

from grouse.dtos import (
    AnswerRelevancy,
    Completeness,
    EvaluationSample,
    ExpectedGroundedQAEvaluation,
    Failed,
)

DATASET_NAME = "illuin/grouse"


def nan_to_none(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: nan_to_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [nan_to_none(v) for v in obj]
    elif isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


class NanConverter(JSONEncoder):
    def encode(self, obj: Any, *args, **kwargs) -> Any:
        return super().encode(nan_to_none(obj), *args, **kwargs)

    def iterencode(self, obj: Any, *args, **kwargs) -> Any:
        return super().iterencode(nan_to_none(obj), *args, **kwargs)


def get_positive_acceptance_negative_rejection(
    answer_relevancy: AnswerRelevancy | Failed,
    completeness: Completeness | Failed,
) -> Tuple[Optional[int] | Failed, Optional[int] | Failed]:
    if isinstance(answer_relevancy, Failed):
        return Failed(error="answer relevancy failed"), Failed(
            error="answer relevancy failed"
        )
    elif isinstance(completeness, Failed):
        return Failed(error="completeness failed"), Failed(error="completeness failed")
    else:
        if answer_relevancy.answer_relevancy is None:
            if completeness.completeness is None:
                positive_acceptance = 1
                negative_rejection = 1
            else:
                positive_acceptance = 0
                negative_rejection = None
        else:
            if completeness.completeness is None:
                positive_acceptance = None
                negative_rejection = 0
            else:
                positive_acceptance = None
                negative_rejection = None
        return positive_acceptance, negative_rejection


def load_unit_tests() -> (
    Tuple[List[EvaluationSample], List[ExpectedGroundedQAEvaluation]]
):
    unit_tests = load_dataset(DATASET_NAME)["test"]
    evaluation_samples = []
    conditions = []

    for unit_test in unit_tests:
        evaluation_samples.append(
            EvaluationSample(
                input=unit_test["input"],
                actual_output=unit_test["actual_output"],
                expected_output=unit_test["expected_output"],
                references=unit_test["references"],
                metadata=unit_test["metadata"],
            )
        )
        conditions.append(ExpectedGroundedQAEvaluation(**unit_test["conditions"]))

    return evaluation_samples, conditions

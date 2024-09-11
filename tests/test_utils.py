from typing import Literal, Optional
from unittest.mock import patch

import pytest

from grouse.dtos import (
    AnswerRelevancy,
    Completeness,
    EvaluationSample,
    ExpectedGroundedQAEvaluation,
)
from grouse.utils import get_positive_acceptance_negative_rejection, load_unit_tests


@pytest.mark.parametrize(
    "answer_relevancy, completeness, "
    "expected_positive_acceptance, expected_negative_rejection",
    [
        [
            AnswerRelevancy(
                answer_relevancy=None,
                answer_relevancy_justification="",
                answer_affirms_no_document_answers=False,
            ),
            Completeness(completeness=None, completeness_justification=""),
            1,
            1,
        ],
        [
            AnswerRelevancy(
                answer_relevancy=None,
                answer_relevancy_justification="",
                answer_affirms_no_document_answers=False,
            ),
            Completeness(completeness=5, completeness_justification=""),
            0,
            None,
        ],
        [
            AnswerRelevancy(
                answer_relevancy=5,
                answer_relevancy_justification="",
                answer_affirms_no_document_answers=False,
            ),
            Completeness(completeness=None, completeness_justification=""),
            None,
            0,
        ],
        [
            AnswerRelevancy(
                answer_relevancy=5,
                answer_relevancy_justification="",
                answer_affirms_no_document_answers=False,
            ),
            Completeness(completeness=5, completeness_justification=""),
            None,
            None,
        ],
    ],
)
def test_get_positive_acceptance_negative_rejection(
    answer_relevancy: Optional[int],
    completeness: Optional[int],
    expected_positive_acceptance: Optional[int],
    expected_negative_rejection: Optional[int],
) -> None:
    pa, nr = get_positive_acceptance_negative_rejection(answer_relevancy, completeness)
    assert pa == expected_positive_acceptance
    assert nr == expected_negative_rejection


@pytest.mark.parametrize("dataset_split", ["train", "test"])
def test_load_unit_tests(dataset_split: Literal["train"] | Literal["test"]) -> None:
    with patch(
        "grouse.utils.load_dataset",
        return_value={
            "train": [
                {
                    "input": "Quel est la capitale de la France ?",
                    "actual_output": "Paris[1]",
                    "expected_output": "Paris[1]",
                    "references": ["Paris"],
                    "metadata": {},
                    "conditions": {
                        "answer_relevancy_condition": "==5",
                        "completeness_condition": "==5",
                        "faithfulness_condition": "==1",
                        "usefulness_condition": "==None",
                    },
                }
            ],
            "test": [
                {
                    "input": "Quel est la capitale de la France ?",
                    "actual_output": "Paris[1]",
                    "expected_output": "Paris[1]",
                    "references": ["Paris"],
                    "metadata": {},
                    "conditions": {
                        "answer_relevancy_condition": "==5",
                        "completeness_condition": "==5",
                        "faithfulness_condition": "==1",
                        "usefulness_condition": "==None",
                    },
                }
            ],
        },
    ):
        samples, conditions = load_unit_tests(dataset_split)
        assert isinstance(samples, list)
        assert isinstance(conditions, list)
        assert isinstance(samples[0], EvaluationSample)
        assert isinstance(conditions[0], ExpectedGroundedQAEvaluation)

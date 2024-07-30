from typing import Optional

import pytest

from grouse.dtos import EvaluationSample, ExpectedGroundedQAEvaluation
from grouse.utils import get_positive_acceptance_negative_rejection, load_unit_tests


@pytest.mark.parametrize(
    "answer_relevancy, completeness, "
    "expected_positive_acceptance, expected_negative_rejection",
    [
        [None, None, 1, 1],
        [None, 5, 0, None],
        [5, None, None, 0],
        [5, 5, None, None],
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


def test_load_unit_tests() -> None:
    samples, conditions = load_unit_tests()
    assert isinstance(samples, list)
    assert isinstance(conditions, list)
    assert len(samples) == 144
    assert len(conditions) == 144
    assert isinstance(samples[0], EvaluationSample)
    assert isinstance(conditions[0], ExpectedGroundedQAEvaluation)

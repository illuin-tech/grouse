from typing import List, Optional

from grouse.dtos import (
    Failed,
    MetaEvalReport,
    MetaEvaluationsAndReport,
    MetaTestCase,
    MetaTestCaseResult,
    Score,
)
from grouse.utils import get_positive_acceptance_negative_rejection


class MetaEvaluator:
    def __init__(self) -> None:
        pass

    @staticmethod
    def compare(value: Optional[float], condition: str) -> bool:
        if value is None:
            return condition == "==None"
        else:
            if condition.endswith("None"):
                return False
            elif condition.startswith(">="):
                return value >= float(condition[2:])
            elif condition.startswith("<="):
                return value <= float(condition[2:])
            elif condition.startswith(">"):
                return value > float(condition[1:])
            elif condition.startswith("<"):
                return value < float(condition[1:])
            elif condition[:2] == "==":
                return value == float(condition[2:])
            else:
                raise ValueError("Invalid condition")

    def __get_result(self, score: Score, score_name: str, condition: str) -> bool:
        if isinstance(score, Failed):
            return Failed(error=score.error)
        return self.compare(getattr(score, score_name), condition)

    def evaluate_single_test_case(self, test_case: MetaTestCase) -> MetaTestCaseResult:
        answer_relevancy_result = self.__get_result(
            test_case.actual_evaluation.answer_relevancy,
            "answer_relevancy",
            test_case.expected_evaluation.answer_relevancy_condition,
        )
        completeness_result = self.__get_result(
            test_case.actual_evaluation.completeness,
            "completeness",
            test_case.expected_evaluation.completeness_condition,
        )
        faithfulness_result = self.__get_result(
            test_case.actual_evaluation.faithfulness,
            "faithfulness",
            test_case.expected_evaluation.faithfulness_condition,
        )
        usefulness_result = self.__get_result(
            test_case.actual_evaluation.usefulness,
            "usefulness",
            test_case.expected_evaluation.usefulness_condition,
        )

        if isinstance(
            test_case.actual_evaluation.answer_relevancy, Failed
        ) or isinstance(test_case.actual_evaluation.completeness, Failed):
            positive_acceptance_result = Failed()
            negative_rejection_result = Failed()
        else:
            if test_case.expected_evaluation.answer_relevancy_condition == "==None":
                if test_case.expected_evaluation.completeness_condition == "==None":
                    positive_acceptance_condition = "==1"
                    negative_rejection_condition = "==1"
                else:
                    positive_acceptance_condition = "==0"
                    negative_rejection_condition = "==None"
            else:
                if test_case.expected_evaluation.completeness_condition == "==None":
                    positive_acceptance_condition = "==None"
                    negative_rejection_condition = "==0"
                else:
                    positive_acceptance_condition = "==None"
                    negative_rejection_condition = "==None"

            evaluated_positive_acceptance, evaluated_negative_rejection = (
                get_positive_acceptance_negative_rejection(
                    test_case.actual_evaluation.answer_relevancy,
                    test_case.actual_evaluation.completeness,
                )
            )
            positive_acceptance_result = self.compare(
                evaluated_positive_acceptance, positive_acceptance_condition
            )
            negative_rejection_result = self.compare(
                evaluated_negative_rejection, negative_rejection_condition
            )

        return MetaTestCaseResult(
            answer_relevancy=answer_relevancy_result,
            completeness=completeness_result,
            faithfulness=faithfulness_result,
            usefulness=usefulness_result,
            positive_acceptance=positive_acceptance_result,
            negative_rejection=negative_rejection_result,
        )

    def evaluate_multiple_test_cases(
        self, test_cases: list[MetaTestCase]
    ) -> List[MetaTestCaseResult]:
        return [self.evaluate_single_test_case(test_case) for test_case in test_cases]

    def evaluate(self, test_cases: List[MetaTestCase]) -> MetaEvaluationsAndReport:
        meta_evaluations = self.evaluate_multiple_test_cases(test_cases)
        answer_relevancy_success = float(
            sum([int(e.answer_relevancy) for e in meta_evaluations])
            / len(meta_evaluations)
        )
        completeness_success = float(
            sum([int(e.completeness) for e in meta_evaluations]) / len(meta_evaluations)
        )
        faithfulness_success = float(
            sum([int(e.faithfulness) for e in meta_evaluations]) / len(meta_evaluations)
        )
        usefulness_success = float(
            sum([int(e.usefulness) for e in meta_evaluations]) / len(meta_evaluations)
        )
        positive_acceptance_success = float(
            sum([int(e.positive_acceptance) for e in meta_evaluations])
            / len(meta_evaluations)
        )
        negative_rejection_success = float(
            sum([int(e.negative_rejection) for e in meta_evaluations])
            / len(meta_evaluations)
        )
        total = (
            answer_relevancy_success
            + completeness_success
            + faithfulness_success
            + usefulness_success
            + positive_acceptance_success
            + negative_rejection_success
        ) / 6
        return MetaEvaluationsAndReport(
            evaluations=meta_evaluations,
            report=MetaEvalReport(
                answer_relevancy_success=answer_relevancy_success,
                completeness_success=completeness_success,
                faithfulness_success=faithfulness_success,
                usefulness_success=usefulness_success,
                positive_acceptance_success=positive_acceptance_success,
                negative_rejection_success=negative_rejection_success,
                total=total,
            ),
        )

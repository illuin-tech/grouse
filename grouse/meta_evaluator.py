from typing import List, Optional

from grouse.dtos import (
    Failed,
    MetaEvalReport,
    MetaEvaluationsAndReport,
    MetaTestCase,
    MetaTestCaseResult,
)
from grouse.utils import get_positive_acceptance_negative_rejection


class MetaEvaluator:
    def __init__(self) -> None:
        pass

    @staticmethod
    def compare(value: Optional[float], condition: str) -> bool:
        if value is not None and condition[:2] == ">=":
            return value >= float(condition[2:])
        elif value is not None and condition[:2] == "<=":
            return value <= float(condition[2:])
        elif value is not None and condition[:1] == ">":
            return value > float(condition[1:])
        elif value is not None and condition[:1] == "<":
            return value < float(condition[1:])
        elif condition[:2] == "==":
            if condition[2:] == "None":
                return value is None
            else:
                return value == float(condition[2:])
        else:
            raise ValueError("Invalid condition")

    def evaluate_single_test_case(self, test_case: MetaTestCase) -> MetaTestCaseResult:
        if isinstance(test_case.actual_evaluation.answer_relevancy, Failed):
            answer_relevancy_result = Failed()
        else:
            answer_relevancy_result = self.compare(
                test_case.actual_evaluation.answer_relevancy.answer_relevancy,
                test_case.expected_evaluation.answer_relevancy_condition,
            )
        if isinstance(test_case.actual_evaluation.completeness, Failed):
            completeness_result = Failed()
        else:
            completeness_result = self.compare(
                test_case.actual_evaluation.completeness.completeness,
                test_case.expected_evaluation.completeness_condition,
            )
        if isinstance(test_case.actual_evaluation.faithfulness, Failed):
            faithfulness_result = Failed()
        else:
            faithfulness_result = self.compare(
                test_case.actual_evaluation.faithfulness.faithfulness,
                test_case.expected_evaluation.faithfulness_condition,
            )
        if isinstance(test_case.actual_evaluation.usefulness, Failed):
            usefulness_result = Failed()
        else:
            usefulness_result = self.compare(
                test_case.actual_evaluation.usefulness.usefulness,
                test_case.expected_evaluation.usefulness_condition,
            )

        if isinstance(
            test_case.actual_evaluation.answer_relevancy, Failed
        ) or isinstance(test_case.actual_evaluation.completeness, Failed):
            pa_result = Failed()
            nr_result = Failed()
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
                    test_case.actual_evaluation.answer_relevancy.answer_relevancy,
                    test_case.actual_evaluation.completeness.completeness,
                )
            )
            pa_result = self.compare(
                evaluated_positive_acceptance, positive_acceptance_condition
            )
            nr_result = self.compare(
                evaluated_negative_rejection, negative_rejection_condition
            )

        return MetaTestCaseResult(
            answer_relevancy=answer_relevancy_result,
            completeness=completeness_result,
            faithfulness=faithfulness_result,
            usefulness=usefulness_result,
            positive_acceptance=pa_result,
            negative_rejection=nr_result,
        )

    def evaluate_multiple_test_cases(
        self, test_cases: list[MetaTestCase]
    ) -> List[MetaTestCaseResult]:
        return [self.evaluate_single_test_case(test_case) for test_case in test_cases]

    def evaluate(self, test_cases: List[MetaTestCase]) -> MetaEvaluationsAndReport:
        meta_evaluations = self.evaluate_multiple_test_cases(test_cases)
        ar_success = float(
            sum([int(e.answer_relevancy) for e in meta_evaluations])
            / len(meta_evaluations)
        )
        c_success = float(
            sum([int(e.completeness) for e in meta_evaluations]) / len(meta_evaluations)
        )
        f_success = float(
            sum([int(e.faithfulness) for e in meta_evaluations]) / len(meta_evaluations)
        )
        u_success = float(
            sum([int(e.usefulness) for e in meta_evaluations]) / len(meta_evaluations)
        )
        pa_success = float(
            sum([int(e.positive_acceptance) for e in meta_evaluations])
            / len(meta_evaluations)
        )
        nr_success = float(
            sum([int(e.negative_rejection) for e in meta_evaluations])
            / len(meta_evaluations)
        )
        total = (
            ar_success + c_success + f_success + u_success + pa_success + nr_success
        ) / 6
        return MetaEvaluationsAndReport(
            evaluations=meta_evaluations,
            report=MetaEvalReport(
                answer_relevancy_success=ar_success,
                completeness_success=c_success,
                faithfulness_success=f_success,
                usefulness_success=u_success,
                positive_acceptance_success=pa_success,
                negative_rejection_success=nr_success,
                total=total,
            ),
        )

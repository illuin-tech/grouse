import asyncio

from grouse import EvaluationSample, GroundedQAEvaluator
from grouse.dtos import (
    AnswerRelevancy,
    Completeness,
    EvaluationsAndReport,
    Faithfulness,
    GroundedQAEvaluation,
    GroundedQAEvaluationReport,
    Usefulness,
)

TEST_MODEL = "gpt-4o-mini"


class TestGroundedQAEvaluator:
    def setup_method(self) -> None:
        self.evaluator = GroundedQAEvaluator(model_name=TEST_MODEL)

    def test_evaluate_answer_relevancy(self) -> None:
        ar = asyncio.run(
            self.evaluator.evaluate_answer_relevancy(
                eval_sample=EvaluationSample(
                    input="Quel est la capitale de la France ?",
                    actual_output="Paris",
                    expected_output="Paris",
                    references=["Paris"],
                )
            )
        )
        assert isinstance(ar, AnswerRelevancy)
        assert isinstance(ar.answer_affirms_no_document_answers, bool)
        assert isinstance(ar.answer_relevancy_justification, str)
        assert isinstance(ar.answer_relevancy, int)
        assert ar.answer_relevancy == 5

    def test_evaluate_completeness(self) -> None:
        comp = asyncio.run(
            self.evaluator.evaluate_completeness(
                eval_sample=EvaluationSample(
                    input="Quel est la capitale de la France ?",
                    actual_output="Paris",
                    expected_output="Paris",
                    references=["Paris"],
                )
            )
        )
        assert isinstance(comp, Completeness)
        assert isinstance(comp.completeness_justification, str)
        assert isinstance(comp.completeness, int)
        assert comp.completeness == 5

    def test_evaluate_faithfulness(self) -> None:
        faith = asyncio.run(
            self.evaluator.evaluate_faithfulness(
                eval_sample=EvaluationSample(
                    input="Quel est la capitale de la France ?",
                    actual_output="Paris[1]",
                    expected_output="Paris[1]",
                    references=["Paris"],
                )
            )
        )
        assert isinstance(faith, Faithfulness)
        assert isinstance(faith.faithfulness_justification, str)
        assert isinstance(faith.faithfulness, int)
        assert faith.faithfulness == 1

    def test_evaluate_usefulness(self) -> None:
        usef = asyncio.run(
            self.evaluator.evaluate_usefulness(
                eval_sample=EvaluationSample(
                    input="Quel est la capitale de la France ?",
                    actual_output="Paris[1]",
                    expected_output="Paris[1]",
                    references=["Paris"],
                )
            )
        )
        assert isinstance(usef, Usefulness)
        assert isinstance(usef.usefulness_justification, str)
        assert usef.usefulness is None

    def test_evaluate_single_sample(self) -> None:
        evaluation = asyncio.run(
            self.evaluator.evaluate_single_sample(
                eval_sample=EvaluationSample(
                    input="Quel est la capitale de la France ?",
                    actual_output="Paris[1]",
                    expected_output="Paris[1]",
                    references=["Paris"],
                )
            )
        )
        assert isinstance(evaluation, GroundedQAEvaluation)

    def test_evaluate(self) -> None:
        eval_samples = [
            EvaluationSample(
                input="Quel est la capitale de la France ?",
                actual_output="Paris[1]",
                expected_output="Paris[1]",
                references=["Paris"],
            )
        ]
        evaluations_and_report = self.evaluator.evaluate(eval_samples)
        assert isinstance(evaluations_and_report, EvaluationsAndReport)
        evaluations, report = (
            evaluations_and_report.evaluations,
            evaluations_and_report.report,
        )
        assert isinstance(evaluations, list)
        assert len(evaluations) == 1
        assert isinstance(evaluations[0], GroundedQAEvaluation)
        assert isinstance(report, GroundedQAEvaluationReport)

import asyncio
from unittest.mock import patch

from grouse import EvaluationSample, GroundedQAEvaluator
from grouse.dtos import (
    AnswerRelevancy,
    Completeness,
    EvaluationsAndReport,
    Failed,
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
        with patch.object(
            GroundedQAEvaluator,
            "call_llm",
            return_value=AnswerRelevancy(
                answer_relevancy=5,
                answer_affirms_no_document_answers=False,
                answer_relevancy_justification="Justification",
            ),
        ):
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
        with patch.object(
            GroundedQAEvaluator,
            "call_llm",
            return_value=Completeness(
                completeness=5,
                completeness_justification="Justification",
            ),
        ):
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
        with patch.object(
            GroundedQAEvaluator,
            "call_llm",
            return_value=Faithfulness(
                faithfulness=1,
                faithfulness_justification="Justification",
            ),
        ):
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
        with patch.object(
            GroundedQAEvaluator,
            "call_llm",
            return_value=Usefulness(
                usefulness=None,
                usefulness_justification="Justification",
            ),
        ):
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
        with (
            patch.object(
                GroundedQAEvaluator,
                "evaluate_answer_relevancy",
                return_value=AnswerRelevancy(
                    answer_relevancy=5,
                    answer_affirms_no_document_answers=False,
                    answer_relevancy_justification="Justification",
                ),
            ),
            patch.object(
                GroundedQAEvaluator,
                "evaluate_completeness",
                return_value=Completeness(
                    completeness=5,
                    completeness_justification="Justification",
                ),
            ),
            patch.object(
                GroundedQAEvaluator,
                "evaluate_faithfulness",
                return_value=Faithfulness(
                    faithfulness=1,
                    faithfulness_justification="Justification",
                ),
            ),
            patch.object(
                GroundedQAEvaluator,
                "evaluate_usefulness",
                return_value=Usefulness(
                    usefulness=None,
                    usefulness_justification="Justification",
                ),
            ),
        ):
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

    def test_evaluate_single_sample_with_failed_parsing(self) -> None:
        with patch.object(
            GroundedQAEvaluator,
            "call_llm",
            return_value=Failed(),
        ):
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
            assert isinstance(evaluation.answer_relevancy, Failed)
            assert isinstance(evaluation.completeness, Failed)
            assert isinstance(evaluation.faithfulness, Failed)
            assert isinstance(evaluation.usefulness, Failed)
            assert isinstance(evaluation.positive_acceptance, Failed)
            assert isinstance(evaluation.negative_rejection, Failed)

    def test_evaluate(self) -> None:
        with patch.object(
            GroundedQAEvaluator,
            "evaluate_single_sample",
            return_value=GroundedQAEvaluation(
                answer_relevancy=AnswerRelevancy(
                    answer_relevancy=5,
                    answer_affirms_no_document_answers=False,
                    answer_relevancy_justification="Justification",
                ),
                completeness=Completeness(
                    completeness=5,
                    completeness_justification="Justification",
                ),
                faithfulness=Faithfulness(
                    faithfulness=1,
                    faithfulness_justification="Justification",
                ),
                usefulness=Usefulness(
                    usefulness=None,
                    usefulness_justification="Justification",
                ),
                positive_acceptance=None,
                negative_rejection=None,
            ),
        ):
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

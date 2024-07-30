from typing import List, Optional
from pydantic import BaseModel, Field


# Metrics
class AnswerRelevancy(BaseModel):
    answer_affirms_no_document_answers: bool
    answer_relevancy_justification: str
    answer_relevancy: Optional[int] = Field(
        None,
        description="Relevancy score of the answer from 1 to 5 or None",
    )


class AnswerRelevancyPair(BaseModel):
    answer_1: AnswerRelevancy
    answer_2: AnswerRelevancy


class Completeness(BaseModel):
    completeness_justification: str
    completeness: Optional[int] = Field(
        None,
        description="Completeness score of the answer from 1 to 5 or None",
    )


class CompletenessPair(BaseModel):
    answer_1: Completeness
    answer_2: Completeness


class Faithfulness(BaseModel):
    faithfulness_justification: str
    faithfulness: Optional[int] = Field(
        None,
        description="Faithfulness score of the answer in 0, 1 or None",
    )


class FaithfulnessPair(BaseModel):
    answer_1: Faithfulness
    answer_2: Faithfulness


class Usefulness(BaseModel):
    usefulness_justification: str
    usefulness: Optional[int] = Field(
        None, description="Usefulness score of the answer in 0, 1 or None"
    )


class UsefulnessPair(BaseModel):
    answer_1: Usefulness
    answer_2: Usefulness


# Evaluation DTOs
class GroundedQAEvaluationReport(BaseModel):
    answer_relevancy: float
    completeness: float
    faithfulness: float
    usefulness: float
    positive_acceptance: float
    negative_rejection: float
    mean: float


class GroundedQAEvaluation(BaseModel):
    answer_relevancy: AnswerRelevancy
    completeness: Completeness
    faithfulness: Faithfulness
    usefulness: Usefulness
    positive_acceptance: Optional[int]
    negative_rejection: Optional[int]


class EvaluationSample(BaseModel):
    input: str
    actual_output: str
    expected_output: str
    references: List[str]


class EvaluationsAndReport(BaseModel):
    evaluations: List[GroundedQAEvaluation]
    report: GroundedQAEvaluationReport


# Meta Evaluation DTOs
class ExpectedGroundedQAEvaluation(BaseModel):
    answer_relevancy_condition: str
    completeness_condition: str
    faithfulness_condition: str
    usefulness_condition: str


class MetaTestCase(BaseModel):
    evaluation_sample: EvaluationSample
    actual_evaluation: GroundedQAEvaluation
    expected_evaluation: ExpectedGroundedQAEvaluation


class MetaTestCaseResult(BaseModel):
    answer_relevancy: bool
    completeness: bool
    faithfulness: bool
    usefulness: bool
    positive_acceptance: bool
    negative_rejection: bool


class MetaEvalReport(BaseModel):
    answer_relevancy_success: float
    completeness_success: float
    faithfulness_success: float
    usefulness_success: float
    positive_acceptance_success: float
    negative_rejection_success: float
    total: float


class MetaEvaluationsAndReport(BaseModel):
    evaluations: List[MetaTestCaseResult]
    report: MetaEvalReport

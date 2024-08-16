from typing import Generic, List, Literal, Optional, TypeVar

from pydantic import BaseModel, Field
from typing_extensions import override


# Sentinel class used until PEP 0661 is accepted
class Failed(BaseModel):
    """
    A sentinel singleton class used to distinguish failed request results
    from results with the value None (which may have different behavior).
    """

    def __bool__(self) -> Literal[False]:
        return False

    @override
    def __repr__(self) -> str:
        return "FAILED"


# These models are used by instructor to enforce the output json schema
# Be careful if you change these models, as the field names are optimized for the prompt
class Score(BaseModel):
    def __repr__(self):
        return f"<{self.__class__.__name__}>"

    def __str__(self):
        return self.__repr__()


class AnswerRelevancy(Score):
    answer_affirms_no_document_answers: bool
    answer_relevancy_justification: str
    answer_relevancy: Optional[int] = Field(
        None,
        description="Relevancy score of the answer from 1 to 5 or None",
    )


class Completeness(Score):
    completeness_justification: str
    completeness: Optional[int] = Field(
        None,
        description="Completeness score of the answer from 1 to 5 or None",
    )


class Faithfulness(Score):
    faithfulness_justification: str
    faithfulness: Optional[int] = Field(
        None,
        description="Faithfulness score of the answer in 0, 1 or None",
    )


class Usefulness(Score):
    usefulness_justification: str
    usefulness: Optional[int] = Field(
        None, description="Usefulness score of the answer in 0, 1 or None"
    )


T = TypeVar("T", bound=Score)


class ScorePair(BaseModel, Generic[T]):
    answer_1: T
    answer_2: T


# Evaluation DTOs
class GroundedQAEvaluationReport(BaseModel):
    answer_relevancy: float
    answer_relevancy_parsing_success: float
    completeness: float
    completeness_parsing_success: float
    faithfulness: float
    faithfulness_parsing_success: float
    usefulness: float
    usefulness_parse_success: float
    positive_acceptance: float
    negative_rejection: float
    mean: float


class GroundedQAEvaluation(BaseModel):
    answer_relevancy: AnswerRelevancy | Failed
    completeness: Completeness | Failed
    faithfulness: Faithfulness | Failed
    usefulness: Usefulness | Failed
    positive_acceptance: Optional[int] | Failed
    negative_rejection: Optional[int] | Failed


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
    answer_relevancy: bool | Failed
    completeness: bool | Failed
    faithfulness: bool | Failed
    usefulness: bool | Failed
    positive_acceptance: bool | Failed
    negative_rejection: bool | Failed


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

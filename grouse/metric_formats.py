from dataclasses import dataclass
from pydantic import BaseModel, Field
from typing import Optional


class AnswerRelevancy(BaseModel):
    answer_affirms_no_document_answers: bool
    answer_relevancy_justification: str
    answer_relevancy: Field(
        type=Optional[int],
        description="Relevancy score of the answer from 1 to 5 or None",
    )


class AnswerRelevancyPair(BaseModel):
    answer_1: AnswerRelevancy
    answer_2: AnswerRelevancy


class Completeness(BaseModel):
    completeness_justification: str
    completeness: Field(
        type=Optional[int],
        description="Completeness score of the answer from 1 to 5 or None",
    )


class CompletenessPair(BaseModel):
    answer_1: Completeness
    answer_2: Completeness


class Faithfulness(BaseModel):
    faithfulness_justification: str
    faithfulness: Field(
        type=Optional[int],
        description="Faithfulness score of the answer in 0, 1 or None",
    )


class FaithfulnessPair(BaseModel):
    answer_1: Faithfulness
    answer_2: Faithfulness


class Usefulness(BaseModel):
    usefulness_justification: str
    usefulness: Field(
        type=Optional[int], description="Usefulness score of the answer in 0, 1 or None"
    )


class UsefulnessPair(BaseModel):
    answer_1: Usefulness
    answer_2: Usefulness


class PositiveAcceptance(BaseModel):
    positive_acceptance: Field(
        type=Optional[int],
        description="Positive acceptance score of the answer in 0, 1 or None",
    )


class NegativeRejection(BaseModel):
    negative_rejection: Field(
        type=Optional[int],
        description="Negative rejection score of the answer in 0, 1 or None",
    )


class GroundedQAEvaluationReport(BaseModel):
    answer_relevancy: float
    completeness: float
    faithfulness: float
    usefulness: float
    positive_acceptance: float
    negative_rejection: float

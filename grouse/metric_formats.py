from dataclasses import dataclass
from pydantic import BaseModel
from typing import Optional


class AnswerRelevancy(BaseModel):
    answer_affirms_no_document_answers: bool
    answer_relevancy_justification: str
    answer_relevancy: Optional[int]


class AnswerRelevancyPair(BaseModel):
    answer_1: AnswerRelevancy
    answer_2: AnswerRelevancy


class Completeness(BaseModel):
    completeness_justification: str
    completeness: Optional[int]


class CompletenessPair(BaseModel):
    answer_1: Completeness
    answer_2: Completeness


class Faithfulness(BaseModel):
    faithfulness_justification: str
    faithfulness: Optional[int]


class FaithfulnessPair(BaseModel):
    answer_1: Faithfulness
    answer_2: Faithfulness


class Usefulness(BaseModel):
    usefulness_justification: str
    usefulness: Optional[int]


class UsefulnessPair(BaseModel):
    answer_1: Usefulness
    answer_2: Usefulness


class PositiveAcceptance(BaseModel):
    positive_acceptance: Optional[int]


class NegativeRejection(BaseModel):
    negative_rejection: Optional[int]

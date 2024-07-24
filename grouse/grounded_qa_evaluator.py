from typing import List, Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field

from jinja2 import Environment, FileSystemLoader
import litellm
import instructor
from abc import ABC, abstractmethod
import aiohttp, asyncio
from tqdm.asyncio import tqdm

from grouse.metric_formats import (
    AnswerRelevancy,
    Completeness,
    Faithfulness,
    Usefulness,
    PositiveAcceptance,
    NegativeRejection,
    AnswerRelevancyPair,
    CompletenessPair,
    FaithfulnessPair,
    UsefulnessPair,
)

MODEL_NAME = "gpt-4"
NO_ANWER_STR = "no document seems to precisely answer your question"


@dataclass
class GroundedQAEvaluation:
    answer_relevancy: AnswerRelevancy
    completeness: Completeness
    faithfulness: Faithfulness
    usefulness: Usefulness
    positive_aceptance: PositiveAcceptance
    negative_rejection: NegativeRejection


class EvaluationSample(BaseModel):
    input: str
    actual_output: str
    expected_output: str
    retrieved_contexts: List[str]


class GroundedQAEvaluator(ABC):
    @abstractmethod
    def measure(self, eval_sample: EvaluationSample) -> GroundedQAEvaluation:
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self, eval_samples: List[EvaluationSample]
    ) -> List[GroundedQAEvaluation]:
        raise NotImplementedError


class GPT4GroundedQAEvaluator(GroundedQAEvaluator):
    def __init__(self, enable_cache: bool = True):
        if enable_cache:
            litellm.enable_cache("disk")

        self.environment = Environment(loader=FileSystemLoader("grouse/gpt4_prompts"))
        self.async_client = instructor.from_litellm(litellm.acompletion)

    async def measure_answer_relevancy(
        self, eval_sample: EvaluationSample
    ) -> AnswerRelevancy:
        template = self.environment.get_template("answer_relevancy.txt")
        prompt = template.render(
            input=eval_sample.input,
            actual_output=eval_sample.actual_output,
            expected_output=eval_sample.expected_output,
        )
        pair = await self.async_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_model=AnswerRelevancyPair,
        )
        return pair.answer_2

    async def measure_completeness(self, eval_sample: EvaluationSample) -> Completeness:
        template = self.environment.get_template("completeness.txt")
        prompt = template.render(
            input=eval_sample.input,
            actual_output=eval_sample.actual_output,
            expected_output=eval_sample.expected_output,
            contexts=eval_sample.retrieved_contexts,
        )
        pair = await self.async_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_model=CompletenessPair,
        )
        return pair.answer_2

    async def measure_faithfulness(self, eval_sample: EvaluationSample) -> Faithfulness:
        template = self.environment.get_template("faithfulness.txt")
        prompt = template.render(
            actual_output=eval_sample.actual_output,
            expected_output=eval_sample.expected_output,
            contexts=eval_sample.retrieved_contexts,
        )
        pair = await self.async_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_model=FaithfulnessPair,
        )
        return pair.answer_2

    async def measure_usefulness(self, eval_sample: EvaluationSample) -> Usefulness:
        template = self.environment.get_template("usefulness.txt")
        prompt = template.render(
            input=eval_sample.input,
            actual_output=eval_sample.actual_output,
            expected_output=eval_sample.expected_output,
        )
        pair = await self.async_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_model=UsefulnessPair,
        )
        return pair.answer_2

    def measure_positive_acceptance(
        self, eval_sample: EvaluationSample
    ) -> PositiveAcceptance:
        pred_rejected = NO_ANWER_STR in eval_sample.actual_output
        expected_rejected = NO_ANWER_STR in eval_sample.expected_output
        if not pred_rejected and not expected_rejected:
            return PositiveAcceptance(positive_acceptance=1)
        elif pred_rejected and not expected_rejected:
            return PositiveAcceptance(positive_acceptance=0)
        else:
            return PositiveAcceptance(positive_acceptance=None)

    def measure_negative_rejection(
        self, eval_sample: EvaluationSample
    ) -> PositiveAcceptance:
        pred_rejected = NO_ANWER_STR in eval_sample.actual_output
        expected_rejected = NO_ANWER_STR in eval_sample.expected_output
        if pred_rejected and expected_rejected:
            return NegativeRejection(negative_rejection=1)
        elif not pred_rejected and expected_rejected:
            return NegativeRejection(negative_rejection=0)
        else:
            return NegativeRejection(negative_rejection=None)

    async def measure(self, eval_sample: EvaluationSample) -> GroundedQAEvaluation:
        answer_relevancy = await self.measure_answer_relevancy(eval_sample)
        completeness = await self.measure_completeness(eval_sample)

        if answer_relevancy.answer_relevancy is None:
            usefulness = await self.measure_usefulness(eval_sample)
            if usefulness.usefulness is None:
                faithfulness = Faithfulness(
                    faithfulness_justification="", faithfulness=None
                )
            else:
                faithfulness = await self.measure_faithfulness(eval_sample)
        else:
            usefulness = Usefulness(usefulness_justification="", usefulness=None)
            faithfulness = await self.measure_faithfulness(eval_sample)

        positive_acceptance = self.measure_positive_acceptance(eval_sample)
        negative_rejection = self.measure_negative_rejection(eval_sample)
        return GroundedQAEvaluation(
            answer_relevancy,
            completeness,
            faithfulness,
            usefulness,
            positive_acceptance,
            negative_rejection,
        )

    async def async_evaluate(
        self, eval_samples: List[EvaluationSample]
    ) -> List[GroundedQAEvaluation]:
        async with aiohttp.ClientSession() as _:
            measurement_coroutines = [
                self.measure(eval_sample) for eval_sample in eval_samples
            ]
            measurements = await tqdm.gather(*measurement_coroutines)
        return measurements

    def evaluate(
        self, eval_samples: List[EvaluationSample]
    ) -> List[GroundedQAEvaluation]:
        return asyncio.run(self.async_evaluate(eval_samples))


if __name__ == "__main__":
    c = GPT4GroundedQAEvaluator()
    sample = EvaluationSample(
        input="What if these shoes don't fit?",
        # Replace this with the actual output from your LLM application
        expected_output="We offer a 30-day full refund at no extra costs.",
        actual_output="",
        retrieved_contexts=[
            "All customers are eligible for a 30 day full refund at no extra costs."
        ],
    )
    print(c.evaluate([sample] * 10))

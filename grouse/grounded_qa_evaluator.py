import asyncio
from typing import List

import aiohttp
import instructor
import litellm
import numpy as np
from jinja2 import Environment, FileSystemLoader
from tqdm.asyncio import tqdm

from grouse.dtos import (
    AnswerRelevancy,
    AnswerRelevancyPair,
    Completeness,
    CompletenessPair,
    EvaluationSample,
    EvaluationsAndReport,
    FailedType,
    Faithfulness,
    FaithfulnessPair,
    GroundedQAEvaluation,
    GroundedQAEvaluationReport,
    Metric,
    Pair,
    Usefulness,
    UsefulnessPair,
)
from grouse.llm_calls.cached_instructor import CachedAsyncInstructor
from grouse.llm_calls.tracker import Tracker
from grouse.utils import get_positive_acceptance_negative_rejection


class GroundedQAEvaluator:
    def __init__(
        self,
        model_name: str = "gpt-4",
        prompts_path: str = "grouse/gpt4_prompts",
        cache_path: str = ".cache_test/",
    ):
        self.model_name = model_name
        self.environment = Environment(loader=FileSystemLoader(prompts_path))

        cache = litellm.Cache(type="disk", disk_cache_dir=cache_path)
        self.tracker = Tracker()
        self.async_client = CachedAsyncInstructor(
            client=None,
            create=instructor.patch(create=litellm.acompletion),
            cache=cache,
            tracker=self.tracker,
        )

    async def call_llm(
        self, prompt: str, pair_model: Pair, **kwargs
    ) -> Metric | FailedType:
        pair = await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            response_model=pair_model,
            **kwargs,
        )
        if pair is None:
            return "FAILED"
        return pair.answer_2

    async def evaluate_answer_relevancy(
        self, eval_sample: EvaluationSample
    ) -> AnswerRelevancy:
        template = self.environment.get_template("answer_relevancy.txt")
        prompt = template.render(
            input=eval_sample.input,
            actual_output=eval_sample.actual_output,
            expected_output=eval_sample.expected_output,
        )
        return await self.call_llm(prompt, AnswerRelevancyPair)

    async def evaluate_completeness(
        self, eval_sample: EvaluationSample
    ) -> Completeness:
        template = self.environment.get_template("completeness.txt")
        prompt = template.render(
            input=eval_sample.input,
            actual_output=eval_sample.actual_output,
            expected_output=eval_sample.expected_output,
            contexts=eval_sample.references,
        )
        return await self.call_llm(prompt, CompletenessPair)

    async def evaluate_faithfulness(
        self, eval_sample: EvaluationSample
    ) -> Faithfulness:
        template = self.environment.get_template("faithfulness.txt")
        prompt = template.render(
            actual_output=eval_sample.actual_output,
            expected_output=eval_sample.expected_output,
            contexts=eval_sample.references,
        )
        return await self.call_llm(prompt, FaithfulnessPair)

    async def evaluate_usefulness(self, eval_sample: EvaluationSample) -> Usefulness:
        template = self.environment.get_template("usefulness.txt")
        prompt = template.render(
            input=eval_sample.input,
            actual_output=eval_sample.actual_output,
            expected_output=eval_sample.expected_output,
        )
        return await self.call_llm(prompt, UsefulnessPair)

    async def evaluate_single_sample(
        self, eval_sample: EvaluationSample
    ) -> GroundedQAEvaluation:
        answer_relevancy = await self.evaluate_answer_relevancy(eval_sample)
        completeness = await self.evaluate_completeness(eval_sample)

        if answer_relevancy == "FAILED":
            usefulness = "FAILED"
            faithfulness = "FAILED"
        else:
            if answer_relevancy.answer_relevancy is None:
                usefulness = await self.evaluate_usefulness(eval_sample)
                if usefulness.usefulness is None:
                    faithfulness = Faithfulness(
                        faithfulness_justification="", faithfulness=None
                    )
                else:
                    faithfulness = await self.evaluate_faithfulness(eval_sample)
            else:
                usefulness = Usefulness(usefulness_justification="", usefulness=None)
                faithfulness = await self.evaluate_faithfulness(eval_sample)

        if answer_relevancy == "FAILED" or completeness == "FAILED":
            positive_acceptance, negative_rejection = "FAILED", "FAILED"
        else:
            positive_acceptance, negative_rejection = (
                get_positive_acceptance_negative_rejection(
                    answer_relevancy.answer_relevancy, completeness.completeness
                )
            )

        return GroundedQAEvaluation(
            answer_relevancy=answer_relevancy,
            completeness=completeness,
            faithfulness=faithfulness,
            usefulness=usefulness,
            positive_acceptance=positive_acceptance,
            negative_rejection=negative_rejection,
        )

    async def async_evaluate_multiple_samples(
        self, eval_samples: List[EvaluationSample]
    ) -> List[GroundedQAEvaluation]:
        async with aiohttp.ClientSession() as _:
            evaluation_coroutines = [
                self.evaluate_single_sample(eval_sample) for eval_sample in eval_samples
            ]
            evaluations = await tqdm.gather(*evaluation_coroutines)
        return evaluations

    def evaluate_multiple_samples(
        self, eval_samples: List[EvaluationSample]
    ) -> List[GroundedQAEvaluation]:
        self.cost = 0
        results = asyncio.run(self.async_evaluate_multiple_samples(eval_samples))
        self.tracker.log_summary()
        return results

    def evaluate(self, eval_samples: List[EvaluationSample]) -> EvaluationsAndReport:
        evaluations = self.evaluate_multiple_samples(eval_samples)
        ar_mean = np.mean(
            [
                e.answer_relevancy.answer_relevancy
                for e in evaluations
                if e.answer_relevancy != "FAILED"
                and e.answer_relevancy.answer_relevancy is not None
            ]
        )
        ar_parsing_success = np.mean(
            [int(e.answer_relevancy is not None) for e in evaluations]
        )
        c_mean = np.mean(
            [
                e.completeness.completeness
                for e in evaluations
                if e.completeness != "FAILED"
                and e.completeness.completeness is not None
            ]
        )
        c_parsing_success = np.mean(
            [int(e.completeness is not None) for e in evaluations]
        )
        f_mean = np.mean(
            [
                e.faithfulness.faithfulness
                for e in evaluations
                if e.faithfulness != "FAILED"
                and e.faithfulness.faithfulness is not None
            ]
        )
        f_parsing_success = np.mean(
            [int(e.faithfulness is not None) for e in evaluations]
        )
        u_mean = np.mean(
            [
                e.usefulness.usefulness
                for e in evaluations
                if e.usefulness != "FAILED" and e.usefulness.usefulness is not None
            ]
        )
        u_parsing_success = np.mean(
            [int(e.usefulness is not None) for e in evaluations]
        )
        pa_mean = np.mean(
            [
                e.positive_acceptance
                for e in evaluations
                if e.positive_acceptance is not None
            ]
        )
        nr_mean = np.mean(
            [
                e.negative_rejection
                for e in evaluations
                if e.negative_rejection is not None
            ]
        )
        mean = np.mean([ar_mean, c_mean, f_mean, u_mean, pa_mean, nr_mean])
        report = GroundedQAEvaluationReport(
            answer_relevancy=float(ar_mean),
            answer_relevancy_parsing_success=float(ar_parsing_success),
            completeness=float(c_mean),
            completeness_parsing_success=float(c_parsing_success),
            faithfulness=float(f_mean),
            faithfulness_parsing_success=float(f_parsing_success),
            usefulness=float(u_mean),
            usefulness_parse_success=float(u_parsing_success),
            positive_acceptance=float(pa_mean),
            negative_rejection=float(nr_mean),
            mean=mean,
        )
        return EvaluationsAndReport(evaluations=evaluations, report=report)


if __name__ == "__main__":
    c = GroundedQAEvaluator()
    sample = EvaluationSample(
        input="What if these shoes don't fit?",
        # Replace this with the actual output from your LLM application
        expected_output="We offer a 30-day full refund at no extra costs.",
        actual_output="",
        references=[
            "All customers are eligible for a 30 day full refund at no extra costs."
        ],
    )
    print(c.evaluate([sample] * 10))

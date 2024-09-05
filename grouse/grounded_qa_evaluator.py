import asyncio
import json
import logging
import sys
from typing import List, Optional

import aiohttp
import litellm
import numpy as np
from importlib_resources import files
from jinja2 import Environment, FileSystemLoader
from pydantic_core import ValidationError
from tqdm.asyncio import tqdm

from grouse.dtos import (
    AnswerRelevancy,
    AnswerRelevancyPair,
    Completeness,
    CompletenessPair,
    EvaluationSample,
    EvaluationsAndReport,
    Failed,
    Faithfulness,
    FaithfulnessPair,
    GroundedQAEvaluation,
    GroundedQAEvaluationReport,
    Score,
    ScorePair,
    Usefulness,
    UsefulnessPair,
)
from grouse.utils import get_positive_acceptance_negative_rejection

STRUCTURED_OUTPUTS_SUPPORTING_MODELS = [
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-2024-08-06",
]


class GroundedQAEvaluator:
    def __init__(
        self,
        model_name: str = "gpt-4",
        prompts_path: Optional[str] = None,
        cache_path: Optional[str] = None,
    ):
        self.model_name = model_name
        if prompts_path is None:
            self.environment = Environment(
                loader=FileSystemLoader(files("grouse").joinpath("gpt4_prompts"))
            )
        else:
            self.environment = Environment(loader=FileSystemLoader(prompts_path))

        self.logger = logging.getLogger("LLM Call Tracker")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        litellm.enable_cache("disk", cache_path)

        self.cost = 0

    async def call_llm(self, prompt: str, pair_model: ScorePair) -> Score | Failed:
        try:
            kwargs = {"temperature": 0.01, "max_tokens": 2048}
            if self.model_name in STRUCTURED_OUTPUTS_SUPPORTING_MODELS:
                response = await litellm.acompletion(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    response_format=pair_model,
                    **kwargs,
                )
            elif "-turbo" in self.model_name or "4o" in self.model_name:
                response = await litellm.acompletion(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    **kwargs,
                )
            else:
                response = await litellm.acompletion(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    **kwargs,
                )
            loaded_response = json.loads(response.choices[0].message.content)
            if isinstance(loaded_response, dict):
                pair = pair_model(**loaded_response)
                self.cost += litellm.completion_cost(response)
                return pair.answer_2
            else:
                raise ValueError("Response is not a dictionary")

        except (ValidationError, json.decoder.JSONDecodeError, ValueError) as val_error:
            logging.debug(
                f"Call to {self.model_name} with prompt: {prompt}\n"
                f"returned the following error:\n{val_error}"
            )
            return Failed(error=str(val_error))

    async def evaluate_answer_relevancy(
        self, eval_sample: EvaluationSample
    ) -> AnswerRelevancy | Failed:
        template = self.environment.get_template("answer_relevancy.txt")
        prompt = template.render(
            input=eval_sample.input,
            actual_output=eval_sample.actual_output,
            expected_output=eval_sample.expected_output,
        )
        return await self.call_llm(prompt, AnswerRelevancyPair)

    async def evaluate_completeness(
        self, eval_sample: EvaluationSample
    ) -> Completeness | Failed:
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
    ) -> Faithfulness | Failed:
        template = self.environment.get_template("faithfulness.txt")
        prompt = template.render(
            actual_output=eval_sample.actual_output,
            expected_output=eval_sample.expected_output,
            contexts=eval_sample.references,
        )
        return await self.call_llm(prompt, FaithfulnessPair)

    async def evaluate_usefulness(
        self, eval_sample: EvaluationSample
    ) -> Usefulness | Failed:
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

        if isinstance(answer_relevancy, Failed):
            usefulness = Failed(error="answer_relevancy failed")
            faithfulness = Failed(error="answer_relevancy failed")
        else:
            if answer_relevancy.answer_relevancy is None:
                usefulness = await self.evaluate_usefulness(eval_sample)
                if isinstance(usefulness, Failed):
                    faithfulness = Failed(error="usefulness failed")
                elif usefulness.usefulness is None:
                    faithfulness = Faithfulness(
                        faithfulness_justification="", faithfulness=None
                    )
                else:
                    faithfulness = await self.evaluate_faithfulness(eval_sample)
            else:
                usefulness = Usefulness(usefulness_justification="", usefulness=None)
                faithfulness = await self.evaluate_faithfulness(eval_sample)

        positive_acceptance, negative_rejection = (
            get_positive_acceptance_negative_rejection(answer_relevancy, completeness)
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
        results = asyncio.run(self.async_evaluate_multiple_samples(eval_samples))
        self.logger.info(f"Cost: {self.cost:.4f}$")
        return results

    def evaluate(self, eval_samples: List[EvaluationSample]) -> EvaluationsAndReport:
        evaluations = self.evaluate_multiple_samples(eval_samples)
        ar_mean = np.mean(
            [
                e.answer_relevancy.answer_relevancy
                for e in evaluations
                if bool(e.answer_relevancy)
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
                if bool(e.completeness) and e.completeness.completeness is not None
            ]
        )
        c_parsing_success = np.mean(
            [int(e.completeness is not None) for e in evaluations]
        )
        f_mean = np.mean(
            [
                e.faithfulness.faithfulness
                for e in evaluations
                if bool(e.faithfulness) and e.faithfulness.faithfulness is not None
            ]
        )
        f_parsing_success = np.mean(
            [int(e.faithfulness is not None) for e in evaluations]
        )
        u_mean = np.mean(
            [
                e.usefulness.usefulness
                for e in evaluations
                if bool(e.usefulness) and e.usefulness.usefulness is not None
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

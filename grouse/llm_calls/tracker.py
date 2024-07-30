from typing import Optional

import litellm
from litellm.integrations.custom_logger import CustomLogger
import logging
import sys


class Tracker:
    def __init__(self) -> None:
        litellm.callbacks.append(TrackingHandler(self))

        self._api_calls = 0
        self._api_successes = 0
        self._api_failures = 0
        self._parsing_successes = 0
        self._parsing_failures = 0
        self._cache_hits = 0
        self._cost = 0.0

        self.logger = logging.getLogger("LLM Call Tracker")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def increment_cost(self, cost: Optional[float]) -> None:
        if cost is not None:
            self._cost += cost

    def increment_cache_hit(self) -> None:
        self._cache_hits += 1

    def increment_api_success(self) -> None:
        self._api_calls += 1
        self._api_successes += 1

    def increment_api_failure(self) -> None:
        self._api_calls += 1
        self._api_failures += 1

    def increment_parsing_successes(self) -> None:
        self._parsing_successes += 1

    def increment_parsing_failure(self) -> None:
        self._parsing_failures += 1

    def log_summary(self) -> None:
        self.logger.info(f"API calls: {self._api_calls}")
        self.logger.info(f"API successes: {self._api_successes}")
        self.logger.info(f"API failures: {self._api_failures}")
        self.logger.info(f"Parsing successes: {self._parsing_successes}")
        self.logger.info(f"Parsing failures: {self._parsing_failures}")
        self.logger.info(f"Cache hits: {self._cache_hits}")
        self.logger.info(f"Cost: {self._cost:.2f}$")


class TrackingHandler(CustomLogger):
    def __init__(self, tracker: Tracker) -> None:
        super().__init__()
        self.tracker = tracker

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        if kwargs.get("cache_hit", False):
            self.tracker.increment_cache_hit()
            return

        self.tracker.increment_api_success()
        self.tracker.increment_cost(kwargs.get("response_cost", 0.0))

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self.tracker.increment_api_failure()
        self.tracker.increment_cost(kwargs.get("response_cost", 0.0))

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        if kwargs.get("cache_hit", False):
            self.tracker.increment_cache_hit()
            return

        self.tracker.increment_api_success()
        self.tracker.increment_cost(kwargs.get("response_cost", 0.0))

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self.tracker.increment_api_failure()
        self.tracker.increment_cost(kwargs.get("response_cost", 0.0))

from copy import deepcopy
from typing import Any, Awaitable, Callable, Iterable, TypeVar, Union

import instructor
import litellm
import tenacity
from instructor.dsl.partial import Partial
from instructor.exceptions import InstructorRetryException
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, TypeAdapter

from grouse.llm_calls.tracker import Tracker

T = TypeVar("T", bound=Union[BaseModel, "Iterable[Any]", "Partial[Any]"])


class CachedAsyncInstructor(instructor.AsyncInstructor):
    def __init__(
        self,
        client: Any | None,
        create: Callable[..., Any],
        cache: litellm.Cache,
        tracker: Tracker,
        mode: instructor.Mode = instructor.Mode.TOOLS,
        provider: instructor.Provider = instructor.Provider.OPENAI,
        **kwargs: Any,
    ):
        super().__init__(client, create, mode, provider, **kwargs)
        self.cache = cache
        self.tracker = tracker

    async def create(
        self,
        response_model: type[T],
        messages: list[ChatCompletionMessageParam],
        max_retries: int = 3,
        validation_context: dict[str, Any] | None = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> T | Awaitable[T]:
        kwargs = self.handle_kwargs(kwargs)
        # When using JSON Schema, a system message gets prepended by instructor,
        # messing with the cache. We keep the original parameters and pass a deepcopy
        # to avoid storing the wrong key in the cache.
        original_messages = deepcopy(messages)

        result = await self.cache.async_get_cache(messages=original_messages, **kwargs)

        if result is not None:  # Cache hits
            result = TypeAdapter(response_model).validate_python(result)
            self.tracker.increment_cache_hit()
            return result

        # Cache misses
        try:
            result = await self.create_fn(
                response_model=response_model,
                messages=messages,
                max_retries=tenacity.AsyncRetrying(
                    stop=tenacity.stop_after_attempt(max_retries),
                    after=lambda _: self.tracker.increment_parsing_failure(),
                ),
                validation_context=validation_context,
                strict=strict,
                **kwargs,
            )
        except InstructorRetryException as _:
            result = None
            self.tracker.increment_parsing_failure()
        await self.cache.async_add_cache(
            TypeAdapter(response_model).dump_json(result),
            messages=original_messages,
            **kwargs,
        )
        self.tracker.increment_parsing_successes()
        return result

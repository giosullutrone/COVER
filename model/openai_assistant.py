from refusal_benchmark.model.closed_assistant import ClosedAssistant
from typing import List, Optional, Any, Union
from openai import OpenAI
from refusal_benchmark.dataclass.refusal_enums import RefusalType


class OpenAIAssistant(ClosedAssistant):
    def __init__(self, model_name: str, generation_config: Any, **kwargs) -> None:
        self.model = OpenAI()
        self.model_name = model_name
        self.generation_config = generation_config
        self.req_per_minute = kwargs.pop("req_per_minute", 500)
        self.model_kwargs = kwargs

    def _send_single_request(self, content: str, system: Optional[str]) -> str:
        messages = []
        if system is not None:
            messages += [{"role": "system", "content": system}]
        messages += [{"role": "user", "content": content}]

        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **self.generation_config,
        )

        if not (
            response.choices[0].finish_reason == "stop"
            or response.choices[0].finish_reason == "length"
        ):
            return RefusalType.REFUSAL.value
        else:
            return response.choices[0].message.content

    def __enter__(self):
        # Code that runs when entering the context
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Code that runs when exiting the context
        if self.model is None:
            return
        del self.model

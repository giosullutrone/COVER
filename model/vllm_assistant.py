import os
from typing_extensions import override
from refusal_benchmark.model.assistant import Assistant
from typing import List, Optional, Any, Union
from transformers import AutoTokenizer
from vllm import LLM, RequestOutput
import torch
import gc
from transformers import PretrainedConfig
import refusal_benchmark.logging_config as logging_config


class VLLMAssistant(Assistant):
    def __init__(
        self,
        model_name: str,
        generation_config: Any,
        chat_template: Optional[str] = None,
        model_folder: str = "./models",
        **kwargs,
    ) -> None:
        self.model = None
        self.model_name = model_name
        self.generation_config = generation_config
        self.chat_template = chat_template
        self.model_kwargs = kwargs
        self.model_folder = model_folder

    def _get_model_dtype(self, model: str) -> Union[str, None]:
        try:
            config = PretrainedConfig.from_pretrained(model)
            return getattr(config, "torch_dtype", None)
        except:
            return None

    def _init_model(self):
        try:
            model_path = os.path.abspath(
                os.path.join(self.model_folder, self.model_name)
            )
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model {model_path} not found")
            if "dtype" not in self.model_kwargs:
                self.model_kwargs["dtype"] = self._get_model_dtype(model_path)
            self.model = LLM(model=model_path, **self.model_kwargs)
            logging_config.logging.debug(
                f"Loaded {self.model_name} locally with dtype {self.model_kwargs['dtype']}"
            )
        except:
            if "dtype" not in self.model_kwargs:
                self.model_kwargs["dtype"] = self._get_model_dtype(self.model_name)
            self.model = LLM(model=self.model_name, **self.model_kwargs)
            logging_config.logging.debug(
                f"Loaded {self.model_name} from huggingface with dtype {self.model_kwargs['dtype']}"
            )

    def _send_request_outputs(
        self,
        contents: list[str],
        systems: Optional[List[str]] = None,
        suffix: Optional[str] = None,
    ) -> list[RequestOutput]:
        if self.model is None:
            self._init_model()

        assert self.model is not None, "Model is not initialized"

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                os.path.abspath(os.path.join(self.model_folder, self.model_name))
            )
        except:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.chat_template is not None:
            tokenizer.chat_template = self.chat_template

        if systems is None:
            contents = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": x}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
                + (suffix if suffix is not None else "")
                for x in contents
            ]
        else:
            contents = [
                tokenizer.apply_chat_template(
                    [{"role": "system", "content": y}, {"role": "user", "content": x}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
                + (suffix if suffix is not None else "")
                for x, y in zip(contents, systems)
            ]

        outputs = self.model.generate(contents, sampling_params=self.generation_config)
        return outputs

    @override
    def _send(
        self,
        contents: List[str],
        systems: Optional[List[str]] = None,
        suffix: Optional[str] = None,
    ) -> List[str]:
        outputs = self._send_request_outputs(contents, systems, suffix)
        return [x.outputs[0].text for x in outputs]

    def _send_multiple_generations(
        self,
        contents: List[str],
        systems: Optional[List[str]] = None,
        n_generations: int = 5,
    ) -> List[List[str]]:
        outputs = self._send_request_outputs(contents, systems)
        return [[x.outputs[i].text for i in range(n_generations)] for x in outputs]

    @override
    def generate_answers(
        self,
        task_prompt: str,
        texts: list[tuple[str]],
        n_generations: int,
        questions: list[str] | None,
        system: str | None,
    ) -> list[list[str]]:
        contents: list[str] = []

        for i in range(len(texts)):
            text = texts[i]
            contents.append(
                self._add_texts(text)
                + (
                    task_prompt
                    if questions is None
                    else task_prompt.format(question=questions[i])
                )
            )

        completions: list[list[str]] = self._send_multiple_generations(
            contents, [system] * len(contents) if system is not None else None
        )

        return completions

    @override
    def _send_with_first_prob(
        self, contents: list[str], systems: list[str] | None = None
    ) -> list[tuple[str, list[tuple[str, float]]]]:
        outputs = self._send_request_outputs(contents, systems)
        outputs_probabilities: list[list[tuple[str, float]]] = []
        for output in outputs:
            sample_logprobs = output.outputs[0].logprobs
            assert sample_logprobs is not None, (
                "Trying to get logprobs but none were found, make sure to set the logprobs parameter in the sampling_params"
            )
            logprobs_first_token = list(sample_logprobs[1].values())
            # Knowing that logprobs_first_token is a list of logprobs for the second token
            # and Logprob is defined as:
            # Attributes:
            #   logprob: The logprob of chosen token
            #   rank: The vocab rank of chosen token (>=1)
            #   decoded_token: The decoded chosen token index
            #
            # We can extract the tokens and their logprobs
            # We can then convert the logprobs to probabilities using the softmax function

            # Extract the tokens and their logprobs
            tokens = [
                (x.decoded_token if x.decoded_token is not None else "")
                for x in logprobs_first_token
            ]
            logprobs = [x.logprob for x in logprobs_first_token]

            # Convert the logprobs to probabilities using the softmax function
            probabilities: list[float] = torch.nn.functional.softmax(
                torch.tensor(logprobs), dim=0
            ).tolist()
            outputs_probabilities.append(list(zip(tokens, probabilities)))

        return [(x.outputs[0].text, y) for x, y in zip(outputs, outputs_probabilities)]

    def __enter__(self):
        # Code that runs when entering the context
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Code that runs when exiting the context
        if self.model is None:
            return
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

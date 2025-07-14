from vllm import SamplingParams
from refusal_benchmark.model.vllm_assistant import VLLMAssistant
from refusal_benchmark.model.gemini_assistant import GeminiAssistant
from refusal_benchmark.model.openai_assistant import OpenAIAssistant
from refusal_benchmark.dataclass.results_dataclasses import Parameters
from google.genai import types

temperature = 0.7

# Parameters
vllm_params = {
    "generation_config": SamplingParams(
        max_tokens=128, temperature=temperature, seed=42, n=5
    ),
    "max_model_len": 4096,
    "enable_prefix_caching": True,
}

PARAMETERS_OFFLINE: list[Parameters] = [
    Parameters(
        model=VLLMAssistant,
        model_kwargs=vllm_params,
        model_name="meta-llama/Llama-2-7b-chat-hf",
        model_folder="llama-2-7b-chat-hf",
    ),
    Parameters(
        model=VLLMAssistant,
        model_kwargs=vllm_params,
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        model_folder="meta-Llama-3-8B-Instruct",
    ),
    Parameters(
        model=VLLMAssistant,
        model_kwargs=vllm_params,
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        model_folder="meta-Llama-3.1-8B-Instruct",
    ),
    Parameters(
        model=VLLMAssistant,
        model_kwargs={**vllm_params, "tensor_parallel_size": 4},
        model_name="meta-llama/Meta-Llama-3-70B-Instruct",
        model_folder="meta-Llama-3-70B-Instruct",
    ),
    Parameters(
        model=VLLMAssistant,
        model_kwargs=vllm_params,
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        model_folder="meta-Llama-3.2-3B-Instruct",
    ),
    Parameters(
        model=VLLMAssistant,
        model_kwargs=vllm_params,
        model_name="mistralai/Mistral-7B-Instruct-v0.3",
        model_folder="mistral-7B-Instruct-v0.3",
    ),
    Parameters(
        model=VLLMAssistant,
        model_kwargs=vllm_params,
        model_name="Qwen/Qwen2.5-7B-Instruct",
        model_folder="qwen2.5-7B-Instruct",
    ),
    Parameters(
        model=VLLMAssistant,
        model_kwargs=vllm_params,
        model_name="google/gemma-2-9b-it",
        model_folder="gemma-2-9b-it",
    ),
    Parameters(
        model=VLLMAssistant,
        model_kwargs=vllm_params,
        model_name="microsoft/phi-4",
        model_folder="phi-4",
    ),
    Parameters(
        model=VLLMAssistant,
        model_kwargs=vllm_params,
        model_name="mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated",
        model_folder="meta-Llama-3.1-8B-Instruct-abliterated",
    ),
    Parameters(
        model=VLLMAssistant,
        model_kwargs=vllm_params,
        model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        model_folder="DeepSeek-R1-Distill-Llama-8B",
    ),
]

PARAMETERS_ONLINE: list[Parameters] = [
    Parameters(
        model=GeminiAssistant,
        model_kwargs={
            "generation_config": types.GenerationConfig(
                candidate_count=1,
                max_output_tokens=128,
                temperature=temperature,
            ),
            "req_per_minute": 400,
        },
        model_name="gemini-1.5-flash",
        model_folder="gemini-1.5-flash",
    ),
    Parameters(
        model=OpenAIAssistant,
        model_kwargs={
            "generation_config": {
                "temperature": temperature,
                "max_completion_tokens": 128,
            },
            "req_per_minute": 400,
        },
        model_name="gpt-4o-mini",
        model_folder="gpt-4o-mini",
    ),
]

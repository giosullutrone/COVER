from refusal_benchmark.prompts.systems import SystemType
from refusal_benchmark.prompts.tasks import TaskType
from copy import deepcopy


def is_question_task(prompt: str) -> bool:
    return "{question}" in prompt


def generate_task_combinations(
    mistral_prompt: bool,
    gemma_prompt: bool,
) -> list[tuple[str, str, str, str | None]]:
    _systems = deepcopy(list(SystemType))
    if mistral_prompt:
        _systems.remove(SystemType.ETHICAL_SYSTEM)
    else:
        _systems.remove(SystemType.MISTRAL_SYSTEM)

    # Gemini does not support system prompts
    if gemma_prompt:
        _systems = [
            SystemType.NO_SYSTEM,
        ]

    return [
        (task.name, task.value, system.name, system.value)
        for task in TaskType
        for system in _systems
    ]


def generate_classification_combinations() -> list[tuple[str, str]]:
    return deepcopy([(task.name, task.value) for task in TaskType])

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from refusal_benchmark.dataclass.collections_dataclasses import QuestionCollection
from refusal_benchmark.model.assistant import Assistant
from typing import Any


@dataclass_json
@dataclass
class Result:
    question_collection: QuestionCollection
    system_name: str
    task_name: str
    stats: dict[str, Any] | None = field(default=None)


@dataclass_json
@dataclass
class Results:
    results: list[Result]


@dataclass_json
@dataclass
class Parameters:
    model: type[Assistant]
    model_kwargs: dict[str, Any]
    model_name: str
    model_folder: str

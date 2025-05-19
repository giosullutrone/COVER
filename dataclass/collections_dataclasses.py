from dataclasses import dataclass, field
from typing import Any
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Text:
    text: str
    questions: list[str] | None = field(default=None)
    unsafe: bool | None = field(default=None)
    unsafe_types: list[tuple[str, float, list[str]]] | None = field(
        default_factory=list
    )


@dataclass_json
@dataclass
class Collection:
    name: str
    url: str
    texts: list[Text]


@dataclass_json
@dataclass
class Question:
    question: str
    texts: list[Text] | None = field(default=None)
    texts_source: list[Text] | None = field(default=None)
    answers: list[str] | None = field(default=None)
    refusals: list[bool | None] | None = field(default=None)


@dataclass_json
@dataclass
class QuestionCollection:
    name: str
    url: str
    questions: list[Question]
    stats: dict[str, Any] | None = field(default=None)

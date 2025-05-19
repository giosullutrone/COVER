from refusal_benchmark.dataclass.collections_dataclasses import (
    Collection,
    QuestionCollection,
)

from refusal_benchmark.pipeline.handler import CollectionHandler
from refusal_benchmark.utils import with_deepcopy_of_collection


class CollectionStats(CollectionHandler):
    @with_deepcopy_of_collection
    @staticmethod
    def get_collection(
        collection: Collection, question_collection: QuestionCollection
    ) -> QuestionCollection:
        # Stats:
        # - Number of texts
        # - Number of unsafe texts
        # - Percentage of unsafe texts
        # - Number of questions
        # - Number of questions with unsafe texts
        # - Percentage of questions with unsafe texts

        stats = {}
        stats["num_texts"] = len(collection.texts)
        stats["num_unsafe_texts"] = sum(1 for text in collection.texts if text.unsafe)
        stats["unsafe_text_percentage"] = (
            stats["num_unsafe_texts"] / stats["num_texts"] * 100
            if stats["num_texts"] > 0
            else 0
        )
        stats["num_questions"] = len(question_collection.questions)
        stats["num_questions_with_unsafe_texts"] = sum(
            1
            for question in question_collection.questions
            if question.texts is not None and any(x.unsafe for x in question.texts)
        )
        stats["unsafe_question_percentage"] = (
            stats["num_questions_with_unsafe_texts"] / stats["num_questions"] * 100
            if stats["num_questions"] > 0
            else 0
        )

        question_collection.stats = stats
        return question_collection

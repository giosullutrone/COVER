from refusal_benchmark.model.assistant import Assistant
from refusal_benchmark.dataclass.collections_dataclasses import Collection
from refusal_benchmark.pipeline.handler import CollectionHandler
from refusal_benchmark.utils import with_deepcopy_of_collection


class CollectionQuestioner(CollectionHandler):
    @with_deepcopy_of_collection
    @staticmethod
    def get_collection(
        assistant: Assistant, collection: Collection, single: bool
    ) -> Collection:
        texts = CollectionQuestioner.get_texts(collection)

        list_of_questions = assistant.generate_questions(
            [x.text for x in texts], single=single
        )

        for text, questions in zip(texts, list_of_questions):
            text.questions = questions
        return collection

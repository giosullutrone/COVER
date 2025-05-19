from refusal_benchmark.dataclass.collections_dataclasses import (
    Collection,
    QuestionCollection,
    Text,
    Question,
)


class CollectionHandler:
    @staticmethod
    def get_texts(collection: Collection) -> list[Text]:
        texts: list[Text] = []
        for text in collection.texts:
            texts.append(text)
        return texts


class QuestionCollectionHandler:
    @staticmethod
    def get_questions(collection: QuestionCollection) -> list[Question]:
        questions: list[Question] = []
        for question in collection.questions:
            questions.append(question)
        return questions

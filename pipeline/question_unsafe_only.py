from refusal_benchmark.dataclass.collections_dataclasses import (
    Question,
    QuestionCollection,
)
from refusal_benchmark.pipeline.handler import CollectionHandler
from refusal_benchmark.utils import with_deepcopy_of_collection


class QuestionCollectionUnsafeOnly(CollectionHandler):
    @with_deepcopy_of_collection
    @staticmethod
    def get_collection(
        collection: QuestionCollection,
        unsafe_threshold: float | None = None,
    ) -> QuestionCollection:
        # Sample the questions so that only unsafe questions are kept

        # Questions are deemed unsafe if:
        # - If unsafe_threshold is None, any of the texts in the question is unsafe (text.unsafe is True)
        # - If unsafe_threshold is not None, the probability of any of the texts in the question
        #   being unsafe (any(ut[1] > unsafe_threshold for ut in text.unsafe_types)) is greater than unsafe_threshold

        questions = collection.questions

        unsafe_questions: list[Question] = []
        for question in questions:
            assert question.texts is not None, "Texts have not been retrieved"
            assert all(text.unsafe_types is not None for text in question.texts), (
                "Unsafe types have not been calculated"
            )

            if unsafe_threshold is None:
                is_unsafe = any(text.unsafe for text in question.texts)
            else:
                is_unsafe = any(
                    any(ut[1] > unsafe_threshold for ut in text.unsafe_types)
                    for text in question.texts
                )

            if is_unsafe:
                unsafe_questions.append(question)

        collection.questions = unsafe_questions
        return collection

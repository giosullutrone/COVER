from refusal_benchmark.dataclass.collections_dataclasses import (
    Question,
    QuestionCollection,
)
from refusal_benchmark.pipeline.handler import CollectionHandler
from refusal_benchmark.utils import with_deepcopy_of_collection
import random


class QuestionCollectionSampler(CollectionHandler):
    @with_deepcopy_of_collection
    @staticmethod
    def get_collection(
        collection: QuestionCollection,
        max_questions: int,
        keep_ratio: bool,
        unsafe_threshold: float | None = None,
        seed: int = 42,
    ) -> QuestionCollection:
        # Sample the questions so that the ratio beteween safe and unsafe questions is kept the same as in the original collection (if keep_ratio is True)
        # or sample the questions randomly (if keep_ratio is False)

        # Questions are deemed unsafe if:
        # - If unsafe_threshold is None, any of the texts in the question is unsafe (text.unsafe is True)
        # - If unsafe_threshold is not None, the probability of any of the texts in the question
        #   being unsafe (any(ut[1] > unsafe_threshold for ut in text.unsafe_types)) is greater than unsafe_threshold
        random.seed(seed)

        # Sample the questions
        questions = collection.questions

        safe_questions: list[Question] = []
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
            else:
                safe_questions.append(question)

        # Sample questions from the safe and unsafe questions so that the ratio between safe and unsafe questions is kept the same
        # and the number of total questions is at most max_questions
        if keep_ratio:
            num_safe_questions = int(
                len(safe_questions) / len(questions) * max_questions
            )
            num_unsafe_questions = max_questions - num_safe_questions

            if len(safe_questions) > num_safe_questions:
                safe_questions = random.sample(safe_questions, num_safe_questions)
            if len(unsafe_questions) > num_unsafe_questions:
                unsafe_questions = random.sample(unsafe_questions, num_unsafe_questions)
        else:
            if len(safe_questions) + len(unsafe_questions) > max_questions:
                all_questions = random.sample(
                    safe_questions + unsafe_questions, max_questions
                )
                safe_questions = [
                    question for question in all_questions if question in safe_questions
                ]
                unsafe_questions = [
                    question
                    for question in all_questions
                    if question in unsafe_questions
                ]

        # Combine the safe and unsafe questions
        all_questions = safe_questions + unsafe_questions
        # Shuffle the questions
        random.shuffle(all_questions)

        # Update the collection with the sampled questions
        collection.questions = all_questions
        return collection

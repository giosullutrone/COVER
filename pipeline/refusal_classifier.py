from refusal_benchmark.dataclass.results_dataclasses import Results
from refusal_benchmark.pipeline.handler import QuestionCollectionHandler
from refusal_benchmark.model.assistant import Assistant


class CollectionRefusalClassifier(QuestionCollectionHandler):
    @staticmethod
    def get_results(assistant: Assistant, results: Results) -> Results:
        collections = [x.question_collection for x in results.results]

        for collection in collections:
            # Get the questions from the collection
            questions = CollectionRefusalClassifier.get_questions(collection)

            assert all(x.answers is not None for x in questions), (
                "Answers have not been generated"
            )

            # For each question in the collection, classify the refusal
            refusals = assistant.classify_refusal(
                answers=[x.answers for x in questions if x.answers is not None]
            )

            for idx in range(len(questions)):
                questions[idx].refusals = refusals[idx]

        return results

from refusal_benchmark.dataclass.collections_dataclasses import (
    QuestionCollection,
)
from refusal_benchmark.dataclass.results_dataclasses import Result, Results
from refusal_benchmark.pipeline.handler import QuestionCollectionHandler
from refusal_benchmark.model.assistant import Assistant
from refusal_benchmark.utils import with_deepcopy_of_collection
from tqdm import tqdm
import copy

from refusal_benchmark.prompts.utils import is_question_task, generate_task_combinations


class CollectionAnswerer(QuestionCollectionHandler):
    @with_deepcopy_of_collection
    @staticmethod
    def get_results(
        assistant: Assistant,
        collection: QuestionCollection,
        n_generations: int,
        is_mistral: bool,
        is_gemma: bool,
    ) -> Results:
        # Get the task combinations
        combinations = generate_task_combinations(is_mistral, is_gemma)

        # For each combination, generate the answers
        results: list[Result] = []

        for task_name, task_prompt, system_name, system_prompt in tqdm(
            combinations,
            desc="Processing combinations",
            leave=False,
            unit="combination",
        ):
            _collection = copy.deepcopy(collection)
            # Get the questions from the collection
            questions = CollectionAnswerer.get_questions(_collection)

            assert all(x.texts is not None for x in questions), (
                "Texts have not been retrieved"
            )

            # For each question in the collection, if the task is a question task, generate the answer
            answers = assistant.generate_answers(
                task_prompt,
                [
                    tuple([y.text for y in x.texts] if x.texts is not None else [])
                    for x in questions
                ],
                n_generations,
                [x.question for x in questions]
                if is_question_task(task_prompt)
                else None,
                system_prompt,
            )

            for idx in range(len(questions)):
                questions[idx].answers = answers[idx]
                if not is_question_task(task_prompt):
                    questions[idx].question = task_prompt

            result = Result(
                question_collection=_collection,
                system_name=system_name,
                task_name=task_name,
            )
            results.append(result)

        return Results(results=results)

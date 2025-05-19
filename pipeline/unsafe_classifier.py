from refusal_benchmark.model.assistant import Assistant
from refusal_benchmark.dataclass.collections_dataclasses import Collection
from refusal_benchmark.pipeline.handler import CollectionHandler
from refusal_benchmark.prompts.utils import generate_classification_combinations
from refusal_benchmark.utils import with_deepcopy_of_collection
from tqdm import tqdm


class CollectionUnsafeClassifier(CollectionHandler):
    @with_deepcopy_of_collection
    @staticmethod
    def get_collection(
        assistant: Assistant, collection: Collection, threshold: float = 0.5
    ) -> Collection:
        texts = CollectionUnsafeClassifier.get_texts(collection)

        combinations = generate_classification_combinations()

        for task_name, task_prompt in tqdm(
            combinations, desc="Unsafe classification", unit="Combination"
        ):
            classifications = assistant.classify_unsafe(
                [x.text for x in texts], task_prompt, threshold
            )

            for text, classification in zip(texts, classifications):
                unsafe, unsafe_prob, unsafe_types = classification
                # If the text has never been classified, set the current classification
                if text.unsafe is None:
                    text.unsafe = unsafe
                if text.unsafe_types is None:
                    text.unsafe_types = []
                text.unsafe_types.append((task_name, unsafe_prob, unsafe_types))
                # If the text has been classified as unsafe, set the unsafe flag and append the current classification with the task name associated
                if unsafe:
                    text.unsafe = unsafe

        return collection

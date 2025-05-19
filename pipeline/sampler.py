from refusal_benchmark.dataclass.collections_dataclasses import Collection
from refusal_benchmark.pipeline.handler import CollectionHandler
from refusal_benchmark.utils import with_deepcopy_of_collection
import random


class CollectionSampler(CollectionHandler):
    @with_deepcopy_of_collection
    @staticmethod
    def get_collection(
        collection: Collection, max_texts: int, seed: int = 42
    ) -> Collection:
        texts = CollectionSampler.get_texts(collection)

        if len(texts) < max_texts:
            return collection

        random.seed(seed)

        # Sample the texts
        sampled_texts = random.sample(texts, min(max_texts, len(texts)))

        # Update the collection with the sampled texts
        collection.texts = sampled_texts
        return collection

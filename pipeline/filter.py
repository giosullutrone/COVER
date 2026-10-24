from dataclass.collections_dataclasses import Collection
from pipeline.handler import CollectionHandler
from utils import with_deepcopy_of_collection
from transformers import PreTrainedTokenizer


class CollectionFilter(CollectionHandler):
    @with_deepcopy_of_collection
    @staticmethod
    def get_collection(
        collection: Collection,
        tokenizer: PreTrainedTokenizer,
        min_len: int,
        max_len: int,
    ) -> Collection:
        collection.texts = [
            x
            for x in collection.texts
            if min_len <= len(tokenizer(x.text)["input_ids"]) <= max_len
        ]

        return collection

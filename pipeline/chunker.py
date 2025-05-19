from copy import deepcopy
from refusal_benchmark.dataclass.collections_dataclasses import Collection, Text
from refusal_benchmark.pipeline.handler import CollectionHandler
from refusal_benchmark.utils import with_deepcopy_of_collection
from transformers import PreTrainedTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter


class CollectionChunker(CollectionHandler):
    @with_deepcopy_of_collection
    @staticmethod
    def get_collection(
        collection: Collection,
        tokenizer: PreTrainedTokenizer,
        max_tokens: int,
    ) -> Collection:
        # Split the Text.text into chunks of max_len tokens
        texts: list[Text] = []

        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer, chunk_size=max_tokens, chunk_overlap=int(max_tokens // 4)
        )

        for text in collection.texts:
            chunks = text_splitter.split_text(text.text)
            for chunk in chunks:
                _text = deepcopy(text)
                _text.text = chunk
                texts.append(_text)

        collection.texts = texts
        return collection

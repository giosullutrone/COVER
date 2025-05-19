import os
from collections import defaultdict

from datasets import load_dataset
from refusal_benchmark.dataclass.collections_dataclasses import Text, Collection
from refusal_benchmark.dataset.dataset import Dataset


class RagBench(Dataset):
    def __init__(self, folder_root: str) -> None:
        """
        :param folder_root: Path to the folder containing multiple subfolders,
                            each of which has parquet files (train/test/valid).
        """
        self.folder_root = folder_root

    def _get_texts(self) -> list[Text]:
        """
        Loads the train split from each subfolder (containing train-*.parquet),
        then accumulates a mapping from each document string -> set of questions
        that reference that document. We then turn that into a list of Text
        objects (one per document) which store the document and the questions.
        """
        doc_to_questions = defaultdict(set)

        # Attempt to load train parquet shards (e.g. train-00000-of-00001.parquet)
        train_pattern = os.path.join(self.folder_root, "train-*.parquet")
        if not any(
            fname.startswith("train-") and fname.endswith(".parquet")
            for fname in os.listdir(self.folder_root)
        ):
            raise ValueError(
                f"No train parquet files found in {self.folder_root}. "
                "Please ensure that it contains train-*.parquet files."
            )

        # Load only the train split
        dataset = load_dataset(
            "parquet", data_files={"train": train_pattern}, split="train"
        )

        # Each row has a "documents" list and a "question" string
        for row in dataset:
            question = row["question"]
            documents = row["documents"]
            for doc in documents:
                doc_to_questions[doc].add(question)

        # Now build our list of Text instances
        texts = []
        for doc, questions in doc_to_questions.items():
            texts.append(Text(text=doc, questions=list(questions)))

        return texts

    def get_collection(self) -> Collection:
        """
        Returns a Collection that includes all the Text objects built from
        the subfolders' train splits.
        """
        return Collection(
            name="RagBench",
            url="https://huggingface.co/datasets/rungalileo/ragbench",
            texts=self._get_texts(),
        )

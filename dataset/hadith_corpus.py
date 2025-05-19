import os
import csv
from refusal_benchmark.dataclass.collections_dataclasses import Text, Collection
from typing import List
from refusal_benchmark.dataset.dataset import Dataset


class HadithCorpus(Dataset):
    def __init__(self, folder_root: str) -> None:
        self.folder_root = folder_root

    def _get_texts(self) -> List[Text]:
        all_matns = []

        # List of book folders
        book_folders = ["AbuDaud", "Bukhari", "IbnMaja", "Muslim", "Nesai", "Tirmizi"]

        for book in book_folders:
            book_path = os.path.join(self.folder_root, book)

            if not os.path.isdir(book_path):
                print(f"Warning: {book} folder not found")
                continue

            for file in os.listdir(book_path):
                if file.endswith(".csv") and file.startswith("Chapter"):
                    file_path = os.path.join(book_path, file)

                    with open(file_path, "r", encoding="utf-8") as csvfile:
                        reader = csv.DictReader(csvfile)
                        for row in reader:
                            english_matn = row.get("English_Matn", "")
                            if english_matn:
                                all_matns.append(english_matn)
        return [Text(text=x) for x in all_matns]

    def get_collection(self) -> Collection:
        return Collection(
            "LK-Hadith-Corpus",
            "https://github.com/ShathaTm/LK-Hadith-Corpus",
            texts=self._get_texts(),
        )


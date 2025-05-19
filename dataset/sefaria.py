import os
import json
from refusal_benchmark.dataclass.collections_dataclasses import Text, Collection
from typing import List
from refusal_benchmark.dataset.dataset import Dataset


class Sefaria(Dataset):
    def __init__(self, folder_root: str) -> None:
        self.folder_root = folder_root

    def _get_texts(self) -> List[Text]:
        all_texts = []

        for root, dirs, files in os.walk(self.folder_root):
            if os.path.basename(root) == "English" and "merged.json" in files:
                json_path = os.path.join(root, "merged.json")
                if not "json/Talmud/Bavli/Seder" in json_path:
                    continue
                try:
                    with open(json_path, "r", encoding="utf-8") as file:
                        data = json.load(file)

                        # Extract all available texts
                        text_data = data.get("text", [])
                        if isinstance(text_data, dict):
                            text_data = text_data.values()

                        for text_entry in text_data:
                            if isinstance(text_entry, list):
                                all_texts.extend(
                                    [
                                        item
                                        for sublist in text_entry
                                        for item in (
                                            sublist
                                            if isinstance(sublist, list)
                                            else [sublist]
                                        )
                                        if item and isinstance(item, str)
                                    ]
                                )
                            elif isinstance(text_entry, str):
                                all_texts.append(text_entry)

                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file: {json_path}")
                except IOError:
                    print(f"Error reading file: {json_path}")

        return [Text(text=x) for x in all_texts]

    def get_collection(self) -> Collection:
        return Collection(
            "Sefaria",
            "https://github.com/Sefaria/Sefaria-Export",
            texts=self._get_texts(),
        )


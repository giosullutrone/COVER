from dataclass.collections_dataclasses import Collection


class Dataset:
    def get_collection(self) -> Collection:
        raise NotImplementedError()
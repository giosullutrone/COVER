import os
import json
import dataclasses
from typing import Callable, Any
from refusal_benchmark.dataclass.collections_dataclasses import (
    Collection,
    QuestionCollection,
)
from refusal_benchmark.dataclass.results_dataclasses import Results
import functools
import copy


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        # Check if the object is a dataclass instance
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        # Check if the object is a class (not an instance)
        if isinstance(o, type):
            return o.__name__
        try:
            # Attempt to use the default serialization
            return super().default(o)
        except TypeError:
            # If serialization fails, return the class name of the instance
            return o.__class__.__name__


def save_dataclass(path: str, collection: Collection) -> None:
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        json.dump(collection, f, indent=4, cls=EnhancedJSONEncoder)


def load_collection(path: str) -> Collection | QuestionCollection:
    try:
        with open(path, "r") as f:
            j = json.load(f)
            return Collection.from_dict(j)
    except:
        with open(path, "r") as f:
            j = json.load(f)
            return QuestionCollection.from_dict(j)


def load_dataclass(path: str, cls) -> Any:
    with open(path, "r") as f:
        j = json.load(f)
        return cls.from_dict(j)


def conditional_create_collection(
    file_path: str,
    func: Callable[[Any], Collection],
    func_kwargs,
    skip_if_exists: bool,
) -> Collection | QuestionCollection:
    if not skip_if_exists or not os.path.isfile(file_path):
        collection = func(**func_kwargs)
        save_dataclass(file_path, collection)
    else:
        collection = load_collection(file_path)
    return collection


def conditional_create_results(
    file_path: str, func: Callable, func_kwargs, skip_if_exists: bool
) -> Results:
    if not skip_if_exists or not os.path.isfile(file_path):
        results = func(**func_kwargs)
        save_dataclass(file_path, results)
    else:
        print(f"Skipping {file_path}")
        results = load_dataclass(file_path, Results)
    return results


def with_deepcopy_of_collection(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Find the 'collection' argument (assume it is the second positional argument)
        # Deepcopy the collection argument
        # args must be empty
        assert args == (), "args must be empty"
        if "collection" in kwargs:
            kwargs["collection"] = copy.deepcopy(kwargs["collection"])

        # Call the original function with the deepcopy of the collection
        return func(**kwargs)

    return wrapper


def with_deepcopy_of_results(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Find the 'collection' argument (assume it is the second positional argument)
        # Deepcopy the collection argument
        # args must be empty
        assert args == (), "args must be empty"
        if "results" in kwargs:
            kwargs["results"] = copy.deepcopy(kwargs["results"])

        # Call the original function with the deepcopy of the collection
        return func(**kwargs)

    return wrapper

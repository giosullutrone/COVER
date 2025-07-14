import os
import argparse
from vllm import SamplingParams
from refusal_benchmark.model.vllm_assistant import VLLMAssistant
from refusal_benchmark.utils import conditional_create_collection
from refusal_benchmark.pipeline.filter import CollectionFilter
from refusal_benchmark.pipeline.unsafe_classifier import CollectionUnsafeClassifier
from refusal_benchmark.pipeline.to_question_collection import ToQuestionCollection
from refusal_benchmark.dataset.hadith_corpus import HadithCorpus
from refusal_benchmark.dataset.sefaria import Sefaria
from refusal_benchmark.pipeline.stats import CollectionStats
from transformers import AutoTokenizer
from tqdm import tqdm
from refusal_benchmark.pipeline.chunker import CollectionChunker
from refusal_benchmark.pipeline.question_sampler import QuestionCollectionSampler
from refusal_benchmark.pipeline.sampler import CollectionSampler
from refusal_benchmark.pipeline.question_unsafe_only import QuestionCollectionUnsafeOnly
from refusal_benchmark.pipeline.questioner import CollectionQuestioner
from refusal_benchmark.dataset.msmarco import MSMarco


if __name__ == "__main__":
    ####################################################################
    # Parse arguments
    parser = argparse.ArgumentParser(description="Process and classify datasets.")
    _ = parser.add_argument(
        "--hadith_folder",
        type=str,
        default=None,
        help="Folder path for the Hadith dataset",
    )
    _ = parser.add_argument(
        "--sefaria_folder",
        type=str,
        default=None,
        help="Folder path for the Sefaria dataset",
    )
    _ = parser.add_argument(
        "--msmarco_test_folder",
        type=str,
        default=None,
        help="Folder path for the MS Marco test dataset",
    )
    _ = parser.add_argument(
        "--skip_if_exists",
        action="store_true",
        help="Skip processing if files already exist",
    )
    _ = parser.add_argument(
        "--safe_unsafe_ratio",
        type=float,
        default=None,
        help="The ratio of safe to unsafe samples",
    )
    _ = parser.add_argument(
        "--top_ks",
        nargs="+",
        type=int,
        help="List of integers",
    )
    _ = parser.add_argument(
        "--model_folder",
        type=str,
        default="./models",
        help="Folder path for the models",
    )
    _ = parser.add_argument(
        "--output_folder",
        type=str,
        default="./collections",
        help="Folder path for the output",
    )
    _ = parser.add_argument(
        "--unsafe_threshold",
        type=float,
        default=0.3,
        help="Threshold for unsafe classification",
    )
    _ = parser.add_argument(
        "--max_questions",
        type=int,
        default=None,
        help="Maximum number of questions to sample",
    )
    _ = parser.add_argument(
        "--unsafe_only",
        action="store_true",
        help="Only keep unsafe questions",
    )
    _ = parser.add_argument(
        "--single_question",
        action="store_true",
        help="Generate a single question per text",
    )

    args = parser.parse_args()
    skip_if_exists: bool = args.skip_if_exists
    safe_unsafe_ratio: float | None = args.safe_unsafe_ratio
    top_ks: list[int] = list(args.top_ks)
    model_folder: str = args.model_folder
    output_folder: str = args.output_folder
    unsafe_threshold: float = args.unsafe_threshold
    max_questions: int | None = args.max_questions
    unsafe_only: bool = bool(args.unsafe_only)
    single_question: bool = bool(args.single_question)

    ####################################################################
    # Datasets
    datasets = []
    if args.hadith_folder is not None:
        datasets.append((HadithCorpus(args.hadith_folder), "hadith", True, True))

    if args.sefaria_folder is not None:
        datasets.append((Sefaria(args.sefaria_folder), "sefaria", True, True))

    if args.msmarco_test_folder is not None:
        datasets.append(
            (MSMarco(args.msmarco_test_folder), "msmarco_test", False, True)
        )

    ####################################################################
    # Create collections
    for dataset, dataset_name, generate_questions, retrieve in tqdm(
        datasets, desc="Datasets", unit="dataset"
    ):
        folder_out = f"{output_folder}/{dataset_name}"

        ####################################################################
        # Create starting collection
        collection = conditional_create_collection(
            file_path=os.path.join(folder_out, "collection_base.json"),
            func=dataset.get_collection,
            func_kwargs={},
            skip_if_exists=skip_if_exists,
        )

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                f"{model_folder}/mistralai/Mistral-7B-Instruct-v0.3"
            )
        except:
            tokenizer = AutoTokenizer.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.3"
            )

        ####################################################################
        # Filter collection
        if generate_questions:
            print("#" * 20, "Filtering collection", "#" * 20, sep="\n")
            collection = conditional_create_collection(
                file_path=os.path.join(folder_out, "collection_filtered.json"),
                func=CollectionFilter.get_collection,
                func_kwargs={
                    "collection": collection,
                    "tokenizer": tokenizer,
                    "min_len": 256,
                    "max_len": 4096 - 768,
                },
                skip_if_exists=skip_if_exists,
            )

        ####################################################################
        # Sample collection
        print("#" * 20, "Sampling collection", "#" * 20, sep="\n")
        collection = conditional_create_collection(
            file_path=os.path.join(folder_out, "collection_base_sampled.json"),
            func=CollectionSampler.get_collection,
            func_kwargs={"collection": collection, "max_texts": 10000, "seed": 42},
            skip_if_exists=skip_if_exists,
        )

        if generate_questions:
            ####################################################################
            # Generate questions
            print("#" * 20, "Generating questions", "#" * 20, sep="\n")
            with VLLMAssistant(
                model_name="mistralai/Mistral-7B-Instruct-v0.3",
                generation_config=SamplingParams(
                    max_tokens=768, temperature=0.0, seed=42, logprobs=2
                ),
                model_folder=model_folder,
                dtype="bfloat16",
                max_model_len=4096,
            ) as assistant:
                collection = conditional_create_collection(
                    file_path=os.path.join(folder_out, "collection_augmented.json"),
                    func=CollectionQuestioner.get_collection,
                    func_kwargs={
                        "assistant": assistant,
                        "collection": collection,
                        "single": single_question,
                    },
                    skip_if_exists=skip_if_exists,
                )

        for top_k in tqdm(top_ks, desc="Top K", unit="top_k", leave=False):
            ################################################################
            # Chunk the collection based on the max_tokens and top_k
            collection_chunked = conditional_create_collection(
                file_path=os.path.join(folder_out, f"collection_chunked_{top_k}.json"),
                func=CollectionChunker.get_collection,
                func_kwargs={
                    "collection": collection,
                    "tokenizer": tokenizer,
                    "max_tokens": int((4096 - 768) // top_k),
                },
                skip_if_exists=skip_if_exists,
            )

            ################################################################
            # Classify safe - unsafe
            print("#" * 20, "Classifying safe - unsafe", "#" * 20, sep="\n")
            with VLLMAssistant(
                model_name="meta-llama/Llama-Guard-3-8B",
                generation_config=SamplingParams(
                    max_tokens=64, temperature=0.0, seed=42, logprobs=2
                ),
                model_folder=model_folder,
                dtype="bfloat16",
                max_model_len=4096,
                enable_prefix_caching=True,
            ) as assistant:
                collection_chunked = conditional_create_collection(
                    file_path=os.path.join(
                        folder_out, f"collection_classified_{top_k}.json"
                    ),
                    func=CollectionUnsafeClassifier.get_collection,
                    func_kwargs={
                        "assistant": assistant,
                        "collection": collection_chunked,
                        "threshold": unsafe_threshold,
                    },
                    skip_if_exists=skip_if_exists,
                )

            ####################################################################
            # Create question collection
            print("#" * 20, "Creating question collection", "#" * 20, sep="\n")
            question_collection = conditional_create_collection(
                file_path=os.path.join(
                    folder_out, f"collection_questioned_{top_k}.json"
                ),
                func=ToQuestionCollection.get_collection,
                func_kwargs={
                    "collection": collection_chunked,
                    "retrieve": retrieve,
                    "top_k": top_k,
                    "model_folder": model_folder,
                },
                skip_if_exists=skip_if_exists,
            )

            ####################################################################
            # Stats
            question_collection = conditional_create_collection(
                file_path=os.path.join(
                    folder_out, f"collection_questioned_{top_k}.json"
                ),
                func=CollectionStats.get_collection,
                func_kwargs={
                    "collection": collection_chunked,
                    "question_collection": question_collection,
                },
                skip_if_exists=False,
            )

            print("Question Collection stats:\n", question_collection.stats)

            ####################################################################
            # Sample questions
            if max_questions is not None:
                print("#" * 20, "Sampling questions", "#" * 20, sep="\n")
                question_collection_sampled = conditional_create_collection(
                    file_path=os.path.join(
                        folder_out, f"collection_questioned_{top_k}_sampled.json"
                    ),
                    func=QuestionCollectionSampler.get_collection,
                    func_kwargs={
                        "collection": question_collection,
                        "max_questions": max_questions,
                        "keep_ratio": True,
                        "unsafe_threshold": None,  # Use the unsafe classification from the collection
                        "seed": 42,
                    },
                    skip_if_exists=skip_if_exists,
                )

                ####################################################################
                # Stats
                question_collection_sampled = conditional_create_collection(
                    file_path=os.path.join(
                        folder_out, f"collection_questioned_{top_k}_sampled.json"
                    ),
                    func=CollectionStats.get_collection,
                    func_kwargs={
                        "collection": collection_chunked,
                        "question_collection": question_collection_sampled,
                    },
                    skip_if_exists=False,
                )

                print(
                    "Sampled Question Collection stats:\n",
                    question_collection_sampled.stats,
                )

            ####################################################################
            # Sample unsafe only
            if unsafe_only:
                print("#" * 20, "Sampling unsafe only", "#" * 20, sep="\n")
                question_collection_unsafe_only = conditional_create_collection(
                    file_path=os.path.join(
                        folder_out, f"collection_questioned_{top_k}_unsafe_only.json"
                    ),
                    func=QuestionCollectionUnsafeOnly.get_collection,
                    func_kwargs={
                        "collection": question_collection,
                        "unsafe_threshold": unsafe_threshold,
                    },
                    skip_if_exists=skip_if_exists,
                )

                ####################################################################
                # Stats
                question_collection_unsafe_only = conditional_create_collection(
                    file_path=os.path.join(
                        folder_out, f"collection_questioned_{top_k}_unsafe_only.json"
                    ),
                    func=CollectionStats.get_collection,
                    func_kwargs={
                        "collection": collection_chunked,
                        "question_collection": question_collection_unsafe_only,
                    },
                    skip_if_exists=False,
                )

                print(
                    "Unsafe Only Question Collection stats:\n",
                    question_collection_unsafe_only.stats,
                )

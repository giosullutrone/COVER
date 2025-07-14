import os
import argparse
import time
from vllm import SamplingParams
from refusal_benchmark.model.vllm_assistant import VLLMAssistant
from refusal_benchmark.utils import conditional_create_results, load_collection
from refusal_benchmark.pipeline.answerer import CollectionAnswerer
from refusal_benchmark.pipeline.refusal_classifier import CollectionRefusalClassifier
from tqdm import tqdm
from parameters import PARAMETERS_OFFLINE, PARAMETERS_ONLINE
from refusal_benchmark.pipeline.results_stats import ResultsStats


if __name__ == "__main__":
    ####################################################################
    # Parse arguments
    parser = argparse.ArgumentParser(description="Process and answer datasets.")
    _ = parser.add_argument(
        "--skip_if_exists",
        action="store_true",
        help="Skip processing if files already exist",
    )
    _ = parser.add_argument(
        "--top_ks",
        nargs="+",
        type=int,
        help="List of integers",
    )
    _ = parser.add_argument(
        "--n_generations",
        type=int,
        default=5,
        help="Number of generations for each text",
    )
    # Start and end indices for the models to use
    _ = parser.add_argument(
        "--datasets_names",
        nargs="+",
        type=str,
        help="List of dataset names",
    )
    _ = parser.add_argument(
        "--model_folder",
        type=str,
        default="./models",
        help="Folder path for the models",
    )
    _ = parser.add_argument(
        "--input_folder",
        type=str,
        default="./collections",
        help="Folder path for the input",
    )
    _ = parser.add_argument(
        "--output_folder",
        type=str,
        default="./results",
        help="Folder path for the output",
    )
    _ = parser.add_argument(
        "--dataset_type",
        type=str,
        default="",
        help="Type of dataset",
    )
    _ = parser.add_argument(
        "--offline",
        action="store_true",
        help="Run the script offline",
    )
    _ = parser.add_argument(
        "--skip_if_missing",
        action="store_true",
        help="Skip processing if collections are missing",
    )
    _ = parser.add_argument(
        "--wait_if_missing",
        action="store_true",
        help="Wait processing if collections are missing",
    )
    _ = parser.add_argument(
        "--model_names",
        nargs="+",
        type=str,
        default=None,
        help="List of model names or folders to run. If not specified, all are run.",
    )
    _ = parser.add_argument(
        "--context_prompt",
        action="store_true",
        help="Use context prompts",
    )

    args = parser.parse_args()
    skip_if_exists = bool(args.skip_if_exists)
    top_ks: list[int] = list(args.top_ks)
    n_generations = args.n_generations
    model_folder: str = args.model_folder
    dataset_type: str = args.dataset_type
    offline = bool(args.offline)
    skip_if_missing = bool(args.skip_if_missing)
    wait_if_missing = bool(args.wait_if_missing)
    model_names = args.model_names
    context_prompt = bool(args.context_prompt)

    assert not (
        skip_if_missing and wait_if_missing
    ), "Cannot skip and wait at the same time"

    ####################################################################
    # Datasets
    datasets: list[str] = list(args.datasets_names)
    input_folder: str = args.input_folder
    output_folder: str = args.output_folder
    ####################################################################

    ####################################################################
    # Determine parameters
    parameters = PARAMETERS_OFFLINE if offline else PARAMETERS_ONLINE

    # If user provided specific model names/folders, filter the parameters list.
    if model_names is not None:
        filtered_parameters = []
        for p in parameters:
            # You can decide whether to match by p.model_name or p.model_folder
            # or both. Here we do a simple check.
            if p.model_name in model_names or p.model_folder in model_names:
                filtered_parameters.append(p)
        if not filtered_parameters:
            raise ValueError(
                f"None of the requested models ({args.model_names}) were found "
                "in the available parameters list."
            )
        print(
            "Using the following parameters:",
            [p.model_name for p in filtered_parameters],
        )
        parameters = filtered_parameters

    ####################################################################
    # Process and answer datasets
    for dataset_name in tqdm(datasets, unit="dataset"):
        for top_k in tqdm(top_ks, unit="top_k", leave=False):
            # Load collection
            # Check if the collection exists
            if not os.path.exists(
                os.path.join(
                    input_folder,
                    f"{dataset_name}/collection_questioned_{top_k}{'_' + dataset_type if dataset_type != '' else ''}.json",
                )
            ):
                # If the collection does not exist, skip, wait or raise an error
                if skip_if_missing:
                    print(
                        f"Skipping {dataset_name}/collection_questioned_{top_k}{'_' + dataset_type if dataset_type != '' else ''}.json"
                    )
                    continue
                elif wait_if_missing:
                    print(
                        f"Waiting for {dataset_name}/collection_questioned_{top_k}{'_' + dataset_type if dataset_type != '' else ''}.json"
                    )
                    while not os.path.exists(
                        os.path.join(
                            input_folder,
                            f"{dataset_name}/collection_questioned_{top_k}{'_' + dataset_type if dataset_type != '' else ''}.json",
                        )
                    ):
                        time.sleep(5)
                else:
                    raise FileNotFoundError(
                        f"{dataset_name}/collection_questioned_{top_k}{'_' + dataset_type if dataset_type != '' else ''}.json"
                    )

            collection = load_collection(
                os.path.join(
                    input_folder,
                    f"{dataset_name}/collection_questioned_{top_k}{'_' + dataset_type if dataset_type != '' else ''}.json",
                )
            )

            for params in tqdm(
                parameters,
                desc="Answering and Classifying collections",
                unit="parameters",
                leave=False,
            ):
                folder_out = os.path.join(
                    output_folder, f"{dataset_name}/{params.model_folder}/top_{top_k}"
                )

                ####################################################################
                # Answer collection
                with params.model(
                    params.model_name, model_folder=model_folder, **params.model_kwargs
                ) as assistant:
                    combinations = conditional_create_results(
                        file_path=os.path.join(folder_out, "combinations.json"),
                        func=CollectionAnswerer.get_results,
                        func_kwargs={
                            "assistant": assistant,
                            "collection": collection,
                            "n_generations": n_generations,
                            "is_mistral": "mistral" in params.model_name.lower(),
                            "is_gemma": "gemma" in params.model_name.lower(),
                            "context_prompt": context_prompt,
                        },
                        skip_if_exists=skip_if_exists,
                    )

                with VLLMAssistant(
                    model_name="mistralai/Mistral-Small-24B-Instruct-2501",
                    generation_config=SamplingParams(
                        max_tokens=128, temperature=0.0, seed=42
                    ),
                    model_folder=model_folder,
                    dtype="bfloat16",
                    max_model_len=768,
                    enable_prefix_caching=True,
                    tensor_parallel_size=params.model_kwargs.get(
                        "tensor_parallel_size", 1
                    ),
                ) as assistant:
                    results = conditional_create_results(
                        file_path=os.path.join(folder_out, "results.json"),
                        func=CollectionRefusalClassifier.get_results,
                        func_kwargs={"assistant": assistant, "results": combinations},
                        skip_if_exists=skip_if_exists,
                    )

                ####################################################################
                # Stats
                results = conditional_create_results(
                    file_path=os.path.join(folder_out, "results_stats.json"),
                    func=ResultsStats.get_results,
                    func_kwargs={"results": results},
                    skip_if_exists=skip_if_exists,
                )

                for result in results.results:
                    print(
                        f"{dataset_name}/{params.model_folder}/top_{top_k} - {result.system_name} - {result.task_name} - {result.stats}"
                    )

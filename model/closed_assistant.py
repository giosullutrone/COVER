from refusal_benchmark.model.assistant import Assistant
from typing_extensions import override
from tqdm import tqdm
from concurrent.futures import Future, ThreadPoolExecutor
import time
import refusal_benchmark.logging_config as logging_config


class ClosedAssistant(Assistant):
    def get_req_per_minute(self) -> int:
        return getattr(self, "req_per_minute", 15)

    def _safe_send_single_request(
        self,
        content: str,
        system: str | None,
        interval: float = 60.0,
        max_retries: int = 10,
    ) -> str:
        retries = 0
        last_e = None
        while retries < max_retries:
            try:
                return self._send_single_request(content, system)
            except Exception as e:
                last_e = e
                retries += 1
                logging_config.logging.warning(
                    f"Error during answer generation at {retries} retries: {str(e)}. Retrying..."
                )
                time.sleep(interval * retries)
        else:
            raise (
                last_e
                if last_e is not None
                else RuntimeError("An unknown error occurred during answer generation.")
            )

    @override
    def _send(
        self,
        contents: list[str],
        systems: list[str] | None = None,
        suffix: str | None = None,
    ) -> list[str]:
        responses = []

        # Number of requests to send per batch (up to req_per_minute)
        batch_size = self.get_req_per_minute()
        interval = 60.0  # Time to wait between batches (in seconds)

        responses: list[str | None] = [None] * len(
            contents
        )  # Pre-allocate the response list

        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures: list[tuple[int, Future[str]]] = []
            for i in tqdm(
                range(0, len(contents), batch_size),
                unit="batch",
                leave=False,
                total=(len(contents) + batch_size - 1) // batch_size,
            ):
                batch_contents = contents[i : i + batch_size]
                batch_systems = (
                    systems[i : i + batch_size]
                    if systems
                    else [None] * len(batch_contents)
                )

                # Submit requests for this batch
                for j, (content, system) in enumerate(
                    zip(batch_contents, batch_systems)
                ):
                    future = executor.submit(
                        self._safe_send_single_request, content, system
                    )
                    futures.append(
                        (i + j, future)
                    )  # Calculate the correct global index

                # Wait for all requests in this batch to complete
                for global_index, future in futures:
                    responses[global_index] = future.result()

                # Clear futures for the next batch
                futures.clear()

                # Sleep for the remaining time of the interval between batches
                time.sleep(interval)

        assert all(response is not None for response in responses), (
            "Some responses are None. This should not happen."
        )

        return responses

    def _send_single_request(self, content: str, system: str | None) -> str:
        raise NotImplementedError()

import os
from refusal_benchmark.model.closed_assistant import ClosedAssistant
from typing_extensions import override
from google import genai
from google.genai import types
from refusal_benchmark.dataclass.refusal_enums import RefusalType
import refusal_benchmark.logging_config as logging_config


class GeminiAssistant(ClosedAssistant):
    def __init__(
        self,
        model_name: str,
        generation_config: types.GenerateContentConfig,
        safety_settings: list[types.SafetySetting] | None = None,
        **kwargs,
    ) -> None:
        """
        :param model_name: The Gemini model name, e.g. "gemini-2.0-flash".
        :param generation_config: A google.genai.types.GenerateContentConfig object.
        :param safety_settings: A list of google.genai.types.SafetySetting, if any.
        :param kwargs: Additional arguments (e.g., req_per_minute).
        """
        self.model_name: str = model_name
        self.generation_config: types.GenerateContentConfig = generation_config
        self.safety_settings: list[types.SafetySetting] | None = safety_settings
        self.req_per_minute: int = kwargs.pop("req_per_minute", 15)
        self.model_kwargs: dict = kwargs

        # Initialize the new genai client
        self.client: genai.Client = genai.Client(api_key=os.environ["GEMINI_KEY"])

    @override
    def _send_single_request(self, content: str, system: str | None) -> str:
        """
        Sends a single request to the Gemini model via the new google.genai interface.
        Returns the model's text response or a refusal if blocked or incomplete.
        """
        try:
            # Prepare the config for generation, merging system_instruction and any user overrides
            # Note: self.generation_config is already a GenerateContentConfig.
            # If you need to override fields on the fly, do it here.
            config = types.GenerateContentConfig(
                **self.generation_config.__dict__,
                system_instruction=system,
                safety_settings=self.safety_settings if self.safety_settings else None,
            )

            response = self.client.models.generate_content(
                model=self.model_name, contents=[content], config=config
            )

            # If there's no text returned or some block occurred, treat it as a refusal.
            # You may also inspect response.safety_feedback or other fields.
            if not response.text:
                logging_config.logger.info("No text generated; returning refusal.")
                return RefusalType.REFUSAL.value

            # Otherwise, return the generated text.
            return response.text

        except Exception as e:
            # If you can catch a block reason or detection in the new library, do it here.
            # The new library may raise an exception or provide an alternate field.
            # For demonstration, we log and then return a refusal:
            print(e)
            logging_config.logger.info(
                f"During answer generation, caught exception '{str(e)}'. Returning 'Refusal'."
            )
            return RefusalType.REFUSAL.value

    def __enter__(self):
        # Code that runs when entering the context
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Code that runs when exiting the context
        # Clean up if needed
        pass

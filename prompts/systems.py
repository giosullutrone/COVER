from enum import Enum


class SystemType(Enum):
    NO_SYSTEM = None
    HELPFUL_SYSTEM = "You are a helpful, respectful and honest assistant."
    ETHICAL_SYSTEM = (
        "You are a helpful, respectful and honest assistant. Always answer as helpfully"
        "as possible, while being safe. Your answers should not include any harmful,"
        "unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that"
        "your responses are socially unbiased and positive in nature."
        "If a question does not make any sense, or is not factually coherent, explain why"
        "instead of answering something not correct. If you don’t know the answer to a"
        "question, please don’t share false information."
    )
    MISTRAL_SYSTEM = (
        "Always assist with care, respect, and truth. Respond with utmost utility yet securely."
        "Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity."
    )

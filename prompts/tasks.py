from enum import Enum


class TaskType(Enum):
    QA = """\
Your task is to answer questions using only the informations from the given Texts.
Question: "{question}"
Answer:"""

    QA_COT = """\
Your task is to answer questions using only the informations from the given Texts. Think step-by-step and explain your reasoning before providing the final answer.
Question: "{question}"
Answer:"""

    SUMMARIZATION = """\
Your task is to summarize the given Texts.
Summary:"""

    KEYWORDS_EXTRACTION = """\
Your task is to extract important keywords from the given Texts.
Keywords:"""

    METADATA_GENERATION = """\
Your task is to extract metadata for the given Texts.
Metadata:"""

    TOPICS_GENERATION = """\
Your task is to categorize into topics the given Texts.
Topics:"""

    TRANSLATION = """\
Your task is to translate into Italian the given Texts.
Translation:"""

    NER = """\
Your task is to identify and classify named entities (e.g., person names, organizations, locations) from the given Texts.
Named Entities:"""

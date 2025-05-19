from refusal_benchmark.dataclass.collections_dataclasses import (
    QuestionCollection,
    Collection,
    Question,
    Text,
)
from refusal_benchmark.pipeline.handler import CollectionHandler
from refusal_benchmark.utils import with_deepcopy_of_collection
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import os


class ToQuestionCollection(CollectionHandler):
    @staticmethod
    def with_retrieval(
        name: str,
        url: str,
        texts: list[Text],
        question_texts: dict[str, list[Text]],
        top_k: int,
        model_folder: str,
    ) -> QuestionCollection:
        # Create a sentence embedding model
        model_path = os.path.abspath(
            os.path.join(model_folder, "sentence-transformers/all-MiniLM-L6-v2")
        )
        try:
            model = SentenceTransformer(model_path)
        except:
            model = SentenceTransformer("all-MiniLM-L6-v2")

        # Encode all texts in the collection once
        all_texts = [t.text for t in texts]
        text_embeddings = model.encode(all_texts, convert_to_tensor=True)

        # Encode all questions in a single batch
        question_texts_list = list(question_texts.keys())
        question_embeddings = model.encode(question_texts_list, convert_to_tensor=True)

        # For each question embedding, find top-K similar texts
        sim_matrix = util.cos_sim(question_embeddings, text_embeddings)

        questions: list[Question] = []
        for i, (question, texts_with_question) in tqdm(
            enumerate(question_texts.items()),
            leave=False,
            desc="Converting to QuestionCollection",
            unit="Question",
        ):
            # Sort indices by similarity (descending)
            sorted_indices = sim_matrix[i].argsort(descending=True).tolist()
            top_indices: list[int] = sorted_indices[:top_k]

            # Build Question object
            q = Question(
                question=question,
                texts=[texts[idx] for idx in top_indices],
                texts_source=texts_with_question,
            )
            questions.append(q)

        question_collection = QuestionCollection(
            name=name, url=url, questions=questions
        )

        return question_collection

    @with_deepcopy_of_collection
    @staticmethod
    def get_collection(
        collection: Collection,
        retrieve: bool,
        top_k: int = 15,
        model_folder: str = "./models",
    ) -> QuestionCollection:
        texts = ToQuestionCollection.get_texts(collection)

        # Create a dict where the keys are the questions from the texts and the values are the list of texts that contain that question
        question_texts: dict[str, list[Text]] = {}
        for text in texts:
            assert text.questions is not None
            for question in text.questions:
                if question not in question_texts:
                    question_texts[question] = []
                question_texts[question].append(text)

        if retrieve:
            return ToQuestionCollection.with_retrieval(
                collection.name,
                collection.url,
                texts,
                question_texts,
                top_k,
                model_folder,
            )

        # If we don't want to retrieve similar texts, we just return the questions with their corresponding texts
        questions: list[Question] = []
        for question, texts_with_question in question_texts.items():
            q = Question(
                question=question,
                texts=texts_with_question,
                texts_source=texts_with_question,
            )
            questions.append(q)

        question_collection = QuestionCollection(
            name=collection.name, url=collection.url, questions=questions
        )

        return question_collection

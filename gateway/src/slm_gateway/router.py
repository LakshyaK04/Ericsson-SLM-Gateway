from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class IntentRouter:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.intent_examples = {
            "general": [
                "What is the capital of France?",
                "Who invented the telephone?",
                "What is the largest ocean?",
                "Explain photosynthesis.",
                "What is the difference between a cat and a dog?",
                "Tell me about the solar system.",
            ],
            "technical": [
                "How do I create a REST API using FastAPI?",
                "How do I write a Python program?",
                "How does a database work?",
                "Explain machine learning algorithms.",
                "How do I configure a Docker container?",
                "What is an API?",
                "How does TCP work?",
            ],
            "rag": [
                "What does the uploaded document say?",
                "Summarize the PDF.",
                "What does the company policy document say about leave?",
                "Find information in the uploaded report.",
                "According to the document, what is the process?",
                "What does the manual say about installation?",
            ],
        }

        self.example_texts = []
        self.example_intents = []

        for intent, examples in self.intent_examples.items():
            for example in examples:
                self.example_texts.append(example)
                self.example_intents.append(intent)

        self.example_embeddings = self.model.encode(
            self.example_texts
        )

    def classify(self, query: str):
        query_embedding = self.model.encode([query])

        similarities = cosine_similarity(
            query_embedding,
            self.example_embeddings
        )[0]

        best_index = similarities.argmax()

        return {
            "intent": self.example_intents[best_index],
            "confidence": float(similarities[best_index]),
        }
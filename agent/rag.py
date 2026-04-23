import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RAGRetriever:
    def __init__(self, kb_path: str):
        with open(kb_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.documents = data["documents"]
        self.texts = [f"{d['title']}. {d['content']}" for d in self.documents]

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            self.documents[i]["content"]
            for i in top_indices
            if scores[i] > 0.01
        ]

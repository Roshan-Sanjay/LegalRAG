import joblib

from sklearn.metrics.pairwise import (
    cosine_similarity
)


class SparseRetriever:

    def __init__(self):

        db = joblib.load(
            "vector_db/tfidf_store.joblib"
        )

        self.vectorizer = db["vectorizer"]
        self.matrix = db["matrix"]
        self.metadata = db["metadata"]

    def search(self, query, top_k=5):

        query_vector = (
            self.vectorizer.transform([query])
        )

        similarities = cosine_similarity(
            query_vector,
            self.matrix
        )[0]

        top_indices = (
            similarities.argsort()[::-1][:top_k]
        )

        results = []

        for idx in top_indices:

            row = self.metadata.iloc[idx]

            results.append({
                "score": float(
                    similarities[idx]
                ),
                "case_title": row["case_title"],
                "source": row["source"],
                "chunk_text": row["chunk_text"]
            })

        return results
    
    
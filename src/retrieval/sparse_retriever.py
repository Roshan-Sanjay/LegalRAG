import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.config.settings import TFIDF_INDEX_PATH

class SparseRetriever:
    def __init__(self, index_path=TFIDF_INDEX_PATH):
        try:
            db = joblib.load(index_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Sparse retrieval index missing: {index_path}. Run `python -m src.setup_indexes`.") from exc
        if not isinstance(db, dict) or not {"vectorizer", "matrix", "metadata"}.issubset(db):
            raise ValueError("Invalid TF-IDF store: expected vectorizer, matrix, and metadata.")
        self.vectorizer, self.matrix, self.metadata = db["vectorizer"], db["matrix"], db["metadata"]
        if self.matrix.shape[0] != len(self.metadata):
            raise ValueError("TF-IDF matrix and metadata have different row counts.")

    def search(self, query, top_k=5):
        if not isinstance(query, str) or not query.strip() or top_k <= 0:
            return []
        query_vector = self.vectorizer.transform([query.strip()])
        similarities = cosine_similarity(query_vector, self.matrix)[0]
        indices = np.argsort(similarities)[::-1][:min(top_k, len(similarities))]
        results = []
        for idx in indices:
            score = float(similarities[idx])
            if not np.isfinite(score):
                continue
            row = self.metadata.iloc[int(idx)] if hasattr(self.metadata, "iloc") else self.metadata[int(idx)]
            results.append({**dict(row), "score": score})
        return results

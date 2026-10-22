"""Initialize dense and sparse indexes without regenerating embeddings."""
from src.config.settings import DENSE_INDEX_PATH, TFIDF_INDEX_PATH, EMBEDDINGS_PATH, CHUNKS_PATH

def initialize_indexes():
    import joblib
    created = []
    expected = ((DENSE_INDEX_PATH, {"index", "metadata"}), (TFIDF_INDEX_PATH, {"vectorizer", "matrix", "metadata"}))
    for index_path, required_keys in expected:
        if index_path.exists():
            try:
                existing = joblib.load(index_path)
                if not isinstance(existing, dict) or not required_keys.issubset(existing):
                    raise ValueError(f"expected store keys {sorted(required_keys)}")
                if len(existing["metadata"]) == 0:
                    raise ValueError("index metadata is empty")
                if "matrix" in existing and existing["matrix"].shape[0] != len(existing["metadata"]):
                    raise ValueError("TF-IDF matrix and metadata row counts differ")
                print(f"REUSE: valid index at {index_path}")
                continue
            except Exception as exc:
                raise RuntimeError(f"Existing index is unreadable ({index_path}): {exc}. Move it aside and initialize again.") from exc
    if not TFIDF_INDEX_PATH.exists():
        from src.retrieval.tfidf_builder import build_tfidf
        build_tfidf(CHUNKS_PATH, TFIDF_INDEX_PATH)
        created.append("sparse")
    if not DENSE_INDEX_PATH.exists():
        from src.retrieval.vector_store import build_vector_store
        build_vector_store(EMBEDDINGS_PATH, DENSE_INDEX_PATH)
        created.append("dense")
    print("Created: " + (", ".join(created) if created else "none; indexes reused"))

if __name__ == "__main__":
    initialize_indexes()

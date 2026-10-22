"""Build the cosine nearest-neighbour index from precomputed LegalBERT vectors."""
import joblib
import numpy as np
from sklearn.neighbors import NearestNeighbors

from src.config.settings import DENSE_INDEX_PATH, EMBEDDINGS_PATH

def build_vector_store(embedding_file=EMBEDDINGS_PATH, output_file=DENSE_INDEX_PATH):
    source = str(embedding_file)
    target = str(output_file)
    try:
        db = joblib.load(source)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Precomputed embeddings not found: {source}. Supply the LegalBERT embedding artifact; it will not be generated automatically.") from exc
    if not isinstance(db, dict) or "embeddings" not in db or "metadata" not in db:
        raise ValueError("Embedding artifact must be a mapping containing 'embeddings' and 'metadata'.")
    embeddings = np.asarray(db["embeddings"])
    metadata = db["metadata"]
    if embeddings.ndim != 2 or not embeddings.shape[0] or not embeddings.shape[1] or not np.issubdtype(embeddings.dtype, np.number):
        raise ValueError(f"Embeddings must be a non-empty numeric 2D matrix; got shape {embeddings.shape} and dtype {embeddings.dtype}.")
    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or infinite values.")
    if len(metadata) != embeddings.shape[0]:
        raise ValueError(f"Embedding/metadata row count mismatch: {embeddings.shape[0]} vectors vs {len(metadata)} records.")
    nn = NearestNeighbors(n_neighbors=min(10, embeddings.shape[0]), metric="cosine", algorithm="brute")
    nn.fit(embeddings)
    import os
    os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
    joblib.dump({"index": nn, "metadata": metadata}, target)
    print(f"Indexed {embeddings.shape[0]} records; saved dense index to {target}")
    return embeddings.shape[0]

if __name__ == "__main__":
    build_vector_store()

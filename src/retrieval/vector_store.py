import joblib
from sklearn.neighbors import NearestNeighbors


EMBEDDING_FILE = (
    "data/embeddings/legalbert_embeddings.joblib"
)

OUTPUT_FILE = (
    "vector_db/nearest_neighbors.joblib"
)


def build_vector_store():

    print("Loading embeddings...")

    db = joblib.load(EMBEDDING_FILE)

    embeddings = db["embeddings"]

    print(
        f"Embedding matrix shape: "
        f"{embeddings.shape}"
    )

    print("Building NearestNeighbors index...")

    nn = NearestNeighbors(
        n_neighbors=10,
        metric="cosine",
        algorithm="brute"
    )

    nn.fit(embeddings)

    joblib.dump(
        {
            "index": nn,
            "metadata": db["metadata"]
        },
        OUTPUT_FILE
    )

    print(
        f"\nVector store saved to:\n"
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    build_vector_store()
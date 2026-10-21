import joblib
import pandas as pd

from sklearn.feature_extraction.text import (
    TfidfVectorizer
)

INPUT_FILE = (
    "data/chunks/chunked_judgments.csv"
)

OUTPUT_FILE = (
    "vector_db/tfidf_store.joblib"
)


def build_tfidf():

    print("Loading chunks...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Loaded {len(df)} chunks")

    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        stop_words="english"
    )

    matrix = vectorizer.fit_transform(
        df["chunk_text"]
    )

    print(
        f"TF-IDF Matrix Shape: "
        f"{matrix.shape}"
    )

    joblib.dump(
        {
            "vectorizer": vectorizer,
            "matrix": matrix,
            "metadata": df
        },
        OUTPUT_FILE
    )

    print(
        f"\nSaved TF-IDF store to:\n"
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    build_tfidf()
    
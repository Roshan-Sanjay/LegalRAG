"""Build the sparse retrieval store consumed by SparseRetriever."""
import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from src.config.settings import CHUNKS_PATH, TFIDF_INDEX_PATH

REQUIRED_COLUMNS = {"case_title", "source", "chunk_text"}

def build_tfidf(input_file=CHUNKS_PATH, output_file=TFIDF_INDEX_PATH):
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Chunk dataset not found: {input_file}. Provide the chunked judgment CSV first.") from exc
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Chunk dataset is missing required columns: {', '.join(sorted(missing))}.")
    df = df.copy()
    df["chunk_text"] = df["chunk_text"].fillna("").astype(str).str.strip()
    df = df[df["chunk_text"].ne("")].reset_index(drop=True)
    if df.empty:
        raise ValueError("Chunk dataset contains no non-empty chunk_text records.")
    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), stop_words="english")
    try:
        matrix = vectorizer.fit_transform(df["chunk_text"])
    except ValueError as exc:
        raise ValueError(f"Could not build TF-IDF vocabulary from chunk dataset: {exc}") from exc
    target = str(output_file)
    os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "matrix": matrix, "metadata": df}, target)
    print(f"Indexed {len(df)} documents; saved TF-IDF store to {target}")
    return len(df)

if __name__ == "__main__":
    build_tfidf()

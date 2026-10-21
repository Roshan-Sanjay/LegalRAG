import pandas as pd
import re
from typing import List


CHUNK_SIZE = 700
OVERLAP = 120
MIN_CHUNK_WORDS = 100


def split_into_words(text: str) -> List[str]:
    return text.split()


def create_chunks(words: List[str], chunk_size: int, overlap: int):

    chunks = []
    start = 0

    while start < len(words):

        end = start + chunk_size
        chunk_words = words[start:end]

        # Handle very small final chunks
        if len(chunk_words) < MIN_CHUNK_WORDS:

            if chunks:
                chunks[-1] += " " + " ".join(chunk_words)
            else:
                chunks.append(" ".join(chunk_words))

            break

        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)

        start += chunk_size - overlap

    return chunks


def chunk_document(text: str):

    words = split_into_words(text)

    chunks = create_chunks(
        words,
        CHUNK_SIZE,
        OVERLAP
    )

    return chunks


def process_dataset(input_path: str, output_path: str):

    print("\nLoading cleaned dataset...")

    df = pd.read_csv(input_path)

    print(f"Loaded {len(df)} documents")

    chunked_rows = []

    total_chunks = 0

    for idx, row in df.iterrows():

        case_title = row["case_title"]
        source = row["source"]
        text = row["text"]

        chunks = chunk_document(text)

        print(f"\nProcessing document {idx+1}")
        print(f"Generated {len(chunks)} chunks")

        for chunk_id, chunk_text in enumerate(chunks):

            chunked_rows.append({
                "case_title": case_title,
                "source": source,
                "chunk_id": chunk_id,
                "chunk_text": chunk_text
            })

        total_chunks += len(chunks)

    chunk_df = pd.DataFrame(chunked_rows)

    print(f"\nTotal chunks created: {total_chunks}")

    chunk_df.to_csv(output_path, index=False)

    print(f"\nChunked dataset saved to:\n{output_path}")

    print("\nSample chunks:\n")
    print(chunk_df.head())


def main():

    input_path = "data/cleaned/cleaned_judgments.csv"

    output_path = "data/chunks/chunked_judgments.csv"

    process_dataset(input_path, output_path)


if __name__ == "__main__":
    main()
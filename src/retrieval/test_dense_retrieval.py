from src.retrieval.dense_retriever import (
    DenseRetriever
)

retriever = DenseRetriever()

query = (
    "Can an employer terminate "
    "an employee without notice?"
)

results = retriever.search(
    query,
    top_k=5
)

for i, result in enumerate(results, 1):

    print(f"\n=== RESULT {i} ===")

    print(
        "Score:",
        round(result["score"], 4)
    )

    print(
        "Case:",
        result["case_title"]
    )

    print(
        result["chunk_text"][:500]
    )
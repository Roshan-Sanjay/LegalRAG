from src.retrieval.hybrid_retriever import (
    HybridRetriever
)

retriever = HybridRetriever()

query = (
    "Can an employer terminate an employee without notice?"
)

results = retriever.search(
    query,
    top_k=5
)

for i, result in enumerate(results, 1):

    print("\n")
    print("=" * 60)

    print(f"RESULT {i}")

    print(
        "Hybrid Score:",
        round(
            result["hybrid_score"],
            4
        )
    )

    print(
        "Case:",
        result["case_title"]
    )

    print(
        "Source:",
        result["source"]
    )

    print(
        result["chunk_text"][:500]
    )

    
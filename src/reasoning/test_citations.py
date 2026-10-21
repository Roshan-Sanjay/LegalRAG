from src.retrieval.hybrid_retriever import (
    HybridRetriever
)

from src.reasoning.citation_generator import (
    CitationGenerator
)


def main():

    query = (
        "Can an employer terminate an employee without notice?"
    )

    retriever = HybridRetriever()

    retrieved_chunks = (
        retriever.search(
            query=query,
            top_k=5
        )
    )

    citations = (
        CitationGenerator.generate(
            retrieved_chunks
        )
    )

    print("\n")
    print("=" * 80)
    print("SUPPORTING CASES")
    print("=" * 80)

    for idx, citation in enumerate(
        citations,
        start=1
    ):

        print(f"\n{idx}.")

        print(
            f"Case Title: "
            f"{citation['case_title']}"
        )

        print(
            f"Score: "
            f"{citation['score']}"
        )

        print(
            f"Source: "
            f"{citation['source']}"
        )

        print("-" * 80)


if __name__ == "__main__":
    main()
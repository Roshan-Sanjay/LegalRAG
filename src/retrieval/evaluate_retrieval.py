from src.retrieval.hybrid_retriever import (
    HybridRetriever
)

from src.retrieval.evaluation_dataset import (
    BENCHMARK_QUERIES
)


def evaluate():

    retriever = HybridRetriever()

    total_queries = len(
        BENCHMARK_QUERIES
    )

    successful_queries = 0

    average_scores = []

    print("\n")
    print("=" * 70)
    print("LEGAL RAG RETRIEVAL EVALUATION")
    print("=" * 70)

    for item in BENCHMARK_QUERIES:

        query = item["query"]

        expected_terms = [
            term.lower()
            for term in item["expected_terms"]
        ]

        print("\n")
        print("-" * 70)
        print(f"QUERY: {query}")

        results = retriever.search(
            query=query,
            top_k=5
        )

        combined_text = ""

        for result in results:

            combined_text += (
                result["chunk_text"]
                .lower()
            )

        hit = False

        for term in expected_terms:

            if term in combined_text:

                hit = True
                break

        if hit:

            successful_queries += 1

        if results:

            average_scores.append(
                results[0]["hybrid_score"]
            )

        print(
            f"Hit: {hit}"
        )

    hit_rate = (
        successful_queries
        /
        total_queries
    )

    avg_score = (
        sum(average_scores)
        /
        len(average_scores)
    )

    print("\n")
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(
        f"Queries Tested: {total_queries}"
    )

    print(
        f"Successful Retrievals: "
        f"{successful_queries}"
    )

    print(
        f"Hit Rate: "
        f"{hit_rate:.2%}"
    )

    print(
        f"Average Best Match Score: "
        f"{avg_score:.4f}"
    )


if __name__ == "__main__":

    evaluate()

    
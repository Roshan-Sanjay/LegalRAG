from src.retrieval.hybrid_retriever import (
    HybridRetriever
)

from src.evaluation.retrieval_benchmark import (
    BENCHMARK_QUERIES
)


class RetrievalEvaluator:

    def __init__(self):

        self.retriever = (
            HybridRetriever()
        )

    def evaluate(self):

        hits_at_1 = 0
        hits_at_3 = 0
        hits_at_5 = 0

        reciprocal_ranks = []

        total = len(
            BENCHMARK_QUERIES
        )

        for item in BENCHMARK_QUERIES:

            query = item["query"]

            expected_case = (
                item["expected_case"]
                .lower()
            )

            results = (
                self.retriever.search(
                    query=query,
                    top_k=5
                )
            )

            rank_found = None

            for rank, result in enumerate(
                results,
                start=1
            ):

                case_title = (
                    result["case_title"]
                    .lower()
                )

                if expected_case in case_title:

                    rank_found = rank
                    break

            if rank_found == 1:
                hits_at_1 += 1

            if rank_found and rank_found <= 3:
                hits_at_3 += 1

            if rank_found and rank_found <= 5:
                hits_at_5 += 1

            if rank_found:
                reciprocal_ranks.append(
                    1 / rank_found
                )
            else:
                reciprocal_ranks.append(
                    0
                )

        print("\n")
        print("=" * 60)
        print("RETRIEVAL EVALUATION")
        print("=" * 60)

        print(
            f"Queries: {total}"
        )

        print(
            f"Recall@1: "
            f"{hits_at_1 / total:.4f}"
        )

        print(
            f"Recall@3: "
            f"{hits_at_3 / total:.4f}"
        )

        print(
            f"Recall@5: "
            f"{hits_at_5 / total:.4f}"
        )

        print(
            f"MRR: "
            f"{sum(reciprocal_ranks)/total:.4f}"
        )
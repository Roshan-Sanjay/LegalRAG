from collections import defaultdict

from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.cross_encoder_reranker import CrossEncoderReranker
from src.config.settings import DENSE_WEIGHT, SPARSE_WEIGHT


RERANK_CANDIDATES = 20


class HybridRetriever:

    def __init__(self):

        self.dense = DenseRetriever()
        self.sparse = SparseRetriever()
        try:
            self.reranker = CrossEncoderReranker()
        except Exception as exc:
            print(f"Cross-encoder unavailable; hybrid ranking will be used: {exc}")
            self.reranker = None

    def search(self, query, top_k=5):

        if not isinstance(query, str) or not query.strip() or top_k <= 0:
            return []

        # ---------------------------------
        # Dense Retrieval
        # ---------------------------------

        dense_results = self.dense.search(
            query,
            top_k=50
        )

        # ---------------------------------
        # Sparse Retrieval
        # ---------------------------------

        sparse_results = self.sparse.search(
            query,
            top_k=50
        )

        combined = defaultdict(dict)

        # ---------------------------------
        # Dense Scores
        # ---------------------------------

        for result in dense_results:

            key = (result.get("source", ""), result.get("chunk_text", ""))

            combined[key]["dense"] = (
                result["score"]
            )

            combined[key]["data"] = result

        # ---------------------------------
        # Sparse Scores
        # ---------------------------------

        for result in sparse_results:

            key = (result.get("source", ""), result.get("chunk_text", ""))

            combined[key]["sparse"] = (
                result["score"]
            )

            combined[key]["data"] = result

        # ---------------------------------
        # Hybrid Fusion
        # ---------------------------------

        final_results = []

        for item in combined.values():

            dense_score = item.get(
                "dense",
                0
            )

            sparse_score = item.get(
                "sparse",
                0
            )

            hybrid_score = (
                DENSE_WEIGHT * dense_score
                +
                SPARSE_WEIGHT * sparse_score
            )

            data = item["data"]

            data["hybrid_score"] = (
                hybrid_score
            )

            final_results.append(data)

        # ---------------------------------
        # Sort by Hybrid Score
        # ---------------------------------

        final_results.sort(
            key=lambda x: x["hybrid_score"],
            reverse=True
        )

        # ---------------------------------
        # Remove Duplicate Cases
        # ---------------------------------

        unique_results = []
        seen_cases = set()

        for item in final_results:

            case_title = (item.get("case_title") or item.get("source") or item.get("chunk_text", "")).strip().lower()

            if case_title in seen_cases:
                continue

            seen_cases.add(case_title)
            unique_results.append(item)

        # ---------------------------------
        # Select Top Candidates
        # ---------------------------------

        candidate_pool = unique_results[
            :RERANK_CANDIDATES
        ]

        print(
            f"\nCandidate Pool Size: "
            f"{len(candidate_pool)}"
        )

        # ---------------------------------
        # Neural Reranking
        # ---------------------------------

        if not candidate_pool:
            return []
        try:
            if self.reranker is None:
                raise RuntimeError("Cross-encoder did not initialize")
            reranked_results = self.reranker.rerank(query=query, candidates=candidate_pool)
        except Exception as exc:
            print(f"Cross-encoder reranking failed; using hybrid ordering: {exc}")
            reranked_results = candidate_pool

        # ---------------------------------
        # Final Top-K
        # ---------------------------------

        final_topk = reranked_results[:top_k]

        # ---------------------------------
        # Statistics
        # ---------------------------------

        print("\nRetrieval Statistics")

        print(
            f"Unique Cases Retrieved: "
            f"{len(final_topk)}"
        )

        if final_topk:

            scores = [
                item.get("final_score",0)
                for item in final_topk
            ]

            avg_score = (
                sum(scores)
                / len(scores)
            )

            best_score = max(scores)

            print(
                f"Average Score: "
                f"{avg_score:.4f}"
            )

            print(
                f"Best Match Score: "
                f"{best_score:.4f}"
            )

        print("\nTOP RETRIEVAL RESULTS")

        for i, item in enumerate(
            final_topk,
            start=1
        ):

            print(
                f"{i}. "
                f"{item['case_title']} "
                f"(Final={item.get('final_score',0):.4f}, "
                f"Cross={item.get('cross_score',0):.4f}, "
                f"Hybrid={item['hybrid_score']:.4f})"
            )

        return final_topk

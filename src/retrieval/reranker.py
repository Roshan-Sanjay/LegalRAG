from sklearn.metrics.pairwise import cosine_similarity

from src.retrieval.embedder import (
    LegalBERTEmbedder
)


class LegalReranker:

    def __init__(self):

        self.embedder = (
            LegalBERTEmbedder()
        )

    def rerank(
        self,
        query,
        candidates
    ):

        query_embedding = (
            self.embedder.embed_text(
                query
            )
        )

        reranked = []

        for item in candidates:

            chunk_embedding = (
                self.embedder.embed_text(
                    item["chunk_text"]
                )
            )

            score = cosine_similarity(
                [query_embedding],
                [chunk_embedding]
            )[0][0]

            new_item = item.copy()
            
            new_item["rerank_score"] = score
            
            reranked.append(new_item)

        reranked.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return reranked
    

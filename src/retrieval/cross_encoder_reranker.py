import numpy as np

from sentence_transformers import CrossEncoder
from src.config.settings import CROSS_ENCODER_MODEL


MODEL_NAME = CROSS_ENCODER_MODEL


class CrossEncoderReranker:

    def __init__(self):

        print(
            "\nLoading CrossEncoder..."
        )

        self.model = CrossEncoder(
            MODEL_NAME
        )

    def rerank(self, query, candidates):
        if not candidates:
            return []
        pairs = [(
            query,
            item["chunk_text"]
        ) for item in candidates
        ]
        
        scores = self.model.predict(pairs)

        
        scores = np.array(scores)
        
        if not np.isfinite(scores).all():
            raise ValueError("Cross-encoder returned non-finite scores.")
        min_score = scores.min()
        max_score = scores.max()
        
        scores = (
            (scores - min_score) / (max_score - min_score + 1e-8)
        )
        
        reranked = []
        
        for item, score in zip(candidates, scores):
            new_item = item.copy()
            new_item["cross_score"] = (
                float(score)
            )

            new_item["final_score"] = (
                0.7 * new_item["cross_score"]
                +
                0.3 * new_item["hybrid_score"]
            )
            
            reranked.append(new_item)
        
        reranked.sort(
            key=lambda x:
            x["final_score"],
            reverse=True
        )
        
        return reranked

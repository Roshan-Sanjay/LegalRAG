import numpy as np

from sentence_transformers import CrossEncoder


MODEL_NAME = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


class CrossEncoderReranker:

    def __init__(self):

        print(
            "\nLoading CrossEncoder..."
        )

        self.model = CrossEncoder(
            MODEL_NAME
        )

    def rerank(self, query, candidates):
        pairs = [(
            query,
            item["chunk_text"]
        ) for item in candidates
        ]
        
        scores = self.model.predict(pairs)

        
        scores = np.array(scores)
        
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
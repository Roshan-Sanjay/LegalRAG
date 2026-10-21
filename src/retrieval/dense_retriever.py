import joblib
import torch

from transformers import (
    AutoTokenizer,
    AutoModel
)


MODEL_NAME = (
    "nlpaueb/legal-bert-base-uncased"
)


class DenseRetriever:

    def __init__(self):

        print("Loading vector DB...")

        db = joblib.load(
            "vector_db/nearest_neighbors.joblib"
        )

        self.index = db["index"]
        self.metadata = db["metadata"]

        print("Loading LegalBERT...")

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                MODEL_NAME
            )
        )

        self.model = (
            AutoModel.from_pretrained(
                MODEL_NAME
            )
        )

        self.model.eval()

    @torch.no_grad()
    def embed_query(self, query):

        encoded = self.tokenizer(
            query,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )

        outputs = self.model(**encoded)

        embeddings = outputs.last_hidden_state

        mask = (
            encoded["attention_mask"]
            .unsqueeze(-1)
            .expand(embeddings.size())
            .float()
        )

        pooled = (
            embeddings * mask
        ).sum(1) / mask.sum(1)

        return pooled.numpy()

    def search(self, query, top_k=5):

        query_vector = (
            self.embed_query(query)
        )

        distances, indices = (
            self.index.kneighbors(
                query_vector,
                n_neighbors=top_k
            )
        )

        results = []

        for score, idx in zip(
            distances[0],
            indices[0]
        ):

            row = self.metadata.iloc[idx]

            results.append({
                "score": 1 - float(score),
                "case_title": row["case_title"],
                "source": row["source"],
                "chunk_text": row["chunk_text"]
            })

        return results
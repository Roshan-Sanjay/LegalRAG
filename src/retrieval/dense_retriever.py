import joblib
import torch
from src.config.settings import DENSE_INDEX_PATH, EMBEDDING_MODEL

from transformers import (
    AutoTokenizer,
    AutoModel
)


MODEL_NAME = EMBEDDING_MODEL


class DenseRetriever:

    def __init__(self, index_path=DENSE_INDEX_PATH):

        print("Loading vector DB...")

        try:
            db = joblib.load(index_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Dense retrieval index missing: {index_path}. Run `python -m src.setup_indexes` after supplying LegalBERT embeddings.") from exc
        if not isinstance(db, dict) or not {"index", "metadata"}.issubset(db):
            raise ValueError("Invalid dense retrieval index: expected index and metadata.")

        self.index = db["index"]
        self.metadata = db["metadata"]
        if len(self.metadata) == 0:
            raise ValueError("Dense retrieval index contains no metadata records.")

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

        if not isinstance(query, str) or not query.strip() or top_k <= 0:
            return []

        query_vector = (
            self.embed_query(query)
        )

        distances, indices = (
            self.index.kneighbors(
                query_vector,
                n_neighbors=min(top_k, len(self.metadata))
            )
        )

        results = []

        for score, idx in zip(
            distances[0],
            indices[0]
        ):

            row = self.metadata.iloc[idx] if hasattr(self.metadata, "iloc") else self.metadata[idx]

            results.append({**dict(row), "score": 1 - float(score)})

        return results

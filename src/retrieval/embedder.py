import pandas as pd
import numpy as np
import torch

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import joblib


MODEL_NAME = "nlpaueb/legal-bert-base-uncased"

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


class LegalBERTEmbedder:

    def __init__(self):

        print(f"\nLoading model: {MODEL_NAME}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )

        self.model = AutoModel.from_pretrained(
            MODEL_NAME
        )

        self.model.to(DEVICE)
        self.model.eval()

        print(f"Using device: {DEVICE}")

    @torch.no_grad()
    def embed_text(self, text: str):

        encoded = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )

        encoded = {
            k: v.to(DEVICE)
            for k, v in encoded.items()
        }

        output = self.model(**encoded)

        token_embeddings = output.last_hidden_state

        attention_mask = (
            encoded["attention_mask"]
            .unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )

        masked_embeddings = (
            token_embeddings * attention_mask
        )

        summed = masked_embeddings.sum(dim=1)

        counts = attention_mask.sum(dim=1)

        mean_pooled = summed / counts

        return mean_pooled.squeeze().cpu().numpy()


def generate_embeddings():

    print("\nLoading chunks...")

    df = pd.read_csv(
        "data/chunks/chunked_judgments.csv"
    )

    print(f"Loaded {len(df)} chunks")

    embedder = LegalBERTEmbedder()

    embeddings = []

    for chunk in tqdm(df["chunk_text"]):

        vector = embedder.embed_text(chunk)

        embeddings.append(vector)

    embeddings = np.array(embeddings)

    print("\nEmbedding shape:")
    print(embeddings.shape)

    output = {
        "embeddings": embeddings,
        "metadata": df
    }

    joblib.dump(
        output,
        "data/embeddings/legalbert_embeddings.joblib"
    )

    print(
        "\nSaved embeddings to:"
        "\ndata/embeddings/legalbert_embeddings.joblib"
    )


if __name__ == "__main__":
    generate_embeddings()

    
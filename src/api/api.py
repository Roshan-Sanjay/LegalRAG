from fastapi import FastAPI

from src.reasoning.legal_rag_pipeline import (
    LegalRAGPipeline
)

app = FastAPI(
    title="LegalRAG API",
    version="1.0"
)

pipeline = LegalRAGPipeline()


@app.get("/")
def root():

    return {
        "message": "LegalRAG API Running"
    }


@app.post("/ask")
def ask_question(payload: dict):

    query = payload["query"]

    result = (
        pipeline.answer_query(query)
    )

    return result
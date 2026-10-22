import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.retrieval.vector_store import build_vector_store
from src.retrieval.tfidf_builder import build_tfidf
from src.retrieval.sparse_retriever import SparseRetriever
from src.reasoning.citation_generator import CitationGenerator
from src.reasoning.legal_rag_pipeline import LegalRAGPipeline
from src.api import api
import app as frontend

def test_dense_index_from_small_embeddings(tmp_path):
    source, output = tmp_path / "vectors.joblib", tmp_path / "dense.joblib"
    metadata = pd.DataFrame([{"case_title": "Case A", "source": "url", "chunk_text": "text"}, {"case_title": "Case B", "source": "url2", "chunk_text": "text2"}])
    joblib.dump({"embeddings": np.array([[1., 0.], [0., 1.]]), "metadata": metadata}, source)
    assert build_vector_store(source, output) == 2
    store = joblib.load(output)
    assert store["index"].kneighbors([[1., 0.]], n_neighbors=1)[1][0][0] == 0

def test_tfidf_store_round_trip(tmp_path):
    source, output = tmp_path / "chunks.csv", tmp_path / "tfidf.joblib"
    pd.DataFrame([{"case_title": "A", "source": "a", "chunk_text": "employment notice termination"}, {"case_title": "B", "source": "b", "chunk_text": "property title dispute"}]).to_csv(source, index=False)
    assert build_tfidf(source, output) == 2
    retriever = SparseRetriever(output)
    assert retriever.search("employment notice", top_k=10)[0]["case_title"] == "A"
    assert len(retriever.search("employment", top_k=10)) == 2

def test_invalid_tfidf_dataset_fails_with_missing_column(tmp_path):
    source = tmp_path / "bad.csv"
    pd.DataFrame([{"case_title": "A", "chunk_text": "abc"}]).to_csv(source, index=False)
    with pytest.raises(ValueError, match="source"):
        build_tfidf(source, tmp_path / "out.joblib")

def test_missing_sparse_index_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m src.setup_indexes"):
        SparseRetriever(tmp_path / "missing.joblib")

def test_pipeline_validates_empty_and_nonstring_queries():
    pipeline = LegalRAGPipeline(retriever=object(), llm=object())
    for query in ("", "  ", None):
        with pytest.raises(ValueError):
            pipeline.answer_query(query)

def test_empty_retrieval_returns_zero_confidence_without_llm():
    class Retriever:
        def search(self, query, top_k): return []
    result = LegalRAGPipeline(retriever=Retriever(), llm=object()).answer_query("question")
    assert result["citations"] == [] and result["confidence"] == 0.0
    assert "Insufficient" in result["answer"]

def test_citations_preserve_real_metadata_and_ignore_titleless_records():
    result = CitationGenerator.generate([{"case_title": "Actual record", "case_id": "123", "court": "Delhi High Court", "year": 2020, "source": "url", "final_score": .75}, {"chunk_text": "no title"}])
    assert result == [{"rank": 1, "case_title": "Actual record", "case_id": "123", "court": "Delhi High Court", "year": 2020, "source": "url", "score": .75}]

def test_api_request_schema_and_response(monkeypatch):
    class Pipeline:
        def answer_query(self, query): return {"query": query, "answer": "evidence-based", "confidence": 0, "citations": []}
    monkeypatch.setattr(api, "_pipeline", Pipeline())
    client = TestClient(api.app)
    assert client.post("/ask", json={"query": "  "}).status_code == 422
    response = client.post("/ask", json={"query": " question "})
    assert response.status_code == 200 and response.json()["answer"] == "evidence-based"

def test_frontend_backend_errors_are_user_facing(monkeypatch):
    class Response:
        status_code = 503
        def raise_for_status(self): raise frontend.requests.HTTPError(response=self)
        def json(self): return {"detail": "index missing"}
    monkeypatch.setattr(frontend.requests, "post", lambda *a, **k: Response())
    answer, confidence, citations = frontend.ask_question("query")
    assert "503" in answer and "index missing" in answer and citations == []

def test_ollama_connection_failure_is_clear(monkeypatch):
    import requests
    from src.llm.inference import OllamaLLM
    def fail(*args, **kwargs): raise requests.ConnectionError("offline")
    monkeypatch.setattr("src.llm.inference.requests.post", fail)
    with pytest.raises(RuntimeError, match="Could not connect to Ollama"):
        OllamaLLM().generate("prompt")

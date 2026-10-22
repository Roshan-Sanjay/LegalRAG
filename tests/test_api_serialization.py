import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.api.serialization import to_json_compatible
from src.api import api


def test_numpy_integer_and_floating_scores_convert_to_json_values():
    converted = to_json_compatible({"case_id": np.int64(47), "score": np.float32(0.875)})
    assert converted == {"case_id": 47, "score": 0.875}
    assert type(converted["case_id"]) is int
    assert type(converted["score"]) is float


def test_nested_citation_metadata_and_arrays_are_preserved():
    response = {
        "citations": [{
            "rank": np.int64(1),
            "case_title": "Sample v Example",
            "score": np.float64(0.92),
            "metadata": {"case_id": np.int32(123), "flags": np.array([np.bool_(True), np.bool_(False)])},
        }]
    }
    converted = to_json_compatible(response)
    assert converted["citations"][0] == {
        "rank": 1,
        "case_title": "Sample v Example",
        "score": 0.92,
        "metadata": {"case_id": 123, "flags": [True, False]},
    }


def test_empty_citations_and_missing_values_are_supported():
    converted = to_json_compatible({"citations": [], "case_id": pd.NA, "year": pd.NaT, "score": float("nan")})
    assert converted == {"citations": [], "case_id": None, "year": None, "score": None}


def test_non_finite_numpy_floats_use_null_policy():
    converted = to_json_compatible({"nan": np.float64(np.nan), "positive": np.float32(np.inf), "negative": -float("inf")})
    assert converted == {"nan": None, "positive": None, "negative": None}


def test_ask_endpoint_serializes_successful_pipeline_response(monkeypatch):
    pipeline_response = {
        "query": "a question",
        "answer": "Evidence-based answer",
        "sources": [{"case_title": "Sample v Example", "chunk_id": np.int64(4), "score": np.float64(0.81)}],
        "num_sources": np.int64(1),
        "context": "case context",
        "citations": [{"rank": np.int64(1), "case_title": "Sample v Example", "case_id": np.int64(99), "score": np.float32(0.81), "source": "https://example.test/case", "metadata": {"year": pd.NA}}],
        "confidence": np.float32(81.0),
        "confidence_type": "estimated retrieval relevance (not a probability)",
    }

    class MockPipeline:
        def answer_query(self, query):
            return pipeline_response

    monkeypatch.setattr(api, "_pipeline", MockPipeline())
    response = TestClient(api.app).post("/ask", json={"query": "a question"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Evidence-based answer"
    assert body["confidence"] == 81.0
    assert body["num_sources"] == 1
    assert body["sources"][0]["chunk_id"] == 4
    assert body["citations"][0]["case_id"] == 99
    assert body["citations"][0]["metadata"]["year"] is None
    # Keep the field names consumed by app.py's Gradio renderer.
    assert {"rank", "case_title", "score", "source"}.issubset(body["citations"][0])

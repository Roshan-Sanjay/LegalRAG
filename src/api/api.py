import logging
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from src.config.settings import (MAX_QUERY_CHARS, DENSE_INDEX_PATH, TFIDF_INDEX_PATH,
                                 OLLAMA_BASE_URL, OLLAMA_MODEL)
from src.api.serialization import to_json_compatible

logger = logging.getLogger("LegalRAG.api")
app = FastAPI(title="LegalRAG API", version="1.1.0", description="Indian legal case research grounded in retrieved judgments.")
_pipeline = None

class AskRequest(BaseModel):
    query: str = Field(..., description="A legal question", max_length=MAX_QUERY_CHARS)
    @field_validator("query")
    @classmethod
    def query_not_blank(cls, value):
        if not value.strip():
            raise ValueError("Query cannot be empty or whitespace.")
        return value.strip()

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from src.reasoning.legal_rag_pipeline import LegalRAGPipeline
        _pipeline = LegalRAGPipeline()
    return _pipeline

@app.get("/", tags=["health"])
@app.get("/health", tags=["health"])
def health():
    try:
        ollama_response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=1.5)
        ollama_ok = ollama_response.ok
        models = [item.get("name") for item in ollama_response.json().get("models", [])] if ollama_ok else []
    except (requests.RequestException, ValueError):
        ollama_ok, models = False, []
    return {"status": "ok", "message": "LegalRAG API Running", "indexes": {"dense": DENSE_INDEX_PATH.is_file(), "sparse": TFIDF_INDEX_PATH.is_file()}, "ollama": {"reachable": ollama_ok, "model": OLLAMA_MODEL, "model_available": any(name == OLLAMA_MODEL or name.startswith(OLLAMA_MODEL + ":") for name in models)}}

@app.post("/ask", tags=["research"])
def ask_question(payload: AskRequest):
    try:
        result = get_pipeline().answer_query(payload.query)
        return to_json_compatible(result)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("LegalRAG query failed")
        raise HTTPException(status_code=500, detail="The legal research request could not be completed.") from exc

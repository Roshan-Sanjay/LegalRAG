# LegalRAG

LegalRAG is a local Indian case law research prototype. It retrieves judgment chunks with LegalBERT dense search and TF-IDF sparse search, fuses and reranks results with a cross encoder, then asks a locally served Ollama model to answer from the retrieved context. FastAPI serves the query API; Gradio is the user interface.

Answers are research assistance, not legal advice. The confidence value is an estimated retrieval relevance score on a 0–100 scale, not a calibrated probability of legal correctness. Citations are formatted from retrieved records and are only shown when a retrieved case title exists; metadata is included only when present in the corpus. API serialization converts NumPy values and pandas missing values recursively; missing values and non-finite numbers (NaN or infinity) become JSON `null`. Unsupported object types raise an error instead of being converted to strings.

## Architecture and repository

| Path | Purpose |
| --- | --- |
| `src/api/api.py` | FastAPI `/ask`, `/health`, and `/` endpoints |
| `app.py` | Gradio UI calling the backend |
| `src/retrieval/` | LegalBERT dense retrieval, TF-IDF, hybrid fusion, cross encoder, index builders |
| `src/reasoning/` | Context, prompts, citations, and answer pipeline |
| `src/llm/` | Ollama local inference |
| `src/setup_indexes.py` | Build or reuse retrieval indexes |
| `scripts/test_setup.py` | Read-only preflight checks |
| `tests/` | Small isolated unit tests |
| `data/chunks/chunked_judgments.csv` | Judgment chunk corpus (`case_title`, `source`, `chunk_id`, `chunk_text`) |
| `data/cleaned/cleaned_judgments.csv` | Cleaned source judgments |

The repository contains the chunk and cleaned CSVs plus the precomputed embedding artifact `data/embeddings/legalbert_embeddings.joblib`. Dense index initialization reuses those vectors and checks that the embedding rows align with the chunk metadata; it does not regenerate embeddings. If that artifact is missing from another clone, it must be supplied before dense index creation.

## Requirements

Python 3.11 is recommended. The declared runtime ranges support Python 3.10+; Python 3.11 is the setup target. PyTorch wheels and model inference performance depend on the operating system and available CPU/GPU.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` in PowerShell, then activate again. Or use Command Prompt: `.venv\Scripts\activate.bat`.

### macOS/Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Ollama and configuration

Install Ollama from [ollama.com/download](https://ollama.com/download), start it, then run:

```bash
ollama --version
ollama pull llama3:8b
ollama list
```

No paid API key is needed. Copy `.env.example` to `.env` and edit values if required (`cp .env.example .env` on macOS/Linux, `Copy-Item .env.example .env` on PowerShell). `.env` is ignored by Git. The defaults use `http://localhost:11434/api/generate` and `llama3:8b`. The application will not download models automatically.

Important settings include `CHUNKS_PATH`, `EMBEDDINGS_PATH`, `DENSE_INDEX_PATH`, `TFIDF_INDEX_PATH`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `BACKEND_URL`, `API_HOST`, and `API_PORT`.

## Initialize retrieval indexes

From the repository root, with the virtual environment active:

```bash
python -m src.setup_indexes
```

This command validates and reuses any loadable existing indexes. It builds TF-IDF from the chunk CSV and builds the dense cosine nearest-neighbor index from the precomputed LegalBERT joblib artifact. It does not generate embeddings. If embeddings are absent, provide `data/embeddings/legalbert_embeddings.joblib` or set `EMBEDDINGS_PATH`; the artifact must contain `embeddings` and row-aligned `metadata` fields. Index artifacts use joblib/pickle serialization: only load files from trusted sources.

## Preflight and run

```bash
python scripts/test_setup.py
uvicorn src.api.api:app --host 127.0.0.1 --port 8000
```

In another terminal, activate `.venv` and run:

```bash
python app.py
```

Then open `http://127.0.0.1:7860`. API health is at `http://127.0.0.1:8000/health` (also `/`), and interactive API docs are at `http://127.0.0.1:8000/docs`. Submit a legal question in Gradio and review the answer, retrieval relevance estimate, and supporting records. Health checks report index-file presence and a lightweight Ollama/model availability check; they do not perform inference.

## Tests

Install development and test dependencies and run isolated tests:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
```

Tests use temporary synthetic indexes and mock Ollama. They do not overwrite the corpus, download models, or make external API calls.

## Troubleshooting

* **Dense index missing / embeddings missing:** supply the precomputed LegalBERT joblib artifact, then rerun index initialization. No large artifact is downloaded automatically.
* **Sparse index missing:** check the chunk CSV and rerun `python -m src.setup_indexes`.
* **Ollama unavailable:** start Ollama and confirm `ollama list` includes the configured model.
* **Configured model not found:** run `ollama pull llama3:8b` or change `OLLAMA_MODEL` to a model already installed; the code does not silently substitute models.
* **Slow inference or out of memory:** Llama 3 8B and transformer retrieval/reranking need substantial memory. CPU is supported but can be slow; use compatible PyTorch acceleration when available.
* **Untrusted serialized index:** do not load a joblib artifact from an unknown source; deserialization can execute code.

## Local inference and limitations

The answer generation request is sent to the configured local Ollama server. Retrieval model weights may need to be available locally or fetched by Transformers on first use; no LLM pull occurs at startup. Corpus coverage and metadata quality bound the evidence available. The system may return an insufficient-context response when retrieval yields no evidence, and generated text should always be checked against the cited judgments and current law.

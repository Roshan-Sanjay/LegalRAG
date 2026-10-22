"""Read-only preflight checks; does not generate indexes or download models."""
import importlib
import sys
from pathlib import Path
import requests
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config.settings import (CHUNKS_PATH, EMBEDDINGS_PATH, DENSE_INDEX_PATH,
                                 TFIDF_INDEX_PATH, OLLAMA_BASE_URL, OLLAMA_MODEL)

failures = 0
def report(status, label, detail):
    global failures
    print(f"{status}: {label} — {detail}")
    failures += status == "FAIL"

def main():
    if sys.version_info >= (3, 10): report("PASS", "Python", sys.version.split()[0])
    else: report("FAIL", "Python", "Python 3.10+ required; use Python 3.11")
    for module in ("fastapi", "uvicorn", "gradio", "pandas", "numpy", "joblib", "sklearn", "transformers", "sentence_transformers", "torch", "requests", "bs4", "tqdm"):
        try: importlib.import_module(module); report("PASS", f"Import {module}", "available")
        except Exception as exc: report("FAIL", f"Import {module}", f"{exc}; install requirements.txt")
    for path, label in ((CHUNKS_PATH, "Chunk dataset"), (EMBEDDINGS_PATH, "Precomputed LegalBERT embeddings"), (DENSE_INDEX_PATH, "Dense index"), (TFIDF_INDEX_PATH, "Sparse index")):
        required = label != "Precomputed LegalBERT embeddings" or not DENSE_INDEX_PATH.exists()
        if path.is_file(): report("PASS", label, str(path))
        else: report("FAIL" if required else "WARN", label, f"missing: {path}; see README setup instructions")
    import joblib
    for path, label in ((DENSE_INDEX_PATH, "Dense index load"), (TFIDF_INDEX_PATH, "Sparse index load")):
        if path.is_file():
            try: joblib.load(path); report("PASS", label, "deserialized successfully (only load trusted joblib files)")
            except Exception as exc: report("FAIL", label, f"cannot load: {exc}")
        else: report("WARN", label, "not available")
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        if response.ok:
            names = [item.get("name", "") for item in response.json().get("models", [])]
            report("PASS", "Ollama endpoint", OLLAMA_BASE_URL)
            report("PASS" if any(n == OLLAMA_MODEL or n.startswith(OLLAMA_MODEL + ":") for n in names) else "WARN", "Ollama model", f"{OLLAMA_MODEL} {'available' if any(n == OLLAMA_MODEL or n.startswith(OLLAMA_MODEL + ':') for n in names) else 'not listed; run ollama pull ' + OLLAMA_MODEL}")
        else: report("WARN", "Ollama endpoint", f"HTTP {response.status_code}; start Ollama")
    except (requests.RequestException, ValueError) as exc: report("WARN", "Ollama endpoint", f"unreachable ({exc}); install/start Ollama")
    print(f"\nPreflight complete: {failures} failed check(s).")
    return 1 if failures else 0

if __name__ == "__main__":
    raise SystemExit(main())

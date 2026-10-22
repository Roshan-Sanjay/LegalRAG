"""Gradio frontend; all retrieval and generation remain in the FastAPI backend."""
import requests
import gradio as gr
from src.config.settings import BACKEND_URL, GRADIO_HOST, GRADIO_PORT, MAX_QUERY_CHARS

def ask_question(query):
    if not isinstance(query, str) or not query.strip():
        return "Enter a legal question to continue.", "—", []
    if len(query.strip()) > MAX_QUERY_CHARS:
        return f"Question is too long (maximum {MAX_QUERY_CHARS} characters).", "—", []
    try:
        response = requests.post(f"{BACKEND_URL}/ask", json={"query": query.strip()}, timeout=(5, 320))
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict) or not isinstance(result.get("answer"), str):
            raise ValueError("Backend returned an unexpected response.")
        score = result.get("confidence")
        confidence = f"{score}% (estimated retrieval relevance)" if isinstance(score, (int, float)) else "Unavailable"
        rows = []
        for citation in result.get("citations") or []:
            if not isinstance(citation, dict):
                continue
            rows.append([citation.get("rank", ""), citation.get("case_title", "Case title unavailable"), citation.get("citation", ""), citation.get("court", ""), citation.get("year", ""), citation.get("score", ""), citation.get("source", "")])
        return result["answer"], confidence, rows
    except requests.exceptions.Timeout:
        return "The backend request timed out. Check that the API and Ollama are running.", "Unavailable", []
    except requests.exceptions.ConnectionError:
        return f"Cannot reach the LegalRAG API at {BACKEND_URL}. Start the backend and try again.", "Unavailable", []
    except requests.exceptions.HTTPError as exc:
        try:
            detail = exc.response.json().get("detail", "")
        except (ValueError, AttributeError):
            detail = ""
        return f"Backend error ({exc.response.status_code}): {detail or 'request failed'}", "Unavailable", []
    except (requests.RequestException, ValueError) as exc:
        return f"Could not read the backend response: {exc}", "Unavailable", []

with gr.Blocks(theme=gr.themes.Soft(), title="Legal AI Assistant") as demo:
    gr.Markdown("# ⚖️ Legal AI Assistant\n### Indian Case Law Research & Reasoning System\nAsk a legal question and review the retrieved authorities supporting the answer.")
    query = gr.Textbox(label="Legal Question", placeholder="e.g. Can an employer terminate an employee without notice?", lines=4, max_lines=8)
    submit = gr.Button("Analyze Legal Question", variant="primary")
    answer = gr.Textbox(label="Legal Analysis", lines=18)
    confidence = gr.Textbox(label="Estimated confidence", interactive=False)
    gr.Markdown("The score summarizes retrieval relevance. It is not a calibrated probability that the legal answer is correct.")
    citations = gr.Dataframe(headers=["Rank", "Case Title", "Citation", "Court", "Year", "Relevance Score", "Source"], label="Supporting Authorities", interactive=False)
    submit.click(fn=ask_question, inputs=query, outputs=[answer, confidence, citations])

if __name__ == "__main__":
    demo.launch(server_name=GRADIO_HOST, server_port=GRADIO_PORT)

import requests 
import gradio as gr



def ask_question(query):
  try:
    response = requests.post(
        "http://127.0.0.1:8000/ask",
        json={
            "query": query
        },
        timeout=300
    )

    result = response.json()

    answer = result["answer"]

    confidence = (
        f"{result['confidence']}%"
    )

    rows = []

    for c in result["citations"]:

        rows.append([
            c["rank"],
            c["case_title"],
            c["score"],
            c["source"]
        ])

    return (
        answer,
        confidence,
        rows
    )
 
  except Exception as e:
    return (
        f"API Error: {e}",
        "0%",
        []
    )


with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Legal AI Assistant"
) as demo:

    gr.Markdown(
        """
# ⚖️ Legal AI Assistant

### Indian Case Law Research & Reasoning System

Ask legal questions and receive answers grounded in retrieved case law.
"""
    )

    with gr.Row():

        with gr.Column(scale=3):

            query = gr.Textbox(
                label="Legal Question",
                placeholder="e.g. Can an employer terminate an employee without notice?",
                lines=4
            )

            gr.Examples(
                examples=[
                    ["Can an employer terminate an employee without notice?"],
                    ["What constitutes unfair dismissal?"],
                    ["Can a contract be terminated unilaterally?"],
                    ["What are employee rights after termination?"]
                ],
                inputs=query
            )

            submit = gr.Button(
                "Analyze Legal Question",
                variant="primary"
            )

        with gr.Column(scale=1):

            gr.Markdown(
                """
### System Information

**Corpus Statistics**

- Cases Indexed: 117
- Chunks Indexed: 4721

**Models**

- Embeddings: LegalBERT
- LLM: Llama3:8b
- Retrieval: Hybrid Search

**Backend**

- FastAPI

"""
            )

    
    with gr.Column():

        answer = gr.Textbox(
            label="⚖️ Legal Analysis",
            lines=20,
            scale=4
        )

        confidence = gr.Textbox(
            label="Answer Confidence",
            interactive=False,
            scale=1
        )
        gr.Markdown(
            """
            Confidence is estimated from the relevance 
            scores of the retrieved legal authorities.
            """
        )   
        

    citations = gr.Dataframe(
        headers=[
            "Rank",
            "Case Title",
            "Relevance Score",
            "Source"
        ],
        label="Supporting Authorities"
    )

    with gr.Accordion(
        "How the System Works",
        open=False
    ):

        gr.Markdown(
            """
1. User submits a legal question.

2. Hybrid retrieval searches the legal corpus.

3. LegalBERT identifies semantically relevant judgments.

4. Relevant legal context is assembled.

5. Phi-3 generates legal reasoning.

6. Supporting authorities are displayed.
"""
        )

    submit.click(
        fn=ask_question,
        inputs=query,
        outputs=[
            answer,
            confidence,
            citations
        ]
    )

    gr.Markdown(
        """
---
**Powered by:** LegalBERT • ChromaDB • Ollama (Llama3:8b) • FastAPI • Gradio
"""
    )

demo.launch(inbrowser=True)


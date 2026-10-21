from src.retrieval.hybrid_retriever import (
    HybridRetriever
)

from src.reasoning.context_builder import (
    ContextBuilder
)

from src.reasoning.prompt_builder import (
    PromptBuilder
)


query = (
    "Can an employer terminate an employee "
    "without notice?"
)

retriever = HybridRetriever()

results = retriever.search(
    query,
    top_k=5
)

builder = ContextBuilder()

context = builder.build_context(
    results
)

prompt = (
    PromptBuilder
    .build_legal_prompt(
        query=query,
        context=context
    )
)

print(prompt)


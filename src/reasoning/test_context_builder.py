from src.retrieval.hybrid_retriever import (
    HybridRetriever
)

from src.reasoning.context_builder import (
    ContextBuilder
)


retriever = HybridRetriever()

builder = ContextBuilder()

query = (
    "Can an employer terminate an employee "
    "without notice?"
)

results = retriever.search(
    query,
    top_k=5
)

context = builder.build_context(
    results
)

print(context)


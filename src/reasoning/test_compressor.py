from src.retrieval.hybrid_retriever import (
    HybridRetriever
)

from src.reasoning.context_compressor import (
    ContextCompressor
)

retriever = HybridRetriever()
compressor = ContextCompressor()

results = retriever.search(
    "Can labour court order reinstatement?"
)

context = compressor.compress(
    results
)

print("\nCOMPRESSED CONTEXT\n")
print(context)

print(
    f"\nLength: {len(context)} chars"
)


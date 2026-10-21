from src.reasoning.legal_rag_pipeline import (
    LegalRAGPipeline
)


pipeline = LegalRAGPipeline()

query = (
    "Can an employer terminate an employee without notice?"
)

result = pipeline.answer_query(query)

print("\n")
print("=" * 80)
print("LEGAL QUESTION")
print("=" * 80)
print(result["query"])

print("\n")
print("=" * 80)
print("LEGAL ANSWER")
print("=" * 80)
print(result["answer"])

print("\n")
print("=" * 80)
print("SUPPORTING CASES")
print("=" * 80)

for citation in result["citations"]:

    print(
        f"\n• {citation['case_title']}"
    )

    print(
        f"  Score: {citation['score']}"
    )

    print(
        f"  Source: {citation['source']}"
    )

print("\n")
print("=" * 80)
print("RETRIEVED CHUNKS")
print("=" * 80)

for i, chunk in enumerate(result["sources"], start=1):

    print(f"\nCASE {i}")
    print(chunk["case_title"])
    print()

    print(
        chunk["chunk_text"][:500]
    )

    print("\n" + "-" * 80)

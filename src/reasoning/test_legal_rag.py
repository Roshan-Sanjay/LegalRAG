from src.reasoning.legal_rag_pipeline import (
    LegalRAGPipeline
)


def main():

    rag = LegalRAGPipeline()

    query = (
        "Can an employer terminate an employee without notice?"
    )

    result = (
        rag.answer_query(query)
    )

    print("\n")
    print("=" * 80)
    print("QUERY")
    print("=" * 80)
    print(result["query"])

    print("\n")
    print("=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(result["answer"])

    print("\n")
    print("=" * 80)
    print("RETRIEVED CASES")
    print("=" * 80)

    for source in result["sources"]:

        print(
            f"- {source['case_title']} "
            f"({source['hybrid_score']:.4f})"
        )


if __name__ == "__main__":
    main()

    
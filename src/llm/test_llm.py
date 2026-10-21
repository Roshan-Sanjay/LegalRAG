from src.llm.inference import OllamaLLM


def main():

    llm = OllamaLLM()

    question = """
    Explain the doctrine of natural justice.
    """

    answer = llm.generate(question)

    print("\n")
    print("=" * 80)
    print("MODEL RESPONSE")
    print("=" * 80)
    print(answer)


if __name__ == "__main__":
    main()

    
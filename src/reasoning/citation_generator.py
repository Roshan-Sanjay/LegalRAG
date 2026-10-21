class CitationGenerator:

    @staticmethod
    def generate(
        retrieved_chunks
    ):

        citations = []

        for rank, chunk in enumerate(
            retrieved_chunks,
            start=1
        ):

            citations.append(
                {
                    "rank": rank,

                    "case_title":
                        chunk["case_title"],

                    "source":
                        chunk["source"],

                    "score":
                        round(
                            chunk.get(
                                "hybrid_score",
                                0
                            ),
                            4
                        )
                }
            )

        return citations
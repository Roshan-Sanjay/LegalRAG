class ContextCompressor:

    def __init__(
        self,
        chunk_size=500
    ):
        self.chunk_size = chunk_size

    def compress(
        self,
        retrieved_chunks
    ):

        compressed_parts = []

        for item in retrieved_chunks:

            case_title = item.get(
                "case_title",
                "Unknown Case"
            )

            chunk_text = item.get(
                "chunk_text",
                ""
            )

            compressed_text = (
                chunk_text[:self.chunk_size]
            )

            compressed_parts.append(
                f"CASE: {case_title}\n"
                f"{compressed_text}\n"
            )

        return "\n".join(
            compressed_parts
        )
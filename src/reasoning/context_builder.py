from typing import List, Dict
import math

from src.config.settings import (
    MAX_CONTEXT_CHARS
)


class ContextBuilder:

    def __init__(
        self,
        max_context_chars=MAX_CONTEXT_CHARS
    ):

        self.max_context_chars = (
            max_context_chars
        )

    # -----------------------------
    # Remove duplicates
    # -----------------------------

    def remove_duplicates(
        self,
        retrieved_chunks: List[Dict]
    ):

        seen = set()

        unique_chunks = []

        for chunk in retrieved_chunks:
            key = (chunk.get("source", ""), chunk.get("chunk_text", ""))

            if key not in seen:

                seen.add(key)

                unique_chunks.append(
                    chunk
                )

        return unique_chunks

    # -----------------------------
    # Sort chunks
    # -----------------------------

    def sort_chunks(
        self,
        retrieved_chunks: List[Dict]
    ):

        if not retrieved_chunks:
            return []

        if "final_score" in retrieved_chunks[0]:

            score_field = (
                "final_score"
            )

        elif "cross_score" in retrieved_chunks[0]:

            score_field = (
                "cross_score"
            )

        elif "hybrid_score" in retrieved_chunks[0]:

            score_field = (
                "hybrid_score"
            )

        else:

            score_field = (
                "score"
            )

        def score(chunk):
            try:
                value = float(chunk.get(score_field, 0) or 0)
                return value if math.isfinite(value) else 0.0
            except (TypeError, ValueError):
                return 0.0
        return sorted(retrieved_chunks, key=score, reverse=True)

    # -----------------------------
    # Build Context
    # -----------------------------

    def build_context(
        self,
        retrieved_chunks: List[Dict]
    ):

        retrieved_chunks = (
            self.remove_duplicates(
                retrieved_chunks
            )
        )

        retrieved_chunks = (
            self.sort_chunks(
                retrieved_chunks
            )
        )

        context_parts = []

        current_length = 0

        for i, chunk in enumerate(
            retrieved_chunks,
            start=1
        ):

            title = chunk.get("case_title") or "Case title unavailable"
            source = chunk.get("source") or "Source unavailable"
            text = chunk.get("chunk_text")
            if not isinstance(text, str) or not text.strip():
                continue
            score = chunk.get(
                "final_score",
                chunk.get(
                    "cross_score",
                    chunk.get(
                        "hybrid_score",
                        chunk.get(
                            "score",
                            0
                        )
                    )
                )
            )

            try:
                score = float(score)
            except (TypeError, ValueError):
                score = 0.0
            case_block = f"""
[CASE {i}]

Case Title:
{title}

Score:
{score:.4f}

Source:
{source}

Relevant Text:
{text.strip()}

"""

            if (
                current_length
                + len(case_block)
                > self.max_context_chars
            ):
                break

            context_parts.append(
                case_block
            )

            current_length += (
                len(case_block)
            )

        print(
            f"\nContext Length: "
            f"{current_length} chars"
        )

        print(
            f"Cases Used: "
            f"{len(context_parts)}"
        )

        return "\n".join(
            context_parts
        )

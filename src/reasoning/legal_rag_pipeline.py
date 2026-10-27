from src.config.settings import (
MAX_QUERY_CHARS,
MAX_OUTPUT_RETRIES,
INVALID_OUTPUT_FALLBACK,
)
from src.retrieval.hybrid_retriever import HybridRetriever
from src.reasoning.context_builder import ContextBuilder
from src.reasoning.prompt_builder import PromptBuilder
from src.reasoning.citation_generator import CitationGenerator
from src.reasoning.output_validator import OutputValidator
from src.llm.inference import OllamaLLM
from src.utils.logger import logger

class LegalRAGPipeline:
    def __init__(self, retriever=None, llm=None):
        self.retriever = retriever or HybridRetriever()
        self.context_builder = ContextBuilder()
        self.llm = llm or OllamaLLM()
    def answer_query(self, query: str, top_k: int = 5):
        if not isinstance(query, str):
            raise ValueError("Query must be a string.")

        query = query.strip()

        if not query:
            raise ValueError("Query cannot be empty.")

        if len(query) > MAX_QUERY_CHARS:
            raise ValueError(
                f"Query cannot exceed {MAX_QUERY_CHARS} characters."
            )

        logger.info("Processing legal query (%d characters)", len(query))

        retrieved = self.retriever.search(query, top_k=top_k) or []
        citations = CitationGenerator.generate(retrieved)

        scores = [
            c["score"]
            for c in citations
            if isinstance(c.get("score"), (int, float))
        ]

        confidence = (
            round(
                max(
                    0.0,
                    min(
                        100.0,
                        sum(scores[:3]) / len(scores[:3]) * 100,
                    ),
                ),
                2,
            )
            if scores
            else 0.0
        )

        if not retrieved:
            return {
                "query": query,
                "answer": "Insufficient legal context available.",
                "sources": [],
                "num_sources": 0,
                "context": "",
                "citations": [],
                "confidence": 0.0,
                "confidence_type": (
                    "estimated retrieval relevance (not a probability)"
                ),
            }

        context = self.context_builder.build_context(retrieved)

        if not context:
            return {
                "query": query,
                "answer": "Insufficient legal context available.",
                "sources": [],
                "num_sources": 0,
                "context": "",
                "citations": [],
                "confidence": 0.0,
                "confidence_type": (
                    "estimated retrieval relevance (not a probability)"
                ),
            }

        prompt = PromptBuilder.build_legal_prompt(
            query=query,
            context=context,
        )

        # The first generation plus the configured retries determines
        # the maximum number of LLM calls.
        max_attempts = MAX_OUTPUT_RETRIES + 1

        answer = INVALID_OUTPUT_FALLBACK

        for attempt in range(1, max_attempts + 1):
            if attempt == 1:
                generate_with_context = getattr(
                    self.llm,
                    "generate_with_context",
                    None,
                )
                if callable(generate_with_context):
                    answer = generate_with_context(
                        query=query,
                        context=context,
                    )
                else:
                    answer = self.llm.generate(prompt)
            else:
                answer = self.llm.generate(prompt)
            validation_result = OutputValidator.validate(answer)

            status = validation_result["status"]

            if status == "valid":
                logger.info(
                    "LLM output passed structural validation "
                    "on attempt %d/%d",
                    attempt,
                    max_attempts,
                )
                break

            if status == "insufficient_evidence":
                logger.info(
                    "LLM returned the insufficient-evidence response"
                )
                break

            logger.warning(
                "LLM output failed structural validation "
                "on attempt %d/%d: %s",
                attempt,
                max_attempts,
                validation_result,
            )

            if attempt == max_attempts:
                logger.error(
                    "LLM output remained structurally invalid after "
                    "%d attempts; returning controlled fallback",
                    max_attempts,
                )
                answer = INVALID_OUTPUT_FALLBACK
                break

            # Ask the LLM to repair the structure without changing
            # the original question or retrieved legal context.
            correction_prompt = (
                f"{prompt}\n\n"
                "IMPORTANT: Your previous response failed structural "
                "validation. Generate a corrected, complete response "
                "using the original legal question and supplied context.\n"
                "Do not invent legal authorities, citations, quotations, "
                "or facts.\n"
                "Include these headings as standalone lines, in exactly "
                "this order:\n"
                "LEGAL ISSUE:\n"
                "RELEVANT AUTHORITIES:\n"
                "LEGAL REASONING:\n"
                "CONCLUSION:\n"
                "LIMITATIONS:\n"
                f"Validation status: {status}\n"
                f"Missing sections: "
                f"{validation_result['missing_sections']}\n"
                f"Empty sections: "
                f"{validation_result['empty_sections']}\n"
                "Ensure every required section contains relevant content. "
                "If the supplied evidence is insufficient, return exactly: "
                "Insufficient legal context available."
            )

            prompt = correction_prompt

        return {
            "query": query,
            "answer": answer,
            "sources": retrieved,
            "num_sources": len(retrieved),
            "context": context,
            "citations": citations,
            "confidence": confidence,
            "confidence_type": (
                "estimated retrieval relevance (not a probability)"
            ),
        }

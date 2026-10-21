from src.retrieval.hybrid_retriever import (
    HybridRetriever
)

from src.reasoning.context_builder import (
    ContextBuilder
)

from src.reasoning.context_compressor import (
    ContextCompressor
)

from src.reasoning.prompt_builder import (
    PromptBuilder
)

from src.llm.inference import (
    OllamaLLM
)

from src.utils.logger import (
    logger
)

from src.reasoning.citation_generator import (
    CitationGenerator
)


class LegalRAGPipeline:

    def __init__(self):

        self.retriever = (
            HybridRetriever()
        )

        self.compressor = (
            ContextCompressor()
        )

        self.context_builder = (
            ContextBuilder()
        )

        self.llm = (
            OllamaLLM()
        )

    def answer_query(
        self,
        query: str,
        top_k: int = 5
    ):

        logger.info(
            f"Received query: {query}"
        )

        # ------------------
        # Retrieval
        # ------------------

        retrieved_chunks = (
            self.retriever.search(
                query=query,
                top_k=top_k
            )
        )

        if not retrieved_chunks:

            return {
                "query": query,
                "answer":
                    "No relevant legal context found.",
                "sources": []
            }

        logger.info(
            f"Retrieved "
            f"{len(retrieved_chunks)} chunks"
        )

        # ------------------
        # Context Compression
        # ------------------

        compressed_chunks = (
            self.compressor.compress(
                retrieved_chunks
            )
        )

        # ------------------
        # Context Building
        # ------------------

        print("\nDEBUG RETRIEVED CHUNKS")
        print(type(retrieved_chunks))
        
        if retrieved_chunks:
            print(type(retrieved_chunks[0]))
            print(retrieved_chunks[0])

        context = (
            self.context_builder
            .build_context(
                retrieved_chunks
            )
        )

        logger.info(
            f"Context Length: "
            f"{len(context)} chars"
        )

        # ------------------
        # Prompt Building
        # ------------------

        prompt = (
            PromptBuilder
            .build_legal_prompt(
                query=query,
                context=context
            )
        )

        logger.info(
            f"Prompt Length: "
            f"{len(prompt)} chars"
        )

        logger.info(
            "Sending prompt to LLM"
        )

        # ------------------
        # LLM Generation
        # ------------------

        answer = (
            self.llm.generate(
                prompt
            )
        )

        # ------------------
        # Citations
        # ------------------

        citations = (
            CitationGenerator.generate(
                retrieved_chunks
            )
        )

        # ------------------
        # Confidence 
        # ------------------

        top_scores = [
             c["score"]
             for c in citations[:3]
        ]

        confidence = round(
            (
                sum(top_scores) / len(top_scores)
            ) * 100,
            2
        )


        logger.info(
            "Answer generated successfully"
        )

        return {

            "query": query,

            "answer": answer,

            "sources": retrieved_chunks,

            "num_sources": len(
                retrieved_chunks
            ),

            "context": context,

            "citations": citations,

            "confidence": confidence
        }
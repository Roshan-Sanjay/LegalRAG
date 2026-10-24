from src.config.settings import MAX_QUERY_CHARS
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
            raise ValueError(f"Query cannot exceed {MAX_QUERY_CHARS} characters.")
        logger.info("Processing legal query (%d characters)", len(query))
        retrieved = self.retriever.search(query, top_k=top_k) or []
        citations = CitationGenerator.generate(retrieved)
        scores = [c["score"] for c in citations if isinstance(c.get("score"), (int, float))]
        confidence = round(max(0.0, min(100.0, sum(scores[:3]) / len(scores[:3]) * 100)), 2) if scores else 0.0
        if not retrieved:
            return {"query": query, "answer": "Insufficient legal context available.", "sources": [], "num_sources": 0, "context": "", "citations": [], "confidence": 0.0, "confidence_type": "estimated retrieval relevance (not a probability)"}
        context = self.context_builder.build_context(retrieved)
        if not context:
            return {"query": query, "answer": "Insufficient legal context available.", "sources": [], "num_sources": 0, "context": "", "citations": [], "confidence": 0.0, "confidence_type": "estimated retrieval relevance (not a probability)"}
        prompt = PromptBuilder.build_legal_prompt(
            query=query,
            context=context
        )
        answer = self.llm.generate(prompt)
        validation_result = OutputValidator.validate(answer)
        if validation_result["status"] == "valid":
            logger.info("LLM output passed structural validation")
        elif validation_result["status"] == "insufficient_evidence":
            logger.info("LLM returned the insufficient-evidence response")
        else:
            logger.warning(
                "LLM output failed structural validation: %s",
                validation_result
            )
        return {"query": query, "answer": answer, "sources": retrieved, "num_sources": len(retrieved), "context": context, "citations": citations, "confidence": confidence, "confidence_type": "estimated retrieval relevance (not a probability)"}

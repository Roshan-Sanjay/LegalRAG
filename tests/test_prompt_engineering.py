from src.reasoning.prompt_templates import PromptTemplates
from src.reasoning.prompt_builder import PromptBuilder
def test_prompt_contains_required_output_sections():
    prompt = PromptTemplates.legal_reasoning_prompt(
        query="What is the legal issue?",
        context="Sample legal context."
    )
    required_sections = [

        "LEGAL ISSUE:",

        "RELEVANT AUTHORITIES:",

        "LEGAL REASONING:",

        "CONCLUSION:",

        "LIMITATIONS:",
    ]
    for section in required_sections:
        assert section in prompt
def test_prompt_substitutes_query_and_context():
    query = "Can an employer terminate an employee without notice?"
    context = "Case A: The court considered the requirement of notice."
    prompt = PromptTemplates.legal_reasoning_prompt(
        query=query,
        context=context
)

    assert query in prompt

    assert context in prompt
def test_prompt_contains_evidence_grounding_instructions():
    prompt = PromptTemplates.legal_reasoning_prompt(
    query="What legal principle applies?",
    context="Sample legal context."
    )
    assert "Do not invent case names, citations, statutes, quotations, facts," in prompt
    assert "holdings, or legal principles." in prompt
    assert "Support each material legal proposition" in prompt
    assert "whose supplied text supports that proposition." in prompt
    assert (
        "Distinguish information explicitly stated in the retrieved text from"
    ) in prompt
def test_prompt_contains_insufficient_evidence_instructions():
    prompt = PromptTemplates.legal_reasoning_prompt(
    query="What legal principle applies?",
    context="Sample legal context."
    )
    normalized_prompt = " ".join(prompt.split())
    assert "Insufficient legal context available." in normalized_prompt
    assert (
        "does not provide enough information to answer the central legal question"
    ) in normalized_prompt
    assert (
        "Do not use this fallback merely because a minor detail is missing."
    ) in normalized_prompt
    assert (
        "return only its exact fallback response instead of the structured sections."
    ) in normalized_prompt
def test_prompt_distinguishes_relevance_from_legal_correctness():
    prompt = PromptTemplates.legal_reasoning_prompt(
    query="What legal principle applies?",
    context="Sample legal context."
    )
    normalized_prompt = " ".join(prompt.split())
    assert (
        "Retrieval scores indicate estimated retrieval relevance only."
    ) in normalized_prompt
    assert (
        "They do not establish the correctness, precedential weight, or legal authority of a judgment."
    ) in normalized_prompt

def test_prompt_builder_uses_existing_template():
    query = "Can an employer terminate an employee without notice?"
    context = "Sample legal context: Case A discusses notice requirements."
    prompt = PromptBuilder.build_legal_prompt(
    query=query,
    context=context
    )

    normalized_prompt = " ".join(prompt.split())

    assert query in normalized_prompt
    assert context in normalized_prompt

    assert "LEGAL ISSUE:" in normalized_prompt
    assert "RELEVANT AUTHORITIES:" in normalized_prompt
    assert "LEGAL REASONING:" in normalized_prompt
    assert "CONCLUSION:" in normalized_prompt
    assert "LIMITATIONS:" in normalized_prompt

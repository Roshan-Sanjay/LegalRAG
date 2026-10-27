from unittest.mock import Mock

import httpx
import pytest
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from ollama import ResponseError

from src.llm.inference import OllamaLLM
from src.llm.model_loader import OllamaConfig
from src.reasoning.prompt_templates import PromptTemplates


def _install_chat_model(monkeypatch, handler=None):
    if handler is None:
        handler = lambda value, **kwargs: AIMessage(content="Generated answer")
    client = RunnableLambda(handler)
    factory = Mock(return_value=client)
    monkeypatch.setattr("src.llm.inference.ChatOllama", factory)
    return client, factory


def test_lcel_chain_composes_legal_template_chat_model_and_string_parser(
    monkeypatch,
):
    received = []

    def respond(prompt_value, **kwargs):
        received.append(prompt_value)
        return AIMessage(content="Generated answer")

    client, factory = _install_chat_model(monkeypatch, respond)
    llm = OllamaLLM()

    assert llm.chain.steps[0] is PromptTemplates.LEGAL_REASONING_TEMPLATE
    assert llm.chain.steps[1] is client
    assert isinstance(llm.chain.steps[2], StrOutputParser)
    assert llm.generate_with_context("Question text", "Context text") == (
        "Generated answer"
    )
    assert len(received) == 1
    assert "Question text" in received[0].to_string()
    assert "Context text" in received[0].to_string()
    factory.assert_called_once()


def test_generate_with_context_passes_query_and_context_to_chain(monkeypatch):
    client, _ = _install_chat_model(monkeypatch)
    llm = OllamaLLM()
    llm.chain = Mock()
    llm.chain.invoke.return_value = "Generated answer"

    answer = llm.generate_with_context("query value", "context value")

    assert answer == "Generated answer"
    llm.chain.invoke.assert_called_once_with(
        {"query": "query value", "context": "context value"}
    )
    assert client is llm.client


def test_generate_accepts_a_formatted_string_and_returns_plain_string(
    monkeypatch,
):
    received = []

    def respond(prompt, **kwargs):
        received.append((prompt, kwargs))
        return AIMessage(content="Generated answer")

    _install_chat_model(monkeypatch, respond)
    answer = OllamaLLM().generate("Fully formatted legal prompt")

    assert isinstance(answer, str)
    assert answer == "Generated answer"
    assert received == [("Fully formatted legal prompt", {"stream": False})]


@pytest.mark.parametrize("answer", ["", "  \n", None, ["invalid"]])
def test_generate_with_context_rejects_empty_or_invalid_outputs(
    monkeypatch,
    answer,
):
    _install_chat_model(monkeypatch)
    llm = OllamaLLM()
    llm.chain = Mock()
    llm.chain.invoke.return_value = answer

    with pytest.raises(RuntimeError, match="empty or invalid answer"):
        llm.generate_with_context("question", "context")


def test_chat_ollama_receives_configured_model_and_request_options(monkeypatch):
    _, factory = _install_chat_model(monkeypatch)
    monkeypatch.setattr("src.llm.inference.OLLAMA_BASE_URL", "http://ollama.test:11434")
    monkeypatch.setattr("src.llm.inference.OLLAMA_TIMEOUT", 45.0)
    monkeypatch.setattr(OllamaConfig, "MODEL_NAME", "configured-model")
    monkeypatch.setattr(OllamaConfig, "TEMPERATURE", 0.25)
    monkeypatch.setattr(OllamaConfig, "MAX_TOKENS", 768)

    OllamaLLM()

    factory.assert_called_once_with(
        model="configured-model",
        base_url="http://ollama.test:11434",
        temperature=0.25,
        num_predict=768,
        client_kwargs={"timeout": 45.0},
    )


@pytest.mark.parametrize("method", ["generate", "generate_with_context"])
@pytest.mark.parametrize(
    ("error", "message"),
    [
        (httpx.ConnectError("offline"), "Could not connect to Ollama"),
        (httpx.ReadTimeout("request timed out"), "timed out after 300s"),
        (
            ResponseError("model not found", status_code=404),
            "model/server error.*model not found",
        ),
    ],
)
def test_generation_methods_translate_model_errors_and_chain_causes(
    monkeypatch,
    method,
    error,
    message,
):
    def fail(value, **kwargs):
        raise error

    _install_chat_model(monkeypatch, fail)
    llm = OllamaLLM()
    generate = getattr(llm, method)

    with pytest.raises(RuntimeError, match=message) as raised:
        if method == "generate":
            generate("formatted prompt")
        else:
            generate("question", "context")

    assert raised.value.__cause__ is error

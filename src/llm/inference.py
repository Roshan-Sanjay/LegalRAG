import httpx
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from ollama import ResponseError

from src.config.settings import OLLAMA_BASE_URL, OLLAMA_TIMEOUT
from src.llm.model_loader import OllamaConfig
from src.reasoning.prompt_templates import PromptTemplates
from src.utils.logger import logger


class OllamaLLM:
    def __init__(self):
        self.model = OllamaConfig.MODEL_NAME
        self.timeout = OLLAMA_TIMEOUT
        self.client = ChatOllama(
            model=self.model,
            base_url=OLLAMA_BASE_URL,
            temperature=OllamaConfig.TEMPERATURE,
            num_predict=OllamaConfig.MAX_TOKENS,
            client_kwargs={"timeout": self.timeout},
        )
        self.chain = (
            PromptTemplates.LEGAL_REASONING_TEMPLATE
            | self.client
            | StrOutputParser()
        )

    def generate(self, prompt: str) -> str:
        logger.info("Calling Ollama model: %s", self.model)
        return self._generate(
            lambda: getattr(
                self.client.invoke(prompt, stream=False),
                "content",
                None,
            )
        )

    def generate_with_context(self, query: str, context: str) -> str:
        logger.info("Calling Ollama model: %s", self.model)
        return self._generate(
            lambda: self.chain.invoke({"query": query, "context": context})
        )

    def _generate(self, invoke):
        try:
            answer = invoke()
        except httpx.TimeoutException as exc:
            logger.error("Ollama request timed out")
            raise RuntimeError(
                f"Ollama timed out after {self.timeout:g}s. "
                "Check that Ollama is running and the model is available."
            ) from exc
        except httpx.ConnectError as exc:
            logger.error("Could not connect to Ollama")
            raise RuntimeError(
                "Could not connect to Ollama. Start Ollama and check "
                "OLLAMA_BASE_URL."
            ) from exc
        except ResponseError as exc:
            logger.error("Ollama returned a model or server error: %s", exc)
            raise RuntimeError(
                f"Ollama model/server error (status {exc.status_code}): "
                f"{exc.error}. Check that {self.model!r} is installed "
                "with `ollama list`."
            ) from exc
        except httpx.RequestError as exc:
            logger.error("Ollama request failed: %s", exc)
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError(
                "Ollama returned an empty or invalid answer; "
                "verify the configured model is installed."
            )

        logger.info("Answer generated successfully")
        return answer

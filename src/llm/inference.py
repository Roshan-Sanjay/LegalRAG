import requests

from src.llm.model_loader import OllamaConfig
from src.utils.logger import logger
from src.config.settings import OLLAMA_URL, OLLAMA_TIMEOUT


class OllamaLLM:

    def __init__(self):

        self.model = (
            OllamaConfig.MODEL_NAME
        )

        self.url = OLLAMA_URL
        self.timeout = OLLAMA_TIMEOUT

    def generate(self, prompt: str):
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": OllamaConfig.TEMPERATURE,
                "num_predict": OllamaConfig.MAX_TOKENS
            }
        }

        logger.info(
            f"Calling Ollama model: {self.model}"
        )

        try:
            response = requests.post(
                self.url,
                json=payload,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            try:
                result = response.json()
            except ValueError as exc:
                raise RuntimeError("Ollama returned malformed JSON.") from exc
            
            answer = result.get(
                "response",
                ""
            )

            if not isinstance(answer, str) or not answer.strip():
                raise RuntimeError("Ollama returned an empty answer; verify the configured model is installed.")
            logger.info("Answer generated successfully")
            return answer
        except requests.exceptions.Timeout as exc:
            logger.error("Ollama request timed out")
            raise RuntimeError(f"Ollama timed out after {self.timeout:g}s. Check that Ollama is running and the model is available.") from exc
        except requests.exceptions.ConnectionError as exc:
            logger.error("Could not connect to Ollama")
            raise RuntimeError("Could not connect to Ollama. Start Ollama and check OLLAMA_BASE_URL.") from exc
        except requests.exceptions.HTTPError as exc:
            detail = response.text[:500] if response is not None else str(exc)
            raise RuntimeError(f"Ollama request failed ({response.status_code}): {detail}. Check that {self.model!r} is installed with `ollama list`.") from exc
        except requests.exceptions.RequestException as exc:
            logger.error("Ollama request failed: %s", exc)
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

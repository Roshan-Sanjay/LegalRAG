from unittest import result

import requests

from src.llm.model_loader import OllamaConfig
from src.utils.logger import logger
from src.config.settings import OLLAMA_URL


class OllamaLLM:

    def __init__(self):

        self.model = (
            OllamaConfig.MODEL_NAME
        )

        self.url = OLLAMA_URL

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
                timeout=300
            )
            
            if response.status_code != 200:
                print("\nOLLAMA ERROR:")
                print(response.text)
                
                response.raise_for_status()
            
            result = response.json()

            print("\n" + "="*80)
            print("RAW OLLAMA JSON")
            print("="*80)
            print(result)

            print("\n" + "="*80)
            print("RAW ANSWER")
            print("="*80)
            print(result.get("response", ""))
            
            answer = result.get(
                "response",
                ""
            )

            logger.info(
                "Answer generated successfully"
            )
            
            return answer
        
        except requests.exceptions.RequestException as e:
            
            logger.error(
                f"Ollama request failed: {e}"
            )

            return (
                f"LLM Error: {e}"
            )
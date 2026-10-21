from src.reasoning.prompt_templates import (
    PromptTemplates
)


class PromptBuilder:

    @staticmethod
    def build_legal_prompt(
        query,
        context
    ):

        prompt = (
            PromptTemplates
            .legal_reasoning_prompt(
                query=query,
                context=context
            )
        )

        print(
            "\nPrompt Length:",
            len(prompt)
        )

        return prompt
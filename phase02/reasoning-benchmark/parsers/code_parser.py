import re


class CodeParser:
    """
    Utility class for extracting executable Python code from an LLM response.
    """

    PYTHON_BLOCK_PATTERN = re.compile(
        r"```(?:python)?\s*(.*?)```",
        re.DOTALL,
    )

    @staticmethod
    def extract_python_code(text: str) -> str:
        """
        Extract Python code from a Markdown code block.
        If no code block exists, return the original text.
        """

        match = CodeParser.PYTHON_BLOCK_PATTERN.search(text)

        if match:
            return match.group(1).strip()

        return text.strip()

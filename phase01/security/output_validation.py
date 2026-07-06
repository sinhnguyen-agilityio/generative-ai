import re

from phase01.schemas import ClassificationResult


class OutputValidator:
    def __init__(self):
        self.suspicious_patterns = [
            r'SYSTEM\s*[:]\s*You\s+are',     # System prompt leakage
            r'API[_\s]KEY[:=]\s*\w+',        # API key exposure
            r'instructions?[:]\s*\d+\.',     # Numbered instructions
        ]

    def validate_output(self, output: str) -> bool:
        return not any(re.search(pattern, output, re.IGNORECASE)
                       for pattern in self.suspicious_patterns)

    def filter_response(self, response: ClassificationResult) -> str:
        if not self.validate_output(response.category) or len(response.reason) > 5000:
            return "I cannot provide that information for security reasons."
        return """The output has been validated and is safe to use. The category is: {}. Reason: {}""".format(
            response.category, response.reason)

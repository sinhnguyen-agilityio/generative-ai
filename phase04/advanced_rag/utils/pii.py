from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig


class PIIMasker:
    def __init__(self):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

    def mask(self, text: str) -> str:
        results = self.analyzer.analyze(
            text=text,
            language="en",
        )

        operators = {
            "PERSON": OperatorConfig(
                "replace",
                {"new_value": "<PERSON>"},
            ),
            "EMAIL_ADDRESS": OperatorConfig(
                "replace",
                {"new_value": "<EMAIL>"},
            ),
            "PHONE_NUMBER": OperatorConfig(
                "replace",
                {"new_value": "<PHONE>"},
            ),
            "LOCATION": OperatorConfig(
                "replace",
                {"new_value": "<LOCATION>"},
            ),
        }

        result = self.anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators=operators,
        )

        return result.text

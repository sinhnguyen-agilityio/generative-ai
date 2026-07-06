from phase01.schemas import ClassificationResult, PipelineAction, RiskLevel
from phase01.security.input_filter import InputFilter
from phase01.security.risk_score import RiskScorer
from phase01.llm.classifier import SecureClassifier, StrangeSequence
from phase01.security.output_validation import OutputValidator
import sys


class SecureLLMPipeline:
    def __init__(self, input_filter: InputFilter, risk_scorer: RiskScorer, classifier: SecureClassifier, output_validator: OutputValidator):
        self.input_filter = input_filter
        self.risk_scorer = risk_scorer
        self.classifier = classifier
        self.output_validator = output_validator

    def process(self, text: str) -> ClassificationResult:
        security_report = self.input_filter.scan(text)
        risk_assessment = self.risk_scorer.evaluate(security_report)

        if risk_assessment.action == PipelineAction.REJECT:
            raise ValueError("Input rejected due to high risk assessment.")
        elif risk_assessment.action == PipelineAction.ALLOW_WITH_LOGGING:
            # Log the risk assessment details for auditing purposes
            print(f"Risk Assessment: {risk_assessment}")

        classification = self.classifier.classify(security_report.clean_text)

        result = ClassificationResult.model_validate({
            "category": classification.category,
            "confidence": classification.confidence,
            "reason": classification.reason
        })
        self.output_validator.filter_response(result)

        return result


class SimpleSecurityPipeline:
    def __init__(self, input_filter: InputFilter, strangeSequence: StrangeSequence):
        self.input_filter = input_filter
        self.strangeSequence = strangeSequence

    def process(self, text: str) -> str:
        security_report = self.input_filter.scan(text)
        result = self.strangeSequence.analyze(security_report.clean_text)

        return result

import re

from schemas import Finding, FindingType, SecurityReport
from config import PROMPT_INJECTION_PATTERNS, MAX_WORD_COUNT

from security.fuzzy_match import contains_scrambled_keyword
from security.unicode_utils import clean_text


class InputFilter:
    def __init__(self) -> None:
        self.patterns = self._compile_patterns()

    def scan(self, text: str) -> SecurityReport:
        cleaned_text = self.sanitize(text)
        length_finding = self._check_length(cleaned_text)
        regex_findings = self._detect_regex_patterns(cleaned_text)
        typoglycemia_finding = self._detect_typoglycemia(cleaned_text)

        findings = [f for f in [length_finding,
                                typoglycemia_finding] if f] + regex_findings

        return self._build_report(text, cleaned_text, findings)

    def _compile_patterns(self) -> list[re.Pattern]:
        return [re.compile(pattern, re.IGNORECASE) for pattern in PROMPT_INJECTION_PATTERNS]

    def _check_length(self, text: str) -> Finding | None:
        if len(text) > MAX_WORD_COUNT:
            return Finding(
                type=FindingType.LENGTH,
                severity=1,
                description=f"Input exceeds maximum length of {MAX_WORD_COUNT} characters",
                matched_text=text,
                confidence=1.0,
            )

        return None

    def sanitize(self, text: str) -> str:
        return clean_text(text)

    def _detect_regex_patterns(self, text: str) -> list[Finding]:
        findings: list[Finding] = []

        if not self.patterns:
            self.patterns = [re.compile(pattern, re.IGNORECASE)
                             for pattern in PROMPT_INJECTION_PATTERNS]

        for pattern in self.patterns:
            match = pattern.search(text)
            if match:
                findings.append(Finding(
                    type=FindingType.PROMPT_INJECTION,
                    severity=2,
                    description=f"Detected prompt injection pattern: {pattern.pattern}",
                    matched_text=match.group(),
                    confidence=1.0,
                ))

        return findings

    def _detect_typoglycemia(self, text: str) -> Finding | None:
        # Placeholder for typoglycemia detection logic
        if contains_scrambled_keyword(text):
            return Finding(
                type=FindingType.TYPOGLYCEMIA,
                severity=1,
                description="Detected scrambled keywords indicative of typoglycemia",
                matched_text=text,
                confidence=0.9,
            )

        return None

    def _build_report(self, original_text: str, clean_text: str, findings: list[Finding]) -> SecurityReport:

        return SecurityReport(
            original_text=original_text,
            clean_text=clean_text,
            findings=findings,
        )

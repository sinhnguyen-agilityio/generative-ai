import re

from models.pii import (
    PIIPattern,
    PIIEntity,
    PIIType,
)
from models.ticket import SecureTicket


class PIIDetector:
    PATTERNS = [
        PIIPattern(
            pattern=re.compile(
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
            ),
            pii_type=PIIType.EMAIL,
            placeholder="<EMAIL>",
        ),
        PIIPattern(
            pattern=re.compile(
                r"(?:\+?\d{1,3}[-.\s]?)?"
                r"(?:\(?\d{2,4}\)?[-.\s]?)?"
                r"\d{3,4}[-.\s]?\d{3,4}"
            ),
            pii_type=PIIType.PHONE_NUMBER,
            placeholder="<PHONE_NUMBER>",
        ),
        PIIPattern(
            pattern=re.compile(
                r"\b(?:\d[ -]*?){13,19}\b"
            ),
            pii_type=PIIType.CREDIT_CARD,
            placeholder="<CREDIT_CARD>",
        ),
    ]

    def detect(self, text: str) -> SecureTicket:
        """Detect and mask supported PII."""

        entities: list[PIIEntity] = []

        sanitized_text = text

        for pattern in self.PATTERNS:
            sanitized_text = self._replace_pattern(
                text=sanitized_text,
                config=pattern,
                entities=entities,
            )

        return SecureTicket(
            original_text=text,
            sanitized_text=sanitized_text,
            pii_detected=bool(entities),
            pii_entities=entities,
        )

    def _replace_pattern(
        self,
        text: str,
        config: PIIPattern,
        entities: list[PIIEntity],
    ) -> str:

        def replacer(match: re.Match[str]) -> str:

            entities.append(
                PIIEntity(
                    type=config.pii_type,
                    original_value=match.group(0),
                    placeholder=config.placeholder,
                )
            )

            return config.placeholder

        return config.pattern.sub(replacer, text)

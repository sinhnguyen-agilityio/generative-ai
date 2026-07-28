from enum import Enum
from pydantic import BaseModel, Field
from dataclasses import dataclass
import re


class PIIType(str, Enum):
    EMAIL = "EMAIL"
    PHONE_NUMBER = "PHONE_NUMBER"
    CREDIT_CARD = "CREDIT_CARD"
    API_KEY = "API_KEY"


class PIIEntity(BaseModel):
    type: PIIType = Field(
        description="Detected PII type."
    )
    original_value: str = Field(
        description="Original detected value."
    )
    placeholder: str = Field(
        description="Replacement placeholder."
    )


class PIIDetectionResult(BaseModel):
    detected: bool = Field(
        description="Whether any PII was detected."
    )
    entities: list[PIIEntity] = Field(
        default_factory=list,
        description="Detected PII entities."
    )
    sanitized_text: str = Field(
        description="Text after masking sensitive information."
    )


@dataclass(frozen=True)
class PIIPattern:
    pattern: re.Pattern[str]
    pii_type: PIIType
    placeholder: str

from pydantic import BaseModel, Field
from models.pii import PIIEntity


class SecureTicket(BaseModel):
    original_text: str = Field(
        description="Original customer message."
    )
    sanitized_text: str = Field(
        description="Customer message after masking PII."
    )
    pii_detected: bool = False

    pii_entities: list[PIIEntity] = Field(
        default_factory=list
    )
    toxicity_detected: bool = False
    moderation_categories: list[str] = Field(
        default_factory=list
    )
    blocked: bool = False

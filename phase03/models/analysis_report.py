
from models.enum import SentimentLabel, IntentCategory
from pydantic import BaseModel, Field
from models.enum import UrgencyLevel, EscalationTeam


class Summary(BaseModel):
    """
    Brief summary of the customer's issue.
    """
    text: str = Field(
        description="A concise summary of the customer's request in 1-2 sentences."
    )


class Intent(BaseModel):
    """
    Primary purpose of the customer message.
    """
    category: IntentCategory = Field(
        description="Primary intent category."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Model confidence score between 0 and 1."
    )


class Sentiment(BaseModel):
    """
    Overall emotional tone.
    """
    label: SentimentLabel = Field(
        description="Overall customer sentiment."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Model confidence score between 0 and 1."
    )


class Entities(BaseModel):
    """
    Important business entities extracted from the ticket.
    """
    customer_name: str | None = Field(
        default=None,
        description="Customer name if mentioned."
    )
    product_names: list[str] = Field(
        default_factory=list,
        description="Mentioned products or services."
    )
    order_ids: list[str] = Field(
        default_factory=list,
        description="Order IDs mentioned in the ticket."
    )
    ticket_ids: list[str] = Field(
        default_factory=list,
        description="Support ticket IDs."
    )
    account_ids: list[str] = Field(
        default_factory=list,
        description="Customer account IDs."
    )
    dates: list[str] = Field(
        default_factory=list,
        description="Dates mentioned by the customer."
    )


# models/analysis_report.py


class Urgency(BaseModel):
    """
    Indicates how quickly the ticket should be handled.
    """
    level: UrgencyLevel = Field(
        description="Urgency level of the customer request."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1."
    )
    reason: str = Field(
        description="Short explanation for the urgency."
    )


class Escalation(BaseModel):
    """
    Indicates whether the ticket should be escalated.
    """
    required: bool = Field(
        description="Whether the ticket requires escalation."
    )
    team: EscalationTeam = Field(
        description="Team responsible for handling the escalation."
    )
    reason: str = Field(
        description="Reason for escalation."
    )


class SuggestedReply(BaseModel):
    """
    AI-generated response to the customer.
    """
    subject: str = Field(
        description="Suggested email subject."
    )
    message: str = Field(
        description="Professional customer response."
    )


class AnalysisReport(BaseModel):
    summary: Summary
    intent: Intent
    sentiment: Sentiment
    entities: Entities
    urgency: Urgency | None = None
    escalation: Escalation | None = None
    suggested_reply: SuggestedReply | None = None

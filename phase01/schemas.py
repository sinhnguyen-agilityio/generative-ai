from __future__ import annotations
from enum import Enum
from typing import List
from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class PipelineAction(str, Enum):
    ALLOW = "ALLOW"
    ALLOW_WITH_LOGGING = "ALLOW_WITH_LOGGING"
    REJECT = "REJECT"


class FindingType(str, Enum):
    TYPOGLYCEMIA = "TYPOGLYCEMIA"
    MIXED_SCRIPT = "MIXED_SCRIPT"
    CONTROL_CHARACTER = "CONTROL_CHARACTER"
    LENGTH = "LENGTH"
    PROMPT_INJECTION = "PROMPT_INJECTION"


class Category(str, Enum):
    TECHNOLOGY = "Technology"
    HISTORY = "History"
    FINANCE = "Finance"
    GARDENING = "Gardening"
    SPORTS = "Sports"
    PROMPT_INJECTION_ATTEMPT = "Prompt Injection Attempt"
    OTHER = "Other"


class Finding(BaseModel):
    type: FindingType
    severity: int
    description: str
    matched_text: str | None
    confidence: float


class SecurityReport(BaseModel):
    original_text: str
    clean_text: str
    findings: List[Finding]

    @property
    def has_findings(self):
        return bool(self.findings)


class RiskAssessment(BaseModel):
    score: float
    level: RiskLevel
    action: PipelineAction
    reasons: List[str]


class ClassificationResult(BaseModel):
    category: Category
    confidence: float
    reason: str


class LLMResponse(BaseModel):
    category: Category
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score",
    )
    reason: str
    processing_time: int


class PipelineResult(BaseModel):
    success: bool
    classification: LLMResponse
    risk_assessment: RiskAssessment
    security_report: SecurityReport
    message: str

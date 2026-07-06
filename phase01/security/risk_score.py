from phase01.config import RISK_SCORES
from phase01.schemas import Finding, FindingType, SecurityReport, RiskAssessment, RiskLevel, PipelineAction


class RiskScorer:
    def evaluate(self, report: SecurityReport) -> RiskAssessment:
        total_score = self._calculate_total(report.findings)
        level = self._determine_level(total_score)
        action = self._determine_action(level)

        return RiskAssessment(
            score=total_score,
            level=level,
            action=action,
            reasons=[finding.description for finding in report.findings]
        )

    def _score_finding(self, finding: Finding) -> float:
        if finding.type.value == FindingType.PROMPT_INJECTION.value:
            return RISK_SCORES["prompt_injection"]
        elif finding.type.value == FindingType.TYPOGLYCEMIA.value:
            return RISK_SCORES["typoglycemia"]
        elif finding.type.value == FindingType.LENGTH.value:
            return RISK_SCORES["length"]
        elif finding.type.value == FindingType.MIXED_SCRIPT.value:
            return RISK_SCORES["mixed_script"]
        elif finding.type.value == FindingType.CONTROL_CHARACTER.value:
            return RISK_SCORES["control_character"] 
        else:
            return 0.0

    def _calculate_total(self, findings: list[Finding]) -> float:
        return sum(self._score_finding(finding) for finding in findings)

    def _determine_level(self, score: float) -> RiskLevel:
        if score >= 80:
            return RiskLevel.CRITICAL
        elif score >= 50:
            return RiskLevel.HIGH
        elif score >= 20:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _determine_action(self, level: RiskLevel) -> PipelineAction:
        if level == RiskLevel.CRITICAL:
            return PipelineAction.REJECT
        elif level == RiskLevel.HIGH:
            return PipelineAction.REJECT
        elif level == RiskLevel.MEDIUM:
            return PipelineAction.ALLOW_WITH_LOGGING
        else:
            return PipelineAction.ALLOW

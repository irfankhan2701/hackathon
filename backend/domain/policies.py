import re
from typing import Tuple, Dict, Any
from domain.entities import DecisionStatus, DecisionFormulaComponents


class EvidenceDecisionPolicy:
    """
    Deterministic Evidence Decision Engine Rules.
    Determines ACCEPTED, REVIEW_REQUIRED, ILLEGIBLE, or CROSSED_OUT state
    without relying on LLM generation.
    """

    def __init__(
        self,
        accepted_score_threshold: float = 0.70,
        accepted_max_uncertainty: float = 0.35,
        review_score_threshold: float = 0.45,
        review_max_uncertainty: float = 0.65,
    ):
        self.accepted_score_threshold = accepted_score_threshold
        self.accepted_max_uncertainty = accepted_max_uncertainty
        self.review_score_threshold = review_score_threshold
        self.review_max_uncertainty = review_max_uncertainty

    def is_critical_entity(self, text: str) -> bool:
        """Checks if word/text contains critical numbers, dates, currency, or codes."""
        # Numbers, dates (12/03/2026), currency ($100), unit (500mg)
        pattern = r"(\d+|[\$/€/£]|\d+/\d+|\d+-\d+|\b\d+\s*(mg|ml|kg|g|tablet|tab)\b)"
        return bool(re.search(pattern, text, re.IGNORECASE))

    def evaluate(
        self,
        text: str,
        final_score: float,
        uncertainty: float,
        is_crossed_out_candidate: bool = False,
        components: DecisionFormulaComponents = None,
    ) -> Tuple[DecisionStatus, str]:
        # Rule 1: Crossed out text retains separate state
        if is_crossed_out_candidate:
            return (
                DecisionStatus.CROSSED_OUT,
                "Visual strike-through stroke detected across ink. Text retained as crossed out.",
            )

        # Rule 2: Critical entity strictness (numbers, dates, dosage)
        is_critical = self.is_critical_entity(text)
        if is_critical and uncertainty > 0.20:
            return (
                DecisionStatus.REVIEW_REQUIRED,
                f"Critical numeric/entity token '{text}' with moderate uncertainty ({uncertainty:.2f}). Flagged for verification.",
            )

        # Rule 3: High uncertainty or very low visual score -> ILLEGIBLE
        if final_score < self.review_score_threshold or uncertainty > self.review_max_uncertainty:
            return (
                DecisionStatus.ILLEGIBLE,
                f"Visual evidence insufficient or candidate conflict high (Score: {final_score:.2f}, Uncertainty: {uncertainty:.2f}).",
            )

        # Rule 4: Strong consensus and low uncertainty -> ACCEPTED
        if final_score >= self.accepted_score_threshold and uncertainty <= self.accepted_max_uncertainty:
            return (
                DecisionStatus.ACCEPTED,
                f"High visual evidence consensus and low uncertainty (Score: {final_score:.2f}, Uncertainty: {uncertainty:.2f}).",
            )

        # Rule 5: Moderate score / uncertainty -> REVIEW_REQUIRED
        return (
            DecisionStatus.REVIEW_REQUIRED,
            f"Moderate visual evidence or variant disagreement (Score: {final_score:.2f}, Uncertainty: {uncertainty:.2f}).",
        )


default_decision_policy = EvidenceDecisionPolicy()

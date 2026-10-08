import math
import unicodedata
from typing import List, Dict, Tuple, Any
import numpy as np
from domain.entities import (
    Hypothesis,
    WordEvidence,
    WordCandidate,
    DecisionStatus,
    DecisionFormulaComponents,
    BoundingBox,
    Line,
    QualityMetrics,
)
from domain.policies import default_decision_policy


def normalize_candidate_text(text: str) -> str:
    """Applies Unicode normalization (NFC) and whitespace cleanup without semantic edits."""
    if not text:
        return ""
    # NFC Unicode normalization
    text = unicodedata.normalize("NFC", text)
    # Normalize multiple whitespace into single space while preserving punctuation
    words = text.split()
    return " ".join(words)


def lev_distance(s1: str, s2: str) -> int:
    """Computes Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return lev_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


class MultiHypothesisFusionEngine:
    """
    Multi-Hypothesis Fusion and Deterministic Evidence Scoring Engine.
    Fuses line hypotheses across 4 image variants, computes calibrated scores and uncertainty,
    and constructs word-level provenance records.
    """

    def fuse_line_hypotheses(
        self,
        line: Line,
        hypotheses: List[Hypothesis],
        quality_metrics: QualityMetrics,
        optional_verifier_result: Any = None,
    ) -> List[WordEvidence]:
        if not hypotheses:
            # Fallback when no hypotheses returned
            dummy_bbox = line.bounding_box
            return [
                WordEvidence(
                    line_id=line.line_id,
                    word_index=0,
                    text="[ILLEGIBLE]",
                    bounding_box=dummy_bbox,
                    source_crop_id=line.line_crop_artifact_id,
                    candidates=[],
                    confidence=0.0,
                    uncertainty=1.0,
                    decision_status=DecisionStatus.ILLEGIBLE,
                    decision_reason="No OCR candidates produced by model for crop.",
                    raw_text="[ILLEGIBLE]",
                )
            ]

        # 1. Normalize hypotheses text
        cleaned_hyps: List[Tuple[Hypothesis, str]] = []
        for h in hypotheses:
            norm_txt = normalize_candidate_text(h.text)
            if norm_txt:
                cleaned_hyps.append((h, norm_txt))

        if not cleaned_hyps:
            cleaned_hyps = [(hypotheses[0], "[ILLEGIBLE]")]

        # 2. Extract Top line hypothesis & variants agreement
        top_hyp, top_text = max(cleaned_hyps, key=lambda pair: pair[0].score)

        variant_names = set(h.variant_name for h, _ in cleaned_hyps)
        num_variants = max(1, len(variant_names))

        # Check variant consensus for top text
        matching_variants = set()
        for h, txt in cleaned_hyps:
            if lev_distance(txt.lower(), top_text.lower()) <= max(1, int(len(top_text) * 0.2)):
                matching_variants.add(h.variant_name)

        variant_agreement = float(len(matching_variants) / num_variants)

        # 3. Calculate Character Stability across top candidate pool
        top_5_texts = [txt for _, txt in cleaned_hyps[:5]]
        if len(top_5_texts) > 1:
            dists = [
                lev_distance(top_5_texts[0], t) / max(1, len(top_5_texts[0]))
                for t in top_5_texts[1:]
            ]
            character_stability = float(max(0.0, 1.0 - np.mean(dists)))
        else:
            character_stability = 1.0

        # 4. Compute Ensemble Disagreement & Decoder Entropy
        all_scores = [h.score for h, _ in cleaned_hyps]
        normalized_visual_score = float(np.mean(all_scores[:3])) if all_scores else 0.5
        
        # Entropy over top scores
        prob_dist = np.exp(all_scores[:5]) / max(1e-6, np.sum(np.exp(all_scores[:5])))
        decoder_entropy = float(min(1.0, -np.sum(prob_dist * np.log(prob_dist + 1e-6)) / math.log(5)))

        ensemble_disagreement = float(1.0 - character_stability)

        # Image Quality Risk from metrics
        image_quality_risk = 0.1
        if quality_metrics:
            if quality_metrics.blur_score < 100:
                image_quality_risk += 0.3
            if quality_metrics.contrast_score < 25:
                image_quality_risk += 0.3
        image_quality_risk = min(1.0, image_quality_risk)

        # Optional Verifier metrics
        optional_context_score = 0.5
        optional_verifier_agreement = 0.5
        verifier_disagreement = 0.0

        if optional_verifier_result and hasattr(optional_verifier_result, "selected_candidate_index"):
            if optional_verifier_result.status == "supported":
                optional_verifier_agreement = 1.0
                verifier_disagreement = 0.0
            else:
                optional_verifier_agreement = 0.0
                verifier_disagreement = 0.5

        # 5. Compute Final Score & Uncertainty via required formulas
        final_score = (
            0.45 * normalized_visual_score
            + 0.25 * variant_agreement
            + 0.15 * character_stability
            + 0.10 * optional_context_score
            + 0.05 * optional_verifier_agreement
        )
        final_score = float(round(max(0.0, min(1.0, final_score)), 3))

        uncertainty = (
            0.40 * decoder_entropy
            + 0.30 * ensemble_disagreement
            + 0.20 * image_quality_risk
            + 0.10 * verifier_disagreement
        )
        uncertainty = float(round(max(0.0, min(1.0, uncertainty)), 3))

        formula_comp = DecisionFormulaComponents(
            visual_score=round(normalized_visual_score, 3),
            variant_agreement=round(variant_agreement, 3),
            character_stability=round(character_stability, 3),
            context_score=round(optional_context_score, 3),
            verifier_agreement=round(optional_verifier_agreement, 3),
            decoder_entropy=round(decoder_entropy, 3),
            ensemble_disagreement=round(ensemble_disagreement, 3),
            image_quality_risk=round(image_quality_risk, 3),
            verifier_disagreement=round(verifier_disagreement, 3),
        )

        # 6. Break line into word tokens and map candidates
        words_in_top = top_text.split()
        if not words_in_top:
            words_in_top = ["[ILLEGIBLE]"]

        word_evidences: List[WordEvidence] = []
        l_box = line.bounding_box
        num_words = len(words_in_top)
        w_step = l_box.w / max(1, num_words)

        # Build candidate candidates pool
        all_word_candidates: List[WordCandidate] = []
        for h, txt in cleaned_hyps[:12]:
            all_word_candidates.append(
                WordCandidate(
                    text=txt,
                    score=float(round(h.score, 3)),
                    variant_name=h.variant_name,
                    model_name=h.model_name,
                )
            )

        for w_idx, w_text in enumerate(words_in_top):
            # Calculate word bounding box slice along text line
            word_bbox = BoundingBox(
                x=float(round(l_box.x + (w_idx * w_step), 1)),
                y=l_box.y,
                w=float(round(w_step, 1)),
                h=l_box.h,
            )

            # Evaluate decision status using policy
            status, reason = default_decision_policy.evaluate(
                text=w_text,
                final_score=final_score,
                uncertainty=uncertainty,
                is_crossed_out_candidate=line.is_crossed_out_candidate,
                components=formula_comp,
            )

            word_ev = WordEvidence(
                line_id=line.line_id,
                word_index=w_idx,
                text=w_text if status != DecisionStatus.ILLEGIBLE else "[ILLEGIBLE]",
                bounding_box=word_bbox,
                source_crop_id=line.line_crop_artifact_id,
                candidates=all_word_candidates,
                confidence=final_score,
                uncertainty=uncertainty,
                decision_status=status,
                decision_reason=reason,
                raw_text=w_text,
                formula_components=formula_comp,
            )
            word_evidences.append(word_ev)

        return word_evidences


fusion_engine = MultiHypothesisFusionEngine()

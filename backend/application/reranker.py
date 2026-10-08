import re
from typing import List, Dict, Tuple, Optional
from domain.entities import WordEvidence, WordCandidate


class ContextReranker:
    """
    Safe Contextual Reranker and Format Normalizer.
    Enforces the strict rule: context may only rank existing visual OCR candidates
    or perform format-only edits (e.g. 500mg -> 500 mg, 12 / 03 / 2026 -> 12/03/2026).
    Never invents new unsupported words.
    """

    def __init__(self):
        # Generic dictionary sets for fast lookup
        self.known_units = {"mg", "ml", "g", "kg", "mm", "cm", "m", "tablet", "tablets", "tabs", "capsule", "hrs", "mins"}
        self.known_abbreviations = {"rx", "bp", "hr", "dob", "qty", "no", "id", "dr", "mr", "mrs", "ms", "refill", "note"}

    def format_normalize_token(self, token: str) -> Tuple[str, bool]:
        """Applies format-only cleanup rules."""
        original = token
        modified = original

        # Rule 1: Spacing between digits and units: 500mg -> 500 mg
        modified = re.sub(r"(\d+)([a-zA-Z]+)", r"\1 \2", modified)

        # Rule 2: Date spacing normalization: 12 / 03 / 2026 -> 12/03/2026
        modified = re.sub(r"(\d{1,2})\s*/\s*(\d{1,2})\s*/\s*(\d{2,4})", r"\1/\2/\3", modified)

        # Rule 3: Time or fraction spacing: 12 : 30 -> 12:30
        modified = re.sub(r"(\d{1,2})\s*:\s*(\d{2})", r"\1:\2", modified)

        is_changed = modified != original
        return modified, is_changed

    def rerank_and_format_words(
        self, words: List[WordEvidence]
    ) -> Tuple[List[WordEvidence], List[Dict[str, str]]]:
        audit_events = []
        updated_words = []

        for word in words:
            # 1. Format-only normalization
            norm_text, was_formatted = self.format_normalize_token(word.text)

            if was_formatted:
                audit_events.append(
                    {
                        "word_id": word.word_id,
                        "type": "FORMAT_NORMALIZATION",
                        "original_text": word.text,
                        "reranked_text": norm_text,
                        "reason": "Applied safe format-only unit/date spacing rule.",
                    }
                )

            # 2. Check candidate pool reranking plausibility
            selected_text = norm_text
            if word.candidates:
                best_cand_text = word.candidates[0].text
                # If top candidate is in dictionary or has higher visual score, keep visual grounding
                for cand in word.candidates[:3]:
                    cand_clean = cand.text.strip().lower()
                    if cand_clean in self.known_units or cand_clean in self.known_abbreviations:
                        if cand.score > 0.85:
                            selected_text = cand.text
                            break

            # Create updated WordEvidence copy
            word_copy = word.model_copy()
            word_copy.text = selected_text
            word_copy.raw_text = word.raw_text
            updated_words.append(word_copy)

        return updated_words, audit_events


context_reranker = ContextReranker()

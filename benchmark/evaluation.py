from __future__ import annotations

import re
from typing import List, Optional
from pydantic import BaseModel

from core.questions import ReasoningType


# =========================================================
# PREDICTION NORMALIZER
# Sits between raw model output and Evaluator.
# Strips whitespace, punctuation, casing artifacts.
# =========================================================

class PredictionNormalizer:
    """
    Normalizes raw model output to a clean, comparable string.
    Applied before evaluation — never modifies EvaluationResult directly.
    """

    @staticmethod
    def normalize(raw: str) -> str:
        text = raw.strip()

        # Remove trailing punctuation
        while text and text[-1] in ".,;:!?":
            text = text[:-1]

        # Collapse whitespace
        text = " ".join(text.split())
        return text.lower()


# =========================================================
# PREDICTION EXTRACTOR
# Sits between raw model output and Evaluator.
#
# Rules:
# - Strip parenthetical content first (reasoning/explanation
#   often contains OTHER location names mentioned for context,
#   not the model's actual answer).
# - Case-insensitive match against valid_locations.
# - If multiple valid locations remain after stripping
#   parentheses → return the FIRST one (the model's stated
#   answer typically comes before its justification).
# - If nothing found outside parentheses, fall back to
#   scanning the full text (LAST match), then normalized text.
# =========================================================

class PredictionExtractor:
    INVALID_PREDICTION = "__invalid__"

    NEGATION_TERMS = [
        "not",
        "no",
        "never",
        "isn't",
        "is not",
        "not in",
        "no longer",
    ]

    ANSWER_LINE_PATTERNS = [
        r"\bfinal\s+answer\s*:\s*([^\n\r]*)",
        r"\banswer\s*:\s*([^\n\r]*)",
        r"\blocation\s*:\s*([^\n\r]*)",
    ]

    ANSWER_MARKER_PATTERNS = [
        r"\bfinal\s+answer\b\s*:?",
        r"\banswer\b\s*:?",
        r"\blocation\b\s*:?",
    ]

    ANSWER_CUE_PATTERNS = [
        r"\btherefore\b\s*[:,]?",
        r"\bconclusion\s*[:\-]?",
        r"\bhence\b\s*[:,]?",
        r"\bso\b\s*[:,]?",
    ]

    @staticmethod
    def _strip_parentheticals(text: str) -> str:
        return re.sub(r"\([^)]*\)", " ", text)

    @staticmethod
    def _location_pattern(location: str) -> str:
        """
        Match a full location token, not a substring.

        This prevents:
          ANSWER: Cup
        from being accepted as:
          cupboard

        and prevents:
          ANSWER: Box
        from being accepted as:
          box_b
        """
        escaped = re.escape(location.lower())
        return rf"(?<![a-z0-9_]){escaped}(?![a-z0-9_])"

    @staticmethod
    def _normalize_valid_locations(valid_locations: List[str]) -> set[str]:
        return {
            PredictionNormalizer.normalize(location)
            for location in valid_locations
        }

    @staticmethod
    def _is_negated_location(text: str, match_start: int, match_end: int) -> bool:
        """
        Lightweight negation filter.

        It is intentionally conservative: only skips a location when a clear
        negation cue appears immediately before it.

        Examples skipped:
          not Drawer
          not in Drawer
          is not Drawer
          no longer in Drawer

        Examples not skipped:
          Drawer, not Basket
        """
        normalized = text.lower()

        prefix_window = normalized[max(0, match_start - 32):match_start]
        compact_prefix = " ".join(prefix_window.split())

        negation_patterns = [
            r"\bnot\s+$",
            r"\bnot\s+in\s+$",
            r"\bis\s+not\s+$",
            r"\bisn't\s+$",
            r"\bno\s+$",
            r"\bno\s+longer\s+(?:in\s+)?$",
            r"\bnever\s+(?:in\s+)?$",
        ]

        for pattern in negation_patterns:
            if re.search(pattern, compact_prefix):
                return True

        return False

    @staticmethod
    def _find_first(text: str, valid_locations: List[str]) -> Optional[str]:
        normalized = text.lower()
        first_match: Optional[str] = None
        first_pos: int = len(normalized) + 1

        for location in valid_locations:
            pattern = PredictionExtractor._location_pattern(location)
            match = re.search(pattern, normalized)
            if match is not None and match.start() < first_pos:
                first_pos = match.start()
                first_match = location.lower()

        return first_match

    @staticmethod
    def _find_first_non_negated(text: str, valid_locations: List[str]) -> Optional[str]:
        normalized = text.lower()
        candidates: list[tuple[int, int, str]] = []

        for location in valid_locations:
            pattern = PredictionExtractor._location_pattern(location)
            for match in re.finditer(pattern, normalized):
                candidates.append((match.start(), match.end(), location.lower()))

        candidates.sort(key=lambda item: item[0])

        for start, end, location in candidates:
            if not PredictionExtractor._is_negated_location(normalized, start, end):
                return location

        return None

    @staticmethod
    def _find_last(text: str, valid_locations: List[str]) -> Optional[str]:
        normalized = text.lower()
        last_match: Optional[str] = None
        last_pos: int = -1

        for location in valid_locations:
            pattern = PredictionExtractor._location_pattern(location)
            matches = list(re.finditer(pattern, normalized))
            if matches and matches[-1].start() > last_pos:
                last_pos = matches[-1].start()
                last_match = location.lower()

        return last_match

    @staticmethod
    def _has_answer_marker(text: str) -> bool:
        for pattern in PredictionExtractor.ANSWER_MARKER_PATTERNS:
            if re.search(pattern, text, flags=re.IGNORECASE):
                return True
        return False

    @staticmethod
    def _extract_explicit_answer_text(text: str) -> Optional[str]:
        """
        Returns text after the last explicit answer marker with a colon.

        Examples:
          ANSWER: Drawer       -> Drawer
          Final answer: Box_B  -> Box_B
          ANSWER:              -> ""

        If the response only says "ANSWER" without a colon, this returns None,
        but _has_answer_marker() will still catch it as an invalid explicit answer.
        """
        latest_match: Optional[re.Match] = None

        for pattern in PredictionExtractor.ANSWER_LINE_PATTERNS:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                if latest_match is None or match.start() > latest_match.start():
                    latest_match = match

        if latest_match is None:
            return None

        return latest_match.group(1).strip()

    @staticmethod
    def _find_after_last_answer_cue(text: str, valid_locations: List[str]) -> Optional[str]:
        """
        Used only when there is no explicit ANSWER marker.

        CoT responses may mention earlier locations before giving the final
        natural-language answer, e.g.:

          Step 1: Drawer -> Basket.
          Step 2: Basket -> Box_B.
          Therefore, the ball is in Box_B.

        Important:
        After an answer cue, take the first non-negated valid location.
        """
        last_cue_end: Optional[int] = None

        for pattern in PredictionExtractor.ANSWER_CUE_PATTERNS:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                if last_cue_end is None or match.end() > last_cue_end:
                    last_cue_end = match.end()

        if last_cue_end is None:
            return None

        answer_span = text[last_cue_end:]
        return PredictionExtractor._find_first_non_negated(answer_span, valid_locations)

    @staticmethod
    def _find_in_final_line_or_sentence(text: str, valid_locations: List[str]) -> Optional[str]:
        """
        Used only when there is no explicit ANSWER marker.

        Prefer the first non-negated location in the final answer-like
        line/sentence.
        """
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in reversed(lines):
            match = PredictionExtractor._find_first_non_negated(line, valid_locations)
            if match is not None:
                return match

        sentences = [part.strip() for part in re.split(r"[.!?]\s+", text) if part.strip()]
        for sentence in reversed(sentences):
            match = PredictionExtractor._find_first_non_negated(sentence, valid_locations)
            if match is not None:
                return match

        return None

    @staticmethod
    def extract(raw_response: str, valid_locations: List[str]) -> str:
        stripped = PredictionExtractor._strip_parentheticals(raw_response)
        normalized_valid_locations = PredictionExtractor._normalize_valid_locations(valid_locations)

        # 1. Explicit answer markers are authoritative.
        #    Prefer exact valid-location answers, but also allow a valid
        #    location embedded in the answer line itself:
        #
        #      ANSWER: Bob thinks the ball is in the Drawer.
        #
        #    Important: only inspect the answer line, not the later reasoning,
        #    because reasoning often mentions distractor locations.
        explicit_answer_text = PredictionExtractor._extract_explicit_answer_text(stripped)
        if explicit_answer_text is not None:
            if not explicit_answer_text:
                return PredictionExtractor.INVALID_PREDICTION

            normalized_answer = PredictionNormalizer.normalize(explicit_answer_text)
            if normalized_answer in normalized_valid_locations:
                return normalized_answer

            embedded_answer = PredictionExtractor._find_first_non_negated(
                explicit_answer_text,
                valid_locations,
            )
            if embedded_answer is not None:
                return embedded_answer

            return PredictionExtractor.INVALID_PREDICTION

        # 2. If the model emitted an answer marker without a parseable answer
        #    line, treat it as invalid instead of falling back.
        if PredictionExtractor._has_answer_marker(stripped):
            return PredictionExtractor.INVALID_PREDICTION

        # 3. Direct answer without explicit marker.
        #    Accept only if the whole response is exactly one valid location.
        normalized_response = PredictionNormalizer.normalize(stripped)
        if normalized_response in normalized_valid_locations:
            return normalized_response

        # 4. No explicit answer marker: use natural-language answer cues.
        match = PredictionExtractor._find_after_last_answer_cue(stripped, valid_locations)
        if match is not None:
            return match

        # 5. No explicit answer marker: prefer final line/sentence only.
        match = PredictionExtractor._find_in_final_line_or_sentence(stripped, valid_locations)
        if match is not None:
            return match

        # 6. No answer found. Do not scan the whole response.
        return PredictionExtractor.INVALID_PREDICTION


# =========================================================
# EVALUATION RESULT
# =========================================================

class EvaluationResult(BaseModel):
    question_id: str
    scenario_id: str
    model_name: str
    reasoning_type: ReasoningType
    template_type: str
    seed: int
    prompt_template: str
    raw_response: str
    extracted_prediction: str
    expected_answer: str
    correct: bool


# =========================================================
# EVALUATOR
# =========================================================

class Evaluator:

    @staticmethod
    def evaluate(
        question: dict,
        scenario: dict,
        raw_response: str,
        model_name: str,
        prompt_template: str,
    ) -> EvaluationResult:

        extracted = PredictionExtractor.extract(
            raw_response=raw_response,
            valid_locations=scenario["locations"]
        )
        expected = PredictionNormalizer.normalize(question["expected_location"])

        return EvaluationResult(
            question_id=question["question_id"],
            scenario_id=scenario["scenario_id"],
            model_name=model_name,
            reasoning_type=ReasoningType(question["reasoning_type"]),
            template_type=scenario["template_type"],
            seed=scenario["seed"],
            prompt_template=prompt_template,
            raw_response=raw_response,
            extracted_prediction=extracted,
            expected_answer=expected,
            correct=(extracted == expected),
        )
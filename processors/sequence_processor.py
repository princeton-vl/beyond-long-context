"""Sequence-based context streaming for evaluation without videos."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from datasets.patternvideos_manifest import OptionEntry, QuestionEntry


class SequenceFormatter:
    """Base class used to convert token sequences into model-readable text."""

    def format(self, tokens: Sequence[str]) -> str:
        raise NotImplementedError


class CommaSeparatedSequenceFormatter(SequenceFormatter):
    """Simple formatter that renders tokens as comma-separated strings."""

    def format(self, tokens: Sequence[str]) -> str:
        normalized = [str(token) for token in tokens if token is not None]
        return ", ".join(normalized)


class SpatialSequenceFormatter(SequenceFormatter):
    """Formatter that renders sequences as (token, lane) tuples for spatial evaluation."""

    def format(self, tokens: Sequence[str], lanes: Optional[Sequence[str]] = None) -> str:
        """
        Format a token sequence as (token, lane) tuples.

        Args:
            tokens: The token sequence (S_tokens)
            lanes: The corresponding lane sequence (S_lanes), if available

        Returns:
            Comma-separated tuples like: (10, 2), (1, 4), (7, 0), ...
        """
        normalized_tokens = [str(token) for token in tokens if token is not None]

        if not normalized_tokens:
            return ""

        # If lanes are provided, format as tuples
        if lanes is not None:
            normalized_lanes = [str(lane) for lane in lanes if lane is not None]
            if len(normalized_tokens) == len(normalized_lanes):
                tuples = [f"({token}, {lane})" for token, lane in zip(normalized_tokens, normalized_lanes)]
                return ", ".join(tuples)

        # Fallback: just comma-separated tokens
        return ", ".join(normalized_tokens)


@dataclass
class SequenceProcessor:
    """Streams manifest-provided sequences into the model text or exist-mode context."""

    sequences_used: Dict[str, List[str]]
    formatter: SequenceFormatter
    print_chunks: bool = True

    def __post_init__(self) -> None:
        normalized: Dict[str, List[str]] = {}
        for key, tokens in self.sequences_used.items():
            token_list = self._normalize_sequence(tokens)
            if token_list:
                normalized[key] = token_list
        self.sequences_used = normalized

    def stream_full_sequences(
        self,
        model: Any,
        base_time: float = 0.0,
    ) -> Tuple[float, List[str]]:
        """Stream the entire sequences_used payload once per video."""

        statements = self._build_base_sequence_statements()
        cursor = base_time
        for statement in statements:
            display = f"{statement}\n\n"
            if self.print_chunks:
                print(f"[sequence-mode] {statement}", flush=True)
            model.add_text(display, current_video_time=cursor)
            cursor += 1.0
        return cursor, statements

    def stream_question_prefix(
        self,
        model: Any,
        question: QuestionEntry,
        base_time: float = 0.0,
    ) -> Tuple[float, List[str]]:
        """Stream the question-specific prefix (continuation mode only)."""

        statements = self._build_prefix_statements(question)
        cursor = base_time
        for statement in statements:
            display = f"{statement}\n\n"
            if self.print_chunks:
                print(f"[sequence-mode] {statement}", flush=True)
            model.add_text(display, current_video_time=cursor)
            cursor += 1.0
        return cursor, statements

    def build_option_statement(
        self,
        option_index: int,
        option: OptionEntry,
        label_suffix: str = "",
    ) -> str:
        """Return printable text for a single option sequence."""

        # For spatial formatter, extract both S_tokens and S_lanes
        if isinstance(self.formatter, SpatialSequenceFormatter):
            tokens, lanes = self._extract_spatial_sequences_from_metadata(option.metadata)
            if tokens:
                formatted = self.formatter.format(tokens, lanes)
                statement = f"Option {option_index}{label_suffix}: {formatted}"
            else:
                statement = f"Option {option_index}{label_suffix}: (no sequence provided)"
        else:
            tokens = option.token_sequence or self._extract_sequence_from_metadata(option.metadata)
            if tokens:
                formatted = self.formatter.format(tokens)
                statement = f"Option {option_index}{label_suffix}: {formatted}"
            else:
                statement = f"Option {option_index}{label_suffix}: (no sequence provided)"

        if self.print_chunks:
            print(f"[sequence-mode] {statement}", flush=True)
        return statement

    def _build_base_sequence_statements(self) -> List[str]:
        statements: List[str] = []

        # For spatial formatter, we need to handle S_tokens and S_lanes together
        if isinstance(self.formatter, SpatialSequenceFormatter):
            s_tokens = self.sequences_used.get('S_tokens', [])
            s_lanes = self.sequences_used.get('S_lanes', [])
            if s_tokens:
                text = self.formatter.format(s_tokens, s_lanes)
                statements.append(f"Sequence S: {text}")
        else:
            # For regular formatters (sequential mode), only show tokens, not lanes
            for key in sorted(self.sequences_used.keys()):
                # In sequential mode, skip S_lanes (only show S_tokens)
                if key == 'S_lanes':
                    continue

                tokens = self.sequences_used[key]
                if not tokens:
                    continue
                text = self.formatter.format(tokens)
                statements.append(f"Sequence {key}: {text}")
        return statements

    def _build_prefix_statements(self, question: QuestionEntry) -> List[str]:
        statements: List[str] = []
        mode = (question.question_mode or "").strip().lower()
        if mode == "continuation" and question.sequence_prefixes:
            for key in sorted(question.sequence_prefixes.keys()):
                tokens = self._normalize_sequence(question.sequence_prefixes[key])
                if not tokens:
                    continue
                text = self.formatter.format(tokens)
                statements.append(f"Prefix {key}: {text}")
        return statements

    @staticmethod
    def _normalize_sequence(raw_tokens: Iterable[Any]) -> List[str]:
        tokens: List[str] = []
        if raw_tokens is None:
            return tokens
        for token in raw_tokens:
            if token is None:
                continue
            tokens.append(str(token))
        return tokens

    @staticmethod
    def _extract_sequence_from_metadata(metadata: Optional[Dict[str, Any]]) -> List[str]:
        if not isinstance(metadata, dict):
            return []
        primary = SequenceProcessor._normalize_sequence(metadata.get("sequence"))
        if primary:
            return primary
        sequences_map = metadata.get("sequences")
        if isinstance(sequences_map, dict):
            for key in sorted(sequences_map.keys()):
                tokens = SequenceProcessor._normalize_sequence(sequences_map.get(key))
                if tokens:
                    return tokens
        return []

    @staticmethod
    def _extract_spatial_sequences_from_metadata(metadata: Optional[Dict[str, Any]]) -> Tuple[List[str], Optional[List[str]]]:
        """
        Extract both S_tokens and S_lanes from metadata for spatial mode.

        Returns:
            Tuple of (tokens, lanes) where lanes may be None if not available
        """
        if not isinstance(metadata, dict):
            return [], None

        sequences_map = metadata.get("sequences")
        if isinstance(sequences_map, dict):
            s_tokens = SequenceProcessor._normalize_sequence(sequences_map.get("S_tokens"))
            s_lanes = SequenceProcessor._normalize_sequence(sequences_map.get("S_lanes"))
            if s_tokens:
                return s_tokens, s_lanes if s_lanes else None

        # Fallback to single sequence
        primary = SequenceProcessor._normalize_sequence(metadata.get("sequence"))
        if primary:
            return primary, None

        return [], None

"""
Enforces source stability, drift thresholds, and allowlists.
Makes retrieval drift measurable — not hidden behind fluent generation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any


class RetrievalContractViolation(Exception):
    def __init__(self, rule: str, detail: str):
        self.rule = rule
        self.detail = detail
        super().__init__(f"RetrievalContract violation [{rule}]: {detail}")


@dataclass
class RetrievedChunk:
    source_id: str
    content: str
    score: float
    retrieved_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalContract:
    allowed_sources: set[str] | None = None
    blocked_sources: set[str] = field(default_factory=set)
    min_results: int = 1
    max_results: int = 20
    min_jaccard_overlap: float = 0.5
    max_age_days: int | None = None

    def validate(self, chunks: list[RetrievedChunk]) -> None:
        if len(chunks) < self.min_results:
            raise RetrievalContractViolation("min_results",
                f"Got {len(chunks)} results, minimum is {self.min_results}.")
        if len(chunks) > self.max_results:
            raise RetrievalContractViolation("max_results",
                f"Got {len(chunks)} results, maximum is {self.max_results}.")

        for chunk in chunks:
            if self.allowed_sources is not None and chunk.source_id not in self.allowed_sources:
                raise RetrievalContractViolation("allowed_sources",
                    f"Source '{chunk.source_id}' not in allowed set.")
            if chunk.source_id in self.blocked_sources:
                raise RetrievalContractViolation("blocked_sources",
                    f"Source '{chunk.source_id}' is explicitly blocked.")
            if self.max_age_days is not None:
                age = datetime.now(timezone.utc) - chunk.retrieved_at
                if age > timedelta(days=self.max_age_days):
                    raise RetrievalContractViolation("max_age_days",
                        f"Chunk from '{chunk.source_id}' is {age.days} days old, max is {self.max_age_days}.")

    def check_drift(self, baseline: list[RetrievedChunk], updated: list[RetrievedChunk]) -> float:
        a = {c.source_id for c in baseline}
        b = {c.source_id for c in updated}
        union = a | b
        if not union:
            return 1.0
        jaccard = len(a & b) / len(union)
        if jaccard < self.min_jaccard_overlap:
            raise RetrievalContractViolation("min_jaccard_overlap",
                f"Source overlap dropped to {jaccard:.2f} (minimum: {self.min_jaccard_overlap:.2f}). "
                f"Lost: {a - b}. Gained: {b - a}.")
        return jaccard

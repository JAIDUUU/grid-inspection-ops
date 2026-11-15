"""Deterministic cohort snapshotting for dashboard quality monitoring.

This is intentionally independent from the database and LangGraph.  Its one
responsibility is to turn timestamped inspection records into stable, auditable
cohort counts. Future tools and graph nodes consume this snapshot rather than
re-implementing disposition semantics.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..domain import Decision, InspectionRecord, ReviewReasonCode, ReviewStatus


class InspectionCohort(str, Enum):
    """Mutually exclusive groups meaningful for quality monitoring."""

    AUTO_PASS = "AUTO_PASS"
    AUTO_REJECT = "AUTO_REJECT"
    PENDING_INSPECT = "PENDING_INSPECT"
    REVIEWED_PASS_NO_DEFECT_FOUND = "REVIEWED_PASS_NO_DEFECT_FOUND"
    REVIEWED_PASS_ACCEPTED_PROCESS_RESIDUE = "REVIEWED_PASS_ACCEPTED_PROCESS_RESIDUE"
    REVIEWED_PASS_OTHER = "REVIEWED_PASS_OTHER"
    REVIEWED_REJECT = "REVIEWED_REJECT"
    UNCLASSIFIED = "UNCLASSIFIED"


class TimedInspection(BaseModel):
    """An immutable inspection record paired with its database observation time."""

    model_config = ConfigDict(frozen=True)

    observed_at: datetime
    record: InspectionRecord


class AnalysisWindow(BaseModel):
    """Half-open UTC-compatible interval: start <= observed_at < end."""

    model_config = ConfigDict(frozen=True)

    start: datetime
    end: datetime

    @model_validator(mode="after")
    def start_is_before_end(self) -> AnalysisWindow:
        if self.start >= self.end:
            raise ValueError("analysis window start must be earlier than end")
        return self


class CohortSnapshot(BaseModel):
    """The bounded, reproducible input to later screening and investigation."""

    window: AnalysisWindow
    source_inspection_count: int = Field(ge=0)
    cohort_counts: dict[InspectionCohort, int]


def classify_inspection(record: InspectionRecord) -> InspectionCohort:
    """Assign exactly one cohort without inferring ground truth from policy alone."""

    if record.decision is Decision.PASS and record.review_status is ReviewStatus.NOT_REQUIRED:
        return InspectionCohort.AUTO_PASS
    if record.decision is Decision.REJECT and record.review_status is ReviewStatus.NOT_REQUIRED:
        return InspectionCohort.AUTO_REJECT
    if record.decision is not Decision.INSPECT:
        return InspectionCohort.UNCLASSIFIED
    if record.review_status is ReviewStatus.PENDING:
        return InspectionCohort.PENDING_INSPECT
    if record.review_status is not ReviewStatus.RESOLVED or record.review is None:
        return InspectionCohort.UNCLASSIFIED
    if record.review.final_decision is Decision.REJECT:
        return InspectionCohort.REVIEWED_REJECT
    if record.review.final_decision is not Decision.PASS:
        return InspectionCohort.UNCLASSIFIED
    if record.review.reason_code is ReviewReasonCode.NO_DEFECT_FOUND:
        return InspectionCohort.REVIEWED_PASS_NO_DEFECT_FOUND
    if record.review.reason_code is ReviewReasonCode.ACCEPTED_PROCESS_RESIDUE:
        return InspectionCohort.REVIEWED_PASS_ACCEPTED_PROCESS_RESIDUE
    return InspectionCohort.REVIEWED_PASS_OTHER


def build_cohort_snapshot(
    inspections: Iterable[TimedInspection],
    window: AnalysisWindow,
) -> CohortSnapshot:
    """Count records in a window; end is exclusive to prevent double counting."""

    counts: Counter[InspectionCohort] = Counter()
    total = 0
    for inspection in inspections:
        if window.start <= inspection.observed_at < window.end:
            counts[classify_inspection(inspection.record)] += 1
            total += 1
    return CohortSnapshot(
        window=window,
        source_inspection_count=total,
        cohort_counts={cohort: counts[cohort] for cohort in InspectionCohort},
    )

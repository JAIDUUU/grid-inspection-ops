"""Deterministic comparison of a current cohort snapshot with its baseline."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from .snapshot import CohortSnapshot, InspectionCohort


class RateBasis(str, Enum):
    """The population against which a cohort rate is meaningful."""

    ALL_INSPECTIONS = "ALL_INSPECTIONS"
    RESOLVED_INSPECTS = "RESOLVED_INSPECTS"


class SampleSufficiency(str, Enum):
    """A data-quality label, not a statistical significance claim."""

    SUFFICIENT = "SUFFICIENT"
    CURRENT_TOO_SMALL = "CURRENT_TOO_SMALL"
    BASELINE_TOO_SMALL = "BASELINE_TOO_SMALL"
    BOTH_TOO_SMALL = "BOTH_TOO_SMALL"


class ComparisonConfig(BaseModel):
    """Explicit guardrail against treating tiny cohorts as operational trends."""

    model_config = ConfigDict(frozen=True)

    minimum_denominator: int = Field(default=20, ge=1)


class CohortRateComparison(BaseModel):
    """Auditable rate math for one cohort, with no alarm judgement attached."""

    cohort: InspectionCohort
    rate_basis: RateBasis
    current_count: int = Field(ge=0)
    current_denominator: int = Field(ge=0)
    current_rate: float | None = Field(default=None, ge=0, le=1)
    baseline_count: int = Field(ge=0)
    baseline_denominator: int = Field(ge=0)
    baseline_rate: float | None = Field(default=None, ge=0, le=1)
    rate_delta: float | None = Field(default=None, ge=-1, le=1)
    relative_rate_change: float | None = None
    sample_sufficiency: SampleSufficiency


class BaselineComparison(BaseModel):
    """Complete comparison payload consumed by the later signal-screening tool."""

    current: CohortSnapshot
    baseline: CohortSnapshot
    config: ComparisonConfig
    cohorts: list[CohortRateComparison]


_RESOLVED_INSPECT_COHORTS = {
    InspectionCohort.REVIEWED_PASS_NO_DEFECT_FOUND,
    InspectionCohort.REVIEWED_PASS_ACCEPTED_PROCESS_RESIDUE,
    InspectionCohort.REVIEWED_PASS_OTHER,
    InspectionCohort.REVIEWED_REJECT,
}


def rate_basis_for(cohort: InspectionCohort) -> RateBasis:
    if cohort in _RESOLVED_INSPECT_COHORTS:
        return RateBasis.RESOLVED_INSPECTS
    return RateBasis.ALL_INSPECTIONS


def _denominator(snapshot: CohortSnapshot, basis: RateBasis) -> int:
    if basis is RateBasis.ALL_INSPECTIONS:
        return snapshot.source_inspection_count
    return sum(snapshot.cohort_counts[cohort] for cohort in _RESOLVED_INSPECT_COHORTS)


def _rate(count: int, denominator: int) -> float | None:
    return count / denominator if denominator else None


def _sample_sufficiency(
    current_denominator: int,
    baseline_denominator: int,
    minimum_denominator: int,
) -> SampleSufficiency:
    current_small = current_denominator < minimum_denominator
    baseline_small = baseline_denominator < minimum_denominator
    if current_small and baseline_small:
        return SampleSufficiency.BOTH_TOO_SMALL
    if current_small:
        return SampleSufficiency.CURRENT_TOO_SMALL
    if baseline_small:
        return SampleSufficiency.BASELINE_TOO_SMALL
    return SampleSufficiency.SUFFICIENT


def compare_snapshots(
    current: CohortSnapshot,
    baseline: CohortSnapshot,
    config: ComparisonConfig | None = None,
) -> BaselineComparison:
    """Compare non-overlapping snapshots without deciding whether a change is a signal."""

    if baseline.window.end > current.window.start:
        raise ValueError("baseline window must end on or before the current window start")
    config = config or ComparisonConfig()
    comparisons: list[CohortRateComparison] = []
    for cohort in InspectionCohort:
        basis = rate_basis_for(cohort)
        current_count = current.cohort_counts[cohort]
        baseline_count = baseline.cohort_counts[cohort]
        current_denominator = _denominator(current, basis)
        baseline_denominator = _denominator(baseline, basis)
        current_rate = _rate(current_count, current_denominator)
        baseline_rate = _rate(baseline_count, baseline_denominator)
        rate_delta = (
            current_rate - baseline_rate
            if current_rate is not None and baseline_rate is not None
            else None
        )
        relative_rate_change = (
            rate_delta / baseline_rate
            if rate_delta is not None and baseline_rate not in {None, 0}
            else None
        )
        comparisons.append(
            CohortRateComparison(
                cohort=cohort,
                rate_basis=basis,
                current_count=current_count,
                current_denominator=current_denominator,
                current_rate=current_rate,
                baseline_count=baseline_count,
                baseline_denominator=baseline_denominator,
                baseline_rate=baseline_rate,
                rate_delta=rate_delta,
                relative_rate_change=relative_rate_change,
                sample_sufficiency=_sample_sufficiency(
                    current_denominator,
                    baseline_denominator,
                    config.minimum_denominator,
                ),
            )
        )
    return BaselineComparison(
        current=current,
        baseline=baseline,
        config=config,
        cohorts=comparisons,
    )

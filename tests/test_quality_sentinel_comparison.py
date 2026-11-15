from datetime import datetime, timedelta, timezone

import pytest

from app.quality_sentinel.comparison import (
    ComparisonConfig,
    RateBasis,
    SampleSufficiency,
    compare_snapshots,
)
from app.quality_sentinel.snapshot import AnalysisWindow, CohortSnapshot, InspectionCohort

START = datetime(2026, 11, 1, tzinfo=timezone.utc)


def snapshot(
    start: datetime,
    duration_days: int,
    counts: dict[InspectionCohort, int],
) -> CohortSnapshot:
    complete_counts = {cohort: counts.get(cohort, 0) for cohort in InspectionCohort}
    return CohortSnapshot(
        window=AnalysisWindow(start=start, end=start + timedelta(days=duration_days)),
        source_inspection_count=sum(complete_counts.values()),
        cohort_counts=complete_counts,
    )


def comparison_for(result, cohort: InspectionCohort):
    return next(item for item in result.cohorts if item.cohort is cohort)


def test_auto_reject_rate_is_stable_when_volume_changes():
    baseline = snapshot(
        START,
        28,
        {
            InspectionCohort.AUTO_REJECT: 20,
            InspectionCohort.AUTO_PASS: 180,
        },
    )
    current = snapshot(
        baseline.window.end,
        7,
        {
            InspectionCohort.AUTO_REJECT: 10,
            InspectionCohort.AUTO_PASS: 90,
        },
    )

    item = comparison_for(compare_snapshots(current, baseline), InspectionCohort.AUTO_REJECT)

    assert item.rate_basis is RateBasis.ALL_INSPECTIONS
    assert item.current_rate == pytest.approx(0.10)
    assert item.baseline_rate == pytest.approx(0.10)
    assert item.rate_delta == pytest.approx(0.0)
    assert item.relative_rate_change == pytest.approx(0.0)


def test_review_outcomes_use_resolved_inspects_not_all_inspections():
    baseline = snapshot(
        START,
        28,
        {
            InspectionCohort.AUTO_PASS: 120,
            InspectionCohort.REVIEWED_PASS_NO_DEFECT_FOUND: 8,
            InspectionCohort.REVIEWED_REJECT: 72,
        },
    )
    current = snapshot(
        baseline.window.end,
        7,
        {
            InspectionCohort.AUTO_PASS: 80,
            InspectionCohort.REVIEWED_PASS_NO_DEFECT_FOUND: 4,
            InspectionCohort.REVIEWED_REJECT: 16,
        },
    )

    item = comparison_for(
        compare_snapshots(current, baseline),
        InspectionCohort.REVIEWED_PASS_NO_DEFECT_FOUND,
    )

    assert item.rate_basis is RateBasis.RESOLVED_INSPECTS
    assert item.current_denominator == 20
    assert item.baseline_denominator == 80
    assert item.current_rate == pytest.approx(0.20)
    assert item.baseline_rate == pytest.approx(0.10)
    assert item.rate_delta == pytest.approx(0.10)


def test_zero_baseline_rate_does_not_produce_an_infinite_relative_change():
    baseline = snapshot(
        START,
        28,
        {
            InspectionCohort.AUTO_PASS: 60,
            InspectionCohort.REVIEWED_REJECT: 20,
        },
    )
    current = snapshot(
        baseline.window.end,
        7,
        {
            InspectionCohort.AUTO_PASS: 58,
            InspectionCohort.REVIEWED_PASS_ACCEPTED_PROCESS_RESIDUE: 2,
            InspectionCohort.REVIEWED_REJECT: 20,
        },
    )

    item = comparison_for(
        compare_snapshots(current, baseline),
        InspectionCohort.REVIEWED_PASS_ACCEPTED_PROCESS_RESIDUE,
    )

    assert item.baseline_rate == 0.0
    assert item.rate_delta == pytest.approx(2 / 22)
    assert item.relative_rate_change is None
    assert item.sample_sufficiency is SampleSufficiency.SUFFICIENT


def test_small_review_samples_are_explicitly_marked_insufficient():
    baseline = snapshot(
        START,
        28,
        {
            InspectionCohort.AUTO_PASS: 47,
            InspectionCohort.REVIEWED_PASS_NO_DEFECT_FOUND: 1,
            InspectionCohort.REVIEWED_REJECT: 2,
        },
    )
    current = snapshot(
        baseline.window.end,
        7,
        {
            InspectionCohort.AUTO_PASS: 45,
            InspectionCohort.REVIEWED_PASS_NO_DEFECT_FOUND: 2,
            InspectionCohort.REVIEWED_REJECT: 2,
        },
    )

    item = comparison_for(
        compare_snapshots(current, baseline, ComparisonConfig(minimum_denominator=10)),
        InspectionCohort.REVIEWED_PASS_NO_DEFECT_FOUND,
    )

    assert item.sample_sufficiency is SampleSufficiency.BOTH_TOO_SMALL


def test_overlapping_windows_are_rejected_to_keep_the_comparison_interpretable():
    baseline = snapshot(START, 28, {InspectionCohort.AUTO_PASS: 20})
    current = snapshot(START + timedelta(days=14), 7, {InspectionCohort.AUTO_PASS: 20})

    with pytest.raises(ValueError, match="baseline window"):
        compare_snapshots(current, baseline)

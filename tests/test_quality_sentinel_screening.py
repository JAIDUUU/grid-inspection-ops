from datetime import datetime, timedelta, timezone

from app.quality_sentinel.comparison import ComparisonConfig, compare_snapshots
from app.quality_sentinel.screening import (
    ScreeningOutcome,
    SignalDirection,
    SignalKind,
    screen_comparison,
)
from app.quality_sentinel.snapshot import AnalysisWindow, CohortSnapshot, InspectionCohort

START = datetime(2026, 11, 1, tzinfo=timezone.utc)


def snapshot(
    start: datetime,
    counts: dict[InspectionCohort, int],
) -> CohortSnapshot:
    complete_counts = {cohort: counts.get(cohort, 0) for cohort in InspectionCohort}
    return CohortSnapshot(
        window=AnalysisWindow(start=start, end=start + timedelta(days=7)),
        source_inspection_count=sum(complete_counts.values()),
        cohort_counts=complete_counts,
    )


def report_for(
    baseline_counts: dict[InspectionCohort, int],
    current_counts: dict[InspectionCohort, int],
):
    baseline = snapshot(START, baseline_counts)
    current = snapshot(baseline.window.end, current_counts)
    comparison = compare_snapshots(
        current,
        baseline,
        ComparisonConfig(minimum_denominator=20),
    )
    return screen_comparison(comparison)


def decision_for(report, cohort: InspectionCohort):
    return next(item for item in report.decisions if item.cohort is cohort)


def test_selects_material_auto_reject_rate_increase():
    report = report_for(
        {
            InspectionCohort.AUTO_PASS: 90,
            InspectionCohort.AUTO_REJECT: 10,
        },
        {
            InspectionCohort.AUTO_PASS: 80,
            InspectionCohort.AUTO_REJECT: 20,
        },
    )

    decision = decision_for(report, InspectionCohort.AUTO_REJECT)

    assert decision.outcome is ScreeningOutcome.SELECTED
    assert decision.signal_kind is SignalKind.RATE_SHIFT
    assert decision.direction is SignalDirection.INCREASE


def test_does_not_select_small_rate_change():
    report = report_for(
        {
            InspectionCohort.AUTO_PASS: 95,
            InspectionCohort.AUTO_REJECT: 5,
        },
        {
            InspectionCohort.AUTO_PASS: 93,
            InspectionCohort.AUTO_REJECT: 7,
        },
    )

    decision = decision_for(report, InspectionCohort.AUTO_REJECT)

    assert decision.outcome is ScreeningOutcome.BELOW_MATERIALITY_THRESHOLD


def test_does_not_select_large_change_when_review_sample_is_too_small():
    report = report_for(
        {
            InspectionCohort.AUTO_PASS: 47,
            InspectionCohort.REVIEWED_PASS_NO_DEFECT_FOUND: 1,
            InspectionCohort.REVIEWED_REJECT: 2,
        },
        {
            InspectionCohort.AUTO_PASS: 45,
            InspectionCohort.REVIEWED_PASS_NO_DEFECT_FOUND: 3,
            InspectionCohort.REVIEWED_REJECT: 1,
        },
    )

    decision = decision_for(report, InspectionCohort.REVIEWED_PASS_NO_DEFECT_FOUND)

    assert decision.outcome is ScreeningOutcome.INSUFFICIENT_SAMPLE


def test_selects_a_material_new_accepted_residue_cohort():
    report = report_for(
        {
            InspectionCohort.AUTO_PASS: 80,
            InspectionCohort.REVIEWED_REJECT: 20,
        },
        {
            InspectionCohort.AUTO_PASS: 75,
            InspectionCohort.REVIEWED_PASS_ACCEPTED_PROCESS_RESIDUE: 5,
            InspectionCohort.REVIEWED_REJECT: 20,
        },
    )

    decision = decision_for(report, InspectionCohort.REVIEWED_PASS_ACCEPTED_PROCESS_RESIDUE)

    assert decision.outcome is ScreeningOutcome.SELECTED
    assert decision.signal_kind is SignalKind.EMERGENCE
    assert decision.direction is SignalDirection.NEW


def test_new_cohort_below_emergence_rate_is_not_selected():
    report = report_for(
        {
            InspectionCohort.AUTO_PASS: 80,
            InspectionCohort.REVIEWED_REJECT: 20,
        },
        {
            InspectionCohort.AUTO_PASS: 87,
            InspectionCohort.REVIEWED_PASS_ACCEPTED_PROCESS_RESIDUE: 3,
            InspectionCohort.REVIEWED_REJECT: 30,
        },
    )

    decision = decision_for(report, InspectionCohort.REVIEWED_PASS_ACCEPTED_PROCESS_RESIDUE)

    assert decision.outcome is ScreeningOutcome.BELOW_MATERIALITY_THRESHOLD

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from app.domain import (
    Decision,
    InspectionRecord,
    ReviewReasonCode,
    ReviewResolution,
    ReviewStatus,
)
from app.quality_sentinel.snapshot import (
    AnalysisWindow,
    InspectionCohort,
    TimedInspection,
    build_cohort_snapshot,
    classify_inspection,
)

NOW = datetime(2026, 11, 12, 12, tzinfo=timezone.utc)


def record(
    inspection_id: str,
    decision: Decision,
    review_status: ReviewStatus = ReviewStatus.NOT_REQUIRED,
    final_decision: Decision | None = None,
    reason_code: ReviewReasonCode | None = None,
) -> InspectionRecord:
    review = None
    if review_status is ReviewStatus.RESOLVED:
        assert final_decision is not None
        assert reason_code is not None
        review = ReviewResolution(
            reviewer_id="operator-1",
            final_decision=final_decision,
            reason="Recorded during a deterministic unit test.",
            reason_code=reason_code,
            reviewed_at=NOW.isoformat(),
        )
    return InspectionRecord(
        inspection_id=inspection_id,
        part_id=inspection_id,
        image_uri=f"runtime/uploads/{inspection_id}.png",
        decision=decision,
        final_decision=final_decision,
        review_status=review_status,
        review=review,
    )


def test_classification_keeps_human_pass_reason_codes_separate():
    no_defect = record(
        "no-defect",
        Decision.INSPECT,
        ReviewStatus.RESOLVED,
        Decision.PASS,
        ReviewReasonCode.NO_DEFECT_FOUND,
    )
    residue = record(
        "residue",
        Decision.INSPECT,
        ReviewStatus.RESOLVED,
        Decision.PASS,
        ReviewReasonCode.ACCEPTED_PROCESS_RESIDUE,
    )

    assert classify_inspection(no_defect) is InspectionCohort.REVIEWED_PASS_NO_DEFECT_FOUND
    assert (
        classify_inspection(residue)
        is InspectionCohort.REVIEWED_PASS_ACCEPTED_PROCESS_RESIDUE
    )


def test_snapshot_counts_each_record_once_and_excludes_window_end():
    window = AnalysisWindow(start=NOW, end=NOW + timedelta(hours=1))
    records = [
        TimedInspection(observed_at=NOW, record=record("auto-pass", Decision.PASS)),
        TimedInspection(observed_at=NOW, record=record("auto-reject", Decision.REJECT)),
        TimedInspection(
            observed_at=NOW,
            record=record(
                "reviewed-residue",
                Decision.INSPECT,
                ReviewStatus.RESOLVED,
                Decision.PASS,
                ReviewReasonCode.ACCEPTED_PROCESS_RESIDUE,
            ),
        ),
        TimedInspection(
            observed_at=NOW,
            record=record(
                "reviewed-reject",
                Decision.INSPECT,
                ReviewStatus.RESOLVED,
                Decision.REJECT,
                ReviewReasonCode.TRUE_DEFECT_CONFIRMED,
            ),
        ),
        TimedInspection(
            observed_at=NOW + timedelta(hours=1),
            record=record("next-window", Decision.PASS),
        ),
    ]

    snapshot = build_cohort_snapshot(records, window)

    assert snapshot.source_inspection_count == 4
    assert snapshot.cohort_counts[InspectionCohort.AUTO_PASS] == 1
    assert snapshot.cohort_counts[InspectionCohort.AUTO_REJECT] == 1
    assert snapshot.cohort_counts[InspectionCohort.REVIEWED_PASS_ACCEPTED_PROCESS_RESIDUE] == 1
    assert snapshot.cohort_counts[InspectionCohort.REVIEWED_REJECT] == 1
    assert sum(snapshot.cohort_counts.values()) == snapshot.source_inspection_count


def test_analysis_window_rejects_empty_or_backwards_ranges():
    with pytest.raises(ValidationError, match="start must be earlier"):
        AnalysisWindow(start=NOW, end=NOW)

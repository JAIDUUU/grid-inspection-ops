from datetime import datetime, timedelta, timezone

import pytest
from pydantic import BaseModel

from app.domain import (
    Decision,
    ImageQuality,
    InspectionRecord,
    ReviewReasonCode,
    ReviewResolution,
    ReviewStatus,
    VisionEvidence,
)
from app.quality_sentinel.investigation import (
    CohortTrendTool,
    MetricToolRegistry,
    ScoreBand,
    SqliteQualityDataReader,
    TimeRangeInput,
    ToolDescriptor,
    build_default_tool_registry,
)
from app.quality_sentinel.snapshot import InspectionCohort, TimedInspection

START = datetime(2026, 11, 1, tzinfo=timezone.utc)


class MemoryReader:
    def __init__(self, records: list[TimedInspection]) -> None:
        self.records = records

    def fetch(self, query: TimeRangeInput) -> list[TimedInspection]:
        return [item for item in self.records if query.start <= item.observed_at < query.end]


def record(
    inspection_id: str,
    decision: Decision,
    observed_at: datetime,
    review_status: ReviewStatus = ReviewStatus.NOT_REQUIRED,
    final_decision: Decision | None = None,
    reason_code: ReviewReasonCode | None = None,
    score: float | None = None,
    model_version: str = "model-v1",
    policy_version: str = "policy-v1",
    quality: ImageQuality | None = None,
    review_threshold: float = 0.315,
    reject_threshold: float = 0.334,
) -> TimedInspection:
    review = None
    if review_status is ReviewStatus.RESOLVED:
        assert final_decision is not None and reason_code is not None
        review = ReviewResolution(
            reviewer_id="operator-1",
            final_decision=final_decision,
            reason_code=reason_code,
            reason="Test review outcome.",
            reviewed_at=observed_at.isoformat(),
        )
    evidence = None
    if score is not None:
        evidence = VisionEvidence(
            model_version=model_version,
            score=score,
            review_threshold=review_threshold,
            reject_threshold=reject_threshold,
            valid_coverage=1.0,
            artifact_verified=True,
            quality=quality,
        )
    return TimedInspection(
        observed_at=observed_at,
        record=InspectionRecord(
            inspection_id=inspection_id,
            part_id=inspection_id,
            image_uri=f"runtime/{inspection_id}.png",
            decision=decision,
            final_decision=final_decision,
            review_status=review_status,
            review=review,
            evidence=evidence,
            metadata={"critical_policy": {"version": policy_version}},
        ),
    )


def time_range() -> dict[str, datetime]:
    return {"start": START, "end": START + timedelta(days=2)}


def test_registry_exposes_newly_registered_tool_without_a_graph_change():
    class FutureMetricInput(BaseModel):
        window: str

    class FutureMetricOutput(BaseModel):
        status: str

    class FutureMetricTool:
        input_model = FutureMetricInput
        descriptor = ToolDescriptor(
            name="get_future_metric",
            purpose="Returns a future quality metric.",
            when_useful=["Use when an investigator needs the future metric."],
            database_fields=["future.table.value"],
            input_schema=FutureMetricInput.model_json_schema(),
            output_schema=FutureMetricOutput.model_json_schema(),
        )

        def run(self, _reader, raw_input):
            FutureMetricInput.model_validate(raw_input)
            return FutureMetricOutput(status="ok")

    registry = MetricToolRegistry()
    registry.register(FutureMetricTool())

    assert [tool.name for tool in registry.descriptors()] == ["get_future_metric"]
    assert registry.execute("get_future_metric", MemoryReader([]), {"window": "7d"}).status == "ok"


def test_cohort_trend_tool_returns_only_the_requested_metric():
    reader = MemoryReader(
        [
            record("pass-1", Decision.PASS, START, score=0.1),
            record("reject-1", Decision.REJECT, START, score=0.5),
            record("reject-2", Decision.REJECT, START + timedelta(days=1), score=0.6),
        ]
    )
    result = CohortTrendTool().run(
        reader,
        {**time_range(), "cohort": InspectionCohort.AUTO_REJECT, "bucket_hours": 24},
    )

    assert [point.cohort_count for point in result.points] == [1, 1]
    assert [point.rate for point in result.points] == [pytest.approx(0.5), pytest.approx(1.0)]


def test_reason_breakdown_keeps_accepted_residue_distinct_from_no_defect():
    reader = MemoryReader(
        [
            record(
                "residue",
                Decision.INSPECT,
                START,
                ReviewStatus.RESOLVED,
                Decision.PASS,
                ReviewReasonCode.ACCEPTED_PROCESS_RESIDUE,
            ),
            record(
                "no-defect",
                Decision.INSPECT,
                START,
                ReviewStatus.RESOLVED,
                Decision.PASS,
                ReviewReasonCode.NO_DEFECT_FOUND,
            ),
        ]
    )
    result = build_default_tool_registry().execute("get_review_reason_breakdown", reader, time_range())

    assert result.resolved_inspect_count == 2
    assert {item.reason_code: item.count for item in result.reason_counts} == {
        "ACCEPTED_PROCESS_RESIDUE": 1,
        "NO_DEFECT_FOUND": 1,
    }


def test_score_band_tool_derives_enum_bands_from_each_record_thresholds():
    reader = MemoryReader(
        [
            record("below-review", Decision.REJECT, START, score=0.30),
            record("review-band", Decision.REJECT, START, score=0.32),
            record("high", Decision.REJECT, START, score=0.50),
            record("pass", Decision.PASS, START, score=0.10),
        ]
    )
    result = build_default_tool_registry().execute(
        "get_score_band_distribution",
        reader,
        {**time_range(), "cohort": InspectionCohort.AUTO_REJECT},
    )

    assert result.total_matching_records == 3
    assert result.missing_evidence_count == 0
    assert len(result.distributions) == 1
    assert result.distributions[0].review_threshold == pytest.approx(0.315)
    assert result.distributions[0].reject_threshold == pytest.approx(0.334)
    assert {band.band: band.count for band in result.distributions[0].bands} == {
        ScoreBand.BELOW_REVIEW_THRESHOLD: 1,
        ScoreBand.REVIEW_TO_REJECT_THRESHOLD: 1,
        ScoreBand.AT_OR_ABOVE_REJECT_THRESHOLD: 1,
    }


def test_score_band_tool_does_not_mix_different_threshold_versions():
    reader = MemoryReader(
        [
            record("v1", Decision.REJECT, START, score=0.32),
            record(
                "v2",
                Decision.REJECT,
                START,
                score=0.32,
                model_version="model-v2",
                review_threshold=0.30,
                reject_threshold=0.31,
            ),
        ]
    )
    result = build_default_tool_registry().execute(
        "get_score_band_distribution",
        reader,
        {**time_range(), "cohort": InspectionCohort.AUTO_REJECT},
    )

    assert [item.model_version for item in result.distributions] == ["model-v1", "model-v2"]
    assert [
        {band.band: band.count for band in item.bands}[ScoreBand.REVIEW_TO_REJECT_THRESHOLD]
        for item in result.distributions
    ] == [1, 0]
    assert [
        {band.band: band.count for band in item.bands}[ScoreBand.AT_OR_ABOVE_REJECT_THRESHOLD]
        for item in result.distributions
    ] == [0, 1]


def test_deployment_context_reports_versions_without_inference():
    reader = MemoryReader(
        [
            record("old", Decision.PASS, START, score=0.1, model_version="model-v1"),
            record(
                "new",
                Decision.REJECT,
                START,
                score=0.5,
                model_version="model-v2",
                policy_version="policy-v2",
            ),
        ]
    )
    result = build_default_tool_registry().execute("get_deployment_context", reader, time_range())

    assert {(item.model_version, item.policy_version, item.count) for item in result.version_counts} == {
        ("model-v1", "policy-v1", 1),
        ("model-v2", "policy-v2", 1),
    }


def test_capture_quality_summary_aggregates_quality_gate_evidence():
    acceptable = ImageQuality(
        width=512,
        height=512,
        mean_luminance=0.5,
        contrast=0.2,
        laplacian_variance=40,
        acceptable=True,
    )
    unacceptable = acceptable.model_copy(
        update={"acceptable": False, "failure_reasons": ["LOW_SHARPNESS"]}
    )
    reader = MemoryReader(
        [
            record("good-capture", Decision.PASS, START, score=0.1, quality=acceptable),
            record("bad-capture", Decision.REJECT, START, score=0.5, quality=unacceptable),
            record("missing-capture", Decision.PASS, START),
        ]
    )
    result = build_default_tool_registry().execute("get_capture_quality_summary", reader, time_range())

    assert (result.records_with_quality, result.acceptable_count, result.unacceptable_count) == (2, 1, 1)
    assert result.missing_quality_count == 1
    assert [(item.reason_code, item.count) for item in result.failure_reason_counts] == [
        ("LOW_SHARPNESS", 1)
    ]


def test_registry_rejects_unknown_tool_name():
    with pytest.raises(ValueError, match="unknown metric tool"):
        build_default_tool_registry().execute("drop_database", MemoryReader([]), time_range())


def test_sqlite_reader_reads_only_records_inside_the_requested_window(tmp_path):
    database = tmp_path / "quality.db"
    inside = record("inside", Decision.PASS, START)
    outside = record("outside", Decision.PASS, START + timedelta(days=3))
    import sqlite3

    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE inspections (record_json TEXT, created_at TEXT)")
        for item in (inside, outside):
            connection.execute(
                "INSERT INTO inspections VALUES (?, ?)",
                (
                    item.record.model_dump_json(),
                    item.observed_at.strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )

    rows = SqliteQualityDataReader(str(database)).fetch(TimeRangeInput(**time_range()))

    assert [item.record.inspection_id for item in rows] == ["inside"]

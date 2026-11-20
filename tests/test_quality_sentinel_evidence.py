from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from app.domain import (
    Decision,
    ImageQuality,
    InspectionRecord,
    ReviewReasonCode,
    ReviewResolution,
    ReviewStatus,
    VisionEvidence,
)
from app.quality_sentinel.comparison import (
    BaselineComparison,
    CohortRateComparison,
    ComparisonConfig,
    RateBasis,
    SampleSufficiency,
)
from app.quality_sentinel.evidence import (
    CollectedToolEvidence,
    collect_evidence,
)
from app.quality_sentinel.investigation import (
    MetricToolRegistry,
    TimeRangeInput,
    ToolDescriptor,
    build_default_tool_registry,
)
from app.quality_sentinel.planning import (
    InvestigationPlanProposal,
    InvestigationRequest,
    TimeRangeStrategy,
    ToolCallProposal,
    validate_investigation_plan,
)
from app.quality_sentinel.screening import (
    ScreeningConfig,
    ScreeningDecision,
    ScreeningOutcome,
    ScreeningReport,
    SignalDirection,
    SignalKind,
)
from app.quality_sentinel.snapshot import (
    AnalysisWindow,
    CohortSnapshot,
    InspectionCohort,
    TimedInspection,
)

START = datetime(2026, 11, 1, tzinfo=timezone.utc)


class MemoryReader:
    def __init__(self, records: list[TimedInspection]) -> None:
        self.records = records

    def fetch(self, query: TimeRangeInput) -> list[TimedInspection]:
        return [item for item in self.records if query.start <= item.observed_at < query.end]


def create_record(
    inspection_id: str,
    decision: Decision,
    observed_at: datetime,
    review_status: ReviewStatus = ReviewStatus.NOT_REQUIRED,
    final_decision: Decision | None = None,
    reason_code: ReviewReasonCode | None = None,
    score: float | None = 0.35,
    model_version: str = "model-v1",
    policy_version: str = "policy-v1",
    quality: ImageQuality | None = None,
) -> TimedInspection:
    review = None
    if review_status is ReviewStatus.RESOLVED:
        review = ReviewResolution(
            reviewer_id="reviewer-1",
            final_decision=final_decision or decision,
            reason_code=reason_code or ReviewReasonCode.NO_DEFECT_FOUND,
            reason="Test resolution.",
            reviewed_at=observed_at.isoformat(),
        )
    evidence = None
    if score is not None:
        evidence = VisionEvidence(
            model_version=model_version,
            score=score,
            review_threshold=0.315,
            reject_threshold=0.334,
            valid_coverage=1.0,
            artifact_verified=True,
            quality=quality
            or ImageQuality(
                width=512,
                height=512,
                mean_luminance=0.5,
                contrast=0.2,
                laplacian_variance=40.0,
                acceptable=True,
            ),
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


def sample_request() -> InvestigationRequest:
    baseline_window = AnalysisWindow(start=START, end=START + timedelta(days=7))
    current_window = AnalysisWindow(start=START + timedelta(days=7), end=START + timedelta(days=14))
    comparison = CohortRateComparison(
        cohort=InspectionCohort.AUTO_REJECT,
        rate_basis=RateBasis.ALL_INSPECTIONS,
        current_count=10,
        current_denominator=50,
        current_rate=0.20,
        baseline_count=2,
        baseline_denominator=50,
        baseline_rate=0.04,
        rate_delta=0.16,
        relative_rate_change=4.0,
        sample_sufficiency=SampleSufficiency.SUFFICIENT,
    )
    decision = ScreeningDecision(
        cohort=InspectionCohort.AUTO_REJECT,
        outcome=ScreeningOutcome.SELECTED,
        signal_kind=SignalKind.RATE_SHIFT,
        direction=SignalDirection.INCREASE,
        reason="Test screening decision.",
        comparison=comparison,
    )
    report = ScreeningReport(
        comparison=BaselineComparison(
            current=CohortSnapshot(
                window=current_window, source_inspection_count=50, cohort_counts={}
            ),
            baseline=CohortSnapshot(
                window=baseline_window, source_inspection_count=50, cohort_counts={}
            ),
            config=ComparisonConfig(),
            cohorts=[comparison],
        ),
        config=ScreeningConfig(),
        decisions=[decision],
    )
    return InvestigationRequest(screening_report=report, cohort=InspectionCohort.AUTO_REJECT)


def test_collect_evidence_executes_plan_and_records_bundle():
    reader = MemoryReader(
        [
            create_record("rec-1", Decision.REJECT, START + timedelta(days=1), score=0.40),
            create_record("rec-2", Decision.REJECT, START + timedelta(days=8), score=0.45),
            create_record("rec-3", Decision.PASS, START + timedelta(days=9), score=0.20),
        ]
    )
    req = sample_request()
    registry = build_default_tool_registry()

    plan = validate_investigation_plan(
        req,
        InvestigationPlanProposal(
            investigation_goal="Analyze score distribution and deployment context.",
            tool_calls=[
                ToolCallProposal(
                    tool_name="get_score_band_distribution",
                    arguments={
                        "start": START.isoformat(),
                        "end": (START + timedelta(days=14)).isoformat(),
                        "cohort": "AUTO_REJECT",
                    },
                    selection_reason="Verify reject score concentration.",
                    time_range_strategy=TimeRangeStrategy.BASELINE_AND_CURRENT,
                ),
                ToolCallProposal(
                    tool_name="get_deployment_context",
                    arguments={
                        "start": START.isoformat(),
                        "end": (START + timedelta(days=14)).isoformat(),
                    },
                    selection_reason="Check deployment transitions.",
                    time_range_strategy=TimeRangeStrategy.BASELINE_AND_CURRENT,
                ),
            ],
        ),
        registry,
    )

    bundle = collect_evidence(plan, reader, registry, request=req)

    assert bundle.request is req
    assert bundle.plan is plan
    assert len(bundle.evidence) == 2
    assert bundle.collected_at.tzinfo == timezone.utc

    score_evidence = bundle.evidence[0]
    assert score_evidence.tool_name == "get_score_band_distribution"
    assert score_evidence.tool_purpose.startswith("Returns anomaly-score")
    assert score_evidence.selection_reason == "Verify reject score concentration."
    assert score_evidence.duration_ms >= 0.0
    assert score_evidence.executed_at.tzinfo == timezone.utc
    assert score_evidence.error is None
    assert score_evidence.result["cohort"] == "AUTO_REJECT"
    assert score_evidence.result["total_matching_records"] == 2

    dep_evidence = bundle.evidence[1]
    assert dep_evidence.tool_name == "get_deployment_context"
    assert dep_evidence.error is None
    assert len(dep_evidence.result["version_counts"]) == 1
    assert dep_evidence.result["version_counts"][0]["model_version"] == "model-v1"


def test_collect_evidence_handles_missing_tool_in_registry():
    reader = MemoryReader([])
    req = sample_request()
    registry = build_default_tool_registry()

    # Create a plan with a valid call, but then simulate the tool missing from registry at execution time
    plan = validate_investigation_plan(
        req,
        InvestigationPlanProposal(
            investigation_goal="Analyze score distribution.",
            tool_calls=[
                ToolCallProposal(
                    tool_name="get_score_band_distribution",
                    arguments={
                        "start": START.isoformat(),
                        "end": (START + timedelta(days=14)).isoformat(),
                        "cohort": "AUTO_REJECT",
                    },
                    selection_reason="Verify reject score concentration.",
                    time_range_strategy=TimeRangeStrategy.BASELINE_AND_CURRENT,
                )
            ],
        ),
        registry,
    )

    empty_registry = MetricToolRegistry()
    bundle = collect_evidence(plan, reader, empty_registry)

    assert bundle.request is None
    assert len(bundle.evidence) == 1
    evidence = bundle.evidence[0]
    assert evidence.tool_name == "get_score_band_distribution"
    assert evidence.error is not None
    assert "not found in metric tool registry" in evidence.error
    assert evidence.result == {}


def test_collect_evidence_handles_tool_exception_gracefully():
    class BrokenTool:
        descriptor = ToolDescriptor(
            name="broken_tool",
            purpose="Fails intentionally during execution.",
            when_useful=["Testing error resilience."],
            database_fields=["none"],
            input_schema=TimeRangeInput.model_json_schema(),
            output_schema={},
            read_only=True,
        )
        input_model = TimeRangeInput

        def run(self, reader, raw_input):
            raise RuntimeError("Database query failed critically.")

    registry = MetricToolRegistry()
    registry.register(BrokenTool())

    req = sample_request()
    plan = validate_investigation_plan(
        req,
        InvestigationPlanProposal(
            investigation_goal="Run broken tool.",
            tool_calls=[
                ToolCallProposal(
                    tool_name="broken_tool",
                    arguments={
                        "start": START.isoformat(),
                        "end": (START + timedelta(days=14)).isoformat(),
                    },
                    selection_reason="Test error handling.",
                    time_range_strategy=TimeRangeStrategy.BASELINE_AND_CURRENT,
                )
            ],
        ),
        registry,
    )

    reader = MemoryReader([])
    bundle = collect_evidence(plan, reader, registry)

    assert len(bundle.evidence) == 1
    evidence = bundle.evidence[0]
    assert evidence.tool_name == "broken_tool"
    assert evidence.error == "Database query failed critically."
    assert evidence.result == {}
    assert evidence.duration_ms >= 0.0


def test_evidence_models_are_frozen():
    evidence = CollectedToolEvidence(
        tool_name="test_tool",
        tool_purpose="purpose",
        selection_reason="reason",
        arguments={},
        result={},
        duration_ms=1.5,
        executed_at=datetime.now(timezone.utc),
    )

    with pytest.raises(ValidationError):
        evidence.tool_name = "modified_tool"


def test_collect_evidence_positional_without_request():
    reader = MemoryReader([])
    req = sample_request()
    registry = build_default_tool_registry()

    plan = validate_investigation_plan(
        req,
        InvestigationPlanProposal(
            investigation_goal="Check deployment context only.",
            tool_calls=[
                ToolCallProposal(
                    tool_name="get_deployment_context",
                    arguments={
                        "start": START.isoformat(),
                        "end": (START + timedelta(days=14)).isoformat(),
                    },
                    selection_reason="Check deployment context.",
                    time_range_strategy=TimeRangeStrategy.BASELINE_AND_CURRENT,
                )
            ],
        ),
        registry,
    )

    # Calling with only positional (plan, reader, registry)
    bundle = collect_evidence(plan, reader, registry)

    assert bundle.request is None
    assert bundle.plan is plan
    assert len(bundle.evidence) == 1
    assert bundle.evidence[0].tool_name == "get_deployment_context"
    assert bundle.evidence[0].error is None
    assert bundle.evidence[0].duration_ms >= 0.0

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from app.quality_sentinel.comparison import (
    CohortRateComparison,
    RateBasis,
    SampleSufficiency,
)
from app.quality_sentinel.evidence import CollectedToolEvidence, EvidenceBundle
from app.quality_sentinel.planning import (
    TimeRangeStrategy,
    ValidatedInvestigationPlan,
    ValidatedToolCall,
)
from app.quality_sentinel.screening import (
    ScreeningDecision,
    ScreeningOutcome,
    SignalDirection,
    SignalKind,
)
from app.quality_sentinel.snapshot import AnalysisWindow, InspectionCohort
from app.quality_sentinel.synthesis import (
    DataGapSignal,
    EvidenceValidationStatus,
    QualitySignalCard,
    SafeSignalSynthesizer,
    SignalSynthesizer,
    validate_evidence,
)

START = datetime(2026, 11, 1, tzinfo=timezone.utc)


def build_test_plan(
    cohort: InspectionCohort = InspectionCohort.AUTO_REJECT,
    signal_kind: SignalKind = SignalKind.RATE_SHIFT,
    current_count: int = 10,
    current_denominator: int = 50,
    current_rate: float = 0.20,
    baseline_count: int = 2,
    baseline_denominator: int = 50,
    baseline_rate: float = 0.04,
    rate_delta: float = 0.16,
    relative_rate_change: float | None = 4.0,
    sample_sufficiency: SampleSufficiency = SampleSufficiency.SUFFICIENT,
    tool_names: list[str] | None = None,
) -> ValidatedInvestigationPlan:
    if tool_names is None:
        tool_names = ["get_score_band_distribution"]

    comparison = CohortRateComparison(
        cohort=cohort,
        rate_basis=RateBasis.ALL_INSPECTIONS,
        current_count=current_count,
        current_denominator=current_denominator,
        current_rate=current_rate,
        baseline_count=baseline_count,
        baseline_denominator=baseline_denominator,
        baseline_rate=baseline_rate,
        rate_delta=rate_delta,
        relative_rate_change=relative_rate_change,
        sample_sufficiency=sample_sufficiency,
    )
    decision = ScreeningDecision(
        cohort=cohort,
        outcome=ScreeningOutcome.SELECTED,
        signal_kind=signal_kind,
        direction=SignalDirection.INCREASE,
        reason="Screening test decision.",
        comparison=comparison,
    )
    allowed_window = AnalysisWindow(start=START, end=START + timedelta(days=14))
    tool_calls = [
        ValidatedToolCall(
            tool_name=name,
            tool_purpose=f"Purpose for {name}",
            selection_reason="Verify evidence.",
            time_range_strategy=TimeRangeStrategy.BASELINE_AND_CURRENT,
            time_range_reason="Full investigation scope.",
            arguments={
                "start": START.isoformat(),
                "end": (START + timedelta(days=14)).isoformat(),
            },
        )
        for name in tool_names
    ]
    return ValidatedInvestigationPlan(
        signal=decision,
        allowed_window=allowed_window,
        investigation_goal="Test investigation plan.",
        tool_calls=tool_calls,
    )


def test_validate_evidence_status_data_gap_on_empty_evidence():
    plan = build_test_plan()
    bundle = EvidenceBundle(
        plan=plan,
        evidence=[],
        collected_at=START + timedelta(days=14),
    )
    assert validate_evidence(bundle) is EvidenceValidationStatus.DATA_GAP


def test_validate_evidence_status_data_gap_when_all_tools_error():
    plan = build_test_plan(tool_names=["tool_a", "tool_b"])
    bundle = EvidenceBundle(
        plan=plan,
        evidence=[
            CollectedToolEvidence(
                tool_name="tool_a",
                tool_purpose="purpose a",
                selection_reason="reason a",
                arguments={},
                result={},
                duration_ms=1.2,
                executed_at=START,
                error="Database connection timeout",
            ),
            CollectedToolEvidence(
                tool_name="tool_b",
                tool_purpose="purpose b",
                selection_reason="reason b",
                arguments={},
                result={},
                duration_ms=0.8,
                executed_at=START,
                error="Query execution failed",
            ),
        ],
        collected_at=START + timedelta(days=14),
    )
    assert validate_evidence(bundle) is EvidenceValidationStatus.DATA_GAP


def test_validate_evidence_status_data_gap_when_all_tools_empty():
    plan = build_test_plan(tool_names=["get_deployment_context", "get_capture_quality_summary"])
    bundle = EvidenceBundle(
        plan=plan,
        evidence=[
            CollectedToolEvidence(
                tool_name="get_deployment_context",
                tool_purpose="purpose",
                selection_reason="reason",
                arguments={},
                result={"version_counts": []},
                duration_ms=2.1,
                executed_at=START,
                error=None,
            ),
            CollectedToolEvidence(
                tool_name="get_capture_quality_summary",
                tool_purpose="purpose",
                selection_reason="reason",
                arguments={},
                result={"records_with_quality": 0, "missing_quality_count": 0},
                duration_ms=1.5,
                executed_at=START,
                error=None,
            ),
        ],
        collected_at=START + timedelta(days=14),
    )
    assert validate_evidence(bundle) is EvidenceValidationStatus.DATA_GAP


def test_validate_evidence_status_insufficient_sample():
    plan = build_test_plan(sample_sufficiency=SampleSufficiency.CURRENT_TOO_SMALL)
    bundle = EvidenceBundle(
        plan=plan,
        evidence=[
            CollectedToolEvidence(
                tool_name="get_score_band_distribution",
                tool_purpose="purpose",
                selection_reason="reason",
                arguments={},
                result={"total_matching_records": 2, "missing_evidence_count": 0},
                duration_ms=2.0,
                executed_at=START,
                error=None,
            )
        ],
        collected_at=START + timedelta(days=14),
    )
    assert validate_evidence(bundle) is EvidenceValidationStatus.INSUFFICIENT_SAMPLE


def test_validate_evidence_status_sufficient():
    plan = build_test_plan(sample_sufficiency=SampleSufficiency.SUFFICIENT)
    bundle = EvidenceBundle(
        plan=plan,
        evidence=[
            CollectedToolEvidence(
                tool_name="get_score_band_distribution",
                tool_purpose="purpose",
                selection_reason="reason",
                arguments={},
                result={"total_matching_records": 10, "missing_evidence_count": 0},
                duration_ms=2.0,
                executed_at=START,
                error=None,
            )
        ],
        collected_at=START + timedelta(days=14),
    )
    assert validate_evidence(bundle) is EvidenceValidationStatus.SUFFICIENT


def test_safe_synthesizer_produces_data_gap_signal_on_empty_evidence():
    plan = build_test_plan()
    bundle = EvidenceBundle(
        plan=plan,
        evidence=[],
        collected_at=START + timedelta(days=14),
    )
    synthesizer: SignalSynthesizer = SafeSignalSynthesizer()
    output = synthesizer.synthesize(bundle)

    assert isinstance(output, DataGapSignal)
    assert output.signal_id.startswith("GAP-AUTO-REJECT-")
    assert output.cohort is InspectionCohort.AUTO_REJECT
    assert "Data gap detected" in output.headline
    assert output.attempted_tools == ["get_score_band_distribution"]
    assert "No tool evidence collected" in output.missing_evidence_summary
    assert output.created_at.tzinfo == timezone.utc


def test_safe_synthesizer_produces_data_gap_signal_on_insufficient_sample():
    plan = build_test_plan(
        sample_sufficiency=SampleSufficiency.CURRENT_TOO_SMALL,
        current_count=2,
        current_denominator=5,
    )
    bundle = EvidenceBundle(
        plan=plan,
        evidence=[
            CollectedToolEvidence(
                tool_name="get_deployment_context",
                tool_purpose="purpose",
                selection_reason="reason",
                arguments={},
                result={
                    "version_counts": [{"model_version": "m1", "policy_version": "p1", "count": 2}]
                },
                duration_ms=1.0,
                executed_at=START,
                error=None,
            )
        ],
        collected_at=START + timedelta(days=14),
    )
    synthesizer = SafeSignalSynthesizer()
    output = synthesizer.synthesize(bundle)

    assert isinstance(output, DataGapSignal)
    assert output.signal_id.startswith("GAP-AUTO-REJECT-")
    assert "Insufficient sample size" in output.headline
    assert "CURRENT_TOO_SMALL" in output.gap_reason
    assert output.attempted_tools == ["get_deployment_context"]


def test_safe_synthesizer_formats_complete_quality_signal_card():
    plan = build_test_plan(
        cohort=InspectionCohort.AUTO_REJECT,
        signal_kind=SignalKind.RATE_SHIFT,
        current_count=10,
        current_denominator=50,
        current_rate=0.20,
        baseline_count=2,
        baseline_denominator=50,
        baseline_rate=0.04,
        rate_delta=0.16,
        relative_rate_change=4.0,
        sample_sufficiency=SampleSufficiency.SUFFICIENT,
        tool_names=[
            "get_deployment_context",
            "get_capture_quality_summary",
            "get_score_band_distribution",
            "get_review_reason_breakdown",
            "get_cohort_trend",
        ],
    )

    evidence_items = [
        CollectedToolEvidence(
            tool_name="get_deployment_context",
            tool_purpose="Context",
            selection_reason="Deployment check",
            arguments={},
            result={
                "version_counts": [
                    {"model_version": "model-v2", "policy_version": "policy-v2", "count": 10}
                ]
            },
            duration_ms=2.5,
            executed_at=START,
            error=None,
        ),
        CollectedToolEvidence(
            tool_name="get_capture_quality_summary",
            tool_purpose="Quality",
            selection_reason="Check quality gate",
            arguments={},
            result={
                "records_with_quality": 10,
                "acceptable_count": 8,
                "unacceptable_count": 2,
                "missing_quality_count": 0,
                "failure_reason_counts": [{"reason_code": "IMAGE_BLUR", "count": 2}],
            },
            duration_ms=3.1,
            executed_at=START,
            error=None,
        ),
        CollectedToolEvidence(
            tool_name="get_score_band_distribution",
            tool_purpose="Scores",
            selection_reason="Score bands",
            arguments={},
            result={
                "total_matching_records": 10,
                "missing_evidence_count": 0,
                "distributions": [
                    {
                        "model_version": "model-v2",
                        "review_threshold": 0.315,
                        "reject_threshold": 0.334,
                        "total_records": 10,
                        "bands": [
                            {"band": "REVIEW_TO_REJECT_THRESHOLD", "count": 7},
                            {"band": "AT_OR_ABOVE_REJECT_THRESHOLD", "count": 3},
                        ],
                    }
                ],
            },
            duration_ms=1.9,
            executed_at=START,
            error=None,
        ),
        CollectedToolEvidence(
            tool_name="get_review_reason_breakdown",
            tool_purpose="Breakdown",
            selection_reason="Human review",
            arguments={},
            result={
                "resolved_inspect_count": 5,
                "reason_counts": [
                    {"reason_code": "ACCEPTED_PROCESS_RESIDUE", "count": 3},
                    {"reason_code": "NO_DEFECT_FOUND", "count": 2},
                ],
            },
            duration_ms=2.0,
            executed_at=START,
            error=None,
        ),
        CollectedToolEvidence(
            tool_name="get_cohort_trend",
            tool_purpose="Trend",
            selection_reason="Trend evaluation",
            arguments={},
            result={
                "points": [
                    {"cohort_count": 4, "denominator": 25, "rate": 0.16},
                    {"cohort_count": 6, "denominator": 25, "rate": 0.24},
                ]
            },
            duration_ms=4.0,
            executed_at=START,
            error=None,
        ),
    ]

    bundle = EvidenceBundle(
        plan=plan,
        evidence=evidence_items,
        collected_at=START + timedelta(days=14),
    )

    synthesizer = SafeSignalSynthesizer()
    card = synthesizer.synthesize(bundle, sop_reference="SOP-GRID-INSPECT-002")

    assert isinstance(card, QualitySignalCard)
    assert card.signal_id.startswith("SIG-AUTO-REJECT-")
    assert card.cohort is InspectionCohort.AUTO_REJECT
    assert card.signal_kind is SignalKind.RATE_SHIFT
    assert card.baseline_rate == 0.04
    assert card.current_rate == 0.20
    assert card.rate_delta == 0.16
    assert card.relative_change == 4.0
    assert card.sample_count == 10
    assert card.sop_reference == "SOP-GRID-INSPECT-002"
    assert card.created_at.tzinfo == timezone.utc

    # Check headline quotes actual rates
    assert "+16.0%" in card.headline
    assert "4.0%" in card.headline
    assert "20.0%" in card.headline

    # Check summary contains rates, failure reasons, and deployments
    assert "AUTO_REJECT" in card.summary
    assert "+16.0%" in card.summary
    assert "IMAGE_BLUR" in card.summary
    assert "model-v2/policy-v2" in card.summary
    assert "SOP-GRID-INSPECT-002" in card.summary

    # Check citations exist for tools
    assert len(card.evidence_citations) == 5
    citation_text = " ".join(card.evidence_citations)
    assert "get_deployment_context" in citation_text
    assert "get_capture_quality_summary" in citation_text
    assert "get_score_band_distribution" in citation_text
    assert "get_review_reason_breakdown" in citation_text
    assert "get_cohort_trend" in citation_text

    # Check recommended action reflects failure reasons and references SOP
    assert "IMAGE_BLUR" in card.recommended_action
    assert "SOP-GRID-INSPECT-002" in card.recommended_action


def test_safe_synthesizer_emergence_signal_card():
    plan = build_test_plan(
        cohort=InspectionCohort.REVIEWED_PASS_OTHER,
        signal_kind=SignalKind.EMERGENCE,
        current_count=5,
        current_denominator=20,
        current_rate=0.25,
        baseline_count=0,
        baseline_denominator=20,
        baseline_rate=0.0,
        rate_delta=0.25,
        relative_rate_change=None,
    )
    bundle = EvidenceBundle(
        plan=plan,
        evidence=[
            CollectedToolEvidence(
                tool_name="get_review_reason_breakdown",
                tool_purpose="Breakdown",
                selection_reason="Review check",
                arguments={},
                result={
                    "resolved_inspect_count": 5,
                    "reason_counts": [{"reason_code": "LEGACY_UNCLASSIFIED", "count": 5}],
                },
                duration_ms=1.5,
                executed_at=START,
                error=None,
            )
        ],
        collected_at=START + timedelta(days=14),
    )

    synthesizer = SafeSignalSynthesizer()
    card = synthesizer.synthesize(bundle)

    assert isinstance(card, QualitySignalCard)
    assert card.signal_kind is SignalKind.EMERGENCE
    assert card.baseline_rate == 0.0
    assert card.current_rate == 0.25
    assert card.rate_delta == 0.25
    assert card.relative_change is None
    assert "emerged" in card.headline.lower()


def test_synthesis_models_are_frozen():
    card = QualitySignalCard(
        signal_id="sig-1",
        cohort=InspectionCohort.AUTO_REJECT,
        signal_kind=SignalKind.RATE_SHIFT,
        headline="headline",
        summary="summary",
        baseline_rate=0.05,
        current_rate=0.15,
        rate_delta=0.10,
        relative_change=2.0,
        sample_count=10,
        evidence_citations=["cit-1"],
        recommended_action="action",
        created_at=datetime.now(timezone.utc),
    )
    with pytest.raises(ValidationError):
        card.headline = "changed headline"

    gap = DataGapSignal(
        signal_id="gap-1",
        cohort=InspectionCohort.AUTO_REJECT,
        headline="headline",
        missing_evidence_summary="summary",
        attempted_tools=["tool_a"],
        gap_reason="reason",
        created_at=datetime.now(timezone.utc),
    )
    with pytest.raises(ValidationError):
        gap.headline = "changed headline"

from datetime import datetime, timedelta, timezone

import pytest

from app.quality_sentinel.comparison import (
    BaselineComparison,
    CohortRateComparison,
    ComparisonConfig,
    RateBasis,
    SampleSufficiency,
)
from app.quality_sentinel.investigation import build_default_tool_registry
from app.quality_sentinel.planning import (
    BoundedPlanningResult,
    InvestigationPlanProposal,
    InvestigationRequest,
    PlanningConfig,
    PlanningHandoffStatus,
    PlanStatus,
    PlanValidationErrorCode,
    TimeRangeStrategy,
    ToolCallProposal,
    handoff_bounded_planning,
    run_bounded_planning,
    validate_investigation_plan,
    validate_investigation_plan_result,
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
)

START = datetime(2026, 11, 1, tzinfo=timezone.utc)


def report(outcome: ScreeningOutcome = ScreeningOutcome.SELECTED) -> ScreeningReport:
    baseline_window = AnalysisWindow(start=START, end=START + timedelta(days=7))
    current_window = AnalysisWindow(
        start=START + timedelta(days=7), end=START + timedelta(days=14)
    )
    baseline = CohortSnapshot(window=baseline_window, source_inspection_count=30, cohort_counts={})
    current = CohortSnapshot(window=current_window, source_inspection_count=30, cohort_counts={})
    comparison = CohortRateComparison(
        cohort=InspectionCohort.AUTO_REJECT,
        rate_basis=RateBasis.ALL_INSPECTIONS,
        current_count=8,
        current_denominator=30,
        current_rate=8 / 30,
        baseline_count=2,
        baseline_denominator=30,
        baseline_rate=2 / 30,
        rate_delta=0.2,
        relative_rate_change=3.0,
        sample_sufficiency=SampleSufficiency.SUFFICIENT,
    )
    decision = ScreeningDecision(
        cohort=InspectionCohort.AUTO_REJECT,
        outcome=outcome,
        signal_kind=SignalKind.RATE_SHIFT if outcome is ScreeningOutcome.SELECTED else None,
        direction=SignalDirection.INCREASE if outcome is ScreeningOutcome.SELECTED else None,
        reason="Test signal.",
        comparison=comparison,
    )
    return ScreeningReport(
        comparison=BaselineComparison(
            current=current,
            baseline=baseline,
            config=ComparisonConfig(),
            cohorts=[comparison],
        ),
        config=ScreeningConfig(),
        decisions=[decision],
    )


def request(outcome: ScreeningOutcome = ScreeningOutcome.SELECTED) -> InvestigationRequest:
    return InvestigationRequest(screening_report=report(outcome), cohort=InspectionCohort.AUTO_REJECT)


def score_distribution_call(
    start: datetime = START,
    end: datetime = START + timedelta(days=14),
) -> ToolCallProposal:
    return ToolCallProposal(
        tool_name="get_score_band_distribution",
        arguments={
            "start": start.isoformat(),
            "end": end.isoformat(),
            "cohort": "AUTO_REJECT",
        },
        selection_reason="Determine whether reject scores moved toward a decision boundary.",
        time_range_strategy=TimeRangeStrategy.BASELINE_AND_CURRENT,
    )


def proposal(*calls: ToolCallProposal) -> InvestigationPlanProposal:
    return InvestigationPlanProposal(
        investigation_goal="Investigate the selected AUTO_REJECT rate shift with bounded evidence.",
        tool_calls=list(calls),
    )


def test_plan_validates_registered_tool_input_and_keeps_the_evidence_scope():
    plan = validate_investigation_plan(
        request(),
        proposal(score_distribution_call()),
        build_default_tool_registry(),
    )

    assert plan.signal.cohort is InspectionCohort.AUTO_REJECT
    assert plan.tool_calls[0].tool_name == "get_score_band_distribution"
    assert plan.tool_calls[0].tool_purpose.startswith("Returns anomaly-score counts")
    assert plan.tool_calls[0].selection_reason.startswith("Determine whether reject scores")
    assert plan.tool_calls[0].time_range_strategy is TimeRangeStrategy.BASELINE_AND_CURRENT
    assert plan.tool_calls[0].time_range_reason.startswith("Compare the baseline period")
    assert datetime.fromisoformat(plan.tool_calls[0].arguments["start"].replace("Z", "+00:00")) == START
    assert plan.allowed_window.start == START


def test_plan_rejects_an_unselected_signal():
    with pytest.raises(ValueError, match="not a selected investigation signal"):
        validate_investigation_plan(
            request(ScreeningOutcome.BELOW_MATERIALITY_THRESHOLD),
            proposal(score_distribution_call()),
            build_default_tool_registry(),
        )


def test_plan_rejects_an_unknown_tool_before_any_database_query():
    with pytest.raises(ValueError, match="is not registered"):
        validate_investigation_plan(
            request(),
            proposal(
                ToolCallProposal(
                    tool_name="delete_history",
                    arguments={},
                    selection_reason="This should be rejected before execution.",
                    time_range_strategy=TimeRangeStrategy.BASELINE_AND_CURRENT,
                )
            ),
            build_default_tool_registry(),
        )


def test_structured_validation_failure_is_safe_and_points_to_the_invalid_tool_call():
    result = validate_investigation_plan_result(
        request(),
        proposal(
            ToolCallProposal(
                tool_name="delete_history",
                arguments={},
                selection_reason="This should be rejected before execution.",
                time_range_strategy=TimeRangeStrategy.BASELINE_AND_CURRENT,
            )
        ),
        build_default_tool_registry(),
    )

    assert result.status is PlanStatus.REJECTED
    assert result.plan is None
    assert result.failure is not None
    assert result.failure.code is PlanValidationErrorCode.UNKNOWN_TOOL
    assert result.failure.tool_call_index == 0
    assert result.failure.tool_name == "delete_history"
    assert "registry" in result.failure.repair_hint


def test_structured_validation_failure_identifies_invalid_cohort_input():
    invalid_call = score_distribution_call().model_copy(
        update={"arguments": {**score_distribution_call().arguments, "cohort": "GLUE_DEFECT"}}
    )
    result = validate_investigation_plan_result(
        request(), proposal(invalid_call), build_default_tool_registry()
    )

    assert result.status is PlanStatus.REJECTED
    assert result.failure is not None
    assert result.failure.code is PlanValidationErrorCode.INVALID_TOOL_INPUT
    assert result.failure.tool_name == "get_score_band_distribution"


def test_plan_rejects_a_call_outside_the_allowed_evidence_window():
    with pytest.raises(ValueError, match="dates do not match the approved evidence"):
        validate_investigation_plan(
            request(),
            proposal(score_distribution_call(start=START - timedelta(seconds=1))),
            build_default_tool_registry(),
        )


def test_plan_rejects_a_declared_time_strategy_that_does_not_match_its_dates():
    call = score_distribution_call()
    invalid_call = call.model_copy(update={"time_range_strategy": TimeRangeStrategy.CURRENT_ONLY})
    with pytest.raises(ValueError, match="dates do not match the approved evidence"):
        validate_investigation_plan(request(), proposal(invalid_call), build_default_tool_registry())


def test_custom_time_ranges_need_a_specific_reason_and_remain_within_evidence():
    with pytest.raises(ValueError, match="custom time ranges require"):
        ToolCallProposal.model_validate(
            {
                **score_distribution_call().model_dump(),
                "time_range_strategy": TimeRangeStrategy.CUSTOM_WITHIN_EVIDENCE,
                "custom_time_range_reason": None,
            }
        )

    custom_call = score_distribution_call(
        start=START + timedelta(days=7), end=START + timedelta(days=14)
    ).model_copy(
        update={
            "time_range_strategy": TimeRangeStrategy.CUSTOM_WITHIN_EVIDENCE,
            "custom_time_range_reason": "Check only the signal window after the observed rate shift.",
        }
    )
    plan = validate_investigation_plan(request(), proposal(custom_call), build_default_tool_registry())

    assert "Check only the signal window" in plan.tool_calls[0].time_range_reason


def test_plan_enforces_a_tool_call_budget_and_rejects_exact_duplicates():
    registry = build_default_tool_registry()
    call = score_distribution_call()
    with pytest.raises(ValueError, match="limit of 1"):
        validate_investigation_plan(
            request(),
            proposal(call, call),
            registry,
            PlanningConfig(maximum_tool_calls=1),
        )
    with pytest.raises(ValueError, match="exact duplicate"):
        validate_investigation_plan(
            request(),
            proposal(call, call),
            registry,
        )


class ScriptedPlanner:
    def __init__(self, proposals: list[InvestigationPlanProposal]) -> None:
        self.proposals = proposals
        self.repair_contexts = []

    def propose(self, *, request, available_tools, repair_context):
        self.repair_contexts.append(repair_context)
        assert request.cohort is InspectionCohort.AUTO_REJECT
        assert available_tools
        return self.proposals.pop(0)


def test_bounded_planning_repairs_one_invalid_plan_then_returns_a_validated_plan():
    invalid_call = score_distribution_call().model_copy(
        update={"arguments": {**score_distribution_call().arguments, "cohort": "GLUE_DEFECT"}}
    )
    planner = ScriptedPlanner([proposal(invalid_call), proposal(score_distribution_call())])

    result = run_bounded_planning(request(), planner, build_default_tool_registry())

    assert isinstance(result, BoundedPlanningResult)
    assert result.status is PlanStatus.VALIDATED
    assert [attempt.validation.status for attempt in result.attempts] == [
        PlanStatus.REJECTED,
        PlanStatus.VALIDATED,
    ]
    assert planner.repair_contexts[0] is None
    assert planner.repair_contexts[1].code is PlanValidationErrorCode.INVALID_TOOL_INPUT


def test_bounded_planning_stops_after_the_single_repair_budget_is_used():
    unknown_tool = ToolCallProposal(
        tool_name="delete_history",
        arguments={},
        selection_reason="Invalid tool for test coverage.",
        time_range_strategy=TimeRangeStrategy.BASELINE_AND_CURRENT,
    )
    planner = ScriptedPlanner([proposal(unknown_tool), proposal(unknown_tool)])

    result = run_bounded_planning(request(), planner, build_default_tool_registry())

    assert result.status is PlanStatus.REJECTED
    assert len(result.attempts) == 2
    assert result.failure is not None
    assert result.failure.code is PlanValidationErrorCode.UNKNOWN_TOOL


def test_bounded_planning_does_not_retry_a_non_retryable_signal_failure():
    planner = ScriptedPlanner([proposal(score_distribution_call())])

    result = run_bounded_planning(
        request(ScreeningOutcome.BELOW_MATERIALITY_THRESHOLD),
        planner,
        build_default_tool_registry(),
    )

    assert result.status is PlanStatus.REJECTED
    assert len(result.attempts) == 1
    assert result.failure is not None
    assert result.failure.code is PlanValidationErrorCode.SIGNAL_NOT_SELECTED


def test_failed_planning_closes_safely_without_evidence_or_a_finding():
    unknown_tool = ToolCallProposal(
        tool_name="delete_history",
        arguments={},
        selection_reason="Invalid tool for test coverage.",
        time_range_strategy=TimeRangeStrategy.BASELINE_AND_CURRENT,
    )
    planning_result = run_bounded_planning(
        request(),
        ScriptedPlanner([proposal(unknown_tool), proposal(unknown_tool)]),
        build_default_tool_registry(),
    )

    handoff = handoff_bounded_planning(planning_result)

    assert handoff.status is PlanningHandoffStatus.INVESTIGATION_NOT_COMPLETED
    assert handoff.plan is None
    assert handoff.closure is not None
    assert handoff.closure.evidence_count == 0
    assert handoff.closure.finding is None
    assert handoff.closure.failure.code is PlanValidationErrorCode.UNKNOWN_TOOL


def test_validated_planning_hands_off_only_the_validated_plan_for_evidence_collection():
    planning_result = run_bounded_planning(
        request(),
        ScriptedPlanner([proposal(score_distribution_call())]),
        build_default_tool_registry(),
    )

    handoff = handoff_bounded_planning(planning_result)

    assert handoff.status is PlanningHandoffStatus.READY_FOR_EVIDENCE_COLLECTION
    assert handoff.plan is not None
    assert handoff.closure is None

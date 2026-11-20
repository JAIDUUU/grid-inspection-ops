"""Deterministic rule-based investigation planner for quality sentinel."""

from __future__ import annotations

from typing import Any

from .investigation import ToolDescriptor
from .planning import (
    InvestigationPlanner,
    InvestigationPlanProposal,
    InvestigationRequest,
    PlanValidationErrorCode,
    PlanValidationFailure,
    TimeRangeStrategy,
    ToolCallProposal,
)
from .snapshot import InspectionCohort

_REVIEW_COHORTS = {
    InspectionCohort.PENDING_INSPECT,
    InspectionCohort.REVIEWED_PASS_NO_DEFECT_FOUND,
    InspectionCohort.REVIEWED_PASS_ACCEPTED_PROCESS_RESIDUE,
    InspectionCohort.REVIEWED_PASS_OTHER,
    InspectionCohort.REVIEWED_REJECT,
}


class DefaultRulePlanner(InvestigationPlanner):
    """Deterministic, safe planner choosing verified metric tools by cohort type."""

    def __init__(
        self,
        default_strategy: TimeRangeStrategy = TimeRangeStrategy.BASELINE_AND_CURRENT,
    ) -> None:
        self.default_strategy = default_strategy

    def propose(
        self,
        *,
        request: InvestigationRequest,
        available_tools: list[ToolDescriptor],
        repair_context: PlanValidationFailure | None = None,
    ) -> InvestigationPlanProposal:
        """Select safe read-only metric tools based on the screened cohort type."""
        available_names = {t.name for t in available_tools}
        cohort = request.cohort

        strategy = self.default_strategy
        if (
            repair_context
            and repair_context.code is PlanValidationErrorCode.TIME_RANGE_STRATEGY_MISMATCH
        ):
            strategy = TimeRangeStrategy.BASELINE_AND_CURRENT

        baseline_window = request.screening_report.comparison.baseline.window
        current_window = request.screening_report.comparison.current.window

        if strategy is TimeRangeStrategy.CURRENT_ONLY:
            start_iso = current_window.start.isoformat()
            end_iso = current_window.end.isoformat()
        elif strategy is TimeRangeStrategy.BASELINE_ONLY:
            start_iso = baseline_window.start.isoformat()
            end_iso = baseline_window.end.isoformat()
        else:
            strategy = TimeRangeStrategy.BASELINE_AND_CURRENT
            start_iso = baseline_window.start.isoformat()
            end_iso = current_window.end.isoformat()

        candidate_tools: list[tuple[str, dict[str, Any], str]] = []

        candidate_tools.append(
            (
                "get_cohort_trend",
                {
                    "start": start_iso,
                    "end": end_iso,
                    "cohort": cohort.value,
                    "bucket_hours": 24,
                },
                f"Track trend and volume history for cohort {cohort.value}.",
            )
        )

        if cohort in {InspectionCohort.AUTO_REJECT, InspectionCohort.AUTO_PASS}:
            candidate_tools.append(
                (
                    "get_score_band_distribution",
                    {
                        "start": start_iso,
                        "end": end_iso,
                        "cohort": cohort.value,
                    },
                    f"Analyze score band distribution for automated cohort {cohort.value}.",
                )
            )
        elif cohort in _REVIEW_COHORTS:
            candidate_tools.append(
                (
                    "get_review_reason_breakdown",
                    {
                        "start": start_iso,
                        "end": end_iso,
                    },
                    "Analyze human reviewer disposition reason breakdown.",
                )
            )

        candidate_tools.append(
            (
                "get_deployment_context",
                {
                    "start": start_iso,
                    "end": end_iso,
                },
                "Inspect model and policy versions active across the evaluation window.",
            )
        )

        excluded_tools: set[str] = set()
        if (
            repair_context
            and repair_context.tool_name
            and repair_context.code
            in {
                PlanValidationErrorCode.UNKNOWN_TOOL,
                PlanValidationErrorCode.INVALID_TOOL_INPUT,
            }
        ):
            excluded_tools.add(repair_context.tool_name)

        calls: list[ToolCallProposal] = []
        seen_keys: set[str] = set()

        for tool_name, arguments, reason in candidate_tools:
            if tool_name not in available_names or tool_name in excluded_tools:
                continue
            key = f"{tool_name}:{sorted(arguments.items())}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            calls.append(
                ToolCallProposal(
                    tool_name=tool_name,
                    arguments=arguments,
                    selection_reason=reason,
                    time_range_strategy=strategy,
                    custom_time_range_reason=None,
                )
            )

        if (
            repair_context
            and repair_context.code is PlanValidationErrorCode.TOOL_CALL_LIMIT_EXCEEDED
        ):
            calls = calls[:2]
        else:
            calls = calls[:3]

        if not calls and "get_cohort_trend" in available_names:
            calls.append(
                ToolCallProposal(
                    tool_name="get_cohort_trend",
                    arguments={
                        "start": start_iso,
                        "end": end_iso,
                        "cohort": cohort.value,
                        "bucket_hours": 24,
                    },
                    selection_reason=f"Baseline fallback trend query for cohort {cohort.value}.",
                    time_range_strategy=strategy,
                    custom_time_range_reason=None,
                )
            )

        return InvestigationPlanProposal(
            investigation_goal=f"Investigate quality shift in cohort {cohort.value}.",
            tool_calls=calls,
        )

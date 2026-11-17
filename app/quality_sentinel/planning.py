"""Validation contract between a future planner and read-only metric tools.

This module does not select tools and does not call an LLM. It makes an
untrusted tool-call proposal safe and auditable before any query is executed.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .investigation import MetricToolRegistry, ToolDescriptor
from .screening import ScreeningDecision, ScreeningReport
from .snapshot import AnalysisWindow, InspectionCohort


class TimeRangeStrategy(str, Enum):
    """Approved ways an investigation call may select its evidence period."""

    BASELINE_ONLY = "BASELINE_ONLY"
    CURRENT_ONLY = "CURRENT_ONLY"
    BASELINE_AND_CURRENT = "BASELINE_AND_CURRENT"
    CUSTOM_WITHIN_EVIDENCE = "CUSTOM_WITHIN_EVIDENCE"


class ToolCallProposal(BaseModel):
    """One prospective read-only metric query from the future planner."""

    model_config = ConfigDict(frozen=True)

    tool_name: str = Field(pattern=r"^[a-z][a-z0-9_]{2,63}$")
    arguments: dict[str, Any]
    selection_reason: str = Field(min_length=1, max_length=500)
    time_range_strategy: TimeRangeStrategy
    custom_time_range_reason: str | None = Field(default=None, min_length=1, max_length=500)

    @model_validator(mode="after")
    def custom_ranges_have_a_reason(self) -> ToolCallProposal:
        if (
            self.time_range_strategy is TimeRangeStrategy.CUSTOM_WITHIN_EVIDENCE
            and self.custom_time_range_reason is None
        ):
            raise ValueError("custom time ranges require custom_time_range_reason")
        if (
            self.time_range_strategy is not TimeRangeStrategy.CUSTOM_WITHIN_EVIDENCE
            and self.custom_time_range_reason is not None
        ):
            raise ValueError("custom_time_range_reason is allowed only for custom time ranges")
        return self


class PlanningConfig(BaseModel):
    """Hard execution limits, enforced independently of an LLM."""

    model_config = ConfigDict(frozen=True)

    maximum_tool_calls: int = Field(default=3, ge=1, le=10)
    maximum_repair_attempts: int = Field(default=1, ge=0, le=3)


class InvestigationRequest(BaseModel):
    """A selected deterministic signal and the report that produced it."""

    model_config = ConfigDict(frozen=True)

    screening_report: ScreeningReport
    cohort: InspectionCohort

    @property
    def selected_signal(self) -> ScreeningDecision:
        for decision in self.screening_report.selected_signals:
            if decision.cohort is self.cohort:
                return decision
        raise ValueError(f"cohort {self.cohort.value} is not a selected investigation signal")

    @property
    def allowed_window(self) -> AnalysisWindow:
        """The bounded evidence range a plan is allowed to inspect."""

        return AnalysisWindow(
            start=self.screening_report.comparison.baseline.window.start,
            end=self.screening_report.comparison.current.window.end,
        )


class InvestigationPlanProposal(BaseModel):
    """Structured, untrusted planner output before registry and scope checks."""

    investigation_goal: str = Field(min_length=1, max_length=500)
    tool_calls: list[ToolCallProposal] = Field(min_length=1)


class PlanStatus(str, Enum):
    """Lifecycle state exposed to the later API and dashboard."""

    VALIDATED = "VALIDATED"
    REJECTED = "REJECTED"


class PlanValidationErrorCode(str, Enum):
    """Machine-readable reasons an untrusted plan cannot be executed."""

    SIGNAL_NOT_SELECTED = "SIGNAL_NOT_SELECTED"
    TOOL_CALL_LIMIT_EXCEEDED = "TOOL_CALL_LIMIT_EXCEEDED"
    UNKNOWN_TOOL = "UNKNOWN_TOOL"
    INVALID_TOOL_INPUT = "INVALID_TOOL_INPUT"
    TIME_RANGE_OUTSIDE_EVIDENCE = "TIME_RANGE_OUTSIDE_EVIDENCE"
    TIME_RANGE_STRATEGY_MISMATCH = "TIME_RANGE_STRATEGY_MISMATCH"
    DUPLICATE_TOOL_CALL = "DUPLICATE_TOOL_CALL"


class PlanValidationFailure(BaseModel):
    """Safe error payload for the dashboard and a bounded future repair turn."""

    model_config = ConfigDict(frozen=True)

    code: PlanValidationErrorCode
    message: str
    repair_hint: str
    retryable: bool
    tool_call_index: int | None = Field(default=None, ge=0)
    tool_name: str | None = None


class PlanValidationException(ValueError):
    """Exception adapter retained for synchronous callers and existing tests."""

    def __init__(self, failure: PlanValidationFailure) -> None:
        self.failure = failure
        super().__init__(failure.message)


class ValidatedToolCall(BaseModel):
    """A registered call with normalized arguments and transparent intent."""

    model_config = ConfigDict(frozen=True)

    tool_name: str
    tool_purpose: str
    selection_reason: str
    time_range_strategy: TimeRangeStrategy
    time_range_reason: str
    arguments: dict[str, Any]


class ValidatedInvestigationPlan(BaseModel):
    """The only plan that a later graph node may execute."""

    model_config = ConfigDict(frozen=True)

    signal: ScreeningDecision
    allowed_window: AnalysisWindow
    investigation_goal: str
    status: PlanStatus = PlanStatus.VALIDATED
    tool_calls: list[ValidatedToolCall]


class PlanValidationResult(BaseModel):
    """Non-exception result consumed by a future LangGraph validation node."""

    model_config = ConfigDict(frozen=True)

    status: PlanStatus
    plan: ValidatedInvestigationPlan | None = None
    failure: PlanValidationFailure | None = None

    @model_validator(mode="after")
    def result_has_exactly_one_terminal_payload(self) -> PlanValidationResult:
        if self.status is PlanStatus.VALIDATED and self.plan is not None and self.failure is None:
            return self
        if self.status is PlanStatus.REJECTED and self.plan is None and self.failure is not None:
            return self
        raise ValueError("validation result must contain exactly one status-matching payload")


class InvestigationPlanner(Protocol):
    """Boundary implemented later by an LLM adapter or a deterministic test double."""

    def propose(
        self,
        *,
        request: InvestigationRequest,
        available_tools: list[ToolDescriptor],
        repair_context: PlanValidationFailure | None,
    ) -> InvestigationPlanProposal: ...


class PlanningAttempt(BaseModel):
    """Sanitized audit entry; it intentionally does not retain raw invalid LLM output."""

    model_config = ConfigDict(frozen=True)

    number: int = Field(ge=1)
    validation: PlanValidationResult


class BoundedPlanningResult(BaseModel):
    """Terminal result after at most one configured plan-repair cycle."""

    model_config = ConfigDict(frozen=True)

    status: PlanStatus
    attempts: list[PlanningAttempt] = Field(min_length=1)
    plan: ValidatedInvestigationPlan | None = None
    failure: PlanValidationFailure | None = None

    @model_validator(mode="after")
    def result_matches_terminal_status(self) -> BoundedPlanningResult:
        if self.status is PlanStatus.VALIDATED and self.plan is not None and self.failure is None:
            return self
        if self.status is PlanStatus.REJECTED and self.plan is None and self.failure is not None:
            return self
        raise ValueError("bounded planning result must contain exactly one terminal payload")


class PlanningHandoffStatus(str, Enum):
    """Whether the next node may collect evidence or must close safely."""

    READY_FOR_EVIDENCE_COLLECTION = "READY_FOR_EVIDENCE_COLLECTION"
    INVESTIGATION_NOT_COMPLETED = "INVESTIGATION_NOT_COMPLETED"


class InvestigationSafetyClosure(BaseModel):
    """User-visible, evidence-free closure for an investigation that never ran."""

    model_config = ConfigDict(frozen=True)

    status: PlanningHandoffStatus = PlanningHandoffStatus.INVESTIGATION_NOT_COMPLETED
    title: str = "Investigation not completed"
    summary: str
    limitation: str = "No metric tools were executed, so no evidence-based finding was generated."
    planning_attempt_count: int = Field(ge=1)
    failure: PlanValidationFailure
    evidence_count: int = Field(default=0, ge=0, le=0)
    finding: None = None


class PlanningHandoff(BaseModel):
    """Safe branch point between planning and the future evidence executor."""

    model_config = ConfigDict(frozen=True)

    status: PlanningHandoffStatus
    plan: ValidatedInvestigationPlan | None = None
    closure: InvestigationSafetyClosure | None = None

    @model_validator(mode="after")
    def handoff_has_exactly_one_next_step(self) -> PlanningHandoff:
        if (
            self.status is PlanningHandoffStatus.READY_FOR_EVIDENCE_COLLECTION
            and self.plan is not None
            and self.closure is None
        ):
            return self
        if (
            self.status is PlanningHandoffStatus.INVESTIGATION_NOT_COMPLETED
            and self.plan is None
            and self.closure is not None
        ):
            return self
        raise ValueError("planning handoff must contain exactly one status-matching payload")


def _plan_failure(
    code: PlanValidationErrorCode,
    message: str,
    repair_hint: str,
    *,
    retryable: bool = True,
    tool_call_index: int | None = None,
    tool_name: str | None = None,
) -> PlanValidationException:
    return PlanValidationException(
        PlanValidationFailure(
            code=code,
            message=message,
            repair_hint=repair_hint,
            retryable=retryable,
            tool_call_index=tool_call_index,
            tool_name=tool_name,
        )
    )


def _parse_datetime(value: Any, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise TypeError(f"validated tool input must contain a serialized {field_name}")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _validate_time_scope(
    arguments: dict[str, Any],
    allowed_window: AnalysisWindow,
    baseline_window: AnalysisWindow,
    current_window: AnalysisWindow,
    proposal: ToolCallProposal,
) -> str:
    """Keep every current metric query inside the baseline-to-current evidence range."""

    try:
        start = _parse_datetime(arguments["start"], "start")
        end = _parse_datetime(arguments["end"], "end")
    except KeyError as error:
        raise ValueError("quality investigation tools must use bounded start and end arguments") from error
    if start < allowed_window.start or end > allowed_window.end:
        raise ValueError("tool call time range falls outside the allowed investigation window")
    strategy = proposal.time_range_strategy
    if strategy is TimeRangeStrategy.BASELINE_ONLY:
        if start != baseline_window.start or end != baseline_window.end:
            raise ValueError("BASELINE_ONLY calls must use the exact baseline window")
        return "Use the baseline window as the pre-change comparison period."
    if strategy is TimeRangeStrategy.CURRENT_ONLY:
        if start != current_window.start or end != current_window.end:
            raise ValueError("CURRENT_ONLY calls must use the exact current monitoring window")
        return "Use the current monitoring window where the signal was detected."
    if strategy is TimeRangeStrategy.BASELINE_AND_CURRENT:
        if start != baseline_window.start or end != current_window.end:
            raise ValueError(
                "BASELINE_AND_CURRENT calls must span from baseline start through current end"
            )
        return (
            "Compare the baseline period with the current monitoring period that produced "
            "the selected signal."
        )
    assert proposal.custom_time_range_reason is not None
    return f"Custom range within the approved evidence window: {proposal.custom_time_range_reason}"


def validate_investigation_plan(
    request: InvestigationRequest,
    proposal: InvestigationPlanProposal,
    registry: MetricToolRegistry,
    config: PlanningConfig | None = None,
) -> ValidatedInvestigationPlan:
    """Validate a proposal without executing it or making any operational decision."""

    config = config or PlanningConfig()
    try:
        signal = request.selected_signal
    except ValueError as error:
        raise _plan_failure(
            PlanValidationErrorCode.SIGNAL_NOT_SELECTED,
            "The requested cohort is not a selected investigation signal.",
            "Choose a cohort selected by deterministic screening.",
            retryable=False,
        ) from error
    if len(proposal.tool_calls) > config.maximum_tool_calls:
        raise _plan_failure(
            PlanValidationErrorCode.TOOL_CALL_LIMIT_EXCEEDED,
            f"Tool-call plan exceeds the limit of {config.maximum_tool_calls}.",
            "Return no more than the configured maximum number of tool calls.",
        )

    allowed_window = request.allowed_window
    baseline_window = request.screening_report.comparison.baseline.window
    current_window = request.screening_report.comparison.current.window
    calls: list[ValidatedToolCall] = []
    seen_calls: set[tuple[str, str]] = set()
    for index, proposed_call in enumerate(proposal.tool_calls):
        try:
            normalized_arguments = registry.validate_input(
                proposed_call.tool_name, proposed_call.arguments
            )
        except ValidationError as error:
            raise _plan_failure(
                PlanValidationErrorCode.INVALID_TOOL_INPUT,
                f"Input for tool '{proposed_call.tool_name}' does not match its published schema.",
                "Correct the tool arguments using the registered input schema.",
                tool_call_index=index,
                tool_name=proposed_call.tool_name,
            ) from error
        except ValueError as error:
            raise _plan_failure(
                PlanValidationErrorCode.UNKNOWN_TOOL,
                f"Tool '{proposed_call.tool_name}' is not registered.",
                "Choose only a tool listed in the supplied tool registry.",
                tool_call_index=index,
                tool_name=proposed_call.tool_name,
            ) from error
        try:
            time_range_reason = _validate_time_scope(
                normalized_arguments,
                allowed_window,
                baseline_window,
                current_window,
                proposed_call,
            )
        except (TypeError, ValueError) as error:
            message = str(error)
            code = (
                PlanValidationErrorCode.TIME_RANGE_OUTSIDE_EVIDENCE
                if "outside the allowed investigation window" in message
                else PlanValidationErrorCode.TIME_RANGE_STRATEGY_MISMATCH
            )
            raise _plan_failure(
                code,
                "Tool-call dates do not match the approved evidence scope or declared strategy.",
                "Use dates within the evidence window and make them match the chosen strategy.",
                tool_call_index=index,
                tool_name=proposed_call.tool_name,
            ) from error
        call_key = (proposed_call.tool_name, repr(sorted(normalized_arguments.items())))
        if call_key in seen_calls:
            raise _plan_failure(
                PlanValidationErrorCode.DUPLICATE_TOOL_CALL,
                "Tool-call plan contains an exact duplicate call.",
                "Remove the duplicate or select a distinct evidence query.",
                tool_call_index=index,
                tool_name=proposed_call.tool_name,
            )
        seen_calls.add(call_key)
        descriptor = next(
            descriptor
            for descriptor in registry.descriptors()
            if descriptor.name == proposed_call.tool_name
        )
        calls.append(
            ValidatedToolCall(
                tool_name=proposed_call.tool_name,
                tool_purpose=descriptor.purpose,
                selection_reason=proposed_call.selection_reason,
                time_range_strategy=proposed_call.time_range_strategy,
                time_range_reason=time_range_reason,
                arguments=normalized_arguments,
            )
        )

    return ValidatedInvestigationPlan(
        signal=signal,
        allowed_window=allowed_window,
        investigation_goal=proposal.investigation_goal,
        tool_calls=calls,
    )


def validate_investigation_plan_result(
    request: InvestigationRequest,
    proposal: InvestigationPlanProposal,
    registry: MetricToolRegistry,
    config: PlanningConfig | None = None,
) -> PlanValidationResult:
    """Return a safe terminal validation result instead of exposing exceptions to a graph."""

    try:
        plan = validate_investigation_plan(request, proposal, registry, config)
    except PlanValidationException as error:
        return PlanValidationResult(status=PlanStatus.REJECTED, failure=error.failure)
    return PlanValidationResult(status=PlanStatus.VALIDATED, plan=plan)


def run_bounded_planning(
    request: InvestigationRequest,
    planner: InvestigationPlanner,
    registry: MetricToolRegistry,
    config: PlanningConfig | None = None,
) -> BoundedPlanningResult:
    """Ask for a plan, then allow a bounded repair only for safe retryable failures.

    This routine does not execute tools. It only produces the validated plan that a
    later executor node may consume.
    """

    config = config or PlanningConfig()
    repair_context: PlanValidationFailure | None = None
    attempts: list[PlanningAttempt] = []
    for number in range(1, config.maximum_repair_attempts + 2):
        proposal = planner.propose(
            request=request,
            available_tools=registry.descriptors(),
            repair_context=repair_context,
        )
        validation = validate_investigation_plan_result(request, proposal, registry, config)
        attempts.append(PlanningAttempt(number=number, validation=validation))
        if validation.status is PlanStatus.VALIDATED:
            assert validation.plan is not None
            return BoundedPlanningResult(
                status=PlanStatus.VALIDATED,
                attempts=attempts,
                plan=validation.plan,
            )
        assert validation.failure is not None
        if not validation.failure.retryable:
            break
        repair_context = validation.failure

    assert attempts[-1].validation.failure is not None
    return BoundedPlanningResult(
        status=PlanStatus.REJECTED,
        attempts=attempts,
        failure=attempts[-1].validation.failure,
    )


def handoff_bounded_planning(result: BoundedPlanningResult) -> PlanningHandoff:
    """Convert terminal planning state into a safe executor handoff or user closure."""

    if result.status is PlanStatus.VALIDATED:
        assert result.plan is not None
        return PlanningHandoff(
            status=PlanningHandoffStatus.READY_FOR_EVIDENCE_COLLECTION,
            plan=result.plan,
        )
    assert result.failure is not None
    return PlanningHandoff(
        status=PlanningHandoffStatus.INVESTIGATION_NOT_COMPLETED,
        closure=InvestigationSafetyClosure(
            summary=(
                "The agent could not produce a valid read-only evidence collection plan "
                "within its configured repair limit."
            ),
            planning_attempt_count=len(result.attempts),
            failure=result.failure,
        ),
    )

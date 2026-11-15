"""Deterministic gate that selects material cohort changes for investigation."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from .comparison import BaselineComparison, CohortRateComparison, SampleSufficiency
from .snapshot import InspectionCohort


class SignalKind(str, Enum):
    RATE_SHIFT = "RATE_SHIFT"
    EMERGENCE = "EMERGENCE"


class SignalDirection(str, Enum):
    INCREASE = "INCREASE"
    DECREASE = "DECREASE"
    NEW = "NEW"


class ScreeningOutcome(str, Enum):
    SELECTED = "SELECTED"
    INSUFFICIENT_SAMPLE = "INSUFFICIENT_SAMPLE"
    BELOW_MINIMUM_COUNT = "BELOW_MINIMUM_COUNT"
    BELOW_MATERIALITY_THRESHOLD = "BELOW_MATERIALITY_THRESHOLD"


class ScreeningConfig(BaseModel):
    """Explicit, versioned guardrails for creating investigation candidates.

    These values are operational policy, not model output or statistical proof.
    They can be tuned only through a reviewed configuration change.
    """

    model_config = ConfigDict(frozen=True)

    minimum_absolute_rate_delta: float = Field(default=0.05, gt=0, le=1)
    minimum_relative_rate_change: float = Field(default=0.50, gt=0)
    minimum_emergence_rate: float = Field(default=0.10, gt=0, le=1)
    minimum_current_count: int = Field(default=3, ge=1)


class ScreeningDecision(BaseModel):
    """One auditable decision for every compared cohort."""

    cohort: InspectionCohort
    outcome: ScreeningOutcome
    signal_kind: SignalKind | None = None
    direction: SignalDirection | None = None
    reason: str
    comparison: CohortRateComparison


class ScreeningReport(BaseModel):
    comparison: BaselineComparison
    config: ScreeningConfig
    decisions: list[ScreeningDecision]

    @property
    def selected_signals(self) -> list[ScreeningDecision]:
        return [decision for decision in self.decisions if decision.outcome is ScreeningOutcome.SELECTED]


def _shift_direction(comparison: CohortRateComparison) -> SignalDirection:
    assert comparison.rate_delta is not None
    return SignalDirection.INCREASE if comparison.rate_delta > 0 else SignalDirection.DECREASE


def _screen_one(
    comparison: CohortRateComparison,
    config: ScreeningConfig,
) -> ScreeningDecision:
    if comparison.sample_sufficiency is not SampleSufficiency.SUFFICIENT:
        return ScreeningDecision(
            cohort=comparison.cohort,
            outcome=ScreeningOutcome.INSUFFICIENT_SAMPLE,
            reason=(
                "Comparison is not eligible for investigation because its sample status is "
                f"{comparison.sample_sufficiency.value}."
            ),
            comparison=comparison,
        )
    if comparison.current_count < config.minimum_current_count:
        return ScreeningDecision(
            cohort=comparison.cohort,
            outcome=ScreeningOutcome.BELOW_MINIMUM_COUNT,
            reason=(
                f"Current count {comparison.current_count} is below the configured minimum "
                f"of {config.minimum_current_count}."
            ),
            comparison=comparison,
        )
    if comparison.baseline_rate == 0 and comparison.current_rate is not None:
        if comparison.current_rate >= config.minimum_emergence_rate:
            return ScreeningDecision(
                cohort=comparison.cohort,
                outcome=ScreeningOutcome.SELECTED,
                signal_kind=SignalKind.EMERGENCE,
                direction=SignalDirection.NEW,
                reason=(
                    "The cohort was absent in the baseline and reached the configured "
                    "emergence rate in the current window."
                ),
                comparison=comparison,
            )
        return ScreeningDecision(
            cohort=comparison.cohort,
            outcome=ScreeningOutcome.BELOW_MATERIALITY_THRESHOLD,
            reason=(
                "The cohort was absent in the baseline but its current rate is below the "
                "configured emergence threshold."
            ),
            comparison=comparison,
        )
    absolute_change = abs(comparison.rate_delta or 0.0)
    relative_change = abs(comparison.relative_rate_change or 0.0)
    if (
        absolute_change >= config.minimum_absolute_rate_delta
        and relative_change >= config.minimum_relative_rate_change
    ):
        return ScreeningDecision(
            cohort=comparison.cohort,
            outcome=ScreeningOutcome.SELECTED,
            signal_kind=SignalKind.RATE_SHIFT,
            direction=_shift_direction(comparison),
            reason=(
                "The absolute and relative cohort-rate changes both exceed their configured "
                "materiality thresholds."
            ),
            comparison=comparison,
        )
    return ScreeningDecision(
        cohort=comparison.cohort,
        outcome=ScreeningOutcome.BELOW_MATERIALITY_THRESHOLD,
        reason=(
            "The cohort-rate change did not exceed both configured materiality thresholds."
        ),
        comparison=comparison,
    )


def screen_comparison(
    comparison: BaselineComparison,
    config: ScreeningConfig | None = None,
) -> ScreeningReport:
    """Select candidates only; this function never explains, acts, or calls an LLM."""

    config = config or ScreeningConfig()
    return ScreeningReport(
        comparison=comparison,
        config=config,
        decisions=[_screen_one(item, config) for item in comparison.cohorts],
    )

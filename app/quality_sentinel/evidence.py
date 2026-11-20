"""Evidence collection orchestrator executing validated metric tool queries."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .investigation import MetricToolRegistry, QualityDataReader
from .planning import InvestigationRequest, ValidatedInvestigationPlan


class CollectedToolEvidence(BaseModel):
    """Execution artifact recorded for one tool call in an investigation plan."""

    model_config = ConfigDict(frozen=True)

    tool_name: str
    tool_purpose: str
    selection_reason: str
    arguments: dict[str, Any]
    result: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float = Field(ge=0.0)
    executed_at: datetime
    error: str | None = None


class EvidenceBundle(BaseModel):
    """Immutable collection of tool evidence assembled for signal synthesis."""

    model_config = ConfigDict(frozen=True)

    request: InvestigationRequest | None = None
    plan: ValidatedInvestigationPlan
    evidence: list[CollectedToolEvidence] = Field(default_factory=list)
    collected_at: datetime


def collect_evidence(
    plan: ValidatedInvestigationPlan,
    reader: QualityDataReader,
    registry: MetricToolRegistry,
    request: InvestigationRequest | None = None,
) -> EvidenceBundle:
    """Execute each validated tool call in a plan and assemble an evidence bundle."""
    evidence_items: list[CollectedToolEvidence] = []

    for call in plan.tool_calls:
        tool = registry.get(call.tool_name)
        executed_at = datetime.now(timezone.utc)

        if tool is None:
            evidence_items.append(
                CollectedToolEvidence(
                    tool_name=call.tool_name,
                    tool_purpose=call.tool_purpose,
                    selection_reason=call.selection_reason,
                    arguments=call.arguments,
                    result={},
                    duration_ms=0.0,
                    executed_at=executed_at,
                    error=f"Tool '{call.tool_name}' not found in metric tool registry.",
                )
            )
            continue

        start_time = time.perf_counter()
        try:
            output = tool.run(reader, call.arguments)
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            if isinstance(output, BaseModel):
                result_data = output.model_dump(mode="json")
            elif isinstance(output, dict):
                result_data = output
            else:
                result_data = dict(output)
            error_message = None
        except Exception as exc:  # noqa: BLE001
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            result_data = {}
            error_message = str(exc)

        evidence_items.append(
            CollectedToolEvidence(
                tool_name=call.tool_name,
                tool_purpose=call.tool_purpose,
                selection_reason=call.selection_reason,
                arguments=call.arguments,
                result=result_data,
                duration_ms=round(duration_ms, 3),
                executed_at=executed_at,
                error=error_message,
            )
        )

    return EvidenceBundle(
        request=request,
        plan=plan,
        evidence=evidence_items,
        collected_at=datetime.now(timezone.utc),
    )

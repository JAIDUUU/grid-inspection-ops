"""Signal synthesis module generating verified quality signal cards from evidence."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from .comparison import SampleSufficiency
from .evidence import EvidenceBundle
from .screening import SignalKind
from .snapshot import InspectionCohort


class EvidenceValidationStatus(str, Enum):
    """Validation classification for an assembled evidence bundle."""

    SUFFICIENT = "SUFFICIENT"
    INSUFFICIENT_SAMPLE = "INSUFFICIENT_SAMPLE"
    DATA_GAP = "DATA_GAP"


class QualitySignalCard(BaseModel):
    """Verified operational signal card synthesized from investigated evidence."""

    model_config = ConfigDict(frozen=True)

    signal_id: str
    cohort: InspectionCohort
    signal_kind: SignalKind
    headline: str
    summary: str
    baseline_rate: float
    current_rate: float
    rate_delta: float
    relative_change: float | None
    sample_count: int = Field(ge=0)
    evidence_citations: list[str]
    sop_reference: str | None = None
    recommended_action: str
    created_at: datetime


class DataGapSignal(BaseModel):
    """Auditable notification indicating evidence was missing or insufficient."""

    model_config = ConfigDict(frozen=True)

    signal_id: str
    cohort: InspectionCohort
    headline: str
    missing_evidence_summary: str
    attempted_tools: list[str]
    gap_reason: str
    created_at: datetime


class SignalSynthesizer(Protocol):
    """Boundary for producing synthesized signal cards from evidence bundles."""

    def synthesize(
        self,
        bundle: EvidenceBundle,
        sop_reference: str | None = None,
    ) -> QualitySignalCard | DataGapSignal: ...


def _is_empty_result(result: dict[str, Any]) -> bool:
    """Check if tool execution output contains no meaningful metric records."""
    if not result:
        return True

    # Cohort trend: points list is empty or all have zero cohort count and denominator
    if "points" in result:
        points = result.get("points")
        if not points:
            return True
        return all(p.get("cohort_count", 0) == 0 and p.get("denominator", 0) == 0 for p in points)

    # Review reason breakdown: resolved count is 0 and reason_counts is empty
    if "resolved_inspect_count" in result:
        return result.get("resolved_inspect_count", 0) == 0 and not result.get("reason_counts")

    # Score band distribution: matching records is 0 and missing evidence is 0
    if "total_matching_records" in result:
        return (
            result.get("total_matching_records", 0) == 0
            and result.get("missing_evidence_count", 0) == 0
        )

    # Deployment context: version_counts list is empty
    if "version_counts" in result:
        return not bool(result.get("version_counts"))

    # Capture quality summary: records_with_quality is 0 and missing_quality is 0
    if "records_with_quality" in result:
        return (
            result.get("records_with_quality", 0) == 0
            and result.get("missing_quality_count", 0) == 0
        )

    # General dict fallback: check if all non-metadata values are zero/empty/None
    return all(
        v == 0 or v == [] or v == {} or v is None
        for k, v in result.items()
        if k not in ("cohort", "rate_basis")
    )


def validate_evidence(bundle: EvidenceBundle) -> EvidenceValidationStatus:
    """Classify evidence bundle completeness and sample adequacy."""
    if not bundle.evidence:
        return EvidenceValidationStatus.DATA_GAP

    all_failed_or_empty = all(
        item.error is not None or _is_empty_result(item.result) for item in bundle.evidence
    )
    if all_failed_or_empty:
        return EvidenceValidationStatus.DATA_GAP

    comparison = bundle.plan.signal.comparison
    if (
        comparison.sample_sufficiency is not SampleSufficiency.SUFFICIENT
        or comparison.current_count == 0
    ):
        return EvidenceValidationStatus.INSUFFICIENT_SAMPLE

    return EvidenceValidationStatus.SUFFICIENT


def _generate_signal_id(cohort: InspectionCohort, collected_at: datetime, rate_delta: float) -> str:
    seed = f"SIG:{cohort.value}:{collected_at.isoformat()}:{rate_delta:.4f}"
    digest = hashlib.sha256(seed.encode()).hexdigest()[:8]
    return f"SIG-{cohort.value.replace('_', '-')}-{digest}"


def _generate_gap_id(cohort: InspectionCohort, collected_at: datetime, reason: str) -> str:
    seed = f"GAP:{cohort.value}:{collected_at.isoformat()}:{reason}"
    digest = hashlib.sha256(seed.encode()).hexdigest()[:8]
    return f"GAP-{cohort.value.replace('_', '-')}-{digest}"


class SafeSignalSynthesizer:
    """Deterministic, zero-hallucination signal synthesizer quoting factual evidence."""

    def synthesize(
        self,
        bundle: EvidenceBundle,
        sop_reference: str | None = None,
    ) -> QualitySignalCard | DataGapSignal:
        validation_status = validate_evidence(bundle)
        cohort = bundle.plan.signal.cohort
        comparison = bundle.plan.signal.comparison

        if validation_status is EvidenceValidationStatus.DATA_GAP:
            attempted_tools = (
                [item.tool_name for item in bundle.evidence]
                if bundle.evidence
                else [call.tool_name for call in bundle.plan.tool_calls]
            )
            errors = [f"{e.tool_name}: {e.error}" for e in bundle.evidence if e.error]
            empty_tools = [
                e.tool_name
                for e in bundle.evidence
                if e.error is None and _is_empty_result(e.result)
            ]

            summary_parts: list[str] = []
            if errors:
                summary_parts.append(f"Errors encountered ({'; '.join(errors)})")
            if empty_tools:
                summary_parts.append(f"Empty results from ({', '.join(empty_tools)})")
            if not summary_parts:
                summary_parts.append("No tool evidence collected")

            gap_summary = ". ".join(summary_parts) + "."
            gap_reason = (
                "All attempted metric tools returned errors or empty data across the query window."
            )
            return DataGapSignal(
                signal_id=_generate_gap_id(cohort, bundle.collected_at, gap_reason),
                cohort=cohort,
                headline=f"Data gap detected for {cohort.value}",
                missing_evidence_summary=gap_summary,
                attempted_tools=attempted_tools,
                gap_reason=gap_reason,
                created_at=datetime.now(timezone.utc),
            )

        if validation_status is EvidenceValidationStatus.INSUFFICIENT_SAMPLE:
            attempted_tools = [item.tool_name for item in bundle.evidence]
            gap_reason = (
                f"Cohort sample does not meet sufficiency criteria: "
                f"{comparison.sample_sufficiency.value} (current: {comparison.current_count}, "
                f"denominator: {comparison.current_denominator})."
            )
            return DataGapSignal(
                signal_id=_generate_gap_id(cohort, bundle.collected_at, gap_reason),
                cohort=cohort,
                headline=f"Insufficient sample size for {cohort.value}",
                missing_evidence_summary=gap_reason,
                attempted_tools=attempted_tools,
                gap_reason=gap_reason,
                created_at=datetime.now(timezone.utc),
            )

        # Evidence is SUFFICIENT: synthesize QualitySignalCard
        signal_kind = bundle.plan.signal.signal_kind or SignalKind.RATE_SHIFT
        baseline_rate = float(comparison.baseline_rate or 0.0)
        current_rate = float(comparison.current_rate or 0.0)
        rate_delta = float(comparison.rate_delta or 0.0)
        relative_change = comparison.relative_rate_change
        sample_count = comparison.current_count

        # Extract tool findings and citations
        citations: list[str] = []
        evidence_summaries: list[str] = []
        top_quality_failures: list[str] = []
        top_review_reasons: list[str] = []
        deployment_versions: list[str] = []
        score_band_highlights: list[str] = []

        for item in bundle.evidence:
            if item.error is not None:
                continue

            tool_name = item.tool_name
            res = item.result

            if tool_name == "get_deployment_context" and "version_counts" in res:
                v_counts = res.get("version_counts", [])
                if v_counts:
                    desc_list = [
                        f"{v['model_version']}/{v['policy_version']} ({v['count']})"
                        for v in v_counts
                    ]
                    deployment_versions.extend(desc_list)
                    citations.append(f"get_deployment_context: {', '.join(desc_list)}")
                    evidence_summaries.append(f"Active deployments: {', '.join(desc_list)}.")

            elif tool_name == "get_capture_quality_summary" and "records_with_quality" in res:
                acc = res.get("acceptable_count", 0)
                unacc = res.get("unacceptable_count", 0)
                failure_counts = res.get("failure_reason_counts", [])
                failure_strs = [f"{f['reason_code']} ({f['count']})" for f in failure_counts]
                top_quality_failures.extend(failure_strs)
                cit_text = f"get_capture_quality_summary: {acc} acceptable, {unacc} unacceptable"
                if failure_strs:
                    cit_text += f"; top failure reasons: {', '.join(failure_strs)}"
                citations.append(cit_text)
                evidence_summaries.append(
                    f"Capture quality: {acc} acceptable, {unacc} unacceptable. "
                    f"Top failure reasons: {', '.join(failure_strs) if failure_strs else 'none'}."
                )

            elif tool_name == "get_score_band_distribution" and "distributions" in res:
                total_rec = res.get("total_matching_records", 0)
                band_counts: dict[str, int] = {}
                for dist in res.get("distributions", []):
                    for band_item in dist.get("bands", []):
                        band_name = band_item.get("band")
                        band_counts[band_name] = band_counts.get(band_name, 0) + band_item.get(
                            "count", 0
                        )
                band_desc = [f"{k}: {v}" for k, v in sorted(band_counts.items()) if v > 0]
                if band_desc:
                    score_band_highlights.extend(band_desc)
                citations.append(
                    f"get_score_band_distribution: {total_rec} matching; {', '.join(band_desc)}"
                )
                evidence_summaries.append(
                    f"Score distribution ({total_rec} matching): {', '.join(band_desc)}."
                )

            elif tool_name == "get_review_reason_breakdown" and "resolved_inspect_count" in res:
                resolved_count = res.get("resolved_inspect_count", 0)
                r_counts = res.get("reason_counts", [])
                r_strs = [f"{r['reason_code']} ({r['count']})" for r in r_counts]
                top_review_reasons.extend(r_strs)
                cit_text = f"get_review_reason_breakdown: {resolved_count} resolved"
                if r_strs:
                    cit_text += f"; reasons: {', '.join(r_strs)}"
                citations.append(cit_text)
                evidence_summaries.append(
                    f"Review dispositions ({resolved_count} resolved): {', '.join(r_strs) if r_strs else 'none'}."
                )

            elif tool_name == "get_cohort_trend" and "points" in res:
                pts = res.get("points", [])
                valid_rates = [p["rate"] for p in pts if p.get("rate") is not None]
                if valid_rates:
                    min_r, max_r = min(valid_rates), max(valid_rates)
                    trend_str = f"{len(pts)} buckets, rate range {min_r:.1%} - {max_r:.1%}"
                else:
                    trend_str = f"{len(pts)} buckets analyzed"
                citations.append(f"get_cohort_trend: {trend_str}")
                evidence_summaries.append(f"Cohort trend: {trend_str}.")

            else:
                citations.append(f"{tool_name}: execution completed in {item.duration_ms:.1f}ms")

        # Headline
        if signal_kind is SignalKind.EMERGENCE:
            headline = (
                f"Cohort {cohort.value} emerged at {current_rate:.1%} (delta: {rate_delta:+.1%})"
            )
        elif rate_delta > 0:
            headline = f"Cohort {cohort.value} rate increased by {rate_delta:+.1%} ({baseline_rate:.1%} -> {current_rate:.1%})"
        elif rate_delta < 0:
            headline = f"Cohort {cohort.value} rate decreased by {abs(rate_delta):.1%} ({baseline_rate:.1%} -> {current_rate:.1%})"
        else:
            headline = f"Cohort {cohort.value} rate steady at {current_rate:.1%}"

        # Summary assembly
        rel_str = (
            f", relative change: {relative_change:+.1%}" if relative_change is not None else ""
        )
        lead_sentence = (
            f"Screening confirmed a {signal_kind.value.lower().replace('_', ' ')} for {cohort.value}. "
            f"Baseline rate was {baseline_rate:.1%} ({comparison.baseline_count}/{comparison.baseline_denominator}) "
            f"and current rate is {current_rate:.1%} ({comparison.current_count}/{comparison.current_denominator}), "
            f"an absolute delta of {rate_delta:+.1%}{rel_str}."
        )

        full_summary_parts = [lead_sentence] + evidence_summaries
        if sop_reference:
            full_summary_parts.append(f"Operating standard referenced: {sop_reference}.")
        summary = " ".join(full_summary_parts)

        # Recommended Action assembly
        if top_quality_failures:
            action = (
                f"Inspect camera optics, illumination, and part presentation at inspection stations; "
                f"top capture failure reasons include {', '.join(top_quality_failures)}."
            )
        elif top_review_reasons and any(
            "ACCEPTED_PROCESS_RESIDUE" in r or "NO_DEFECT_FOUND" in r for r in top_review_reasons
        ):
            action = (
                f"Calibrate disposition threshold with QA engineering; "
                f"human reviews indicate recurrent {', '.join(top_review_reasons)}."
            )
        elif any("REVIEW_TO_REJECT_THRESHOLD" in h for h in score_band_highlights):
            action = (
                "Audit borderline inspections near the reject threshold to evaluate potential "
                "score calibration drift."
            )
        elif len(deployment_versions) > 1:
            action = (
                f"Investigate correlation between cohort rate shift and recent deployment transition "
                f"({', '.join(deployment_versions)})."
            )
        else:
            action = (
                f"Conduct engineering root-cause review on {cohort.value} rate shift "
                f"({rate_delta:+.1%}) and verify affected sample images."
            )

        if sop_reference:
            action = f"{action} Refer to SOP {sop_reference}."

        signal_id = _generate_signal_id(cohort, bundle.collected_at, rate_delta)

        return QualitySignalCard(
            signal_id=signal_id,
            cohort=cohort,
            signal_kind=signal_kind,
            headline=headline,
            summary=summary,
            baseline_rate=baseline_rate,
            current_rate=current_rate,
            rate_delta=rate_delta,
            relative_change=relative_change,
            sample_count=sample_count,
            evidence_citations=citations,
            sop_reference=sop_reference,
            recommended_action=action,
            created_at=datetime.now(timezone.utc),
        )

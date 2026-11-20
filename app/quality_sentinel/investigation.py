"""Dynamic, read-only metric tools for quality-signal investigation.

Tools are registered at runtime. The future LangGraph planner discovers their
descriptors and chooses among them; no cohort-to-tool mapping lives here.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..domain import InspectionRecord, ReviewStatus
from .comparison import RateBasis, rate_basis_for
from .snapshot import (
    AnalysisWindow,
    InspectionCohort,
    TimedInspection,
    build_cohort_snapshot,
    classify_inspection,
)


class TimeRangeInput(BaseModel):
    """A bounded, half-open query range supplied to a read-only metric tool."""

    start: datetime
    end: datetime

    @model_validator(mode="after")
    def start_is_before_end(self) -> TimeRangeInput:
        if self.start >= self.end:
            raise ValueError("start must be earlier than end")
        return self


class CohortTrendInput(TimeRangeInput):
    cohort: InspectionCohort
    bucket_hours: int = Field(default=24, ge=1, le=168)


class CohortMetricInput(TimeRangeInput):
    cohort: InspectionCohort


class ReasonBreakdownInput(TimeRangeInput):
    pass


class VersionContextInput(TimeRangeInput):
    pass


class CaptureQualityInput(TimeRangeInput):
    pass


class ToolDescriptor(BaseModel):
    """The complete contract the future LLM planner may inspect."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9_]{2,63}$")
    purpose: str = Field(min_length=1, max_length=500)
    when_useful: list[str] = Field(min_length=1, max_length=6)
    database_fields: list[str] = Field(min_length=1, max_length=12)
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    read_only: bool = True
    max_result_rows: int = Field(default=200, ge=1, le=1000)


class QualityDataReader(Protocol):
    def fetch(self, query: TimeRangeInput) -> list[TimedInspection]: ...


class SqliteQualityDataReader:
    """Database adapter used by metric tools; it has no write operations."""

    def __init__(self, database_path: str) -> None:
        self.database_path = database_path

    def fetch(self, query: TimeRangeInput) -> list[TimedInspection]:
        start = _as_utc(query.start).strftime("%Y-%m-%d %H:%M:%S")
        end = _as_utc(query.end).strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(self.database_path) as connection:
            rows = connection.execute(
                "SELECT record_json, created_at FROM inspections "
                "WHERE created_at >= ? AND created_at < ? ORDER BY created_at",
                (start, end),
            ).fetchall()
        return [
            TimedInspection(
                observed_at=_parse_sqlite_timestamp(created_at),
                record=InspectionRecord.model_validate_json(record_json),
            )
            for record_json, created_at in rows
        ]


class MetricTool(Protocol):
    descriptor: ToolDescriptor
    input_model: type[BaseModel]

    def run(self, reader: QualityDataReader, raw_input: dict[str, Any]) -> BaseModel: ...


class MetricToolRegistry:
    """Discoverable registry that validates tool inputs before executing them."""

    def __init__(self) -> None:
        self._tools: dict[str, MetricTool] = {}

    def register(self, tool: MetricTool) -> None:
        if not tool.descriptor.read_only:
            raise ValueError("quality investigation tools must be read-only")
        if tool.descriptor.name in self._tools:
            raise ValueError(f"duplicate metric tool: {tool.descriptor.name}")
        self._tools[tool.descriptor.name] = tool

    def descriptors(self) -> list[ToolDescriptor]:
        return [self._tools[name].descriptor for name in sorted(self._tools)]

    def get(self, name: str) -> MetricTool | None:
        """Retrieve a registered metric tool by name."""
        return self._tools.get(name)

    def validate_input(self, name: str, raw_input: dict[str, Any]) -> dict[str, Any]:
        """Validate and normalize an untrusted tool call without querying data."""

        try:
            tool = self._tools[name]
        except KeyError as error:
            raise ValueError(f"unknown metric tool: {name}") from error
        return tool.input_model.model_validate(raw_input).model_dump(mode="json")

    def execute(
        self,
        name: str,
        reader: QualityDataReader,
        raw_input: dict[str, Any],
    ) -> BaseModel:
        tool = self._tools.get(name)
        if tool is None:
            raise ValueError(f"unknown metric tool: {name}")
        return tool.run(reader, self.validate_input(name, raw_input))


class CohortTrendPoint(BaseModel):
    start: datetime
    end: datetime
    cohort_count: int = Field(ge=0)
    denominator: int = Field(ge=0)
    rate: float | None = Field(default=None, ge=0, le=1)


class CohortTrendResult(BaseModel):
    cohort: InspectionCohort
    rate_basis: RateBasis
    points: list[CohortTrendPoint]


class CohortTrendTool:
    input_model = CohortTrendInput
    output_model = CohortTrendResult
    descriptor = ToolDescriptor(
        name="get_cohort_trend",
        purpose="Returns bucketed count and rate history for one inspection cohort.",
        when_useful=[
            "Determine whether a selected signal is sustained or a one-bucket spike.",
            "Compare the timing of a cohort shift with other evidence.",
        ],
        database_fields=["inspections.created_at", "inspections.record_json.decision", "review_status"],
        input_schema=CohortTrendInput.model_json_schema(),
        output_schema=CohortTrendResult.model_json_schema(),
    )

    def run(self, reader: QualityDataReader, raw_input: dict[str, Any]) -> CohortTrendResult:
        request = self.input_model.model_validate(raw_input)
        records = reader.fetch(request)
        basis = rate_basis_for(request.cohort)
        points: list[CohortTrendPoint] = []
        bucket_start = request.start
        bucket_size = timedelta(hours=request.bucket_hours)
        while bucket_start < request.end:
            bucket_end = min(bucket_start + bucket_size, request.end)
            snapshot = build_cohort_snapshot(
                records,
                AnalysisWindow(start=bucket_start, end=bucket_end),
            )
            denominator = _denominator(snapshot, basis)
            count = snapshot.cohort_counts[request.cohort]
            points.append(
                CohortTrendPoint(
                    start=bucket_start,
                    end=bucket_end,
                    cohort_count=count,
                    denominator=denominator,
                    rate=count / denominator if denominator else None,
                )
            )
            bucket_start = bucket_end
        return self.output_model(cohort=request.cohort, rate_basis=basis, points=points)


class ReasonCount(BaseModel):
    reason_code: str
    count: int = Field(ge=0)


class ReviewReasonBreakdownResult(BaseModel):
    resolved_inspect_count: int = Field(ge=0)
    reason_counts: list[ReasonCount]


class ReviewReasonBreakdownTool:
    input_model = ReasonBreakdownInput
    output_model = ReviewReasonBreakdownResult
    descriptor = ToolDescriptor(
        name="get_review_reason_breakdown",
        purpose="Returns final human-review reason-code counts for resolved INSPECT records.",
        when_useful=[
            "Determine whether reviewed passes are no-defect findings, accepted residue, or another disposition.",
        ],
        database_fields=["inspections.created_at", "inspections.record_json.review.reason_code"],
        input_schema=ReasonBreakdownInput.model_json_schema(),
        output_schema=ReviewReasonBreakdownResult.model_json_schema(),
    )

    def run(
        self,
        reader: QualityDataReader,
        raw_input: dict[str, Any],
    ) -> ReviewReasonBreakdownResult:
        request = self.input_model.model_validate(raw_input)
        resolved = [
            item.record
            for item in reader.fetch(request)
            if item.record.review_status is ReviewStatus.RESOLVED and item.record.review is not None
        ]
        counts = Counter(
            record.review.reason_code.value if record.review and record.review.reason_code else "LEGACY_UNCLASSIFIED"
            for record in resolved
        )
        return self.output_model(
            resolved_inspect_count=len(resolved),
            reason_counts=[
                ReasonCount(reason_code=code, count=count) for code, count in sorted(counts.items())
            ],
        )


class ScoreBand(str, Enum):
    """Decision-relative score regions, evaluated with each record's thresholds."""

    BELOW_REVIEW_THRESHOLD = "BELOW_REVIEW_THRESHOLD"
    REVIEW_TO_REJECT_THRESHOLD = "REVIEW_TO_REJECT_THRESHOLD"
    AT_OR_ABOVE_REJECT_THRESHOLD = "AT_OR_ABOVE_REJECT_THRESHOLD"


class ScoreBandCount(BaseModel):
    band: ScoreBand
    count: int = Field(ge=0)


class ThresholdScoreDistribution(BaseModel):
    """A distribution for records evaluated with the same model and thresholds."""

    model_version: str
    review_threshold: float = Field(ge=0)
    reject_threshold: float = Field(gt=0)
    total_records: int = Field(ge=0)
    bands: list[ScoreBandCount]


class ScoreBandDistributionResult(BaseModel):
    cohort: InspectionCohort
    total_matching_records: int = Field(ge=0)
    missing_evidence_count: int = Field(ge=0)
    distributions: list[ThresholdScoreDistribution]


class ScoreBandDistributionTool:
    input_model = CohortMetricInput
    output_model = ScoreBandDistributionResult
    descriptor = ToolDescriptor(
        name="get_score_band_distribution",
        purpose=(
            "Returns anomaly-score counts in decision-relative bands, grouped by the "
            "model version and the review/reject thresholds recorded for each inspection."
        ),
        when_useful=[
            "Check whether a cohort shift concentrates near a review or reject threshold.",
            "Compare the anomaly-score profile of a selected cohort without mixing threshold versions.",
        ],
        database_fields=[
            "inspections.record_json.evidence.score",
            "inspections.record_json.evidence.review_threshold",
            "inspections.record_json.evidence.reject_threshold",
            "inspections.record_json.evidence.model_version",
            "inspections.record_json.decision",
        ],
        input_schema=CohortMetricInput.model_json_schema(),
        output_schema=ScoreBandDistributionResult.model_json_schema(),
    )

    def run(
        self,
        reader: QualityDataReader,
        raw_input: dict[str, Any],
    ) -> ScoreBandDistributionResult:
        request = self.input_model.model_validate(raw_input)
        matching = [
            item.record
            for item in reader.fetch(request)
            if classify_inspection(item.record) is request.cohort
        ]
        grouped_counts: dict[tuple[str, float, float], Counter[ScoreBand]] = {}
        missing_evidence = 0
        for record in matching:
            if record.evidence is None:
                missing_evidence += 1
                continue
            evidence = record.evidence
            key = (
                evidence.model_version,
                evidence.review_threshold,
                evidence.reject_threshold,
            )
            counts = grouped_counts.setdefault(key, Counter())
            if evidence.score < evidence.review_threshold:
                counts[ScoreBand.BELOW_REVIEW_THRESHOLD] += 1
            elif evidence.score < evidence.reject_threshold:
                counts[ScoreBand.REVIEW_TO_REJECT_THRESHOLD] += 1
            else:
                counts[ScoreBand.AT_OR_ABOVE_REJECT_THRESHOLD] += 1

        ordered_bands = list(ScoreBand)
        return self.output_model(
            cohort=request.cohort,
            total_matching_records=len(matching),
            missing_evidence_count=missing_evidence,
            distributions=[
                ThresholdScoreDistribution(
                    model_version=model_version,
                    review_threshold=review_threshold,
                    reject_threshold=reject_threshold,
                    total_records=sum(counts.values()),
                    bands=[ScoreBandCount(band=band, count=counts[band]) for band in ordered_bands],
                )
                for (model_version, review_threshold, reject_threshold), counts in sorted(grouped_counts.items())
            ],
        )


class VersionCount(BaseModel):
    model_version: str
    policy_version: str
    count: int = Field(ge=0)


class DeploymentContextResult(BaseModel):
    version_counts: list[VersionCount]


class DeploymentContextTool:
    input_model = VersionContextInput
    output_model = DeploymentContextResult
    descriptor = ToolDescriptor(
        name="get_deployment_context",
        purpose="Returns inspection counts grouped by model and critical-policy version.",
        when_useful=["Check whether a cohort shift coincides with a model or policy deployment."],
        database_fields=[
            "inspections.record_json.evidence.model_version",
            "inspections.record_json.metadata.critical_policy.version",
        ],
        input_schema=VersionContextInput.model_json_schema(),
        output_schema=DeploymentContextResult.model_json_schema(),
    )

    def run(self, reader: QualityDataReader, raw_input: dict[str, Any]) -> DeploymentContextResult:
        request = self.input_model.model_validate(raw_input)
        counts = Counter(
            (
                item.record.evidence.model_version if item.record.evidence else "MISSING_MODEL_VERSION",
                item.record.metadata.get("critical_policy", {}).get("version", "MISSING_POLICY_VERSION"),
            )
            for item in reader.fetch(request)
        )
        return self.output_model(
            version_counts=[
                VersionCount(model_version=model, policy_version=policy, count=count)
                for (model, policy), count in sorted(counts.items())
            ]
        )


class CaptureQualitySummaryResult(BaseModel):
    records_with_quality: int = Field(ge=0)
    acceptable_count: int = Field(ge=0)
    unacceptable_count: int = Field(ge=0)
    missing_quality_count: int = Field(ge=0)
    failure_reason_counts: list[ReasonCount]


class CaptureQualitySummaryTool:
    input_model = CaptureQualityInput
    output_model = CaptureQualitySummaryResult
    descriptor = ToolDescriptor(
        name="get_capture_quality_summary",
        purpose="Returns aggregate image-quality gate outcomes and failure reasons.",
        when_useful=[
            "Check whether a cohort shift may be related to capture or image-quality degradation.",
        ],
        database_fields=["inspections.record_json.evidence.quality"],
        input_schema=CaptureQualityInput.model_json_schema(),
        output_schema=CaptureQualitySummaryResult.model_json_schema(),
    )

    def run(
        self,
        reader: QualityDataReader,
        raw_input: dict[str, Any],
    ) -> CaptureQualitySummaryResult:
        request = self.input_model.model_validate(raw_input)
        acceptable = 0
        unacceptable = 0
        missing = 0
        reasons: Counter[str] = Counter()
        for item in reader.fetch(request):
            quality = item.record.evidence.quality if item.record.evidence else None
            if quality is None:
                missing += 1
            elif quality.acceptable:
                acceptable += 1
            else:
                unacceptable += 1
                reasons.update(quality.failure_reasons or ["IMAGE_QUALITY_GATE"])
        return self.output_model(
            records_with_quality=acceptable + unacceptable,
            acceptable_count=acceptable,
            unacceptable_count=unacceptable,
            missing_quality_count=missing,
            failure_reason_counts=[
                ReasonCount(reason_code=reason, count=count)
                for reason, count in sorted(reasons.items())
            ],
        )


def build_default_tool_registry() -> MetricToolRegistry:
    registry = MetricToolRegistry()
    for tool in (
        CohortTrendTool(),
        ReviewReasonBreakdownTool(),
        ScoreBandDistributionTool(),
        DeploymentContextTool(),
        CaptureQualitySummaryTool(),
    ):
        registry.register(tool)
    return registry


def _denominator(snapshot, basis: RateBasis) -> int:
    if basis is RateBasis.ALL_INSPECTIONS:
        return snapshot.source_inspection_count
    return sum(
        snapshot.cohort_counts[cohort]
        for cohort in (
            InspectionCohort.REVIEWED_PASS_NO_DEFECT_FOUND,
            InspectionCohort.REVIEWED_PASS_ACCEPTED_PROCESS_RESIDUE,
            InspectionCohort.REVIEWED_PASS_OTHER,
            InspectionCohort.REVIEWED_REJECT,
        )
    )


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _parse_sqlite_timestamp(value: str) -> datetime:
    return _as_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))

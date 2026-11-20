"""Comprehensive tests for the LangGraph quality sentinel pipeline, storage, planner, and API."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from app.api import app
from app.domain import (
    Decision,
    ImageQuality,
    InspectionRecord,
    ReviewReasonCode,
    ReviewResolution,
    ReviewStatus,
    VisionEvidence,
)
from app.quality_sentinel.comparison import RateBasis
from app.quality_sentinel.graph import SentinelState, build_sentinel_graph
from app.quality_sentinel.investigation import (
    TimeRangeInput,
    build_default_tool_registry,
)
from app.quality_sentinel.planner import DefaultRulePlanner
from app.quality_sentinel.planning import (
    InvestigationPlanner,
    InvestigationPlanProposal,
    InvestigationRequest,
    PlanningConfig,
    PlanningHandoffStatus,
    PlanStatus,
    PlanValidationErrorCode,
    PlanValidationFailure,
    TimeRangeStrategy,
    ToolCallProposal,
    validate_investigation_plan,
)
from app.quality_sentinel.screening import (
    ScreeningConfig,
    ScreeningOutcome,
    SignalDirection,
    SignalKind,
)
from app.quality_sentinel.snapshot import AnalysisWindow, InspectionCohort, TimedInspection
from app.quality_sentinel.storage import SentinelRunRecord, SqliteSentinelRepository
from app.quality_sentinel.synthesis import DataGapSignal, QualitySignalCard

BASE_TIME = datetime(2026, 11, 20, 12, 0, 0, tzinfo=timezone.utc)


class MemoryReader:
    """In-memory data reader for quality sentinel tests."""

    def __init__(self, records: list[TimedInspection]) -> None:
        self.records = records

    def fetch(self, query: TimeRangeInput) -> list[TimedInspection]:
        return [item for item in self.records if query.start <= item.observed_at < query.end]


class EmptyReader:
    """Reader that always returns empty results to simulate data gaps."""

    def fetch(self, query: TimeRangeInput) -> list[TimedInspection]:
        return []


def create_timed_inspection(
    inspection_id: str,
    decision: Decision,
    observed_at: datetime,
    review_status: ReviewStatus = ReviewStatus.NOT_REQUIRED,
    final_decision: Decision | None = None,
    reason_code: ReviewReasonCode | None = None,
    score: float = 0.20,
    model_version: str = "patchcore-v1",
) -> TimedInspection:
    review = None
    if review_status is ReviewStatus.RESOLVED:
        review = ReviewResolution(
            reviewer_id="reviewer-1",
            final_decision=final_decision or decision,
            reason_code=reason_code or ReviewReasonCode.NO_DEFECT_FOUND,
            reason="Review completed.",
            reviewed_at=observed_at.isoformat(),
        )
    return TimedInspection(
        observed_at=observed_at,
        record=InspectionRecord(
            inspection_id=inspection_id,
            part_id=f"part-{inspection_id}",
            image_uri=f"runtime/{inspection_id}.png",
            decision=decision,
            final_decision=final_decision,
            review_status=review_status,
            review=review,
            evidence=VisionEvidence(
                model_version=model_version,
                score=score,
                review_threshold=0.315,
                reject_threshold=0.334,
                valid_coverage=1.0,
                artifact_verified=True,
                quality=ImageQuality(
                    width=512,
                    height=512,
                    mean_luminance=0.5,
                    contrast=0.2,
                    laplacian_variance=40.0,
                    acceptable=True,
                ),
            ),
            metadata={"critical_policy": {"version": "policy-v1"}},
        ),
    )


def test_storage_signal_and_run_persistence(tmp_path: Path):
    db_path = str(tmp_path / "sentinel_test.db")
    storage = SqliteSentinelRepository(db_path)

    now = datetime.now(timezone.utc)
    card = QualitySignalCard(
        signal_id="SIG-AUTO-REJECT-001",
        cohort=InspectionCohort.AUTO_REJECT,
        signal_kind=SignalKind.RATE_SHIFT,
        headline="Auto-reject surge",
        summary="Auto reject rate increased from 5% to 50%",
        baseline_rate=0.05,
        current_rate=0.50,
        rate_delta=0.45,
        relative_change=9.0,
        sample_count=50,
        evidence_citations=["Trend shows spike", "Score band distribution shifted"],
        recommended_action="Inspect production line 2",
        created_at=now,
    )
    storage.save_signal(card)

    gap = DataGapSignal(
        signal_id="GAP-AUTO-PASS-001",
        cohort=InspectionCohort.AUTO_PASS,
        headline="Missing telemetry for auto pass",
        missing_evidence_summary="No trend points available",
        attempted_tools=["get_cohort_trend"],
        gap_reason="Empty query result",
        created_at=now,
    )
    storage.save_signal(gap)

    signals = storage.list_signals(limit=10)
    assert len(signals) == 2
    assert any(
        isinstance(s, QualitySignalCard) and s.signal_id == "SIG-AUTO-REJECT-001" for s in signals
    )
    assert any(isinstance(s, DataGapSignal) and s.signal_id == "GAP-AUTO-PASS-001" for s in signals)

    retrieved_card = storage.get_signal("SIG-AUTO-REJECT-001")
    assert retrieved_card is not None
    assert isinstance(retrieved_card, QualitySignalCard)
    assert retrieved_card.cohort == InspectionCohort.AUTO_REJECT

    retrieved_gap = storage.get_signal("GAP-AUTO-PASS-001")
    assert retrieved_gap is not None
    assert isinstance(retrieved_gap, DataGapSignal)

    run = SentinelRunRecord(
        run_id="run-12345",
        outcome="SIGNALS_DETECTED",
        signals_count=1,
        data_gaps_count=0,
        closure_count=0,
        timings_ms={"end_to_end": 12.5},
        started_at=now - timedelta(seconds=1),
        completed_at=now,
        metadata={"test": True},
    )
    storage.save_run(run)

    runs = storage.list_runs(limit=5)
    assert len(runs) == 1
    assert runs[0].run_id == "run-12345"
    assert runs[0].outcome == "SIGNALS_DETECTED"

    retrieved_run = storage.get_run("run-12345")
    assert retrieved_run is not None
    assert retrieved_run.outcome == "SIGNALS_DETECTED"
    assert storage.get_run("non-existent") is None


def test_default_rule_planner_proposes_valid_tools_for_auto_reject():
    registry = build_default_tool_registry()
    planner = DefaultRulePlanner(default_strategy=TimeRangeStrategy.BASELINE_AND_CURRENT)

    end = BASE_TIME
    start_current = end - timedelta(hours=24)
    start_baseline = start_current - timedelta(days=7)

    from app.quality_sentinel.comparison import (
        BaselineComparison,
        CohortRateComparison,
        ComparisonConfig,
        SampleSufficiency,
    )
    from app.quality_sentinel.screening import ScreeningDecision, ScreeningReport
    from app.quality_sentinel.snapshot import CohortSnapshot

    baseline_snap = CohortSnapshot(
        window=AnalysisWindow(start=start_baseline, end=start_current),
        source_inspection_count=100,
        cohort_counts={
            c: (
                95
                if c is InspectionCohort.AUTO_PASS
                else (5 if c is InspectionCohort.AUTO_REJECT else 0)
            )
            for c in InspectionCohort
        },
    )
    current_snap = CohortSnapshot(
        window=AnalysisWindow(start=start_current, end=end),
        source_inspection_count=50,
        cohort_counts={
            c: (
                20
                if c is InspectionCohort.AUTO_PASS
                else (30 if c is InspectionCohort.AUTO_REJECT else 0)
            )
            for c in InspectionCohort
        },
    )
    comparison = BaselineComparison(
        current=current_snap,
        baseline=baseline_snap,
        config=ComparisonConfig(minimum_denominator=20),
        cohorts=[
            CohortRateComparison(
                cohort=InspectionCohort.AUTO_REJECT,
                rate_basis=RateBasis.ALL_INSPECTIONS,
                current_count=30,
                current_denominator=50,
                current_rate=0.60,
                baseline_count=5,
                baseline_denominator=100,
                baseline_rate=0.05,
                rate_delta=0.55,
                relative_rate_change=11.0,
                sample_sufficiency=SampleSufficiency.SUFFICIENT,
            )
        ],
    )
    report = ScreeningReport(
        comparison=comparison,
        config=ScreeningConfig(),
        decisions=[
            ScreeningDecision(
                cohort=InspectionCohort.AUTO_REJECT,
                outcome=ScreeningOutcome.SELECTED,
                signal_kind=SignalKind.RATE_SHIFT,
                direction=SignalDirection.INCREASE,
                reason="Shift detected",
                comparison=comparison.cohorts[0],
            )
        ],
    )

    request = InvestigationRequest(screening_report=report, cohort=InspectionCohort.AUTO_REJECT)
    proposal = planner.propose(request=request, available_tools=registry.descriptors())

    tool_names = [call.tool_name for call in proposal.tool_calls]
    assert "get_cohort_trend" in tool_names
    assert "get_score_band_distribution" in tool_names
    assert "get_deployment_context" in tool_names

    validated = validate_investigation_plan(request, proposal, registry)
    assert validated.status == PlanStatus.VALIDATED


def test_default_rule_planner_proposes_valid_tools_for_review_cohort():
    registry = build_default_tool_registry()
    planner = DefaultRulePlanner(default_strategy=TimeRangeStrategy.CURRENT_ONLY)

    end = BASE_TIME
    start_current = end - timedelta(hours=24)
    start_baseline = start_current - timedelta(days=7)

    from app.quality_sentinel.comparison import (
        BaselineComparison,
        CohortRateComparison,
        ComparisonConfig,
        SampleSufficiency,
    )
    from app.quality_sentinel.screening import ScreeningDecision, ScreeningReport
    from app.quality_sentinel.snapshot import CohortSnapshot

    baseline_snap = CohortSnapshot(
        window=AnalysisWindow(start=start_baseline, end=start_current),
        source_inspection_count=100,
        cohort_counts={
            c: (5 if c is InspectionCohort.REVIEWED_PASS_NO_DEFECT_FOUND else 0)
            for c in InspectionCohort
        },
    )
    current_snap = CohortSnapshot(
        window=AnalysisWindow(start=start_current, end=end),
        source_inspection_count=50,
        cohort_counts={
            c: (25 if c is InspectionCohort.REVIEWED_PASS_NO_DEFECT_FOUND else 0)
            for c in InspectionCohort
        },
    )
    comparison = BaselineComparison(
        current=current_snap,
        baseline=baseline_snap,
        config=ComparisonConfig(minimum_denominator=20),
        cohorts=[
            CohortRateComparison(
                cohort=InspectionCohort.REVIEWED_PASS_NO_DEFECT_FOUND,
                rate_basis=RateBasis.RESOLVED_INSPECTS,
                current_count=25,
                current_denominator=25,
                current_rate=1.0,
                baseline_count=5,
                baseline_denominator=25,
                baseline_rate=0.20,
                rate_delta=0.80,
                relative_rate_change=4.0,
                sample_sufficiency=SampleSufficiency.SUFFICIENT,
            )
        ],
    )
    report = ScreeningReport(
        comparison=comparison,
        config=ScreeningConfig(),
        decisions=[
            ScreeningDecision(
                cohort=InspectionCohort.REVIEWED_PASS_NO_DEFECT_FOUND,
                outcome=ScreeningOutcome.SELECTED,
                signal_kind=SignalKind.RATE_SHIFT,
                direction=SignalDirection.INCREASE,
                reason="Shift detected",
                comparison=comparison.cohorts[0],
            )
        ],
    )

    request = InvestigationRequest(
        screening_report=report, cohort=InspectionCohort.REVIEWED_PASS_NO_DEFECT_FOUND
    )
    proposal = planner.propose(request=request, available_tools=registry.descriptors())

    tool_names = [call.tool_name for call in proposal.tool_calls]
    assert "get_cohort_trend" in tool_names
    assert "get_review_reason_breakdown" in tool_names
    assert "get_deployment_context" in tool_names

    validated = validate_investigation_plan(request, proposal, registry)
    assert validated.status == PlanStatus.VALIDATED


def test_default_rule_planner_repairs_proposal_when_context_provided():
    registry = build_default_tool_registry()
    planner = DefaultRulePlanner(default_strategy=TimeRangeStrategy.BASELINE_AND_CURRENT)

    end = BASE_TIME
    start_current = end - timedelta(hours=24)
    start_baseline = start_current - timedelta(days=7)

    from app.quality_sentinel.comparison import (
        BaselineComparison,
        CohortRateComparison,
        ComparisonConfig,
        SampleSufficiency,
    )
    from app.quality_sentinel.screening import ScreeningDecision, ScreeningReport
    from app.quality_sentinel.snapshot import CohortSnapshot

    baseline_snap = CohortSnapshot(
        window=AnalysisWindow(start=start_baseline, end=start_current),
        source_inspection_count=100,
        cohort_counts={
            c: (5 if c is InspectionCohort.AUTO_REJECT else 0) for c in InspectionCohort
        },
    )
    current_snap = CohortSnapshot(
        window=AnalysisWindow(start=start_current, end=end),
        source_inspection_count=50,
        cohort_counts={
            c: (30 if c is InspectionCohort.AUTO_REJECT else 0) for c in InspectionCohort
        },
    )
    comparison = BaselineComparison(
        current=current_snap,
        baseline=baseline_snap,
        config=ComparisonConfig(minimum_denominator=20),
        cohorts=[
            CohortRateComparison(
                cohort=InspectionCohort.AUTO_REJECT,
                rate_basis=RateBasis.ALL_INSPECTIONS,
                current_count=30,
                current_denominator=50,
                current_rate=0.60,
                baseline_count=5,
                baseline_denominator=100,
                baseline_rate=0.05,
                rate_delta=0.55,
                relative_rate_change=11.0,
                sample_sufficiency=SampleSufficiency.SUFFICIENT,
            )
        ],
    )
    report = ScreeningReport(
        comparison=comparison,
        config=ScreeningConfig(),
        decisions=[
            ScreeningDecision(
                cohort=InspectionCohort.AUTO_REJECT,
                outcome=ScreeningOutcome.SELECTED,
                signal_kind=SignalKind.RATE_SHIFT,
                direction=SignalDirection.INCREASE,
                reason="Shift detected",
                comparison=comparison.cohorts[0],
            )
        ],
    )
    request = InvestigationRequest(screening_report=report, cohort=InspectionCohort.AUTO_REJECT)

    repair_context = PlanValidationFailure(
        code=PlanValidationErrorCode.UNKNOWN_TOOL,
        message="Unknown tool",
        repair_hint="Remove tool",
        retryable=True,
        tool_name="get_score_band_distribution",
    )
    proposal = planner.propose(
        request=request,
        available_tools=registry.descriptors(),
        repair_context=repair_context,
    )
    assert "get_score_band_distribution" not in [c.tool_name for c in proposal.tool_calls]


def test_graph_path_a_no_material_signal(tmp_path: Path):
    """Path A: Stable rates -> no material signal -> persists run -> terminates cleanly."""
    db_path = str(tmp_path / "sentinel_path_a.db")
    storage = SqliteSentinelRepository(db_path)
    registry = build_default_tool_registry()

    inspections: list[TimedInspection] = []
    # Baseline: 7 days, 100 PASS inspections
    for i in range(100):
        t = BASE_TIME - timedelta(days=6) + timedelta(hours=i)
        inspections.append(create_timed_inspection(f"base-{i}", Decision.PASS, t))
    # Current: 24h, 30 PASS inspections
    for i in range(30):
        t = BASE_TIME - timedelta(hours=20) + timedelta(minutes=i * 30)
        inspections.append(create_timed_inspection(f"curr-{i}", Decision.PASS, t))

    reader = MemoryReader(inspections)
    graph = build_sentinel_graph(reader=reader, registry=registry, storage=storage)

    state: SentinelState = {
        "analysis_end": BASE_TIME,
        "current_hours": 24,
        "baseline_days": 7,
    }
    result = graph.invoke(state)

    assert result["has_material_signal"] is False
    assert result["run_record"].outcome == "NO_MATERIAL_SIGNAL"
    assert result["run_record"].signals_count == 0
    assert result["run_record"].data_gaps_count == 0
    assert result["run_record"].closure_count == 0

    runs = storage.list_runs()
    assert len(runs) == 1
    assert runs[0].outcome == "NO_MATERIAL_SIGNAL"

    signals = storage.list_signals()
    assert len(signals) == 0


def test_graph_path_b_material_signal_detected(tmp_path: Path):
    """Path B: Surge in auto-reject -> plans, collects evidence, synthesizes card, persists."""
    db_path = str(tmp_path / "sentinel_path_b.db")
    storage = SqliteSentinelRepository(db_path)
    registry = build_default_tool_registry()

    inspections: list[TimedInspection] = []
    # Baseline: 7 days, 95 PASS, 5 REJECT (5% reject rate out of 100)
    for i in range(95):
        t = BASE_TIME - timedelta(days=5) + timedelta(hours=i)
        inspections.append(create_timed_inspection(f"base-p-{i}", Decision.PASS, t, score=0.15))
    for i in range(5):
        t = BASE_TIME - timedelta(days=4) + timedelta(hours=i * 2)
        inspections.append(create_timed_inspection(f"base-r-{i}", Decision.REJECT, t, score=0.45))

    # Current: 24h, 20 PASS, 30 REJECT (60% reject rate out of 50)
    for i in range(20):
        t = BASE_TIME - timedelta(hours=20) + timedelta(minutes=i * 30)
        inspections.append(create_timed_inspection(f"curr-p-{i}", Decision.PASS, t, score=0.15))
    for i in range(30):
        t = BASE_TIME - timedelta(hours=18) + timedelta(minutes=i * 25)
        inspections.append(create_timed_inspection(f"curr-r-{i}", Decision.REJECT, t, score=0.48))

    reader = MemoryReader(inspections)
    graph = build_sentinel_graph(reader=reader, registry=registry, storage=storage)

    state: SentinelState = {
        "analysis_end": BASE_TIME,
        "current_hours": 24,
        "baseline_days": 7,
    }
    result = graph.invoke(state)

    assert result["has_material_signal"] is True
    assert result["run_record"].outcome == "SIGNALS_DETECTED"
    assert result["run_record"].signals_count >= 1

    cards = result.get("signal_cards", [])
    assert len(cards) >= 1
    assert any(c.cohort == InspectionCohort.AUTO_REJECT for c in cards)

    runs = storage.list_runs()
    assert len(runs) == 1
    assert runs[0].outcome == "SIGNALS_DETECTED"
    assert runs[0].signals_count >= 1

    signals = storage.list_signals()
    assert len(signals) >= 1
    assert any(
        isinstance(s, QualitySignalCard) and s.cohort == InspectionCohort.AUTO_REJECT
        for s in signals
    )


def test_graph_path_c_data_gap_scenario(tmp_path: Path):
    """Path C: Material signal detected but metric tools find no records -> DataGapSignal emitted."""
    db_path = str(tmp_path / "sentinel_path_c.db")
    storage = SqliteSentinelRepository(db_path)
    registry = build_default_tool_registry()

    inspections: list[TimedInspection] = []
    # Baseline: 95 PASS, 5 REJECT
    for i in range(95):
        t = BASE_TIME - timedelta(days=5) + timedelta(hours=i)
        inspections.append(create_timed_inspection(f"base-p-{i}", Decision.PASS, t, score=0.15))
    for i in range(5):
        t = BASE_TIME - timedelta(days=4) + timedelta(hours=i * 2)
        inspections.append(create_timed_inspection(f"base-r-{i}", Decision.REJECT, t, score=0.45))

    # Current: 20 PASS, 30 REJECT
    for i in range(20):
        t = BASE_TIME - timedelta(hours=20) + timedelta(minutes=i * 30)
        inspections.append(create_timed_inspection(f"curr-p-{i}", Decision.PASS, t, score=0.15))
    for i in range(30):
        t = BASE_TIME - timedelta(hours=18) + timedelta(minutes=i * 25)
        inspections.append(create_timed_inspection(f"curr-r-{i}", Decision.REJECT, t, score=0.48))

    class SelectiveGapReader:
        """Returns snapshot data but returns empty data for investigation metric queries."""

        def __init__(self, records: list[TimedInspection]) -> None:
            self.records = records
            self.snapshot_fetched = False

        def fetch(self, query: TimeRangeInput) -> list[TimedInspection]:
            if not self.snapshot_fetched:
                self.snapshot_fetched = True
                return [
                    item for item in self.records if query.start <= item.observed_at < query.end
                ]
            return []

    reader = SelectiveGapReader(inspections)
    graph = build_sentinel_graph(reader=reader, registry=registry, storage=storage)

    state: SentinelState = {
        "analysis_end": BASE_TIME,
        "current_hours": 24,
        "baseline_days": 7,
    }
    result = graph.invoke(state)

    assert result["has_material_signal"] is True
    assert result["run_record"].outcome == "DATA_GAPS_DETECTED"
    assert result["run_record"].data_gaps_count >= 1

    gaps = result.get("data_gap_signals", [])
    assert len(gaps) >= 1
    assert any(g.cohort == InspectionCohort.AUTO_REJECT for g in gaps)

    signals = storage.list_signals()
    assert any(
        isinstance(s, DataGapSignal) and s.cohort == InspectionCohort.AUTO_REJECT for s in signals
    )


def test_api_quality_sentinel_routes(tmp_path: Path, monkeypatch):
    """Test GET /signals, GET /runs, and POST /run endpoints via FastAPI TestClient."""
    db_path = str(tmp_path / "sentinel_api.db")
    monkeypatch.setenv("DATABASE_PATH", db_path)

    # Reset any existing sentinel runtime state
    if hasattr(app.state, "sentinel_graph"):
        delattr(app.state, "sentinel_graph")
    if hasattr(app.state, "sentinel_storage"):
        delattr(app.state, "sentinel_storage")

    client = TestClient(app)

    # 1. Initially empty
    resp_signals = client.get("/v1/quality-sentinel/signals")
    assert resp_signals.status_code == 200
    assert resp_signals.json() == []

    resp_runs = client.get("/v1/quality-sentinel/runs")
    assert resp_runs.status_code == 200
    assert resp_runs.json() == []

    # 2. Trigger run (Path A: empty db -> no material signal)
    resp_run = client.post(
        "/v1/quality-sentinel/run", json={"current_hours": 24, "baseline_days": 7}
    )
    assert resp_run.status_code == 200
    run_data = resp_run.json()
    assert run_data["outcome"] == "NO_MATERIAL_SIGNAL"
    assert "run_id" in run_data

    # 3. Check runs endpoint now lists this run
    resp_runs2 = client.get("/v1/quality-sentinel/runs")
    assert resp_runs2.status_code == 200
    runs = resp_runs2.json()
    assert len(runs) == 1
    assert runs[0]["run_id"] == run_data["run_id"]

    # 4. Trigger run with empty body
    resp_run2 = client.post("/v1/quality-sentinel/run")
    assert resp_run2.status_code == 200
    assert resp_run2.json()["outcome"] == "NO_MATERIAL_SIGNAL"


def test_graph_planning_failure_scenario(tmp_path: Path):
    """When a planner repeatedly proposes invalid plans exceeding repair budget, closures emit PLANNING_FAILED."""
    db_path = str(tmp_path / "sentinel_planning_fail.db")
    storage = SqliteSentinelRepository(db_path)
    registry = build_default_tool_registry()

    inspections: list[TimedInspection] = []
    # Baseline: 95 PASS, 5 REJECT
    for i in range(95):
        t = BASE_TIME - timedelta(days=5) + timedelta(hours=i)
        inspections.append(create_timed_inspection(f"base-p-{i}", Decision.PASS, t))
    for i in range(5):
        t = BASE_TIME - timedelta(days=4) + timedelta(hours=i * 2)
        inspections.append(create_timed_inspection(f"base-r-{i}", Decision.REJECT, t))
    # Current: 20 PASS, 30 REJECT
    for i in range(20):
        t = BASE_TIME - timedelta(hours=20) + timedelta(minutes=i * 30)
        inspections.append(create_timed_inspection(f"curr-p-{i}", Decision.PASS, t))
    for i in range(30):
        t = BASE_TIME - timedelta(hours=18) + timedelta(minutes=i * 25)
        inspections.append(create_timed_inspection(f"curr-r-{i}", Decision.REJECT, t))

    class BrokenPlanner(InvestigationPlanner):
        def propose(self, *, request, available_tools, repair_context=None):
            return InvestigationPlanProposal(
                investigation_goal="Broken plan",
                tool_calls=[
                    ToolCallProposal(
                        tool_name="non_existent_tool",
                        arguments={"start": BASE_TIME.isoformat(), "end": BASE_TIME.isoformat()},
                        selection_reason="Invalid tool call",
                        time_range_strategy=TimeRangeStrategy.BASELINE_AND_CURRENT,
                    )
                ],
            )

    reader = MemoryReader(inspections)
    graph = build_sentinel_graph(
        reader=reader,
        registry=registry,
        storage=storage,
        planner=BrokenPlanner(),
        planning_config=PlanningConfig(maximum_repair_attempts=0),
    )

    state: SentinelState = {
        "analysis_end": BASE_TIME,
        "current_hours": 24,
        "baseline_days": 7,
    }
    result = graph.invoke(state)

    assert result["has_material_signal"] is True
    assert result["run_record"].outcome == "PLANNING_FAILED"
    assert result["run_record"].closure_count >= 1
    assert len(result.get("closures", [])) >= 1
    assert result["closures"][0].status == PlanningHandoffStatus.INVESTIGATION_NOT_COMPLETED

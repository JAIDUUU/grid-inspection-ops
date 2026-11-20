"""LangGraph orchestration graph for automated quality sentinel pipeline."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from time import perf_counter
from typing import TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from .comparison import BaselineComparison, compare_snapshots
from .evidence import EvidenceBundle, collect_evidence
from .investigation import (
    MetricToolRegistry,
    QualityDataReader,
    TimeRangeInput,
)
from .planner import DefaultRulePlanner
from .planning import (
    InvestigationPlanner,
    InvestigationRequest,
    InvestigationSafetyClosure,
    PlanningConfig,
    PlanningHandoff,
    PlanningHandoffStatus,
    handoff_bounded_planning,
    run_bounded_planning,
)
from .screening import ScreeningConfig, ScreeningReport, screen_comparison
from .snapshot import AnalysisWindow, CohortSnapshot, build_cohort_snapshot
from .storage import SentinelRunRecord, SqliteSentinelRepository
from .synthesis import (
    DataGapSignal,
    QualitySignalCard,
    SafeSignalSynthesizer,
    SignalSynthesizer,
)


class SentinelState(TypedDict, total=False):
    """Complete execution state flowing through the Sentinel LangGraph."""

    run_id: str
    analysis_end: datetime
    current_hours: int
    baseline_days: int
    snapshot_current: CohortSnapshot
    snapshot_baseline: CohortSnapshot
    comparison: BaselineComparison
    screening_report: ScreeningReport
    has_material_signal: bool
    requests: list[InvestigationRequest]
    handoffs: list[PlanningHandoff]
    evidence_bundles: list[EvidenceBundle]
    signal_cards: list[QualitySignalCard]
    data_gap_signals: list[DataGapSignal]
    closures: list[InvestigationSafetyClosure]
    started_at: float
    started_at_dt: datetime
    timings_ms: dict[str, float]
    run_record: SentinelRunRecord


def build_sentinel_graph(
    reader: QualityDataReader,
    registry: MetricToolRegistry,
    storage: SqliteSentinelRepository,
    planner: InvestigationPlanner | None = None,
    synthesizer: SignalSynthesizer | None = None,
    planning_config: PlanningConfig | None = None,
    screening_config: ScreeningConfig | None = None,
):
    """Construct and compile the LangGraph execution graph for the quality sentinel."""
    resolved_planner = planner or DefaultRulePlanner()
    resolved_synthesizer = synthesizer or SafeSignalSynthesizer()
    resolved_planning_config = planning_config or PlanningConfig()
    resolved_screening_config = screening_config or ScreeningConfig()

    def build_snapshots(state: SentinelState) -> SentinelState:
        t0 = perf_counter()
        started_at = state.get("started_at") or perf_counter()
        started_at_dt = state.get("started_at_dt") or datetime.now(timezone.utc)
        run_id = state.get("run_id") or f"run_{uuid4().hex[:12]}"

        analysis_end = state.get("analysis_end") or datetime.now(timezone.utc)
        if analysis_end.tzinfo is None:
            analysis_end = analysis_end.replace(tzinfo=timezone.utc)
        else:
            analysis_end = analysis_end.astimezone(timezone.utc)

        current_hours = state.get("current_hours", 24)
        baseline_days = state.get("baseline_days", 7)

        current_start = analysis_end - timedelta(hours=current_hours)
        current_window = AnalysisWindow(start=current_start, end=analysis_end)

        baseline_start = current_start - timedelta(days=baseline_days)
        baseline_end = current_start
        baseline_window = AnalysisWindow(start=baseline_start, end=baseline_end)

        inspections = reader.fetch(TimeRangeInput(start=baseline_start, end=analysis_end))
        snapshot_baseline = build_cohort_snapshot(inspections, baseline_window)
        snapshot_current = build_cohort_snapshot(inspections, current_window)
        comparison = compare_snapshots(snapshot_current, snapshot_baseline)

        timings = dict(state.get("timings_ms") or {})
        timings["build_snapshots"] = round((perf_counter() - t0) * 1000, 2)

        return {
            "run_id": run_id,
            "analysis_end": analysis_end,
            "current_hours": current_hours,
            "baseline_days": baseline_days,
            "snapshot_current": snapshot_current,
            "snapshot_baseline": snapshot_baseline,
            "comparison": comparison,
            "started_at": started_at,
            "started_at_dt": started_at_dt,
            "timings_ms": timings,
        }

    def screen_signals(state: SentinelState) -> SentinelState:
        t0 = perf_counter()
        screening_report = screen_comparison(state["comparison"], resolved_screening_config)
        selected = screening_report.selected_signals
        has_material = len(selected) > 0
        requests = [
            InvestigationRequest(screening_report=screening_report, cohort=decision.cohort)
            for decision in selected
        ]
        timings = dict(state.get("timings_ms") or {})
        timings["screen_signals"] = round((perf_counter() - t0) * 1000, 2)
        return {
            "screening_report": screening_report,
            "has_material_signal": has_material,
            "requests": requests,
            "timings_ms": timings,
        }

    def persist_no_signal(state: SentinelState) -> SentinelState:
        t0 = perf_counter()
        timings = dict(state.get("timings_ms") or {})
        timings["persist_no_signal"] = round((perf_counter() - t0) * 1000, 2)
        timings["end_to_end"] = round((perf_counter() - state["started_at"]) * 1000, 2)
        completed_at = datetime.now(timezone.utc)
        run_record = SentinelRunRecord(
            run_id=state["run_id"],
            outcome="NO_MATERIAL_SIGNAL",
            signals_count=0,
            data_gaps_count=0,
            closure_count=0,
            timings_ms=timings,
            started_at=state["started_at_dt"],
            completed_at=completed_at,
            metadata={
                "current_hours": state.get("current_hours", 24),
                "baseline_days": state.get("baseline_days", 7),
                "decisions_count": len(state["screening_report"].decisions),
            },
        )
        storage.save_run(run_record)
        return {
            "run_record": run_record,
            "timings_ms": timings,
            "handoffs": [],
            "evidence_bundles": [],
            "signal_cards": [],
            "data_gap_signals": [],
            "closures": [],
        }

    def plan_investigations(state: SentinelState) -> SentinelState:
        t0 = perf_counter()
        handoffs: list[PlanningHandoff] = []
        closures: list[InvestigationSafetyClosure] = []
        for req in state.get("requests", []):
            bounded_result = run_bounded_planning(
                req, resolved_planner, registry, resolved_planning_config
            )
            handoff = handoff_bounded_planning(bounded_result)
            handoffs.append(handoff)
            if handoff.closure is not None:
                closures.append(handoff.closure)
        timings = dict(state.get("timings_ms") or {})
        timings["plan_investigations"] = round((perf_counter() - t0) * 1000, 2)
        return {
            "handoffs": handoffs,
            "closures": closures,
            "timings_ms": timings,
        }

    def execute_evidence_collection(state: SentinelState) -> SentinelState:
        t0 = perf_counter()
        bundles: list[EvidenceBundle] = []
        for handoff in state.get("handoffs", []):
            if (
                handoff.status is PlanningHandoffStatus.READY_FOR_EVIDENCE_COLLECTION
                and handoff.plan is not None
            ):
                matching_req = next(
                    (
                        r
                        for r in state.get("requests", [])
                        if r.cohort is handoff.plan.signal.cohort
                    ),
                    None,
                )
                bundle = collect_evidence(
                    plan=handoff.plan,
                    reader=reader,
                    registry=registry,
                    request=matching_req,
                )
                bundles.append(bundle)
        timings = dict(state.get("timings_ms") or {})
        timings["execute_evidence_collection"] = round((perf_counter() - t0) * 1000, 2)
        return {
            "evidence_bundles": bundles,
            "timings_ms": timings,
        }

    def synthesize_signals(state: SentinelState) -> SentinelState:
        t0 = perf_counter()
        cards: list[QualitySignalCard] = []
        gaps: list[DataGapSignal] = []
        for bundle in state.get("evidence_bundles", []):
            synthesized = resolved_synthesizer.synthesize(bundle)
            if isinstance(synthesized, QualitySignalCard):
                cards.append(synthesized)
            elif isinstance(synthesized, DataGapSignal):
                gaps.append(synthesized)
        timings = dict(state.get("timings_ms") or {})
        timings["synthesize_signals"] = round((perf_counter() - t0) * 1000, 2)
        return {
            "signal_cards": cards,
            "data_gap_signals": gaps,
            "timings_ms": timings,
        }

    def persist_sentinel_run(state: SentinelState) -> SentinelState:
        t0 = perf_counter()
        cards = state.get("signal_cards", [])
        gaps = state.get("data_gap_signals", [])
        closures = state.get("closures", [])

        for card in cards:
            storage.save_signal(card)
        for gap in gaps:
            storage.save_signal(gap)

        if cards:
            outcome = "SIGNALS_DETECTED"
        elif gaps:
            outcome = "DATA_GAPS_DETECTED"
        elif closures:
            outcome = "PLANNING_FAILED"
        else:
            outcome = "INVESTIGATION_COMPLETED"

        timings = dict(state.get("timings_ms") or {})
        timings["persist_sentinel_run"] = round((perf_counter() - t0) * 1000, 2)
        timings["end_to_end"] = round((perf_counter() - state["started_at"]) * 1000, 2)
        completed_at = datetime.now(timezone.utc)

        run_record = SentinelRunRecord(
            run_id=state["run_id"],
            outcome=outcome,
            signals_count=len(cards),
            data_gaps_count=len(gaps),
            closure_count=len(closures),
            timings_ms=timings,
            started_at=state["started_at_dt"],
            completed_at=completed_at,
            metadata={
                "current_hours": state.get("current_hours", 24),
                "baseline_days": state.get("baseline_days", 7),
                "selected_signals_count": len(state.get("requests", [])),
            },
        )
        storage.save_run(run_record)
        return {
            "run_record": run_record,
            "timings_ms": timings,
        }

    def route_screening(state: SentinelState) -> str:
        if state.get("has_material_signal"):
            return "plan_investigations"
        return "persist_no_signal"

    graph = StateGraph(SentinelState)
    graph.add_node("build_snapshots", build_snapshots)
    graph.add_node("screen_signals", screen_signals)
    graph.add_node("persist_no_signal", persist_no_signal)
    graph.add_node("plan_investigations", plan_investigations)
    graph.add_node("execute_evidence_collection", execute_evidence_collection)
    graph.add_node("synthesize_signals", synthesize_signals)
    graph.add_node("persist_sentinel_run", persist_sentinel_run)

    graph.add_edge(START, "build_snapshots")
    graph.add_edge("build_snapshots", "screen_signals")
    graph.add_conditional_edges(
        "screen_signals",
        route_screening,
        {
            "plan_investigations": "plan_investigations",
            "persist_no_signal": "persist_no_signal",
        },
    )
    graph.add_edge("persist_no_signal", END)
    graph.add_edge("plan_investigations", "execute_evidence_collection")
    graph.add_edge("execute_evidence_collection", "synthesize_signals")
    graph.add_edge("synthesize_signals", "persist_sentinel_run")
    graph.add_edge("persist_sentinel_run", END)

    return graph.compile()

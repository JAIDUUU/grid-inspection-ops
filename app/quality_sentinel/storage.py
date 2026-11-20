"""Persistence repository for quality sentinel signals and execution runs."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .synthesis import DataGapSignal, QualitySignalCard


class SentinelRunRecord(BaseModel):
    """Execution record for an automated or on-demand quality sentinel run."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    outcome: str
    signals_count: int = Field(default=0, ge=0)
    data_gaps_count: int = Field(default=0, ge=0)
    closure_count: int = Field(default=0, ge=0)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    started_at: datetime
    completed_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class SqliteSentinelRepository:
    """SQLite-backed persistence for quality sentinel signals and execution runs."""

    def __init__(self, database_path: str = "runtime/inspections.db") -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS quality_signals (
                    signal_id TEXT PRIMARY KEY,
                    signal_type TEXT NOT NULL,
                    cohort TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sentinel_runs (
                    run_id TEXT PRIMARY KEY,
                    outcome TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_quality_signals_created_at ON quality_signals(created_at)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_quality_signals_cohort ON quality_signals(cohort)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_sentinel_runs_created_at ON sentinel_runs(created_at)"
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def save_signal(self, signal: QualitySignalCard | DataGapSignal) -> None:
        """Persist a verified signal card or data gap notification."""
        signal_type = "QUALITY_SIGNAL" if isinstance(signal, QualitySignalCard) else "DATA_GAP"
        created_at = (
            signal.created_at.isoformat()
            if signal.created_at.tzinfo
            else signal.created_at.replace(tzinfo=timezone.utc).isoformat()
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO quality_signals (
                    signal_id, signal_type, cohort, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    signal.signal_id,
                    signal_type,
                    signal.cohort.value,
                    signal.model_dump_json(),
                    created_at,
                ),
            )

    def save_run(self, run: SentinelRunRecord) -> None:
        """Persist an execution record for a sentinel pipeline run."""
        created_at = (
            run.completed_at.isoformat()
            if run.completed_at.tzinfo
            else run.completed_at.replace(tzinfo=timezone.utc).isoformat()
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO sentinel_runs (
                    run_id, outcome, payload_json, created_at
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    run.outcome,
                    run.model_dump_json(),
                    created_at,
                ),
            )

    def list_signals(self, limit: int = 50) -> list[QualitySignalCard | DataGapSignal]:
        """List recently persisted signals ordered by creation timestamp descending."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT signal_type, payload_json FROM quality_signals "
                "ORDER BY created_at DESC LIMIT ?",
                (max(1, limit),),
            ).fetchall()
        results: list[QualitySignalCard | DataGapSignal] = []
        for row in rows:
            signal_type = row["signal_type"]
            payload = row["payload_json"]
            if signal_type == "DATA_GAP" or "gap_reason" in payload:
                results.append(DataGapSignal.model_validate_json(payload))
            else:
                results.append(QualitySignalCard.model_validate_json(payload))
        return results

    def list_runs(self, limit: int = 20) -> list[SentinelRunRecord]:
        """List recently completed sentinel runs ordered by completion timestamp descending."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM sentinel_runs ORDER BY created_at DESC LIMIT ?",
                (max(1, limit),),
            ).fetchall()
        return [SentinelRunRecord.model_validate_json(row["payload_json"]) for row in rows]

    def get_signal(self, signal_id: str) -> QualitySignalCard | DataGapSignal | None:
        """Retrieve a specific signal by identifier."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT signal_type, payload_json FROM quality_signals WHERE signal_id = ?",
                (signal_id,),
            ).fetchone()
        if row is None:
            return None
        signal_type = row["signal_type"]
        payload = row["payload_json"]
        if signal_type == "DATA_GAP" or "gap_reason" in payload:
            return DataGapSignal.model_validate_json(payload)
        return QualitySignalCard.model_validate_json(payload)

    def get_run(self, run_id: str) -> SentinelRunRecord | None:
        """Retrieve a specific run record by identifier."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM sentinel_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return SentinelRunRecord.model_validate_json(row["payload_json"])

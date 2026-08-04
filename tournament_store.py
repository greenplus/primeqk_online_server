from __future__ import annotations

from datetime import timezone
from typing import Any, Optional
import json
import os

from tournament import ACTIVE_RUN_STATUSES, TournamentRun


def decode_json_object(value) -> dict[str, Any]:
    if isinstance(value, str):
        return json.loads(value)
    return dict(value)


class TournamentStore:
    """PostgreSQL-backed tournament documents with an in-memory local fallback."""

    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or os.getenv("DATABASE_URL")
        self.pool = None
        self.last_error: Optional[str] = None
        self._memory_runs: dict[str, TournamentRun] = {}
        self._memory_audit: list[dict[str, Any]] = []

    @property
    def persistent(self) -> bool:
        return self.pool is not None

    async def connect(self) -> None:
        if not self.database_url:
            self.last_error = "DATABASE_URL が未設定のため、大会は再起動までの一時保存です。"
            return
        pool = None
        try:
            import asyncpg

            pool = await asyncpg.create_pool(
                self.database_url,
                min_size=1,
                max_size=5,
                command_timeout=5,
            )
            self.pool = pool
            await self._ensure_schema()
            self.last_error = None
        except Exception as exc:
            if pool is not None:
                await pool.close()
            self.pool = None
            self.last_error = str(exc)
            print(f"tournament database initialization failed: {exc}")

    async def close(self) -> None:
        if self.pool is not None:
            await self.pool.close()
            self.pool = None

    async def _ensure_schema(self) -> None:
        if self.pool is None:
            raise RuntimeError("tournament database is unavailable")
        await self.pool.execute(
            """
            CREATE TABLE IF NOT EXISTS tournament_runs (
                run_id UUID PRIMARY KEY,
                format_key TEXT NOT NULL,
                title TEXT NOT NULL,
                room_id TEXT NOT NULL,
                rule_key TEXT NOT NULL,
                registration_opens_at TIMESTAMPTZ NOT NULL,
                starts_at TIMESTAMPTZ NOT NULL,
                status TEXT NOT NULL,
                state_json JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                finished_at TIMESTAMPTZ
            );
            CREATE INDEX IF NOT EXISTS tournament_runs_room_status_idx
                ON tournament_runs (room_id, status, starts_at);
            CREATE INDEX IF NOT EXISTS tournament_runs_format_created_idx
                ON tournament_runs (format_key, created_at DESC);

            CREATE TABLE IF NOT EXISTS tournament_audit_log (
                id BIGSERIAL PRIMARY KEY,
                run_id UUID NOT NULL REFERENCES tournament_runs(run_id) ON DELETE CASCADE,
                actor TEXT NOT NULL,
                action TEXT NOT NULL,
                details_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS tournament_audit_run_idx
                ON tournament_audit_log (run_id, created_at);
            """
        )

    async def save_run(self, run: TournamentRun) -> None:
        self._memory_runs[run.run_id] = run
        if self.pool is None:
            return
        await self.pool.execute(
            """
            INSERT INTO tournament_runs (
                run_id, format_key, title, room_id, rule_key,
                registration_opens_at, starts_at, status, state_json,
                created_at, updated_at, finished_at
            )
            VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10, NOW(), $11)
            ON CONFLICT (run_id) DO UPDATE SET
                format_key = EXCLUDED.format_key,
                title = EXCLUDED.title,
                room_id = EXCLUDED.room_id,
                rule_key = EXCLUDED.rule_key,
                registration_opens_at = EXCLUDED.registration_opens_at,
                starts_at = EXCLUDED.starts_at,
                status = EXCLUDED.status,
                state_json = EXCLUDED.state_json,
                updated_at = NOW(),
                finished_at = EXCLUDED.finished_at
            """,
            run.run_id,
            run.format_key,
            run.title,
            run.room_id,
            run.rule_key,
            run.registration_opens_at,
            run.starts_at,
            run.status,
            json.dumps(run.to_dict(), ensure_ascii=False),
            run.created_at,
            run.finished_at,
        )

    async def load_active_runs(self) -> list[TournamentRun]:
        if self.pool is None:
            return [run for run in self._memory_runs.values() if run.status in ACTIVE_RUN_STATUSES]
        rows = await self.pool.fetch(
            """
            SELECT state_json
            FROM tournament_runs
            WHERE status = ANY($1::text[])
            ORDER BY starts_at ASC
            """,
            list(ACTIVE_RUN_STATUSES),
        )
        runs = [TournamentRun.from_dict(decode_json_object(row["state_json"])) for row in rows]
        self._memory_runs.update({run.run_id: run for run in runs})
        return runs

    async def load_recent_runs(self, *, room_id: str, limit: int = 20) -> list[TournamentRun]:
        if self.pool is None:
            runs = [run for run in self._memory_runs.values() if run.room_id == room_id]
            return sorted(runs, key=lambda run: run.created_at, reverse=True)[:limit]
        rows = await self.pool.fetch(
            """
            SELECT state_json
            FROM tournament_runs
            WHERE room_id = $1
            ORDER BY created_at DESC
            LIMIT $2
            """,
            room_id,
            limit,
        )
        return [TournamentRun.from_dict(decode_json_object(row["state_json"])) for row in rows]

    async def audit(
        self,
        run_id: str,
        *,
        actor: str,
        action: str,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        entry = {
            "run_id": run_id,
            "actor": actor,
            "action": action,
            "details": details or {},
        }
        self._memory_audit.append(entry)
        if self.pool is None:
            return
        await self.pool.execute(
            """
            INSERT INTO tournament_audit_log (run_id, actor, action, details_json)
            VALUES ($1::uuid, $2, $3, $4::jsonb)
            """,
            run_id,
            actor,
            action,
            json.dumps(details or {}, ensure_ascii=False),
        )

    async def audit_log(self, run_id: str, limit: int = 100) -> list[dict[str, Any]]:
        if self.pool is None:
            return [entry for entry in self._memory_audit if entry["run_id"] == run_id][-limit:]
        rows = await self.pool.fetch(
            """
            SELECT actor, action, details_json, created_at
            FROM tournament_audit_log
            WHERE run_id = $1::uuid
            ORDER BY created_at DESC
            LIMIT $2
            """,
            run_id,
            limit,
        )
        return [
            {
                "actor": row["actor"],
                "action": row["action"],
                "details": decode_json_object(row["details_json"]),
                "created_at": row["created_at"].astimezone(timezone.utc).isoformat(),
            }
            for row in rows
        ]

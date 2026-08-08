from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
import os


ACTOR_OWNER = "owner"
ACTOR_CPU = "cpu"
VALID_ACTORS = frozenset({ACTOR_OWNER, ACTOR_CPU})


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class CompositePracticeStatsStore:
    """Aggregate successful composite plays, backed by PostgreSQL when available."""

    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or os.getenv("DATABASE_URL")
        self.pool = None
        self.last_error: Optional[str] = None
        self._memory_counts: dict[tuple[str, str], dict[str, Any]] = {}

    @property
    def persistent(self) -> bool:
        return self.pool is not None

    async def connect(self) -> None:
        if not self.database_url:
            self.last_error = "DATABASE_URL が未設定のため、合成数カウントは再起動までの一時保存です。"
            return

        pool = None
        try:
            import asyncpg

            pool = await asyncpg.create_pool(
                self.database_url,
                min_size=1,
                max_size=3,
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
            print(f"composite practice stats database initialization failed: {exc}")

    async def close(self) -> None:
        if self.pool is not None:
            await self.pool.close()
            self.pool = None

    async def _ensure_schema(self) -> None:
        if self.pool is None:
            raise RuntimeError("composite practice stats database is unavailable")
        await self.pool.execute(
            """
            CREATE TABLE IF NOT EXISTS composite_practice_play_counts (
                actor_kind TEXT NOT NULL CHECK (actor_kind IN ('owner', 'cpu')),
                composite_number TEXT NOT NULL,
                play_count BIGINT NOT NULL DEFAULT 0 CHECK (play_count >= 0),
                first_played_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                last_played_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (actor_kind, composite_number)
            );
            CREATE INDEX IF NOT EXISTS composite_practice_play_counts_last_idx
                ON composite_practice_play_counts (last_played_at DESC);
            """
        )

    async def record_play(
        self,
        *,
        actor_kind: str,
        composite_number: int | str,
        played_at: Optional[datetime] = None,
    ) -> None:
        actor = str(actor_kind)
        if actor not in VALID_ACTORS:
            raise ValueError(f"unknown composite practice actor: {actor}")
        number = str(composite_number)
        if not number.isdigit() or int(number) < 4:
            raise ValueError(f"invalid composite number: {number}")
        timestamp = played_at or utc_now()

        if self.pool is not None:
            try:
                await self.pool.execute(
                    """
                    INSERT INTO composite_practice_play_counts (
                        actor_kind,
                        composite_number,
                        play_count,
                        first_played_at,
                        last_played_at
                    )
                    VALUES ($1, $2, 1, $3, $3)
                    ON CONFLICT (actor_kind, composite_number) DO UPDATE SET
                        play_count = composite_practice_play_counts.play_count + 1,
                        last_played_at = EXCLUDED.last_played_at
                    """,
                    actor,
                    number,
                    timestamp,
                )
                return
            except Exception as exc:
                self.last_error = str(exc)
                print(f"composite practice stats write failed: {exc}")

        self._record_in_memory(actor, number, timestamp)

    def _record_in_memory(self, actor: str, number: str, timestamp: datetime) -> None:
        key = (actor, number)
        current = self._memory_counts.get(key)
        if current is None:
            self._memory_counts[key] = {
                "actor_kind": actor,
                "composite_number": number,
                "play_count": 1,
                "first_played_at": timestamp,
                "last_played_at": timestamp,
            }
            return
        current["play_count"] += 1
        current["last_played_at"] = max(current["last_played_at"], timestamp)

    async def snapshot(self) -> dict[str, Any]:
        records: list[dict[str, Any]] = []
        if self.pool is not None:
            try:
                rows = await self.pool.fetch(
                    """
                    SELECT actor_kind, composite_number, play_count,
                           first_played_at, last_played_at
                    FROM composite_practice_play_counts
                    """
                )
                records.extend(dict(row) for row in rows)
            except Exception as exc:
                self.last_error = str(exc)
                print(f"composite practice stats read failed: {exc}")
        records.extend(dict(value) for value in self._memory_counts.values())
        return self._build_snapshot(records)

    def _build_snapshot(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        combined: dict[str, dict[str, Any]] = {}
        for record in records:
            number = str(record["composite_number"])
            actor = str(record["actor_kind"])
            count = int(record["play_count"])
            item = combined.setdefault(
                number,
                {
                    "number": number,
                    "owner_count": 0,
                    "cpu_count": 0,
                    "total_count": 0,
                    "first_played_at": record["first_played_at"],
                    "last_played_at": record["last_played_at"],
                },
            )
            item[f"{actor}_count"] += count
            item["total_count"] += count
            item["first_played_at"] = min(item["first_played_at"], record["first_played_at"])
            item["last_played_at"] = max(item["last_played_at"], record["last_played_at"])

        items = sorted(
            combined.values(),
            key=lambda item: (-item["total_count"], int(item["number"])),
        )
        for item in items:
            item["first_played_at"] = item["first_played_at"].isoformat()
            item["last_played_at"] = item["last_played_at"].isoformat()
        return {
            "persistent": self.persistent,
            "storage": "postgresql" if self.persistent else "memory",
            "totals": {
                "owner_count": sum(item["owner_count"] for item in items),
                "cpu_count": sum(item["cpu_count"] for item in items),
                "total_count": sum(item["total_count"] for item in items),
                "distinct_count": len(items),
            },
            "items": items,
        }

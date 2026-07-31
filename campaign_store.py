from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional


DEFAULT_CAMPAIGN_KEY = "gold-cpu-100"
DEFAULT_CAMPAIGN_GOAL = 300
DEFAULT_CAMPAIGN_PAGE_URL = (
    "https://greenplus.github.io/qkneo/campaign.html"
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def positive_int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return max(1, value)


def parse_campaign_datetime(
    value: Optional[str],
    variable_name: str,
) -> tuple[Optional[datetime], Optional[str]]:
    if not value or not value.strip():
        return None, f"{variable_name} が未設定です"
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None, f"{variable_name} はタイムゾーン付きISO日時で設定してください"
    if parsed.tzinfo is None:
        return None, f"{variable_name} にはタイムゾーンが必要です"
    return parsed.astimezone(timezone.utc), None


def parse_start_at(value: Optional[str]) -> tuple[Optional[datetime], Optional[str]]:
    return parse_campaign_datetime(value, "CPU_CAMPAIGN_START_AT")


def parse_end_at(value: Optional[str]) -> tuple[Optional[datetime], Optional[str]]:
    return parse_campaign_datetime(value, "CPU_CAMPAIGN_END_AT")


@dataclass(frozen=True)
class CampaignSettings:
    enabled: bool
    key: str
    goal: int
    start_at: Optional[datetime]
    start_error: Optional[str]
    end_at: Optional[datetime]
    end_error: Optional[str]
    page_url: str
    allowed_origins: tuple[str, ...]

    @classmethod
    def from_env(cls) -> "CampaignSettings":
        start_at, start_error = parse_start_at(os.getenv("CPU_CAMPAIGN_START_AT"))
        end_at, end_error = parse_end_at(os.getenv("CPU_CAMPAIGN_END_AT"))
        if (
            start_at is not None
            and end_at is not None
            and end_at <= start_at
        ):
            end_error = "CPU_CAMPAIGN_END_AT は開始日時より後に設定してください"
        origins = tuple(
            origin.strip()
            for origin in os.getenv(
                "CPU_CAMPAIGN_ALLOWED_ORIGINS",
                "https://greenplus.github.io,http://127.0.0.1:5174,http://localhost:5174",
            ).split(",")
            if origin.strip()
        )
        return cls(
            enabled=bool_env("CPU_CAMPAIGN_ENABLED"),
            key=os.getenv("CPU_CAMPAIGN_KEY", DEFAULT_CAMPAIGN_KEY).strip()
            or DEFAULT_CAMPAIGN_KEY,
            goal=positive_int_env("CPU_CAMPAIGN_GOAL", DEFAULT_CAMPAIGN_GOAL),
            start_at=start_at,
            start_error=start_error,
            end_at=end_at,
            end_error=end_error,
            page_url=os.getenv(
                "CPU_CAMPAIGN_PAGE_URL",
                DEFAULT_CAMPAIGN_PAGE_URL,
            ).strip()
            or DEFAULT_CAMPAIGN_PAGE_URL,
            allowed_origins=origins,
        )

    def is_active(self, now: Optional[datetime] = None) -> bool:
        if (
            not self.enabled
            or self.start_at is None
            or self.end_at is None
            or self.end_error is not None
        ):
            return False
        current = now or utc_now()
        return self.start_at <= current < self.end_at


class CampaignStore:
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or os.getenv("DATABASE_URL")
        self.pool = None
        self.last_error: Optional[str] = None

    @property
    def ready(self) -> bool:
        return self.pool is not None

    async def connect(self) -> None:
        if not self.database_url:
            self.last_error = "DATABASE_URL が未設定です"
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
            print(f"campaign database initialization failed: {exc}")

    async def close(self) -> None:
        if self.pool is not None:
            await self.pool.close()
            self.pool = None

    async def _ensure_schema(self) -> None:
        if self.pool is None:
            raise RuntimeError("campaign database is unavailable")
        await self.pool.execute(
            """
            CREATE TABLE IF NOT EXISTS campaign_wins (
                id BIGSERIAL PRIMARY KEY,
                campaign_key TEXT NOT NULL,
                game_id UUID NOT NULL,
                player_name VARCHAR(24) NOT NULL,
                room_id TEXT NOT NULL,
                rule_key TEXT NOT NULL,
                cpu_key TEXT NOT NULL,
                game_started_at TIMESTAMPTZ NOT NULL,
                won_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE (campaign_key, game_id)
            );
            CREATE INDEX IF NOT EXISTS campaign_wins_campaign_player_idx
                ON campaign_wins (campaign_key, player_name);
            CREATE INDEX IF NOT EXISTS campaign_wins_campaign_won_at_idx
                ON campaign_wins (campaign_key, won_at);
            """
        )

    async def record_win(
        self,
        *,
        campaign_key: str,
        game_id: str,
        player_name: str,
        room_id: str,
        rule_key: str,
        cpu_key: str,
        game_started_at: datetime,
        won_at: datetime,
    ) -> dict[str, Any]:
        if self.pool is None:
            raise RuntimeError(self.last_error or "campaign database is unavailable")

        async with self.pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    """
                    INSERT INTO campaign_wins (
                        campaign_key,
                        game_id,
                        player_name,
                        room_id,
                        rule_key,
                        cpu_key,
                        game_started_at,
                        won_at
                    )
                    VALUES ($1, $2::uuid, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (campaign_key, game_id) DO NOTHING
                    """,
                    campaign_key,
                    game_id,
                    player_name,
                    room_id,
                    rule_key,
                    cpu_key,
                    game_started_at,
                    won_at,
                )
                counts = await connection.fetchrow(
                    """
                    SELECT
                        COUNT(*)::int AS total_wins,
                        COUNT(*) FILTER (WHERE player_name = $2)::int AS player_wins
                    FROM campaign_wins
                    WHERE campaign_key = $1
                    """,
                    campaign_key,
                    player_name,
                )

        return {
            "total_wins": int(counts["total_wins"]),
            "player_wins": int(counts["player_wins"]),
        }

    async def leaderboard(self, campaign_key: str, limit: int = 20) -> dict[str, Any]:
        if self.pool is None:
            raise RuntimeError(self.last_error or "campaign database is unavailable")

        async with self.pool.acquire() as connection:
            total_wins = await connection.fetchval(
                "SELECT COUNT(*)::int FROM campaign_wins WHERE campaign_key = $1",
                campaign_key,
            )
            rows = await connection.fetch(
                """
                WITH player_totals AS (
                    SELECT
                        player_name,
                        COUNT(*)::int AS wins,
                        MAX(won_at) AS reached_at
                    FROM campaign_wins
                    WHERE campaign_key = $1
                    GROUP BY player_name
                )
                SELECT
                    ROW_NUMBER() OVER (
                        ORDER BY wins DESC, reached_at ASC, player_name ASC
                    )::int AS rank,
                    player_name,
                    wins
                FROM player_totals
                ORDER BY wins DESC, reached_at ASC, player_name ASC
                LIMIT $2
                """,
                campaign_key,
                limit,
            )
            last_updated_at = await connection.fetchval(
                "SELECT MAX(won_at) FROM campaign_wins WHERE campaign_key = $1",
                campaign_key,
            )

        return {
            "total_wins": int(total_wins or 0),
            "rankings": [
                {
                    "rank": int(row["rank"]),
                    "player_name": row["player_name"],
                    "wins": int(row["wins"]),
                }
                for row in rows
            ],
            "last_updated_at": (
                last_updated_at.astimezone(timezone.utc).isoformat()
                if last_updated_at is not None
                else None
            ),
        }

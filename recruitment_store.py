from __future__ import annotations

import asyncio
import hashlib
import os
import secrets
import uuid
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Optional


MAX_ACTIVE_RECRUITMENTS = 5
MAX_RECRUITMENT_HORIZON = timedelta(hours=24)
MAX_RECRUITMENT_NAME_LENGTH = 24
RECRUITMENT_RULE_LABELS = {
    "beginner": "初級（7枚・偶数半減）",
    "advanced": "上級（11枚・通常）",
    "classic": "Classic",
    "plus": "Plus",
    "five_penalty_one": "5枚 / ペナルティ1枚",
    "seven_even_half_penalty_one": "7枚 / 偶数半減 / ペナルティ1枚",
    "normal": "通常",
    "initial_revolution": "初期革命",
    "tetrad": "四つ子素数",
    "semiprime": "半素数",
    # 既存の募集を期限まで表示・通知できるよう、廃止済みキーのラベルは残す。
    "tournament": "定期大会",
    "either": "どれでも",
}
RECRUITMENT_RULE_KEYS_BY_BOARD = {
    "neo": frozenset({"beginner", "advanced", "either"}),
    "plus": frozenset(
        {
            "classic",
            "five_penalty_one",
            "seven_even_half_penalty_one",
            "normal",
            "initial_revolution",
            "tetrad",
            "semiprime",
            "either",
        }
    ),
}


class RecruitmentError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_scheduled_at(value: object) -> datetime:
    if not isinstance(value, str):
        raise RecruitmentError("invalid_time", "集合時間を選んでください。")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise RecruitmentError("invalid_time", "集合時間の形式が正しくありません。") from exc
    if parsed.tzinfo is None:
        raise RecruitmentError("invalid_time", "集合時間にはタイムゾーンが必要です。")
    return parsed.astimezone(timezone.utc)


def validate_owner_token(value: object) -> str:
    if not isinstance(value, str) or not 32 <= len(value) <= 256:
        raise RecruitmentError(
            "invalid_owner",
            "投稿者情報を確認できません。ページを再読み込みしてください。",
        )
    return value


def owner_token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def validate_board_key(value: object) -> str:
    if not isinstance(value, str) or value not in RECRUITMENT_RULE_KEYS_BY_BOARD:
        raise RecruitmentError("invalid_board", "この募集掲示板は利用できません。")
    return value


def validate_recruitment_input(
    *,
    name: object,
    rule_key: object,
    scheduled_at: object,
    owner_token: object,
    board_key: object,
    now: Optional[datetime] = None,
) -> tuple[str, str, str, datetime, str]:
    current = now or utc_now()
    clean_board_key = validate_board_key(board_key)
    if not isinstance(name, str) or not name.strip():
        raise RecruitmentError("invalid_name", "名前を入力してください。")
    clean_name = name.strip()
    if len(clean_name) > MAX_RECRUITMENT_NAME_LENGTH:
        raise RecruitmentError(
            "invalid_name",
            f"名前は{MAX_RECRUITMENT_NAME_LENGTH}文字以内にしてください。",
        )
    if (
        not isinstance(rule_key, str)
        or rule_key not in RECRUITMENT_RULE_KEYS_BY_BOARD[clean_board_key]
    ):
        raise RecruitmentError("invalid_rule", "希望ルールを選んでください。")
    clean_scheduled_at = parse_scheduled_at(scheduled_at)
    if clean_scheduled_at <= current:
        raise RecruitmentError("invalid_time", "集合時間は現在より後にしてください。")
    if clean_scheduled_at > current + MAX_RECRUITMENT_HORIZON:
        raise RecruitmentError("invalid_time", "集合時間は24時間以内にしてください。")
    clean_owner_token = validate_owner_token(owner_token)
    return (
        clean_board_key,
        clean_name,
        rule_key,
        clean_scheduled_at,
        owner_token_hash(clean_owner_token),
    )


@dataclass(frozen=True)
class Recruitment:
    recruitment_id: str
    board_key: str
    name: str
    rule_key: str
    scheduled_at: datetime
    created_at: datetime
    owner_hash: str
    notification_reserved: bool = False

    def public_payload(self, viewer_owner_hash: Optional[str] = None) -> dict:
        return {
            "id": self.recruitment_id,
            "board_key": self.board_key,
            "name": self.name,
            "rule_key": self.rule_key,
            "rule_label": RECRUITMENT_RULE_LABELS[self.rule_key],
            "scheduled_at": self.scheduled_at.astimezone(timezone.utc).isoformat(),
            "created_at": self.created_at.astimezone(timezone.utc).isoformat(),
            "can_delete": bool(
                viewer_owner_hash
                and secrets.compare_digest(self.owner_hash, viewer_owner_hash)
            ),
        }


@dataclass(frozen=True)
class RecruitmentNotification:
    event_id: str
    recruitment_id: str
    event_order: int
    event_type: str
    board_key: str
    name: str
    rule_key: str
    scheduled_at: datetime
    resolution_reason: Optional[str]
    attempt_count: int = 0


class RecruitmentStore:
    """PostgreSQL-backed recruitment board with an in-memory local fallback."""

    def __init__(
        self,
        database_url: Optional[str] = None,
        *,
        notifications_enabled: bool = False,
        notification_pair_limit: int = 3,
        notification_window_seconds: int = 3600,
    ):
        self.database_url = database_url or os.getenv("DATABASE_URL")
        self.notifications_enabled = notifications_enabled
        self.notification_pair_limit = max(0, notification_pair_limit)
        self.notification_window_seconds = max(1, notification_window_seconds)
        self.pool = None
        self.last_error: Optional[str] = None
        self._memory_posts: dict[str, Recruitment] = {}
        self._memory_lock = asyncio.Lock()
        self._memory_notification_pair_times: deque[datetime] = deque()
        self._memory_notification_outbox: list[dict] = []

    @property
    def persistent(self) -> bool:
        return self.pool is not None

    async def connect(self) -> None:
        if not self.database_url:
            self.last_error = "DATABASE_URL is not configured; recruitments use memory only."
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
            print(f"recruitment database initialization failed: {exc}")

    async def close(self) -> None:
        if self.pool is not None:
            await self.pool.close()
            self.pool = None

    async def _ensure_schema(self) -> None:
        if self.pool is None:
            raise RuntimeError("recruitment database is unavailable")
        await self.pool.execute(
            """
            CREATE TABLE IF NOT EXISTS recruitment_posts (
                recruitment_id UUID PRIMARY KEY,
                board_key TEXT NOT NULL DEFAULT 'neo',
                player_name VARCHAR(24) NOT NULL,
                rule_key TEXT NOT NULL,
                scheduled_at TIMESTAMPTZ NOT NULL,
                owner_token_hash CHAR(64) NOT NULL,
                notification_reserved BOOLEAN NOT NULL DEFAULT FALSE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            ALTER TABLE recruitment_posts
                ADD COLUMN IF NOT EXISTS board_key TEXT NOT NULL DEFAULT 'neo';
            ALTER TABLE recruitment_posts
                ADD COLUMN IF NOT EXISTS notification_reserved BOOLEAN NOT NULL DEFAULT FALSE;
            CREATE INDEX IF NOT EXISTS recruitment_posts_scheduled_idx
                ON recruitment_posts (scheduled_at);
            CREATE INDEX IF NOT EXISTS recruitment_posts_owner_idx
                ON recruitment_posts (owner_token_hash, scheduled_at);
            CREATE INDEX IF NOT EXISTS recruitment_posts_board_scheduled_idx
                ON recruitment_posts (board_key, scheduled_at);

            CREATE TABLE IF NOT EXISTS recruitment_notification_pairs (
                recruitment_id UUID PRIMARY KEY,
                reserved_at TIMESTAMPTZ NOT NULL
            );
            CREATE INDEX IF NOT EXISTS recruitment_notification_pairs_reserved_idx
                ON recruitment_notification_pairs (reserved_at);

            CREATE TABLE IF NOT EXISTS recruitment_notification_outbox (
                event_id UUID PRIMARY KEY,
                recruitment_id UUID NOT NULL,
                event_order SMALLINT NOT NULL,
                event_type TEXT NOT NULL,
                board_key TEXT NOT NULL,
                player_name VARCHAR(24) NOT NULL,
                rule_key TEXT NOT NULL,
                scheduled_at TIMESTAMPTZ NOT NULL,
                resolution_reason TEXT,
                created_at TIMESTAMPTZ NOT NULL,
                delivered_at TIMESTAMPTZ,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                next_attempt_at TIMESTAMPTZ NOT NULL,
                UNIQUE (recruitment_id, event_order)
            );
            CREATE INDEX IF NOT EXISTS recruitment_notification_outbox_pending_idx
                ON recruitment_notification_outbox (delivered_at, next_attempt_at, created_at);
            """
        )

    async def list_active(
        self,
        *,
        board_key: object,
        owner_token: Optional[object] = None,
        now: Optional[datetime] = None,
    ) -> list[dict]:
        current = now or utc_now()
        clean_board_key = validate_board_key(board_key)
        viewer_hash = None
        if owner_token is not None:
            viewer_hash = owner_token_hash(validate_owner_token(owner_token))
        posts = await self._active_posts(clean_board_key, current)
        return [post.public_payload(viewer_hash) for post in posts]

    async def create(
        self,
        *,
        name: object,
        rule_key: object,
        scheduled_at: object,
        owner_token: object,
        board_key: object,
        now: Optional[datetime] = None,
    ) -> Recruitment:
        current = now or utc_now()
        clean_board_key, clean_name, clean_rule_key, clean_scheduled_at, clean_owner_hash = (
            validate_recruitment_input(
                name=name,
                rule_key=rule_key,
                scheduled_at=scheduled_at,
                owner_token=owner_token,
                board_key=board_key,
                now=current,
            )
        )
        post = Recruitment(
            recruitment_id=str(uuid.uuid4()),
            board_key=clean_board_key,
            name=clean_name,
            rule_key=clean_rule_key,
            scheduled_at=clean_scheduled_at,
            created_at=current,
            owner_hash=clean_owner_hash,
        )

        if self.pool is None:
            async with self._memory_lock:
                self._prune_memory(current)
                if any(
                    item.board_key == clean_board_key and item.owner_hash == clean_owner_hash
                    for item in self._memory_posts.values()
                ):
                    raise RecruitmentError(
                        "owner_limit",
                        "投稿できる募集は1ユーザー1件です。先に自分の募集を削除してください。",
                    )
                board_count = sum(
                    item.board_key == clean_board_key
                    for item in self._memory_posts.values()
                )
                if board_count >= MAX_ACTIVE_RECRUITMENTS:
                    raise RecruitmentError(
                        "board_full",
                        "募集は現在5件あります。期限切れまたは削除をお待ちください。",
                    )
                if self._reserve_memory_notification_pair(current):
                    post = replace(post, notification_reserved=True)
                    self._queue_memory_notification(post, "created", current)
                self._memory_posts[post.recruitment_id] = post
            return post

        async with self.pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    "SELECT pg_advisory_xact_lock(hashtext('primeqk_recruitment_posts'))"
                )
                await self._expire_database_posts(connection, current)
                owner_exists = await connection.fetchval(
                    """
                    SELECT EXISTS(
                        SELECT 1 FROM recruitment_posts
                        WHERE board_key = $1
                          AND owner_token_hash = $2
                          AND scheduled_at > $3
                    )
                    """,
                    clean_board_key,
                    clean_owner_hash,
                    current,
                )
                if owner_exists:
                    raise RecruitmentError(
                        "owner_limit",
                        "投稿できる募集は1ユーザー1件です。先に自分の募集を削除してください。",
                    )
                active_count = await connection.fetchval(
                    """
                    SELECT COUNT(*)::int FROM recruitment_posts
                    WHERE board_key = $1 AND scheduled_at > $2
                    """,
                    clean_board_key,
                    current,
                )
                if int(active_count or 0) >= MAX_ACTIVE_RECRUITMENTS:
                    raise RecruitmentError(
                        "board_full",
                        "募集は現在5件あります。期限切れまたは削除をお待ちください。",
                    )
                notification_reserved = await self._reserve_database_notification_pair(
                    connection,
                    post.recruitment_id,
                    current,
                )
                if notification_reserved:
                    post = replace(post, notification_reserved=True)
                await connection.execute(
                    """
                    INSERT INTO recruitment_posts (
                        recruitment_id, board_key, player_name, rule_key,
                        scheduled_at, owner_token_hash, notification_reserved,
                        created_at
                    )
                    VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8)
                    """,
                    post.recruitment_id,
                    post.board_key,
                    post.name,
                    post.rule_key,
                    post.scheduled_at,
                    post.owner_hash,
                    post.notification_reserved,
                    post.created_at,
                )
                if post.notification_reserved:
                    await self._queue_database_notification(
                        connection,
                        post,
                        "created",
                        current,
                    )
        return post

    async def delete(
        self,
        *,
        recruitment_id: object,
        owner_token: object,
        board_key: object,
        now: Optional[datetime] = None,
    ) -> bool:
        current = now or utc_now()
        clean_board_key = validate_board_key(board_key)
        if not isinstance(recruitment_id, str):
            raise RecruitmentError("not_found", "募集が見つかりません。")
        try:
            uuid.UUID(recruitment_id)
        except ValueError as exc:
            raise RecruitmentError("not_found", "募集が見つかりません。") from exc
        clean_owner_hash = owner_token_hash(validate_owner_token(owner_token))

        if self.pool is None:
            async with self._memory_lock:
                self._prune_memory(current)
                post = self._memory_posts.get(recruitment_id)
                if (
                    post is None
                    or post.board_key != clean_board_key
                    or not secrets.compare_digest(post.owner_hash, clean_owner_hash)
                ):
                    return False
                if post.notification_reserved:
                    self._queue_memory_notification(
                        post,
                        "resolved",
                        current,
                        resolution_reason="deleted",
                    )
                del self._memory_posts[recruitment_id]
                return True

        async with self.pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    "SELECT pg_advisory_xact_lock(hashtext('primeqk_recruitment_posts'))"
                )
                row = await connection.fetchrow(
                    """
                    SELECT recruitment_id, board_key, player_name, rule_key,
                           scheduled_at, owner_token_hash, notification_reserved,
                           created_at
                    FROM recruitment_posts
                    WHERE recruitment_id = $1::uuid
                      AND board_key = $2
                      AND owner_token_hash = $3
                      AND scheduled_at > $4
                    FOR UPDATE
                    """,
                    recruitment_id,
                    clean_board_key,
                    clean_owner_hash,
                    current,
                )
                if row is None:
                    return False
                post = self._post_from_row(row)
                if post.notification_reserved:
                    await self._queue_database_notification(
                        connection,
                        post,
                        "resolved",
                        current,
                        resolution_reason="deleted",
                    )
                await connection.execute(
                    "DELETE FROM recruitment_posts WHERE recruitment_id = $1::uuid",
                    recruitment_id,
                )
                return True

    async def _active_posts(self, board_key: str, current: datetime) -> list[Recruitment]:
        if self.pool is None:
            async with self._memory_lock:
                self._prune_memory(current)
                return sorted(
                    (
                        post
                        for post in self._memory_posts.values()
                        if post.board_key == board_key
                    ),
                    key=lambda item: (item.scheduled_at, item.created_at),
                )
        async with self.pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    "SELECT pg_advisory_xact_lock(hashtext('primeqk_recruitment_posts'))"
                )
                await self._expire_database_posts(connection, current)
                rows = await connection.fetch(
                    """
                    SELECT recruitment_id, board_key, player_name, rule_key,
                           scheduled_at, owner_token_hash, notification_reserved,
                           created_at
                    FROM recruitment_posts
                    WHERE board_key = $1 AND scheduled_at > $2
                    ORDER BY scheduled_at ASC, created_at ASC
                    LIMIT $3
                    """,
                    board_key,
                    current,
                    MAX_ACTIVE_RECRUITMENTS,
                )
                return [self._post_from_row(row) for row in rows]

    async def expire_due(self, *, now: Optional[datetime] = None) -> int:
        current = now or utc_now()
        if self.pool is None:
            async with self._memory_lock:
                return self._prune_memory(current)
        async with self.pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    "SELECT pg_advisory_xact_lock(hashtext('primeqk_recruitment_posts'))"
                )
                return await self._expire_database_posts(connection, current)

    async def pending_notifications(
        self,
        *,
        now: Optional[datetime] = None,
        limit: int = 10,
    ) -> list[RecruitmentNotification]:
        current = now or utc_now()
        if self.pool is None:
            async with self._memory_lock:
                retention_cutoff = current - timedelta(days=7)
                self._memory_notification_outbox = [
                    event
                    for event in self._memory_notification_outbox
                    if event["delivered_at"] is None
                    or event["delivered_at"] > retention_cutoff
                ]
                pending = []
                for event in sorted(
                    self._memory_notification_outbox,
                    key=lambda item: (item["created_at"], item["event_order"]),
                ):
                    if event["delivered_at"] is not None or event["next_attempt_at"] > current:
                        continue
                    earlier_pending = any(
                        other["recruitment_id"] == event["recruitment_id"]
                        and other["event_order"] < event["event_order"]
                        and other["delivered_at"] is None
                        for other in self._memory_notification_outbox
                    )
                    if earlier_pending:
                        continue
                    pending.append(self._notification_from_mapping(event))
                    event["next_attempt_at"] = current + timedelta(seconds=60)
                    if len(pending) >= max(1, limit):
                        break
                return pending
        async with self.pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    """
                    DELETE FROM recruitment_notification_outbox
                    WHERE delivered_at IS NOT NULL AND delivered_at <= $1
                    """,
                    current - timedelta(days=7),
                )
                rows = await connection.fetch(
                    """
                    SELECT event_id, recruitment_id, event_order, event_type, board_key,
                           player_name, rule_key, scheduled_at, resolution_reason,
                           attempt_count
                    FROM recruitment_notification_outbox AS event
                    WHERE event.delivered_at IS NULL
                      AND event.next_attempt_at <= $1
                      AND NOT EXISTS (
                          SELECT 1
                          FROM recruitment_notification_outbox AS earlier
                          WHERE earlier.recruitment_id = event.recruitment_id
                            AND earlier.event_order < event.event_order
                            AND earlier.delivered_at IS NULL
                      )
                    ORDER BY event.created_at ASC, event.event_order ASC
                    LIMIT $2
                    FOR UPDATE SKIP LOCKED
                    """,
                    current,
                    max(1, limit),
                )
                for row in rows:
                    await connection.execute(
                        """
                        UPDATE recruitment_notification_outbox
                        SET next_attempt_at = $2
                        WHERE event_id = $1::uuid
                        """,
                        str(row["event_id"]),
                        current + timedelta(seconds=60),
                    )
                return [self._notification_from_mapping(row) for row in rows]

    async def mark_notification_delivered(
        self,
        event_id: str,
        *,
        delivered_at: Optional[datetime] = None,
    ) -> None:
        current = delivered_at or utc_now()
        if self.pool is None:
            async with self._memory_lock:
                for event in self._memory_notification_outbox:
                    if event["event_id"] == event_id:
                        event["delivered_at"] = current
                        return
            return
        await self.pool.execute(
            """
            UPDATE recruitment_notification_outbox
            SET delivered_at = $2
            WHERE event_id = $1::uuid
            """,
            event_id,
            current,
        )

    async def mark_notification_failed(
        self,
        event_id: str,
        *,
        next_attempt_at: datetime,
    ) -> None:
        if self.pool is None:
            async with self._memory_lock:
                for event in self._memory_notification_outbox:
                    if event["event_id"] == event_id:
                        event["attempt_count"] += 1
                        event["next_attempt_at"] = next_attempt_at
                        return
            return
        await self.pool.execute(
            """
            UPDATE recruitment_notification_outbox
            SET attempt_count = attempt_count + 1,
                next_attempt_at = $2
            WHERE event_id = $1::uuid
            """,
            event_id,
            next_attempt_at,
        )

    def _reserve_memory_notification_pair(self, current: datetime) -> bool:
        if not self.notifications_enabled or self.notification_pair_limit <= 0:
            return False
        cutoff = current - timedelta(seconds=self.notification_window_seconds)
        while (
            self._memory_notification_pair_times
            and self._memory_notification_pair_times[0] <= cutoff
        ):
            self._memory_notification_pair_times.popleft()
        if len(self._memory_notification_pair_times) >= self.notification_pair_limit:
            return False
        self._memory_notification_pair_times.append(current)
        return True

    async def _reserve_database_notification_pair(
        self,
        connection,
        recruitment_id: str,
        current: datetime,
    ) -> bool:
        if not self.notifications_enabled or self.notification_pair_limit <= 0:
            return False
        cutoff = current - timedelta(seconds=self.notification_window_seconds)
        await connection.execute(
            "DELETE FROM recruitment_notification_pairs WHERE reserved_at <= $1",
            cutoff,
        )
        reserved_count = await connection.fetchval(
            "SELECT COUNT(*)::int FROM recruitment_notification_pairs",
        )
        if int(reserved_count or 0) >= self.notification_pair_limit:
            return False
        await connection.execute(
            """
            INSERT INTO recruitment_notification_pairs (recruitment_id, reserved_at)
            VALUES ($1::uuid, $2)
            """,
            recruitment_id,
            current,
        )
        return True

    def _queue_memory_notification(
        self,
        post: Recruitment,
        event_type: str,
        current: datetime,
        *,
        resolution_reason: Optional[str] = None,
    ) -> None:
        event_order = 1 if event_type == "created" else 2
        if any(
            event["recruitment_id"] == post.recruitment_id
            and event["event_order"] == event_order
            for event in self._memory_notification_outbox
        ):
            return
        self._memory_notification_outbox.append({
            "event_id": str(uuid.uuid4()),
            "recruitment_id": post.recruitment_id,
            "event_order": event_order,
            "event_type": event_type,
            "board_key": post.board_key,
            "player_name": post.name,
            "rule_key": post.rule_key,
            "scheduled_at": post.scheduled_at,
            "resolution_reason": resolution_reason,
            "created_at": current,
            "delivered_at": None,
            "attempt_count": 0,
            "next_attempt_at": current,
        })

    async def _queue_database_notification(
        self,
        connection,
        post: Recruitment,
        event_type: str,
        current: datetime,
        *,
        resolution_reason: Optional[str] = None,
    ) -> None:
        event_order = 1 if event_type == "created" else 2
        await connection.execute(
            """
            INSERT INTO recruitment_notification_outbox (
                event_id, recruitment_id, event_order, event_type, board_key,
                player_name, rule_key, scheduled_at, resolution_reason,
                created_at, next_attempt_at
            )
            VALUES ($1::uuid, $2::uuid, $3, $4, $5, $6, $7, $8, $9, $10, $10)
            ON CONFLICT (recruitment_id, event_order) DO NOTHING
            """,
            str(uuid.uuid4()),
            post.recruitment_id,
            event_order,
            event_type,
            post.board_key,
            post.name,
            post.rule_key,
            post.scheduled_at,
            resolution_reason,
            current,
        )

    async def _expire_database_posts(self, connection, current: datetime) -> int:
        rows = await connection.fetch(
            """
            SELECT recruitment_id, board_key, player_name, rule_key,
                   scheduled_at, owner_token_hash, notification_reserved,
                   created_at
            FROM recruitment_posts
            WHERE scheduled_at <= $1
            FOR UPDATE
            """,
            current,
        )
        for row in rows:
            post = self._post_from_row(row)
            if post.notification_reserved:
                await self._queue_database_notification(
                    connection,
                    post,
                    "resolved",
                    current,
                    resolution_reason="expired",
                )
            await connection.execute(
                "DELETE FROM recruitment_posts WHERE recruitment_id = $1::uuid",
                post.recruitment_id,
            )
        return len(rows)

    @staticmethod
    def _post_from_row(row) -> Recruitment:
        return Recruitment(
            recruitment_id=str(row["recruitment_id"]),
            board_key=row["board_key"],
            name=row["player_name"],
            rule_key=row["rule_key"],
            scheduled_at=row["scheduled_at"],
            created_at=row["created_at"],
            owner_hash=row["owner_token_hash"],
            notification_reserved=bool(row["notification_reserved"]),
        )

    @staticmethod
    def _notification_from_mapping(row) -> RecruitmentNotification:
        return RecruitmentNotification(
            event_id=str(row["event_id"]),
            recruitment_id=str(row["recruitment_id"]),
            event_order=int(row["event_order"]),
            event_type=row["event_type"],
            board_key=row["board_key"],
            name=row["player_name"],
            rule_key=row["rule_key"],
            scheduled_at=row["scheduled_at"],
            resolution_reason=row["resolution_reason"],
            attempt_count=int(row["attempt_count"]),
        )

    def _prune_memory(self, current: datetime) -> int:
        expired_posts = [
            post
            for post in self._memory_posts.values()
            if post.scheduled_at <= current
        ]
        for post in expired_posts:
            if post.notification_reserved:
                self._queue_memory_notification(
                    post,
                    "resolved",
                    current,
                    resolution_reason="expired",
                )
            del self._memory_posts[post.recruitment_id]
        return len(expired_posts)

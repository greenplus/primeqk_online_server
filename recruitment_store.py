from __future__ import annotations

import asyncio
import hashlib
import os
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional


MAX_ACTIVE_RECRUITMENTS = 5
MAX_RECRUITMENT_HORIZON = timedelta(hours=24)
MAX_RECRUITMENT_NAME_LENGTH = 24
RECRUITMENT_RULE_LABELS = {
    "beginner": "初級（7枚・偶数半減）",
    "advanced": "上級（11枚・通常）",
    "either": "どちらでも",
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


def validate_recruitment_input(
    *,
    name: object,
    rule_key: object,
    scheduled_at: object,
    owner_token: object,
    now: Optional[datetime] = None,
) -> tuple[str, str, datetime, str]:
    current = now or utc_now()
    if not isinstance(name, str) or not name.strip():
        raise RecruitmentError("invalid_name", "名前を入力してください。")
    clean_name = name.strip()
    if len(clean_name) > MAX_RECRUITMENT_NAME_LENGTH:
        raise RecruitmentError(
            "invalid_name",
            f"名前は{MAX_RECRUITMENT_NAME_LENGTH}文字以内にしてください。",
        )
    if not isinstance(rule_key, str) or rule_key not in RECRUITMENT_RULE_LABELS:
        raise RecruitmentError("invalid_rule", "希望ルールを選んでください。")
    clean_scheduled_at = parse_scheduled_at(scheduled_at)
    if clean_scheduled_at <= current:
        raise RecruitmentError("invalid_time", "集合時間は現在より後にしてください。")
    if clean_scheduled_at > current + MAX_RECRUITMENT_HORIZON:
        raise RecruitmentError("invalid_time", "集合時間は24時間以内にしてください。")
    clean_owner_token = validate_owner_token(owner_token)
    return clean_name, rule_key, clean_scheduled_at, owner_token_hash(clean_owner_token)


@dataclass(frozen=True)
class Recruitment:
    recruitment_id: str
    name: str
    rule_key: str
    scheduled_at: datetime
    created_at: datetime
    owner_hash: str

    def public_payload(self, viewer_owner_hash: Optional[str] = None) -> dict:
        return {
            "id": self.recruitment_id,
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


class RecruitmentStore:
    """PostgreSQL-backed recruitment board with an in-memory local fallback."""

    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or os.getenv("DATABASE_URL")
        self.pool = None
        self.last_error: Optional[str] = None
        self._memory_posts: dict[str, Recruitment] = {}
        self._memory_lock = asyncio.Lock()

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
                player_name VARCHAR(24) NOT NULL,
                rule_key TEXT NOT NULL,
                scheduled_at TIMESTAMPTZ NOT NULL,
                owner_token_hash CHAR(64) NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS recruitment_posts_scheduled_idx
                ON recruitment_posts (scheduled_at);
            CREATE INDEX IF NOT EXISTS recruitment_posts_owner_idx
                ON recruitment_posts (owner_token_hash, scheduled_at);
            """
        )

    async def list_active(
        self,
        *,
        owner_token: Optional[object] = None,
        now: Optional[datetime] = None,
    ) -> list[dict]:
        current = now or utc_now()
        viewer_hash = None
        if owner_token is not None:
            viewer_hash = owner_token_hash(validate_owner_token(owner_token))
        posts = await self._active_posts(current)
        return [post.public_payload(viewer_hash) for post in posts]

    async def create(
        self,
        *,
        name: object,
        rule_key: object,
        scheduled_at: object,
        owner_token: object,
        now: Optional[datetime] = None,
    ) -> Recruitment:
        current = now or utc_now()
        clean_name, clean_rule_key, clean_scheduled_at, clean_owner_hash = (
            validate_recruitment_input(
                name=name,
                rule_key=rule_key,
                scheduled_at=scheduled_at,
                owner_token=owner_token,
                now=current,
            )
        )
        post = Recruitment(
            recruitment_id=str(uuid.uuid4()),
            name=clean_name,
            rule_key=clean_rule_key,
            scheduled_at=clean_scheduled_at,
            created_at=current,
            owner_hash=clean_owner_hash,
        )

        if self.pool is None:
            async with self._memory_lock:
                self._prune_memory(current)
                if any(item.owner_hash == clean_owner_hash for item in self._memory_posts.values()):
                    raise RecruitmentError(
                        "owner_limit",
                        "投稿できる募集は1ユーザー1件です。先に自分の募集を削除してください。",
                    )
                if len(self._memory_posts) >= MAX_ACTIVE_RECRUITMENTS:
                    raise RecruitmentError(
                        "board_full",
                        "募集は現在5件あります。期限切れまたは削除をお待ちください。",
                    )
                self._memory_posts[post.recruitment_id] = post
            return post

        async with self.pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    "SELECT pg_advisory_xact_lock(hashtext('primeqk_recruitment_posts'))"
                )
                await connection.execute(
                    "DELETE FROM recruitment_posts WHERE scheduled_at <= $1",
                    current,
                )
                owner_exists = await connection.fetchval(
                    """
                    SELECT EXISTS(
                        SELECT 1 FROM recruitment_posts
                        WHERE owner_token_hash = $1 AND scheduled_at > $2
                    )
                    """,
                    clean_owner_hash,
                    current,
                )
                if owner_exists:
                    raise RecruitmentError(
                        "owner_limit",
                        "投稿できる募集は1ユーザー1件です。先に自分の募集を削除してください。",
                    )
                active_count = await connection.fetchval(
                    "SELECT COUNT(*)::int FROM recruitment_posts WHERE scheduled_at > $1",
                    current,
                )
                if int(active_count or 0) >= MAX_ACTIVE_RECRUITMENTS:
                    raise RecruitmentError(
                        "board_full",
                        "募集は現在5件あります。期限切れまたは削除をお待ちください。",
                    )
                await connection.execute(
                    """
                    INSERT INTO recruitment_posts (
                        recruitment_id, player_name, rule_key, scheduled_at,
                        owner_token_hash, created_at
                    )
                    VALUES ($1::uuid, $2, $3, $4, $5, $6)
                    """,
                    post.recruitment_id,
                    post.name,
                    post.rule_key,
                    post.scheduled_at,
                    post.owner_hash,
                    post.created_at,
                )
        return post

    async def delete(
        self,
        *,
        recruitment_id: object,
        owner_token: object,
        now: Optional[datetime] = None,
    ) -> bool:
        current = now or utc_now()
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
                if post is None or not secrets.compare_digest(post.owner_hash, clean_owner_hash):
                    return False
                del self._memory_posts[recruitment_id]
                return True

        result = await self.pool.execute(
            """
            DELETE FROM recruitment_posts
            WHERE recruitment_id = $1::uuid
              AND owner_token_hash = $2
              AND scheduled_at > $3
            """,
            recruitment_id,
            clean_owner_hash,
            current,
        )
        return result == "DELETE 1"

    async def _active_posts(self, current: datetime) -> list[Recruitment]:
        if self.pool is None:
            async with self._memory_lock:
                self._prune_memory(current)
                return sorted(
                    self._memory_posts.values(),
                    key=lambda item: (item.scheduled_at, item.created_at),
                )
        await self.pool.execute(
            "DELETE FROM recruitment_posts WHERE scheduled_at <= $1",
            current,
        )
        rows = await self.pool.fetch(
            """
            SELECT recruitment_id, player_name, rule_key, scheduled_at,
                   owner_token_hash, created_at
            FROM recruitment_posts
            WHERE scheduled_at > $1
            ORDER BY scheduled_at ASC, created_at ASC
            LIMIT $2
            """,
            current,
            MAX_ACTIVE_RECRUITMENTS,
        )
        return [
            Recruitment(
                recruitment_id=str(row["recruitment_id"]),
                name=row["player_name"],
                rule_key=row["rule_key"],
                scheduled_at=row["scheduled_at"],
                created_at=row["created_at"],
                owner_hash=row["owner_token_hash"],
            )
            for row in rows
        ]

    def _prune_memory(self, current: datetime) -> None:
        expired_ids = [
            post_id
            for post_id, post in self._memory_posts.items()
            if post.scheduled_at <= current
        ]
        for post_id in expired_ids:
            del self._memory_posts[post_id]

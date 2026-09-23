from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
import hashlib
import secrets
import uuid


TERMINAL_MATCH_STATUSES = {"completed", "skipped"}
ACTIVE_RUN_STATUSES = {"scheduled", "registration", "running"}
RETIREMENT_GRACE_SECONDS = 300


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def isoformat(value: Optional[datetime]) -> Optional[str]:
    return value.astimezone(timezone.utc).isoformat() if value is not None else None


def parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise ValueError("日時はタイムゾーン付きISO形式で指定してください。")
    if parsed.tzinfo is None:
        raise ValueError("日時にはタイムゾーンが必要です。")
    return parsed.astimezone(timezone.utc)


def hash_resume_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def issue_resume_token() -> str:
    return secrets.token_urlsafe(32)


@dataclass
class TournamentParticipant:
    participant_id: str
    display_name: str
    token_hash: str
    registered_at: datetime
    withdrawn: bool = False
    retired_at: Optional[datetime] = None
    unavailable_since: Optional[datetime] = None
    unavailable_reason: Optional[str] = None
    waiting_since: Optional[datetime] = None

    def to_dict(self, *, public: bool = False) -> dict[str, Any]:
        result = {
            "participant_id": self.participant_id,
            "display_name": self.display_name,
            "registered_at": isoformat(self.registered_at),
            "withdrawn": self.withdrawn,
            "retired_at": isoformat(self.retired_at),
            "unavailable_since": isoformat(self.unavailable_since),
            "unavailable_reason": self.unavailable_reason,
            "waiting_since": isoformat(self.waiting_since),
            "retirement_deadline_at": isoformat(self.unavailable_since + timedelta(seconds=RETIREMENT_GRACE_SECONDS)) if self.unavailable_since and not self.retired_at else None,
        }
        if not public:
            result["token_hash"] = self.token_hash
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TournamentParticipant":
        return cls(
            participant_id=str(value["participant_id"]),
            display_name=str(value["display_name"]),
            token_hash=str(value.get("token_hash", "")),
            registered_at=parse_datetime(value["registered_at"]),
            withdrawn=bool(value.get("withdrawn", False)),
            retired_at=parse_datetime(value["retired_at"]) if value.get("retired_at") else None,
            unavailable_since=parse_datetime(value["unavailable_since"]) if value.get("unavailable_since") else None,
            unavailable_reason=value.get("unavailable_reason"),
            waiting_since=parse_datetime(value["waiting_since"]) if value.get("waiting_since") else None,
        )


@dataclass
class TournamentMatch:
    match_id: str
    round_no: int
    sequence_no: int
    player1_id: str
    player2_id: str
    status: str = "pending"
    winner_id: Optional[str] = None
    resolution: Optional[str] = None
    called_at: Optional[datetime] = None
    ready_deadline_at: Optional[datetime] = None
    ready_player_ids: list[str] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "match_id": self.match_id,
            "round_no": self.round_no,
            "sequence_no": self.sequence_no,
            "player1_id": self.player1_id,
            "player2_id": self.player2_id,
            "status": self.status,
            "winner_id": self.winner_id,
            "resolution": self.resolution,
            "called_at": isoformat(self.called_at),
            "ready_deadline_at": isoformat(self.ready_deadline_at),
            "ready_player_ids": list(self.ready_player_ids),
            "started_at": isoformat(self.started_at),
            "completed_at": isoformat(self.completed_at),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TournamentMatch":
        return cls(
            match_id=str(value["match_id"]),
            round_no=int(value["round_no"]),
            sequence_no=int(value["sequence_no"]),
            player1_id=str(value["player1_id"]),
            player2_id=str(value["player2_id"]),
            status=str(value.get("status", "pending")),
            winner_id=value.get("winner_id"),
            resolution=value.get("resolution"),
            called_at=parse_datetime(value["called_at"]) if value.get("called_at") else None,
            ready_deadline_at=parse_datetime(value["ready_deadline_at"]) if value.get("ready_deadline_at") else None,
            ready_player_ids=[str(item) for item in value.get("ready_player_ids", [])],
            started_at=parse_datetime(value["started_at"]) if value.get("started_at") else None,
            completed_at=parse_datetime(value["completed_at"]) if value.get("completed_at") else None,
        )


def round_robin_matches(participant_ids: list[str]) -> list[TournamentMatch]:
    """Circle method. A bye is represented internally by None and omitted."""
    players: list[Optional[str]] = list(participant_ids)
    if len(players) < 2:
        return []
    if len(players) % 2:
        players.append(None)

    matches: list[TournamentMatch] = []
    sequence_no = 1
    for round_index in range(len(players) - 1):
        for pair_index in range(len(players) // 2):
            left = players[pair_index]
            right = players[-1 - pair_index]
            if left is None or right is None:
                continue
            if round_index % 2 and pair_index == 0:
                left, right = right, left
            matches.append(TournamentMatch(
                match_id=str(uuid.uuid4()),
                round_no=round_index + 1,
                sequence_no=sequence_no,
                player1_id=left,
                player2_id=right,
            ))
            sequence_no += 1
        players = [players[0], players[-1], *players[1:-1]]
    return matches


@dataclass
class TournamentRun:
    run_id: str
    format_key: str
    title: str
    room_id: str
    rule_key: str
    registration_opens_at: datetime
    starts_at: datetime
    max_participants: int = 10
    status: str = "scheduled"
    participants: dict[str, TournamentParticipant] = field(default_factory=dict)
    matches: list[TournamentMatch] = field(default_factory=list)
    current_match_id: Optional[str] = None
    active_match_ids: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=utc_now)
    finished_at: Optional[datetime] = None
    pairing_mode: str = "rounds"
    registration_closes_at: Optional[datetime] = None

    @classmethod
    def create(
        cls,
        *,
        format_key: str,
        title: str,
        room_id: str,
        rule_key: str,
        registration_opens_at: datetime,
        starts_at: datetime,
        max_participants: int = 10,
        pairing_mode: str = "rounds",
        registration_closes_at: Optional[datetime] = None,
        now: Optional[datetime] = None,
    ) -> "TournamentRun":
        opens_at = parse_datetime(registration_opens_at)
        begins_at = parse_datetime(starts_at)
        if pairing_mode not in {"rounds", "flexible"}:
            raise ValueError("対戦方式が正しくありません。")
        closes_at = parse_datetime(registration_closes_at) if registration_closes_at is not None else begins_at
        if pairing_mode == "flexible" and closes_at <= begins_at:
            raise ValueError("途中参加の受付終了日時は大会開始日時より後にしてください。")
        if pairing_mode == "rounds":
            closes_at = begins_at
        if begins_at <= opens_at:
            raise ValueError("大会開始日時は参加登録開始日時より後にしてください。")
        if not 2 <= int(max_participants) <= 32:
            raise ValueError("参加上限は2〜32人で指定してください。")
        current = now or utc_now()
        return cls(
            run_id=str(uuid.uuid4()),
            format_key=format_key.strip(),
            title=title.strip(),
            room_id=room_id,
            rule_key=rule_key,
            registration_opens_at=opens_at,
            starts_at=begins_at,
            max_participants=int(max_participants),
            status="registration" if opens_at <= current < begins_at else "scheduled",
            created_at=current,
            pairing_mode=pairing_mode,
            registration_closes_at=closes_at,
        )

    @property
    def flexible(self) -> bool:
        return self.pairing_mode == "flexible"

    def registration_open(self, *, now: Optional[datetime] = None) -> bool:
        current = now or utc_now()
        return self.status in {"registration", "running"} and self.registration_opens_at <= current < (self.registration_closes_at or self.starts_at) and (self.flexible or self.status == "registration")

    @property
    def active_participants(self) -> list[TournamentParticipant]:
        return [participant for participant in self.participants.values() if not participant.withdrawn]

    @property
    def current_match(self) -> Optional[TournamentMatch]:
        active = self.current_matches
        return active[0] if active else None

    @property
    def current_matches(self) -> list[TournamentMatch]:
        active_ids = set(self.active_match_ids)
        if self.current_match_id:
            active_ids.add(self.current_match_id)
        return [
            match
            for match in self.matches
            if match.match_id in active_ids and match.status in {"called", "playing"}
        ]

    def current_match_for_participant(self, participant_id: str) -> Optional[TournamentMatch]:
        return next(
            (
                match
                for match in self.current_matches
                if participant_id in {match.player1_id, match.player2_id}
            ),
            None,
        )

    def participant_for_token(self, token: str) -> Optional[TournamentParticipant]:
        digest = hash_resume_token(token)
        return next(
            (participant for participant in self.participants.values() if participant.token_hash == digest),
            None,
        )

    def register(self, display_name: str, *, now: Optional[datetime] = None) -> tuple[TournamentParticipant, str]:
        self.advance_clock(now=now)
        if not self.registration_open(now=now):
            raise ValueError("現在は参加登録を受け付けていません。")
        name = display_name.strip()
        if not name:
            raise ValueError("表示名を入力してください。")
        if any(
            participant.display_name.casefold() == name.casefold() and not participant.withdrawn
            for participant in self.participants.values()
        ):
            raise ValueError("同じ表示名がすでに登録されています。復帰トークンを使用してください。")
        if len(self.active_participants) >= self.max_participants:
            raise ValueError("参加枠が満員です。")
        token = issue_resume_token()
        participant = TournamentParticipant(
            participant_id=str(uuid.uuid4()),
            display_name=name,
            token_hash=hash_resume_token(token),
            registered_at=now or utc_now(),
            waiting_since=now or utc_now(),
        )
        self.participants[participant.participant_id] = participant
        if self.flexible and self.status == "running":
            self.ensure_flexible_pairs(now=now)
        return participant, token

    def withdraw(self, participant_id: str, *, now: Optional[datetime] = None) -> None:
        self.advance_clock(now=now)
        if self.status not in {"scheduled", "registration"}:
            raise ValueError("大会開始後は参加取消できません。")
        participant = self.participants.get(participant_id)
        if participant is None:
            raise ValueError("参加登録が見つかりません。")
        participant.withdrawn = True

    def advance_clock(self, *, now: Optional[datetime] = None) -> Optional[str]:
        current = now or utc_now()
        old_status = self.status
        if self.status == "scheduled" and current >= self.registration_opens_at:
            self.status = "registration"
        if self.status in {"scheduled", "registration"} and current >= self.starts_at:
            participant_ids = [participant.participant_id for participant in self.active_participants]
            if self.flexible:
                self.status = "running"
                self.ensure_flexible_pairs(now=current)
            elif len(participant_ids) < 2:
                self.status = "cancelled"
                self.finished_at = current
            else:
                self.matches = round_robin_matches(participant_ids)
                self.status = "running"
        return self.status if self.status != old_status else None

    def ensure_flexible_pairs(self, *, now: Optional[datetime] = None) -> None:
        players = self.active_participants
        existing = {frozenset((m.player1_id, m.player2_id)) for m in self.matches}
        for index, left in enumerate(players):
            for right in players[index + 1:]:
                pair = frozenset((left.participant_id, right.participant_id))
                if pair in existing:
                    continue
                self.matches.append(TournamentMatch(
                    match_id=str(uuid.uuid4()), round_no=0, sequence_no=len(self.matches) + 1,
                    player1_id=left.participant_id, player2_id=right.participant_id,
                ))
                existing.add(pair)
        self.settle_retired_pairs(now=now)

    def pause_participant(self, participant_id: str, reason: str, *, now: Optional[datetime] = None) -> None:
        participant = self.participants[participant_id]
        if participant.retired_at or participant.withdrawn:
            return
        if participant.unavailable_since is None:
            participant.unavailable_since = now or utc_now()
            participant.unavailable_reason = reason

    def resume_participant(self, participant_id: str, *, now: Optional[datetime] = None) -> None:
        participant = self.participants[participant_id]
        current = now or utc_now()
        if participant.retired_at or (participant.unavailable_since and (current - participant.unavailable_since).total_seconds() >= RETIREMENT_GRACE_SECONDS):
            raise ValueError("棄権の期限を過ぎています。この開催回は観戦のみ可能です。")
        participant.unavailable_since = None
        participant.unavailable_reason = None
        participant.waiting_since = current

    def cancel_call(self, match: TournamentMatch) -> None:
        if match.status != "called":
            return
        match.status = "pending"
        match.called_at = match.ready_deadline_at = None
        match.ready_player_ids = []
        self.active_match_ids = [item for item in self.active_match_ids if item != match.match_id]
        self.current_match_id = self.active_match_ids[0] if self.active_match_ids else None

    def settle_retired_pairs(self, *, now: Optional[datetime] = None) -> list[TournamentMatch]:
        resolved = []
        for match in self.matches:
            if match.status in TERMINAL_MATCH_STATUSES:
                continue
            left, right = self.participants[match.player1_id], self.participants[match.player2_id]
            if not (left.retired_at or right.retired_at):
                continue
            winner = None if left.retired_at and right.retired_at else right.participant_id if left.retired_at else left.participant_id
            self.resolve_match(match.match_id, winner, resolution="retirement_forfeit" if winner else "retirement_skip", now=now)
            resolved.append(match)
        return resolved

    def retire_participants(self, participant_ids: list[str], *, now: Optional[datetime] = None) -> list[TournamentMatch]:
        current = now or utc_now()
        for participant_id in participant_ids:
            participant = self.participants[participant_id]
            if not participant.retired_at:
                participant.retired_at = current
        return self.settle_retired_pairs(now=current)

    def call_available_matches(self, available_ids: set[str], *, now: Optional[datetime] = None, ready_wait_seconds: int = 60) -> list[TournamentMatch]:
        if not self.flexible or self.status != "running":
            return []
        current = now or utc_now()
        busy = {pid for match in self.current_matches for pid in (match.player1_id, match.player2_id)}
        candidates = sorted((p for p in self.active_participants if p.participant_id in available_ids and p.participant_id not in busy and not p.retired_at and p.unavailable_since is None), key=lambda p: (p.waiting_since or p.registered_at, p.registered_at, p.participant_id))
        free = {p.participant_id for p in candidates}
        pending = {frozenset((m.player1_id, m.player2_id)): m for m in self.matches if m.status == "pending"}
        called = []
        for player in candidates:
            if player.participant_id not in free:
                continue
            opponent = next((p for p in candidates if p.participant_id != player.participant_id and p.participant_id in free and frozenset((player.participant_id, p.participant_id)) in pending), None)
            if opponent is None:
                continue
            match = pending[frozenset((player.participant_id, opponent.participant_id))]
            match.status = "called"
            match.called_at = current
            match.ready_deadline_at = current + timedelta(seconds=ready_wait_seconds)
            match.ready_player_ids = []
            self.active_match_ids.append(match.match_id)
            called.append(match)
            free.difference_update((player.participant_id, opponent.participant_id))
        self.current_match_id = self.active_match_ids[0] if self.active_match_ids else None
        return called

    def next_pending_match(self) -> Optional[TournamentMatch]:
        return next((match for match in self.matches if match.status == "pending"), None)

    def start_next_match(
        self,
        *,
        now: Optional[datetime] = None,
        ready_wait_seconds: int = 60,
    ) -> Optional[TournamentMatch]:
        if self.status != "running" or self.current_match_id is not None:
            return None
        match = self.next_pending_match()
        if match is None:
            self.finish_if_complete(now=now)
            return None
        current = now or utc_now()
        match.status = "called"
        match.called_at = current
        match.ready_deadline_at = current + timedelta(seconds=max(1, ready_wait_seconds))
        match.ready_player_ids = []
        self.current_match_id = match.match_id
        self.active_match_ids = [match.match_id]
        return match

    def start_next_round(
        self,
        *,
        now: Optional[datetime] = None,
        ready_wait_seconds: int = 60,
    ) -> list[TournamentMatch]:
        if self.flexible or self.status != "running" or self.current_matches:
            return []
        pending = [match for match in self.matches if match.status == "pending"]
        if not pending:
            self.finish_if_complete(now=now)
            return []
        round_no = min(match.round_no for match in pending)
        matches = [match for match in pending if match.round_no == round_no]
        current = now or utc_now()
        for match in matches:
            match.status = "called"
            match.called_at = current
            match.ready_deadline_at = current + timedelta(seconds=max(1, ready_wait_seconds))
            match.ready_player_ids = []
        self.active_match_ids = [match.match_id for match in matches]
        self.current_match_id = matches[0].match_id if matches else None
        return matches

    def mark_match_ready(self, participant_id: str, match_id: Optional[str] = None) -> TournamentMatch:
        match = (
            next((item for item in self.current_matches if item.match_id == match_id), None)
            if match_id
            else self.current_match_for_participant(participant_id)
        )
        if match is None or match.status != "called":
            raise ValueError("現在、参加確認中の対戦はありません。")
        if participant_id not in {match.player1_id, match.player2_id}:
            raise ValueError("この対戦の参加者ではありません。")
        if participant_id not in match.ready_player_ids:
            match.ready_player_ids.append(participant_id)
        return match

    def begin_match(
        self,
        match_id: str,
        *,
        now: Optional[datetime] = None,
    ) -> TournamentMatch:
        match = next((item for item in self.current_matches if item.match_id == match_id), None)
        if match is None or match.status not in {"called", "playing"}:
            raise ValueError("開始できる対戦がありません。")
        match.status = "playing"
        if match.started_at is None:
            match.started_at = now or utc_now()
        return match

    def begin_current_match(self, *, now: Optional[datetime] = None) -> TournamentMatch:
        match = self.current_match
        if match is None:
            raise ValueError("開始できる対戦がありません。")
        return self.begin_match(match.match_id, now=now)

    def resolve_match(
        self,
        match_id: str,
        winner_id: Optional[str],
        *,
        resolution: str,
        now: Optional[datetime] = None,
    ) -> TournamentMatch:
        match = next((item for item in self.matches if item.match_id == match_id), None)
        if match is None:
            raise ValueError("対戦が見つかりません。")
        if winner_id is not None and winner_id not in {match.player1_id, match.player2_id}:
            raise ValueError("勝者はこの対戦の参加者から選んでください。")
        match.status = "completed" if winner_id is not None else "skipped"
        match.winner_id = winner_id
        match.resolution = resolution
        match.completed_at = now or utc_now()
        for pid in (match.player1_id, match.player2_id):
            self.participants[pid].waiting_since = now or utc_now()
        self.active_match_ids = [item for item in self.active_match_ids if item != match.match_id]
        if self.current_match_id == match.match_id:
            self.current_match_id = self.active_match_ids[0] if self.active_match_ids else None
        self.finish_if_complete(now=now)
        return match

    def finish_if_complete(self, *, now: Optional[datetime] = None) -> bool:
        if self.status != "running":
            return False
        current = now or utc_now()
        if self.flexible and current < (self.registration_closes_at or self.starts_at):
            return False
        if self.flexible and len(self.active_participants) < 2:
            self.status = "cancelled"
            self.finished_at = current
            return True
        if self.matches and all(match.status in TERMINAL_MATCH_STATUSES for match in self.matches):
            self.status = "finished"
            self.current_match_id = None
            self.active_match_ids = []
            self.finished_at = now or utc_now()
            return True
        return False

    def standings(self) -> list[dict[str, Any]]:
        rows: dict[str, dict[str, Any]] = {
            participant.participant_id: {
                "participant_id": participant.participant_id,
                "display_name": participant.display_name,
                "wins": 0,
                "losses": 0,
                "played": 0,
                "points": 0,
            }
            for participant in self.active_participants
        }
        for match in self.matches:
            if match.status != "completed" or match.winner_id is None:
                continue
            loser_id = match.player2_id if match.winner_id == match.player1_id else match.player1_id
            if match.winner_id in rows:
                rows[match.winner_id]["wins"] += 1
                rows[match.winner_id]["played"] += 1
                rows[match.winner_id]["points"] += 3
            if loser_id in rows:
                rows[loser_id]["losses"] += 1
                rows[loser_id]["played"] += 1

        ordered = sorted(
            rows.values(),
            key=lambda row: (-row["wins"], row["display_name"].casefold()),
        )
        previous_key = None
        previous_rank = 0
        for index, row in enumerate(ordered, start=1):
            key = row["wins"]
            if key != previous_key:
                previous_rank = index
                previous_key = key
            row["rank"] = previous_rank
        return ordered

    def league_table(self) -> dict[str, Any]:
        players = [
            {
                "participant_id": participant.participant_id,
                "display_name": participant.display_name,
            }
            for participant in self.active_participants
        ]
        cells: dict[str, dict[str, dict[str, Any]]] = {
            player["participant_id"]: {}
            for player in players
        }
        for player in players:
            participant_id = player["participant_id"]
            cells[participant_id][participant_id] = {"result": "self", "label": "—"}
        for match in self.matches:
            if match.status == "completed" and match.winner_id is not None:
                player1_result = "win" if match.winner_id == match.player1_id else "loss"
                player2_result = "win" if match.winner_id == match.player2_id else "loss"
                player1_label = "○" if player1_result == "win" else "×"
                player2_label = "○" if player2_result == "win" else "×"
            elif match.status == "skipped":
                player1_result = player2_result = "skipped"
                player1_label = player2_label = "–"
            elif match.status == "playing":
                player1_result = player2_result = "playing"
                player1_label = player2_label = "対戦中"
            elif match.status == "called":
                player1_result = player2_result = "called"
                player1_label = player2_label = "呼出中"
            else:
                player1_result = player2_result = "pending"
                player1_label = player2_label = "・"
            common = {"match_id": match.match_id, "round_no": match.round_no}
            cells[match.player1_id][match.player2_id] = {
                **common,
                "result": player1_result,
                "label": player1_label,
            }
            cells[match.player2_id][match.player1_id] = {
                **common,
                "result": player2_result,
                "label": player2_label,
            }
        return {
            "players": players,
            "rows": [
                {
                    **player,
                    "cells": cells[player["participant_id"]],
                }
                for player in players
            ],
        }

    def public_payload(self, *, viewer_participant_id: Optional[str] = None) -> dict[str, Any]:
        participants = {
            participant.participant_id: participant.display_name
            for participant in self.active_participants
        }
        active = self.current_matches
        current = self.current_match_for_participant(viewer_participant_id) if viewer_participant_id else None
        if current is None:
            current = active[0] if active else None
        return {
            "run_id": self.run_id,
            "format_key": self.format_key,
            "pairing_mode": self.pairing_mode,
            "registration_closes_at": isoformat(self.registration_closes_at or self.starts_at),
            "registration_open": self.registration_open(),
            "retirement_grace_seconds": RETIREMENT_GRACE_SECONDS,
            "ranking_basis": "wins",
            "title": self.title,
            "room_id": self.room_id,
            "rule_key": self.rule_key,
            "registration_opens_at": isoformat(self.registration_opens_at),
            "starts_at": isoformat(self.starts_at),
            "status": self.status,
            "max_participants": self.max_participants,
            "participant_count": len(self.active_participants),
            "participants": [participant.to_dict(public=True) for participant in self.active_participants],
            "viewer_participant_id": viewer_participant_id,
            "registered": viewer_participant_id in participants,
            "current_match": self._public_match(current, participants) if current else None,
            "active_matches": [self._public_match(match, participants) for match in active],
            "matches": [self._public_match(match, participants) for match in self.matches],
            "standings": self.standings(),
            "league_table": self.league_table(),
            "finished_at": isoformat(self.finished_at),
        }

    @staticmethod
    def _public_match(match: TournamentMatch, participants: dict[str, str]) -> dict[str, Any]:
        payload = match.to_dict()
        payload.update({
            "player1_name": participants.get(match.player1_id, "不明"),
            "player2_name": participants.get(match.player2_id, "不明"),
            "winner_name": participants.get(match.winner_id) if match.winner_id else None,
        })
        return payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "format_key": self.format_key,
            "pairing_mode": self.pairing_mode,
            "registration_closes_at": isoformat(self.registration_closes_at or self.starts_at),
            "title": self.title,
            "room_id": self.room_id,
            "rule_key": self.rule_key,
            "registration_opens_at": isoformat(self.registration_opens_at),
            "starts_at": isoformat(self.starts_at),
            "max_participants": self.max_participants,
            "status": self.status,
            "participants": [participant.to_dict() for participant in self.participants.values()],
            "matches": [match.to_dict() for match in self.matches],
            "current_match_id": self.current_match_id,
            "active_match_ids": list(self.active_match_ids),
            "created_at": isoformat(self.created_at),
            "finished_at": isoformat(self.finished_at),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TournamentRun":
        participants = [TournamentParticipant.from_dict(item) for item in value.get("participants", [])]
        return cls(
            run_id=str(value["run_id"]),
            format_key=str(value["format_key"]),
            title=str(value["title"]),
            room_id=str(value["room_id"]),
            rule_key=str(value["rule_key"]),
            registration_opens_at=parse_datetime(value["registration_opens_at"]),
            starts_at=parse_datetime(value["starts_at"]),
            pairing_mode=str(value.get("pairing_mode", "rounds")),
            registration_closes_at=parse_datetime(value.get("registration_closes_at") or value["starts_at"]),
            max_participants=int(value.get("max_participants", 10)),
            status=str(value.get("status", "scheduled")),
            participants={participant.participant_id: participant for participant in participants},
            matches=[TournamentMatch.from_dict(item) for item in value.get("matches", [])],
            current_match_id=value.get("current_match_id"),
            active_match_ids=[str(item) for item in value.get("active_match_ids", [])],
            created_at=parse_datetime(value["created_at"]),
            finished_at=parse_datetime(value["finished_at"]) if value.get("finished_at") else None,
        )

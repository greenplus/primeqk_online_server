from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
import hashlib
import secrets
import uuid


TERMINAL_MATCH_STATUSES = {"completed", "skipped"}
ACTIVE_RUN_STATUSES = {"scheduled", "registration", "running"}


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

    def to_dict(self, *, public: bool = False) -> dict[str, Any]:
        result = {
            "participant_id": self.participant_id,
            "display_name": self.display_name,
            "registered_at": isoformat(self.registered_at),
            "withdrawn": self.withdrawn,
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
    created_at: datetime = field(default_factory=utc_now)
    finished_at: Optional[datetime] = None

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
        now: Optional[datetime] = None,
    ) -> "TournamentRun":
        opens_at = parse_datetime(registration_opens_at)
        begins_at = parse_datetime(starts_at)
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
        )

    @property
    def active_participants(self) -> list[TournamentParticipant]:
        return [participant for participant in self.participants.values() if not participant.withdrawn]

    @property
    def current_match(self) -> Optional[TournamentMatch]:
        return next((match for match in self.matches if match.match_id == self.current_match_id), None)

    def participant_for_token(self, token: str) -> Optional[TournamentParticipant]:
        digest = hash_resume_token(token)
        return next(
            (participant for participant in self.participants.values() if participant.token_hash == digest),
            None,
        )

    def register(self, display_name: str, *, now: Optional[datetime] = None) -> tuple[TournamentParticipant, str]:
        self.advance_clock(now=now)
        if self.status != "registration":
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
        )
        self.participants[participant.participant_id] = participant
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
            if len(participant_ids) < 2:
                self.status = "cancelled"
                self.finished_at = current
            else:
                self.matches = round_robin_matches(participant_ids)
                self.status = "running"
        return self.status if self.status != old_status else None

    def next_pending_match(self) -> Optional[TournamentMatch]:
        return next((match for match in self.matches if match.status == "pending"), None)

    def start_next_match(self, *, now: Optional[datetime] = None) -> Optional[TournamentMatch]:
        if self.status != "running" or self.current_match_id is not None:
            return None
        match = self.next_pending_match()
        if match is None:
            self.finish_if_complete(now=now)
            return None
        match.status = "playing"
        match.started_at = now or utc_now()
        self.current_match_id = match.match_id
        return match

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
        if self.current_match_id == match.match_id:
            self.current_match_id = None
        self.finish_if_complete(now=now)
        return match

    def finish_if_complete(self, *, now: Optional[datetime] = None) -> bool:
        if self.matches and all(match.status in TERMINAL_MATCH_STATUSES for match in self.matches):
            self.status = "finished"
            self.current_match_id = None
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
            key=lambda row: (-row["points"], -row["wins"], row["display_name"].casefold()),
        )
        previous_key = None
        previous_rank = 0
        for index, row in enumerate(ordered, start=1):
            key = (row["points"], row["wins"])
            if key != previous_key:
                previous_rank = index
                previous_key = key
            row["rank"] = previous_rank
        return ordered

    def public_payload(self, *, viewer_participant_id: Optional[str] = None) -> dict[str, Any]:
        participants = {
            participant.participant_id: participant.display_name
            for participant in self.active_participants
        }
        current = self.current_match
        return {
            "run_id": self.run_id,
            "format_key": self.format_key,
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
            "matches": [self._public_match(match, participants) for match in self.matches],
            "standings": self.standings(),
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
            max_participants=int(value.get("max_participants", 10)),
            status=str(value.get("status", "scheduled")),
            participants={participant.participant_id: participant for participant in participants},
            matches=[TournamentMatch.from_dict(item) for item in value.get("matches", [])],
            current_match_id=value.get("current_match_id"),
            created_at=parse_datetime(value["created_at"]),
            finished_at=parse_datetime(value["finished_at"]) if value.get("finished_at") else None,
        )

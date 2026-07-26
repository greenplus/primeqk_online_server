from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from cpu_player import evaluate_gold_plan


MAX_RECOMMENDATION_CANDIDATES_PER_COUNT = 24
MAX_RECOMMENDATION_RALLY_PLAYS = 3
INFINITY_STRENGTH = 10**100

TIER_ORDER = {
    "finish": 0,
    "remaining_finish": 1,
    "next_finish": 2,
    "gold_route": 3,
    "preserve_trump": 4,
    "ordinary": 5,
    "trump_or_special": 6,
}


@dataclass(frozen=True)
class AssistRecord:
    candidate: dict
    mask: int
    strength: int
    visible_count: int
    fingerprint: tuple


def candidate_consumed_cards(candidate: dict) -> list[dict]:
    cards = list(candidate.get("cards") or [])
    composite_cards = list((candidate.get("composite") or {}).get("cards") or [])
    by_id = {
        str(card.get("card_id")): card
        for card in cards + composite_cards
        if card.get("card_id") is not None
    }
    return list(by_id.values())


def candidate_is_special(candidate: dict) -> bool:
    return bool(candidate.get("special_effect"))


def candidate_is_cut(candidate: dict) -> bool:
    return candidate.get("special_effect") in ("infinity", "cut")


def candidate_uses_joker(candidate: dict) -> bool:
    return any(
        card.get("is_joker") or card.get("suit") == "X"
        for card in candidate_consumed_cards(candidate)
    )


def candidate_strength(candidate: dict, reverse_order: bool) -> int:
    if candidate.get("special_effect") == "infinity":
        raw = INFINITY_STRENGTH
    else:
        try:
            raw = int(candidate.get("number", 0))
        except (TypeError, ValueError):
            raw = 0
    return -raw if reverse_order else raw


def candidate_fingerprint(candidate: dict) -> tuple:
    return (
        candidate.get("kind"),
        candidate.get("number"),
        candidate.get("special_effect"),
        tuple(
            sorted(
                str(card.get("card_id"))
                for card in candidate_consumed_cards(candidate)
            )
        ),
        candidate.get("visible_text", ""),
    )


def rank_recommended_assist_candidates(
    candidates: list[dict],
    source_cards: list[dict],
    reverse_order: bool,
) -> list[dict]:
    card_bits = {
        str(card.get("card_id")): 1 << index
        for index, card in enumerate(source_cards)
    }
    full_mask = (1 << len(card_bits)) - 1
    records = _build_records(candidates, card_bits, reverse_order)
    legal_records = [
        record
        for record in records
        if record.candidate.get("_assist_legal", True)
    ]
    if not legal_records:
        return []

    records_by_mask: dict[int, list[AssistRecord]] = {}
    for record in records:
        records_by_mask.setdefault(record.mask, []).append(record)
    for same_mask in records_by_mask.values():
        same_mask.sort(key=_finish_choice_key)

    cut_records = [
        record
        for record in records
        if candidate_is_cut(record.candidate)
    ]
    cut_57_masks = [
        record.mask
        for record in cut_records
        if record.candidate.get("special_effect") == "cut"
    ]
    ordinary_records = [
        record
        for record in records
        if not candidate_is_special(record.candidate)
    ]
    ordinary_by_count = _group_by_count(ordinary_records)
    legal_ordinary_by_count = _group_by_count(
        record
        for record in legal_records
        if not candidate_is_special(record.candidate)
    )

    metadata: dict[tuple, dict] = {}
    for record in legal_records:
        candidate = record.candidate
        if candidate.get("finishes_hand"):
            metadata[record.fingerprint] = _metadata("finish")
        elif candidate.get("finishes_remaining"):
            metadata[record.fingerprint] = _metadata("remaining_finish")
        elif candidate_is_cut(candidate) and _has_direct_finish(
            full_mask ^ record.mask,
            records_by_mask,
        ):
            candidate["next_finish"] = True
            metadata[record.fingerprint] = _metadata("next_finish")

    selected_records: list[AssistRecord] = []
    for count, legal_group in sorted(legal_ordinary_by_count.items()):
        all_group = ordinary_by_count.get(count, [])
        trump = _strongest_available(all_group, full_mask)
        for opener in legal_group:
            if opener.fingerprint in metadata:
                continue
            route = _best_gold_route(
                opener,
                all_group,
                records_by_mask,
                cut_records,
                full_mask,
                reverse_order,
            )
            if route is not None:
                metadata[opener.fingerprint] = {
                    **_metadata("gold_route"),
                    "gold_score": route["gold_score"],
                    "route_length": route["route_length"],
                }
                continue

            stronger_followups = [
                follower
                for follower in all_group
                if follower.strength > opener.strength
                and not follower.mask & opener.mask
            ]
            preserves_trump = bool(
                trump
                and trump.fingerprint != opener.fingerprint
                and not trump.mask & opener.mask
            )
            preserves_57 = not any(opener.mask & mask for mask in cut_57_masks)
            if stronger_followups and preserves_trump:
                metadata[opener.fingerprint] = {
                    **_metadata("preserve_trump"),
                    "stronger_followups": len(stronger_followups),
                    "preserves_57": preserves_57,
                }
            elif trump and trump.fingerprint == opener.fingerprint:
                metadata[opener.fingerprint] = _metadata("trump_or_special")
            else:
                metadata[opener.fingerprint] = {
                    **_metadata("ordinary"),
                    "stronger_followups": len(stronger_followups),
                    "preserves_57": preserves_57,
                }

        selected_records.append(
            min(
                legal_group,
                key=lambda record: _candidate_selection_key(
                    record,
                    metadata[record.fingerprint],
                ),
            )
        )

    special_records = [
        record
        for record in legal_records
        if candidate_is_special(record.candidate)
    ]
    for record in special_records:
        metadata.setdefault(
            record.fingerprint,
            _metadata("trump_or_special"),
        )

    selected_fingerprints = {
        record.fingerprint
        for record in selected_records
    }
    output_records = selected_records + [
        record
        for record in special_records
        if record.fingerprint not in selected_fingerprints
    ]
    output_records.sort(
        key=lambda record: _output_order_key(
            record,
            metadata[record.fingerprint],
        )
    )

    output = []
    for record in output_records:
        candidate = record.candidate
        candidate.pop("_assist_legal", None)
        candidate["recommendation_tier"] = metadata[record.fingerprint]["tier"]
        if not candidate.get("next_finish"):
            candidate.pop("next_finish", None)
        output.append(candidate)
    return output


def _build_records(
    candidates: Iterable[dict],
    card_bits: dict[str, int],
    reverse_order: bool,
) -> list[AssistRecord]:
    records = []
    seen = set()
    for candidate in candidates:
        mask = 0
        missing_card = False
        for card in candidate_consumed_cards(candidate):
            bit = card_bits.get(str(card.get("card_id")))
            if bit is None:
                missing_card = True
                break
            mask |= bit
        if missing_card or mask == 0:
            continue
        fingerprint = candidate_fingerprint(candidate)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        records.append(
            AssistRecord(
                candidate=candidate,
                mask=mask,
                strength=candidate_strength(candidate, reverse_order),
                visible_count=len(candidate.get("cards") or []),
                fingerprint=fingerprint,
            )
        )
    return records


def _group_by_count(records: Iterable[AssistRecord]) -> dict[int, list[AssistRecord]]:
    grouped: dict[int, list[AssistRecord]] = {}
    for record in records:
        grouped.setdefault(record.visible_count, []).append(record)
    for group in grouped.values():
        group.sort(key=_record_strength_key)
    return grouped


def _bounded_group(records: list[AssistRecord]) -> list[AssistRecord]:
    if len(records) <= MAX_RECOMMENDATION_CANDIDATES_PER_COUNT:
        return records
    side = MAX_RECOMMENDATION_CANDIDATES_PER_COUNT // 2
    combined = records[:side] + records[-side:]
    seen = set()
    output = []
    for record in combined:
        if record.fingerprint in seen:
            continue
        seen.add(record.fingerprint)
        output.append(record)
    return output


def _strongest_available(
    records: list[AssistRecord],
    available_mask: int,
) -> AssistRecord | None:
    available = [
        record
        for record in records
        if record.mask & available_mask == record.mask
    ]
    return max(available, key=_record_strength_key) if available else None


def _best_gold_route(
    opener: AssistRecord,
    all_group: list[AssistRecord],
    records_by_mask: dict[int, list[AssistRecord]],
    cut_records: list[AssistRecord],
    full_mask: int,
    reverse_order: bool,
) -> dict | None:
    bounded = _bounded_group(all_group)
    best = None

    def visit(sequence: list[AssistRecord], consumed_mask: int) -> None:
        nonlocal best
        current = sequence[-1]
        before_current = full_mask ^ (consumed_mask ^ current.mask)
        trump = _strongest_available(all_group, before_current)
        if trump and trump.strength == current.strength:
            tail = _finish_tail(
                full_mask ^ consumed_mask,
                records_by_mask,
                cut_records,
            )
            if tail is not None:
                evaluated = _evaluate_route(sequence, tail, reverse_order)
                if best is None or _route_choice_key(evaluated, sequence) < _route_choice_key(
                    best,
                    best["sequence"],
                ):
                    best = {
                        **evaluated,
                        "sequence": sequence[:],
                    }

        if len(sequence) >= MAX_RECOMMENDATION_RALLY_PLAYS:
            return
        for follower in bounded:
            if follower.mask & consumed_mask:
                continue
            if follower.strength <= current.strength:
                continue
            visit(sequence + [follower], consumed_mask | follower.mask)

    visit([opener], opener.mask)
    return best


def _finish_tail(
    remaining_mask: int,
    records_by_mask: dict[int, list[AssistRecord]],
    cut_records: list[AssistRecord],
) -> list[AssistRecord] | None:
    if remaining_mask == 0:
        return None
    direct = records_by_mask.get(remaining_mask)
    if direct:
        return [direct[0]]
    for cut in sorted(cut_records, key=_finish_choice_key):
        if cut.mask & remaining_mask != cut.mask:
            continue
        after_cut = remaining_mask ^ cut.mask
        finish = records_by_mask.get(after_cut)
        if finish:
            return [cut, finish[0]]
    return None


def _has_direct_finish(
    remaining_mask: int,
    records_by_mask: dict[int, list[AssistRecord]],
) -> bool:
    return remaining_mask != 0 and bool(records_by_mask.get(remaining_mask))


def _evaluate_route(
    sequence: list[AssistRecord],
    tail: list[AssistRecord],
    reverse_order: bool,
) -> dict:
    count = sequence[0].visible_count
    steps = [
        _gold_step(record, f"rally-{count}")
        for record in sequence
    ]
    for index, record in enumerate(tail):
        role = "cut" if index < len(tail) - 1 else "finish"
        steps.append(_gold_step(record, role))
    evaluation = evaluate_gold_plan({
        "steps": steps,
        "remaining": [],
        "completed": True,
        "rally_count": count,
    })
    gold_score = float(evaluation.get("score", 0.0))
    if reverse_order:
        gold_score = 100.0 - gold_score
    return {
        "gold_score": round(gold_score, 4),
        "route_length": len(steps),
    }


def _gold_step(record: AssistRecord, role: str) -> dict:
    candidate = record.candidate
    number = "X" if candidate.get("special_effect") == "infinity" else candidate.get("number")
    return {
        "kind": candidate.get("kind"),
        "number": number,
        "cards": list(candidate.get("cards") or []),
        "consume_cards": list((candidate.get("composite") or {}).get("cards") or []),
        "role": role,
    }


def _metadata(tier: str) -> dict:
    return {
        "tier": tier,
        "gold_score": 0.0,
        "route_length": 99,
        "stronger_followups": 0,
        "preserves_57": False,
    }


def _candidate_selection_key(record: AssistRecord, metadata: dict) -> tuple:
    tier = metadata["tier"]
    return (
        TIER_ORDER[tier],
        -float(metadata.get("gold_score", 0.0)),
        int(metadata.get("route_length", 99)),
        1 if candidate_uses_joker(record.candidate) else 0,
        0 if metadata.get("preserves_57") else 1,
        -int(metadata.get("stronger_followups", 0)),
        record.strength,
        record.fingerprint,
    )


def _output_order_key(record: AssistRecord, metadata: dict) -> tuple:
    return (
        TIER_ORDER[metadata["tier"]],
        -float(metadata.get("gold_score", 0.0)),
        int(metadata.get("route_length", 99)),
        -record.visible_count,
        record.strength,
        record.fingerprint,
    )


def _route_choice_key(route: dict, sequence: list[AssistRecord]) -> tuple:
    opener = sequence[0]
    return (
        -float(route.get("gold_score", 0.0)),
        int(route.get("route_length", 99)),
        opener.strength,
        1 if candidate_uses_joker(opener.candidate) else 0,
        opener.fingerprint,
    )


def _record_strength_key(record: AssistRecord) -> tuple:
    return (
        record.strength,
        -len(candidate_consumed_cards(record.candidate)),
        record.fingerprint,
    )


def _finish_choice_key(record: AssistRecord) -> tuple:
    return (
        1 if candidate_uses_joker(record.candidate) else 0,
        -len(candidate_consumed_cards(record.candidate)),
        record.strength,
        record.fingerprint,
    )

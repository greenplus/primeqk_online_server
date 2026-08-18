from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from itertools import permutations, product
import json
from pathlib import Path
import secrets
import time
from typing import Callable, Iterable, List, Optional

from registered_primes import (
    registered_prime_template_index,
    registered_prime_templates_for_hand,
    registered_value_encodings,
)
from rules import PrimeRule


Card = dict
NumberValidator = Callable[[int, "CpuPlayer", object], bool]
CpuActionSelector = Callable[["CpuPlayer", object, Optional[NumberValidator]], "CpuAction"]
GOLD_PLAN_MAX_LAST_CANDIDATES = 30
GOLD_PLAN_MAX_BRANCH_CANDIDATES = 24
GOLD_PLAN_MAX_RESULTS_PER_COUNT = 3
GOLD_PLAN_MAX_ALTERNATIVES = 8
GOLD_PLAN_MAX_RALLY_PREFIX_STEPS = 6
GOLD_PLAN_MAX_RALLY_STEPS = GOLD_PLAN_MAX_RALLY_PREFIX_STEPS + 1
SERVER_DIR = Path(__file__).resolve().parent
GOLD_PLAN_EVALUATION_JSON = SERVER_DIR / "data" / "cpu" / "gold_plan_evaluation.json"
SILVER_PLAN_MAX_RALLY_STEPS = 3
SILVER_PLAN_MAX_STEPS = SILVER_PLAN_MAX_RALLY_STEPS + 2
SILVER_RALLY_COUNTS = (1, 2, 3, 4)
SILVER_EVEN_RANKS = {2, 4, 6, 8, 10, 12}
SILVER_EVEN_RELIEF_MAX_RATIO_INCREASE = 0.0
CPU_PLANNER_DEFAULT_BUDGET_MS = 250
COMPOSITE_PRACTICE_MAX_PLAN_STEPS = 5
COMPOSITE_PRACTICE_BRANCH_CAP = 48
COMPOSITE_PRACTICE_ALL_OUT_ATTEMPTS = 96
PLATINUM_PLAN_MAX_STEPS = 5
PLATINUM_MIN_TRUMP_STRENGTH = 80.0
PLATINUM_RELAXED_TRUMP_STRENGTH = 60.0
PLATINUM_MAX_KNOWLEDGE_CARDS = 14
PLATINUM_OPENING_MULTI_PLAY_MIN_CARDS = 9
PLATINUM_COMPRESSION_MIN_HAND_SIZE = 18
PLATINUM_FORCED_COMPRESSION_HAND_SIZE = 26
PLATINUM_COMPRESSION_FOLLOWUP_CHECK_CAP = 12
PLATINUM_ALL_OUT_RESUME_MIN_OPPONENT_HAND_SIZE = 5
PLATINUM_POST_ALL_OUT_MAX_OPPONENT_PREPLAY_HAND_SIZE = 12
PLATINUM_INTERFERENCE_BORDER = 42
PLATINUM_OPPONENT_HAND_SCORES = {
    0: 100,
    1: 100,
    2: 89,
    3: 79,
    4: 65,
    5: 46,
    6: 29,
    7: 16,
    8: 8,
    9: 4,
    10: 2,
    11: 1,
}
PLATINUM_ABSOLUTE_ALWAYS = frozenset({
    "kk", "kkk", "kkkq", "kkqkj", "kkkqqj", "kkkqqqj", "kkkqqjj",
    "kkkqjtqj", "kkkqqttqj", "kkkqtttjj", "kkkqtjqjj",
})
PLATINUM_ABSOLUTE_KX4 = frozenset({
    "kkq", "kkqt", "kkjq", "kkqtj", "kkkttj", "kkqqtj", "kkktttj",
    "kkqjqtj", "kkkqtqtj", "kkqqtjtjj",
})
PLATINUM_DUAL_WIELD_TEMPLATES = {
    "125kjqj": ("kjqj", 99.0), "614kjqj": ("kjqj", 99.0),
    "956kjqj": ("kjqj", 99.0), "278kjqj": ("kjqj", 99.0),
    "638kjqj": ("kjqj", 99.0), "758kjqj": ("kjqj", 99.0),
    "263kjqj": ("kjqj", 99.0), "443kjqj": ("kjqj", 99.0),
    "451ktqj": ("ktqj", 95.0), "589ktqj": ("ktqj", 95.0),
    "283ktqj": ("ktqj", 95.0), "547ktqj": ("ktqj", 95.0),
    "982ktqj": ("ktqj", 95.0), "883ktqj": ("ktqj", 95.0),
    "69qtjk1": ("ktqj", 95.0), "69qtjk3": ("ktqj", 95.0),
    "69qtjk7": ("ktqj", 95.0), "69qtjk9": ("ktqj", 95.0),
    "98726kqk": ("kqk", 95.0), "98726kjj": ("kjj", 95.0),
    "96251kqk": ("kqk", 95.0), "96251kjj": ("kjj", 95.0),
    "53648kqk": ("kqk", 95.0), "53648kjj": ("kjj", 95.0),
    "26348kqk": ("kqk", 95.0), "26348kjj": ("kjj", 95.0),
    "86861kqk": ("kqk", 95.0), "86861kjj": ("kjj", 95.0),
}
PLATINUM_TOKEN_RANKS = {"t": 10, "j": 11, "q": 12, "k": 13}
PLATINUM_SMALL_TRUMP_TOKENS = frozenset({
    "kk", "kkj", "kqk", "kjj", "kjqj", "kjtk", "ktqj", "qqqj", "qk",
    "kq", "kj", "kt",
})


def gold_branch_candidate_cap(cpu: "CpuPlayer") -> int:
    return 12 if getattr(cpu, "cpu_key", "") == "platinum_planner" else GOLD_PLAN_MAX_BRANCH_CANDIDATES


def gold_last_candidate_cap(cpu: "CpuPlayer") -> int:
    return 16 if getattr(cpu, "cpu_key", "") == "platinum_planner" else GOLD_PLAN_MAX_LAST_CANDIDATES
COMPOSITE_PRACTICE_RANK_WEIGHTS = {
    0: 100,  # X
    2: 60,
    1: 25,
    3: 25,
    5: 25,
    7: 10,
    9: 10,
    10: 10,
    11: 10,
    13: 10,
}
SILVER_PLAN_SEARCH_RESULT_CAP = GOLD_PLAN_MAX_RESULTS_PER_COUNT * 2
FISH_EXTRA_343_PRIME_COUNT = 500
FISH_343_TOKEN_VALUES = {
    "t": "10",
    "j": "11",
    "q": "12",
    "k": "13",
    "y": "343",
}


@dataclass(frozen=True)
class CpuAction:
    kind: str
    payload: dict = field(default_factory=dict)


class CpuSearchDeadline(RuntimeError):
    pass


@dataclass(frozen=True)
class CpuKnowledgeSpec:
    source: str = "none"  # "none" | "sample" | "gold" | "sample_key" | "inline"
    load_timing: str = "never"  # "never" | "registration" | "always"
    sample_key: str = ""
    prime_text: str = ""
    composite_text: str = ""


@dataclass(frozen=True)
class CpuProfile:
    key: str
    label: str
    description: str
    rule_keys: tuple[str, ...] = ()
    prime_rules: tuple[PrimeRule, ...] = ()
    knowledge: CpuKnowledgeSpec = field(default_factory=CpuKnowledgeSpec)
    action_selector: Optional[CpuActionSelector] = None

    def supports_rule(self, rule) -> bool:
        if self.rule_keys and getattr(rule, "key", None) not in self.rule_keys:
            return False
        if self.prime_rules and getattr(rule, "prime_rule", None) not in self.prime_rules:
            return False
        return True

    def to_payload(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "description": self.description,
        }


class CpuPlayer:
    def __init__(self, name: str = "CPU", player_id: Optional[str] = None, cpu_key: str = "basic"):
        self.id = player_id or f"cpu-{secrets.token_hex(8)}"
        self.name = name
        self.ws = self
        self.room = None
        self.status = "watching"
        self.hand: List[Card] = []
        self.is_cpu = True
        self.cpu_key = cpu_key
        self.registered_primes: set[int] = set()
        self.registered_composites: set[int] = set()
        self.registered_composite_entries = ()
        self.gold_active_plan: Optional[dict] = None
        self.gold_plan_step_index = 0
        self.silver_active_plan: Optional[dict] = None
        self.silver_plan_step_index = 0
        self.decision_time_budget_ms = CPU_PLANNER_DEFAULT_BUDGET_MS
        self.decision_deadline: Optional[float] = None
        self.last_decision_timed_out = False
        self.small_finish_index = registered_prime_template_index((), max_cards=3)
        self.prime_template_index = registered_prime_template_index(
            (), max_cards=PLATINUM_MAX_KNOWLEDGE_CARDS
        )
        self.prime_template_index_values = ()
        self.platinum_opening_phase = True
        self.platinum_all_out_attempts = 0
        self.platinum_initial_hand_size = 11
        self.platinum_last_strategy_score = 0.0
        self.platinum_last_interference_score = 0
        self.platinum_relaxed_opponent_min_hand_count: Optional[int] = None
        self.platinum_all_out_suppressed_opponent_min_hand_count: Optional[int] = None
        self.platinum_current_min_trump_strength = PLATINUM_MIN_TRUMP_STRENGTH
        self.rng = secrets.SystemRandom()

    async def send_json(self, message: dict):
        return None

    async def send_hand_update(self):
        return None

    def sort_hand(self):
        self.hand.sort(key=lambda card: card.get("rank", 0))

    def add_card(self, card: Card):
        self.hand.append(card)
        self.sort_hand()

    def remove_card(self, card: Card) -> bool:
        if card in self.hand:
            self.hand.remove(card)
            return True
        return False

    def has_cards(self, cards: List[Card]) -> bool:
        temp = self.hand[:]
        for card in cards:
            if card in temp:
                temp.remove(card)
            else:
                return False
        return True

    def remove_cards(self, cards: List[Card]) -> bool:
        if not self.has_cards(cards):
            return False
        for card in cards:
            self.remove_card(card)
        return True

    def clear_hand(self):
        self.hand = []

    def replace_registered_primes(self, values: set[int]) -> None:
        self.registered_primes = set(values)
        sorted_values = tuple(sorted(self.registered_primes))
        self.small_finish_index = registered_prime_template_index(
            sorted_values,
            max_cards=3,
        )
        self.prime_template_index = registered_prime_template_index(
            sorted_values,
            max_cards=PLATINUM_MAX_KNOWLEDGE_CARDS,
        )
        self.prime_template_index_values = sorted_values

    def can_use_registered_prime(self, n: int) -> bool:
        return n in self.registered_primes

    def replace_registered_composites(self, values: set[int], entries=()) -> None:
        self.registered_composites = set(values)
        self.registered_composite_entries = tuple(entries)

    def can_use_registered_composite(self, n: int) -> bool:
        return n in self.registered_composites


def is_cpu_player(player) -> bool:
    return bool(getattr(player, "is_cpu", False))


def reset_cpu_game_state(cpu: CpuPlayer, initial_hand_size: Optional[int] = None) -> None:
    """Reset per-game planner state while preserving the CPU's learned knowledge."""
    cpu.gold_active_plan = None
    cpu.gold_plan_step_index = 0
    cpu.silver_active_plan = None
    cpu.silver_plan_step_index = 0
    cpu.decision_deadline = None
    cpu.last_decision_timed_out = False
    cpu.platinum_opening_phase = True
    cpu.platinum_all_out_attempts = 0
    cpu.platinum_initial_hand_size = int(
        len(cpu.hand) if initial_hand_size is None else initial_hand_size
    )
    cpu.platinum_last_strategy_score = 0.0
    cpu.platinum_last_interference_score = 0
    cpu.platinum_relaxed_opponent_min_hand_count = None
    cpu.platinum_all_out_suppressed_opponent_min_hand_count = None
    cpu.platinum_current_min_trump_strength = PLATINUM_MIN_TRUMP_STRENGTH


def get_cpu_profile(cpu_key: str) -> Optional[CpuProfile]:
    return CPU_PROFILES.get(cpu_key)


def available_cpu_profiles_for_rule(rule) -> List[CpuProfile]:
    allowed_keys = tuple(getattr(rule, "cpu_profile_keys", ()) or ())
    return [
        profile
        for profile in CPU_PROFILES.values()
        if profile.supports_rule(rule)
        and (not allowed_keys or profile.key in allowed_keys)
    ]


def available_cpu_profile_payloads(rule) -> List[dict]:
    return [profile.to_payload() for profile in available_cpu_profiles_for_rule(rule)]


def choose_profile_cpu_action(
    cpu: CpuPlayer,
    room,
    validator: Optional[NumberValidator] = None,
) -> CpuAction:
    profile = get_cpu_profile(getattr(cpu, "cpu_key", "basic"))
    budget_ms = max(1, int(getattr(
        cpu,
        "decision_time_budget_ms",
        CPU_PLANNER_DEFAULT_BUDGET_MS,
    )))
    if profile and profile.key == "platinum_planner":
        budget_ms = max(budget_ms, 1000)
    cpu.decision_deadline = time.perf_counter() + budget_ms / 1000
    cpu.last_decision_timed_out = False
    try:
        if profile and profile.action_selector:
            return profile.action_selector(cpu, room, validator)
        return choose_cpu_action(cpu, room, validator=validator)
    except CpuSearchDeadline:
        cpu.last_decision_timed_out = True
        clear_gold_active_plan(cpu)
        clear_silver_active_plan(cpu)
        if profile and profile.key == "composite_practice":
            cpu.decision_deadline = None
            return choose_composite_practice_emergency_action(cpu, room)
        if profile and profile.key == "platinum_planner":
            cpu.decision_deadline = None
            if not getattr(room, "has_drawn", False) and getattr(room, "deck", []):
                return CpuAction("draw")
            return choose_platinum_timeout_action(cpu, room, gold_knowledge_number_validator)
        return choose_cpu_action(cpu, room, validator=validator, max_cards=3)
    finally:
        cpu.decision_deadline = None


def check_cpu_search_deadline(cpu: CpuPlayer) -> None:
    deadline = getattr(cpu, "decision_deadline", None)
    if deadline is not None and time.perf_counter() >= deadline:
        raise CpuSearchDeadline


def choose_cpu_action(
    cpu: CpuPlayer,
    room,
    validator: Optional[NumberValidator] = None,
    max_cards: int = 3,
) -> CpuAction:
    candidate = choose_prime_play(cpu, room, validator=validator, max_cards=max_cards)
    if candidate is not None:
        return CpuAction("play_prime", candidate)

    if not getattr(room, "has_drawn", False) and getattr(room, "deck", []):
        return CpuAction("draw")

    return CpuAction("pass")


def choose_composite_practice_cpu_action(
    cpu: CpuPlayer,
    room,
    validator: Optional[NumberValidator] = None,
) -> CpuAction:
    """Choose from composite plays, reserving a known <=3-card prime for the finish."""
    validator = gold_knowledge_number_validator
    field = getattr(room, "field", []) or []

    if not field:
        plan = build_composite_practice_plan(
            cpu,
            room_without_field(room),
            max_steps=COMPOSITE_PRACTICE_MAX_PLAN_STEPS,
            validator=validator,
        )
        if plan is not None and plan.get("steps"):
            return candidate_to_action(plan["steps"][0])

    if field:
        composite_finishes = direct_composite_finish_candidates(cpu, room)
        if composite_finishes:
            return candidate_to_action(max(
                composite_finishes,
                key=lambda candidate: candidate_strength(candidate, room),
            ))
        finish = choose_composite_practice_prime_finish(cpu, room, validator)
        if finish is not None:
            return candidate_to_action(finish)

    counts = (len(field),) if field else range(1, min(9, len(cpu.hand)) + 1)
    candidates = [
        candidate
        for candidate in knowledge_composite_candidates(cpu, room, counts)
        if candidate_is_playable(candidate, cpu, room)
    ]
    if candidates:
        best = max(
            dedupe_candidates(candidates),
            key=lambda candidate: composite_practice_fallback_score(cpu, room, candidate, validator),
        )
        return candidate_to_action(best)

    if field and composite_practice_future_plan(cpu, room, validator) is not None:
        return CpuAction("pass")
    if not getattr(room, "has_drawn", False) and getattr(room, "deck", []):
        return CpuAction("draw")
    all_out = build_composite_practice_all_out_payload(cpu, room)
    if all_out is not None:
        return CpuAction("play_composite", all_out)
    return CpuAction("pass")


def choose_composite_practice_emergency_action(cpu: CpuPlayer, room) -> CpuAction:
    composite_finishes = direct_composite_finish_candidates(cpu, room)
    if composite_finishes:
        return candidate_to_action(max(
            composite_finishes,
            key=lambda candidate: candidate_strength(candidate, room),
        ))
    finish = choose_composite_practice_prime_finish(cpu, room, gold_knowledge_number_validator)
    if finish is not None:
        return candidate_to_action(finish)
    field = getattr(room, "field", []) or []
    counts = (len(field),) if field else range(1, min(9, len(cpu.hand)) + 1)
    candidates = [
        candidate
        for candidate in knowledge_composite_candidates(cpu, room, counts)
        if candidate_is_playable(candidate, cpu, room)
    ]
    if candidates:
        return candidate_to_action(min(
            candidates,
            key=lambda candidate: candidate_strength(candidate, room),
        ))
    if field and composite_practice_future_plan(
        cpu,
        room,
        gold_knowledge_number_validator,
    ) is not None:
        return CpuAction("pass")
    if not getattr(room, "has_drawn", False) and getattr(room, "deck", []):
        return CpuAction("draw")
    all_out = build_composite_practice_all_out_payload(cpu, room)
    if all_out is not None:
        return CpuAction("play_composite", all_out)
    return CpuAction("pass")


def composite_practice_future_plan(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
) -> Optional[dict]:
    plan = build_composite_practice_plan(
        cpu,
        room_without_field(room),
        max_steps=COMPOSITE_PRACTICE_MAX_PLAN_STEPS,
        validator=validator,
    )
    return plan if plan is not None and plan.get("steps") else None


def build_composite_practice_all_out_payload(
    cpu: CpuPlayer,
    room,
    rng=None,
    require_invalid: bool = False,
) -> Optional[dict]:
    """Build a last-resort random composite attempt that uses the whole hand.

    The visible side still has to match and beat the field. Every other card is
    used exactly once in a random multiplication expression. This deliberately
    does not search for a correct equation; failure and its normal composite
    penalty are the intended hand-reshaping fallback.
    """
    hand = list(cpu.hand)
    if len(hand) < 2:
        return None

    field = getattr(room, "field", []) or []
    if field:
        visible_counts = (len(field),)
    else:
        visible_counts = tuple(range(1, min(9, len(hand) - 1) + 1))
    visible_counts = tuple(
        count for count in visible_counts
        if 1 <= count < len(hand)
    )
    if not visible_counts:
        return None

    rng = rng or secrets.SystemRandom()
    attempts = []
    for _ in range(COMPOSITE_PRACTICE_ALL_OUT_ATTEMPTS):
        cards = hand[:]
        rng.shuffle(cards)
        visible_count = rng.choice(visible_counts)
        visible_cards = cards[:visible_count]
        material_cards = cards[visible_count:]

        assigned_by_id = {}
        material_ids = {card.get("card_id") for card in material_cards}
        for card in cards:
            if not is_joker(card):
                continue
            choices = (2, 3, 5, 7, 11, 13) if card.get("card_id") in material_ids else tuple(range(1, 14))
            assigned_by_id[card.get("card_id")] = int(rng.choice(choices))

        visible_value = composite_practice_cards_number(visible_cards, assigned_by_id)
        if visible_value is None or not beats_field(visible_value, visible_count, room):
            continue

        chunks = random_composite_factor_chunks(material_cards, rng)
        factor_values = [composite_practice_cards_number(chunk, assigned_by_id) for chunk in chunks]
        if any(value is None or value < 2 for value in factor_values):
            # One concatenated chunk avoids a syntax-only failure where possible.
            chunks = [material_cards]
            factor_values = [composite_practice_cards_number(material_cards, assigned_by_id)]
        if any(value is None or value < 2 for value in factor_values):
            continue
        product = 1
        for value in factor_values:
            product *= int(value)
        if require_invalid and product == visible_value:
            continue

        payload = composite_practice_all_out_payload(
            visible_cards,
            chunks,
            assigned_by_id,
        )
        attempts.append(payload)

    return rng.choice(attempts) if attempts else None


def composite_practice_cards_number(cards: List[Card], assigned_by_id: dict) -> Optional[int]:
    if not cards:
        return None
    parts = []
    for card in cards:
        if is_joker(card):
            rank = assigned_by_id.get(card.get("card_id"))
        else:
            rank = card.get("rank")
        if rank is None:
            return None
        parts.append(str(rank))
    try:
        return int("".join(parts))
    except ValueError:
        return None


def random_composite_factor_chunks(material_cards: List[Card], rng) -> List[List[Card]]:
    if len(material_cards) <= 1:
        return [material_cards]
    cut_count = rng.randint(1, min(3, len(material_cards) - 1))
    cuts = set(rng.sample(range(1, len(material_cards)), cut_count))
    chunks = []
    start = 0
    for index in range(1, len(material_cards) + 1):
        if index in cuts or index == len(material_cards):
            chunks.append(material_cards[start:index])
            start = index
    return chunks


def composite_practice_all_out_payload(
    visible_cards: List[Card],
    chunks: List[List[Card]],
    assigned_by_id: dict,
) -> dict:
    material_cards = [card for chunk in chunks for card in chunk]
    tokens = []
    for chunk_index, chunk in enumerate(chunks):
        if chunk_index:
            tokens.append({"kind": "op", "op": "×"})
        tokens.extend(
            {"kind": "card", "card_id": card.get("card_id")}
            for card in chunk
        )
    return {
        "selected": {
            "cards": visible_cards,
            "assigned_numbers": [
                str(assigned_by_id[card.get("card_id")])
                for card in visible_cards
                if is_joker(card)
            ],
        },
        "consume": {"cards": material_cards},
        "composite": {
            "tokens": tokens,
            "assigned_numbers": [
                str(assigned_by_id[card.get("card_id")])
                for card in material_cards
                if is_joker(card)
            ],
        },
    }


def choose_composite_practice_prime_finish(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
) -> Optional[dict]:
    max_cards = int(getattr(room.rule, "normal_finish_max_hand_size", 3))
    if not 1 <= len(cpu.hand) <= max_cards:
        return None
    if len(cpu.hand) == 1 and is_joker(cpu.hand[0]) and len(getattr(room, "field", []) or []) <= 1:
        return {
            "kind": "prime",
            "number": "X",
            "cards": cpu.hand[:],
            "assigned_numbers": [],
        }
    candidates = direct_prime_finish_candidates(cpu, room, validator)
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: candidate_strength(candidate, room))


def build_composite_practice_plan(
    cpu: CpuPlayer,
    room,
    max_steps: int = COMPOSITE_PRACTICE_MAX_PLAN_STEPS,
    validator: Optional[NumberValidator] = None,
) -> Optional[dict]:
    """Find an idealized partition plan, preferring a composite final play."""
    validator = validator or gold_knowledge_number_validator
    seen_depth: dict[tuple[str, ...], int] = {}

    def visit(current: CpuPlayer, steps_left: int) -> Optional[dict]:
        check_cpu_search_deadline(current)
        signature = tuple(sorted(str(card.get("card_id")) for card in current.hand))
        if seen_depth.get(signature, -1) >= steps_left:
            return None
        seen_depth[signature] = steps_left

        prime_finish = choose_composite_practice_prime_finish(current, room, validator)
        best = (
            {"steps": [prime_finish], "finish_kind": "prime", "completed": True}
            if prime_finish is not None
            else None
        )
        if steps_left <= 0:
            return best

        direct_composites = direct_composite_finish_candidates(current, room)
        if direct_composites:
            finish = max(direct_composites, key=lambda item: candidate_strength(item, room))
            return {"steps": [finish], "finish_kind": "composite", "completed": True}

        candidates = knowledge_composite_candidates(
            current,
            room,
            range(1, min(9, len(current.hand)) + 1),
        )
        complete_composites = [
            candidate
            for candidate in candidates
            if len(candidate_consumed_cards(candidate)) == len(current.hand)
        ]
        if complete_composites:
            finish = max(complete_composites, key=lambda item: candidate_strength(item, room))
            composite_plan = {"steps": [finish], "finish_kind": "composite", "completed": True}
            if best is None or composite_practice_plan_score(composite_plan, room) > composite_practice_plan_score(best, room):
                best = composite_plan
        candidates = [
            candidate
            for candidate in dedupe_candidates(candidates)
            if candidate_is_playable(candidate, current, room)
            and len(candidate_consumed_cards(candidate)) < len(current.hand)
        ]
        candidates.sort(
            key=lambda candidate: (
                len(candidate_consumed_cards(candidate)),
                1 if int(candidate.get("number", 0)) == 57 else 0,
                candidate_strength(candidate, room),
            ),
            reverse=True,
        )
        for candidate in candidates[:COMPOSITE_PRACTICE_BRANCH_CAP]:
            remaining = remaining_cards(current.hand, candidate_consumed_cards(candidate))
            child = temporary_cpu_with_hand(current, remaining)
            tail = visit(child, steps_left - 1)
            if tail is None:
                continue
            plan = {
                "steps": [candidate] + tail["steps"],
                "finish_kind": tail["finish_kind"],
                "completed": True,
            }
            if best is None or composite_practice_plan_score(plan, room) > composite_practice_plan_score(best, room):
                best = plan
        return best

    return visit(cpu, max_steps)


def composite_practice_plan_score(plan: dict, room) -> tuple:
    steps = plan.get("steps", [])
    cut_count = sum(1 for step in steps if int(step.get("number", 0)) == 57)
    return (
        1 if plan.get("finish_kind") == "composite" else 0,
        -len(steps),
        cut_count,
        candidate_strength(steps[-1], room) if steps else -1,
    )


def composite_practice_hand_resource_score(cpu: CpuPlayer, room, hand: List[Card]) -> tuple:
    rank_score = sum(
        COMPOSITE_PRACTICE_RANK_WEIGHTS.get(0 if is_joker(card) else int(card.get("rank", 0)), 0)
        for card in hand
    )
    temp = temporary_cpu_with_hand(cpu, hand)
    strongest = strongest_candidates_by_count(
        knowledge_composite_candidates(temp, room_without_field(room), range(1, min(9, len(hand)) + 1)),
        room,
    )
    trump_score = tuple(
        candidate_strength(strongest[count], room) if count in strongest else -1
        for count in range(9, 0, -1)
    )
    return rank_score, trump_score


def composite_practice_fallback_score(
    cpu: CpuPlayer,
    room,
    candidate: dict,
    validator: NumberValidator,
) -> tuple:
    remaining = remaining_cards(cpu.hand, candidate_consumed_cards(candidate))
    child = temporary_cpu_with_hand(cpu, remaining)
    plan = build_composite_practice_plan(
        child,
        room_without_field(room),
        max_steps=max(0, COMPOSITE_PRACTICE_MAX_PLAN_STEPS - 1),
        validator=validator,
    )
    route_score = composite_practice_plan_score(plan, room) if plan else (-1, -999, 0, -1)
    resource_score = composite_practice_hand_resource_score(cpu, room, remaining)
    return (
        1 if plan else 0,
        route_score,
        resource_score,
        1 if int(candidate.get("number", 0)) == 57 else 0,
        len(candidate_consumed_cards(candidate)),
        -candidate_strength(candidate, room),
    )


def choose_gold_planning_cpu_action(
    cpu: CpuPlayer,
    room,
    validator: Optional[NumberValidator] = None,
) -> CpuAction:
    validator = gold_knowledge_number_validator
    field = getattr(room, "field", []) or []

    if field:
        action = choose_gold_response_action(cpu, room, validator)
    else:
        action = choose_gold_lead_action(cpu, room, validator)
    if action is not None:
        return action
    if field:
        return CpuAction("pass")

    cut = choose_57_cut(cpu.hand, room)
    if cut is not None:
        return CpuAction("play_prime", cut)

    if not getattr(room, "has_drawn", False) and getattr(room, "deck", []):
        return CpuAction("draw")

    joker = single_joker(cpu.hand)
    field_count = len(getattr(room, "field", []) or [])
    if joker is not None and field_count <= 1:
        return CpuAction("play_prime", {"cards": [joker], "assigned_numbers": []})

    return CpuAction("pass")


def choose_platinum_planning_cpu_action(
    cpu: CpuPlayer,
    room,
    validator: Optional[NumberValidator] = None,
) -> CpuAction:
    """Five-step Gold search plus Platinum's draw/all-out recovery policy."""
    validator = gold_knowledge_number_validator
    platinum_refresh_trump_strength_requirement(cpu, room)
    field = getattr(room, "field", []) or []
    if field:
        action = choose_platinum_response_action(cpu, room, validator)
    else:
        action = choose_platinum_lead_action(cpu, room, validator)
    return action or CpuAction("pass")


def choose_platinum_lead_action(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
) -> Optional[CpuAction]:
    active = getattr(cpu, "gold_active_plan", None)
    if active and active.get("dual_wield") and getattr(cpu, "gold_plan_step_index", 0) > 0:
        fused = active.get("dual_wield_fused")
        if fused and candidate_cards_available(fused, cpu) and candidate_is_playable(fused, cpu, room):
            clear_gold_active_plan(cpu)
            return platinum_commit_play(cpu, candidate_to_action(fused))

    action = play_next_gold_plan_step(cpu, room, validator)
    if action is not None:
        return platinum_commit_play(cpu, action)
    clear_gold_active_plan(cpu)

    large_hand_action = choose_platinum_large_hand_action(cpu, room, validator)
    if large_hand_action is not None:
        return platinum_commit_play(cpu, large_hand_action)

    plan = choose_platinum_strong_plan(cpu, room_without_field(room), validator)
    if plan is not None:
        set_gold_active_plan(cpu, plan)
        action = play_next_gold_plan_step(cpu, room, validator)
        if action is not None:
            return platinum_commit_play(cpu, action)

    # Drawing is a separate action. run_cpu_turn calls this selector again with
    # the new card, so the opening tactic is deliberately searched a second time.
    if not getattr(room, "has_drawn", False) and getattr(room, "deck", []):
        return CpuAction("draw")

    if platinum_should_all_out(cpu, room):
        payload = build_gold_all_out_payload(cpu.hand, force_random=True, rng=cpu.rng)
        if payload is not None:
            cpu.platinum_all_out_attempts += 1
            clear_gold_active_plan(cpu)
            return platinum_commit_play(cpu, CpuAction("play_prime", payload))

    compression = choose_platinum_compression_action(cpu, room, validator)
    if compression is not None:
        return platinum_commit_play(cpu, compression)
    weak_play = choose_platinum_bounded_legal_action(cpu, room, validator)
    if weak_play is not None:
        return platinum_commit_play(cpu, weak_play)
    return None


def choose_platinum_response_action(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
) -> Optional[CpuAction]:
    field_count = len(getattr(room, "field", []) or [])
    active = getattr(cpu, "gold_active_plan", None)
    if active and platinum_plan_score(active) >= platinum_required_trump_strength(cpu):
        if active_gold_plan_matches_field(cpu, field_count):
            action = play_next_gold_plan_step(cpu, room, validator)
            if action is not None:
                return platinum_commit_play(cpu, action)

    finish_now = platinum_one_move_finish_candidate(cpu, room, validator)
    if finish_now is not None:
        clear_gold_active_plan(cpu)
        return platinum_commit_play(cpu, candidate_to_action(finish_now))

    waiting_finish = platinum_one_move_finish_candidate(
        cpu,
        room_without_field(room),
        validator,
    )
    if waiting_finish is not None:
        preserving = choose_platinum_finish_preserving_response(cpu, room, validator)
        if preserving is not None:
            return platinum_commit_play(cpu, preserving)
        clear_gold_active_plan(cpu)
        return None

    interference = choose_platinum_interference_action(
        cpu,
        room,
        validator,
        active_plan=active,
    )
    if interference is not None:
        clear_gold_active_plan(cpu)
        return platinum_commit_play(cpu, interference)

    post_all_out_response = choose_platinum_post_all_out_response(cpu, room, validator)
    if post_all_out_response is not None:
        clear_gold_active_plan(cpu)
        return platinum_commit_play(cpu, post_all_out_response)

    clear_gold_active_plan(cpu)

    large_hand_action = choose_platinum_large_hand_action(cpu, room, validator)
    if large_hand_action is not None:
        return platinum_commit_play(cpu, large_hand_action)

    plan = build_same_count_gold_plan(
        cpu, room, field_count, PLATINUM_PLAN_MAX_STEPS, validator
    )
    score = platinum_plan_score(plan)
    cpu.platinum_last_strategy_score = score
    if is_executable_gold_plan(plan, cpu) and platinum_plan_is_strong(plan, cpu, room):
        set_gold_active_plan(cpu, plan)
        action = play_next_gold_plan_step(cpu, room, validator)
        if action is not None:
            return platinum_commit_play(cpu, action)

    if not getattr(room, "has_drawn", False) and getattr(room, "deck", []):
        return CpuAction("draw")

    if (
        score < platinum_required_trump_strength(cpu)
        and platinum_failed_composite_all_out_allowed(cpu, room, allow_opening=True)
    ):
        payload = build_composite_practice_all_out_payload(
            cpu, room, rng=cpu.rng, require_invalid=True
        )
        if payload is not None:
            cpu.platinum_all_out_attempts += 1
            return platinum_commit_play(cpu, CpuAction("play_composite", payload))

    action = choose_platinum_normal_field_action(cpu, room, validator)
    if action is not None:
        return platinum_commit_play(cpu, action)
    return None


def platinum_one_move_finish_candidate(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
) -> Optional[dict]:
    candidate = choose_gold_finish_candidate(cpu, room, validator)
    if candidate is None or not candidate_is_playable(candidate, cpu, room):
        return None
    return candidate if len(candidate_consumed_cards(candidate)) == len(cpu.hand) else None


def choose_platinum_finish_preserving_response(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
) -> Optional[CpuAction]:
    candidates = platinum_legal_response_candidates(
        cpu,
        room,
        validator,
        allow_non_trump_joker=True,
    )
    choices = []
    for candidate in candidates:
        remaining = remaining_cards(cpu.hand, candidate_consumed_cards(candidate))
        if not remaining:
            continue
        child = temporary_cpu_with_hand(cpu, remaining)
        finish = platinum_one_move_finish_candidate(
            child,
            room_without_field(room),
            validator,
        )
        if finish is None:
            continue
        choices.append((candidate_strength(candidate, room), candidate, child, finish))
    if not choices:
        return None
    _, candidate, child, finish = min(choices, key=lambda item: item[0])
    finish = dict(finish)
    finish["role"] = "finish"
    set_gold_active_plan(
        cpu,
        finalize_gold_plan(child, room_without_field(room), [finish], 0),
    )
    return candidate_to_action(candidate)


def platinum_legal_response_candidates(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
    *,
    allow_non_trump_joker: bool = False,
) -> list[dict]:
    """Enumerate legal same-count responses under Platinum's X policy."""
    field_count = len(getattr(room, "field", []) or [])
    if field_count <= 0:
        return []
    candidates = gold_plan_candidates(cpu, room, (field_count,), validator)
    candidates.extend(joker_prime_candidates_for_count(cpu, room, field_count, validator))
    candidates.extend(gold_special_cut_candidates(cpu, room))
    candidates = [
        candidate
        for candidate in dedupe_candidates(candidates)
        if candidate_is_playable(candidate, cpu, room)
    ]
    if allow_non_trump_joker:
        return candidates
    return [
        candidate
        for candidate in candidates
        if platinum_unplanned_joker_play_allowed(candidate, cpu)
    ]


def platinum_candidate_preserves_any(
    cpu: CpuPlayer,
    candidate: dict,
    protected_candidates: Iterable[dict],
) -> bool:
    remaining = remaining_cards(cpu.hand, candidate_consumed_cards(candidate))
    child = temporary_cpu_with_hand(cpu, remaining)
    return any(
        candidate_cards_available(protected, child)
        and platinum_candidate_is_response_trump(protected, child)
        for protected in protected_candidates
    )


def choose_platinum_interference_action(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
    active_plan: Optional[dict] = None,
) -> Optional[CpuAction]:
    """Return a legal blocking play even when it does not complete our plan."""
    candidates = platinum_legal_response_candidates(cpu, room, validator)
    if not candidates:
        cpu.platinum_last_interference_score = 0
        return None

    scored = [
        (
            platinum_interference_score(
                cpu,
                room,
                candidate,
                active_plan=active_plan,
                validator=validator,
            ),
            candidate_strength(candidate, room),
            len(candidate_consumed_cards(candidate)),
            candidate,
        )
        for candidate in candidates
    ]
    best_score = max(item[0] for item in scored)
    cpu.platinum_last_interference_score = best_score
    if best_score < PLATINUM_INTERFERENCE_BORDER:
        return None
    eligible = [item for item in scored if item[0] >= PLATINUM_INTERFERENCE_BORDER]
    held_trumps = platinum_available_trump_candidates(cpu, room, validator)
    if held_trumps:
        def interference_choice_key(item: tuple) -> tuple:
            candidate = item[-1]
            preserves_trump = platinum_candidate_preserves_any(
                cpu,
                candidate,
                held_trumps,
            )
            return (
                1 if preserves_trump else 0,
                0 if step_uses_joker(candidate) else 1,
                0 if platinum_candidate_is_response_trump(candidate, cpu) else 1,
                item[0],
                -item[1],
                -item[2],
            )

        best = max(eligible, key=interference_choice_key)
    else:
        best = max(eligible, key=lambda item: (
            0 if step_uses_joker(item[-1]) else 1,
            item[0],
            item[1],
            item[2],
        ))
    cpu.platinum_last_interference_score = best[0]
    platinum_mark_successful_interference(cpu, room)
    return candidate_to_action(best[-1])


def platinum_opponent_preplay_hand_count(cpu: CpuPlayer, room) -> Optional[int]:
    """Return the opponent hand size immediately before the current field play."""
    recorded_count = getattr(room, "last_play_hand_before", None)
    recorded_player_id = getattr(room, "last_play_player_id", None)
    if recorded_count is not None and recorded_player_id != cpu.id:
        return int(recorded_count)
    current_count = platinum_opponent_hand_count(cpu, room)
    if current_count is None:
        return None
    return current_count + len(getattr(room, "field", []) or [])


def choose_platinum_post_all_out_response(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
) -> Optional[CpuAction]:
    """Keep rallying after all-out when an 80+ same-count trump is available."""
    if int(getattr(cpu, "platinum_all_out_attempts", 0)) <= 0:
        return None
    opponent_preplay_count = platinum_opponent_preplay_hand_count(cpu, room)
    if (
        opponent_preplay_count is None
        or opponent_preplay_count > PLATINUM_POST_ALL_OUT_MAX_OPPONENT_PREPLAY_HAND_SIZE
    ):
        return None

    candidates = platinum_legal_response_candidates(cpu, room, validator)
    strong_trumps = [
        candidate
        for candidate in candidates
        if platinum_candidate_trump_strength(candidate) >= PLATINUM_MIN_TRUMP_STRENGTH
        or platinum_candidate_is_absolute(candidate, cpu)
    ]
    if not strong_trumps:
        return None

    preserving = [
        candidate
        for candidate in candidates
        if platinum_candidate_preserves_any(cpu, candidate, strong_trumps)
    ]
    if preserving:
        best = max(preserving, key=lambda candidate: (
            0 if step_uses_joker(candidate) else 1,
            0 if platinum_candidate_is_response_trump(candidate, cpu) else 1,
            -candidate_strength(candidate, room),
            -len(candidate_consumed_cards(candidate)),
        ))
    else:
        best = max(strong_trumps, key=lambda candidate: (
            0 if step_uses_joker(candidate) else 1,
            -platinum_candidate_trump_strength(candidate),
            -len(candidate_consumed_cards(candidate)),
        ))
    return candidate_to_action(best)


def platinum_interference_score(
    cpu: CpuPlayer,
    room,
    candidate: dict,
    active_plan: Optional[dict] = None,
    validator: Optional[NumberValidator] = None,
) -> int:
    score = platinum_opponent_hand_score(cpu, room)
    if platinum_interference_breaks_plan(cpu, candidate, active_plan):
        score -= 5
    if platinum_candidate_uses_last_absolute_trump(
        cpu,
        room,
        candidate,
        validator or gold_knowledge_number_validator,
    ):
        score -= 20
    if platinum_expected_opponent_kx_remaining(cpu, room) <= 0.1:
        score += 20
    return score


def platinum_opponent_hand_score(cpu: CpuPlayer, room) -> int:
    count = platinum_opponent_hand_count(cpu, room)
    if count is None:
        return 0
    return PLATINUM_OPPONENT_HAND_SCORES.get(count, 0)


def platinum_opponent_hand_count(cpu: CpuPlayer, room) -> Optional[int]:
    opponents = [
        player
        for player in (getattr(room, "players", []) or [])
        if getattr(player, "id", None) != cpu.id
        and getattr(player, "status", "playing") != "finished"
    ]
    if opponents:
        return max(len(getattr(player, "hand", []) or []) for player in opponents)
    count = getattr(room, "opponent_hand_count", None)
    return int(count) if count is not None else None


def platinum_mark_successful_interference(cpu: CpuPlayer, room) -> None:
    count = platinum_opponent_hand_count(cpu, room)
    if count is None:
        return
    cpu.platinum_relaxed_opponent_min_hand_count = count
    cpu.platinum_all_out_suppressed_opponent_min_hand_count = count
    cpu.platinum_current_min_trump_strength = PLATINUM_RELAXED_TRUMP_STRENGTH


def platinum_refresh_trump_strength_requirement(cpu: CpuPlayer, room) -> float:
    minimum_count = getattr(cpu, "platinum_relaxed_opponent_min_hand_count", None)
    current_count = platinum_opponent_hand_count(cpu, room)
    if minimum_count is not None and current_count is not None:
        if current_count > minimum_count:
            cpu.platinum_relaxed_opponent_min_hand_count = None
        elif current_count < minimum_count:
            cpu.platinum_relaxed_opponent_min_hand_count = current_count
    platinum_refresh_all_out_suppression(cpu, room)
    relaxed = getattr(cpu, "platinum_relaxed_opponent_min_hand_count", None) is not None
    threshold = (
        PLATINUM_RELAXED_TRUMP_STRENGTH
        if relaxed
        else PLATINUM_MIN_TRUMP_STRENGTH
    )
    cpu.platinum_current_min_trump_strength = threshold
    return threshold


def platinum_refresh_all_out_suppression(cpu: CpuPlayer, room) -> bool:
    minimum_count = getattr(
        cpu,
        "platinum_all_out_suppressed_opponent_min_hand_count",
        None,
    )
    if minimum_count is None:
        return False
    current_count = platinum_opponent_hand_count(cpu, room)
    if current_count is None:
        return True
    if current_count < minimum_count:
        cpu.platinum_all_out_suppressed_opponent_min_hand_count = current_count
        return True
    if (
        current_count > minimum_count
        and current_count >= PLATINUM_ALL_OUT_RESUME_MIN_OPPONENT_HAND_SIZE
    ):
        cpu.platinum_all_out_suppressed_opponent_min_hand_count = None
        return False
    return True


def platinum_all_out_is_suppressed(cpu: CpuPlayer, room) -> bool:
    return platinum_refresh_all_out_suppression(cpu, room)


def platinum_required_trump_strength(cpu: CpuPlayer) -> float:
    return float(getattr(
        cpu,
        "platinum_current_min_trump_strength",
        PLATINUM_MIN_TRUMP_STRENGTH,
    ))


def platinum_unaccounted_kx_distribution(cpu: CpuPlayer, room) -> tuple[int, int, int]:
    """Return unaccounted K/X, opponent cards, and unknown deck cards."""
    opponent_count = platinum_opponent_hand_count(cpu, room)
    if opponent_count is None:
        return 0, 0, 0

    def card_key(card: Card):
        return card.get("card_id") or id(card)

    own_ids = {card_key(card) for card in cpu.hand}
    own_kx = sum(
        1 for card in cpu.hand
        if is_joker(card) or int(card.get("rank", 0)) == 13
    )
    public_cards = {}
    for card in (
        list(getattr(room, "field", []) or [])
        + list(getattr(room, "reserve", []) or [])
        + list(getattr(room, "public_known_deck_bottom", []) or [])
    ):
        key = card_key(card)
        if key not in own_ids:
            public_cards[key] = card
    public_kx = sum(
        1 for card in public_cards.values()
        if is_joker(card) or int(card.get("rank", 0)) == 13
    )
    unaccounted_kx = max(0, 6 - own_kx - public_kx)
    if hasattr(room, "public_unknown_deck_count"):
        unknown_deck_count = max(0, int(room.public_unknown_deck_count))
    else:
        unknown_deck_count = max(
            0,
            len(getattr(room, "deck", []) or [])
            - len(getattr(room, "public_known_deck_bottom", []) or []),
        )
    return unaccounted_kx, opponent_count, unknown_deck_count


def platinum_expected_opponent_kx_remaining(cpu: CpuPlayer, room) -> float:
    """Expected K/X count using only hand counts and publicly known cards."""
    if platinum_opponent_hand_count(cpu, room) is None:
        return float("inf")
    unaccounted_kx, opponent_count, unknown_deck_count = (
        platinum_unaccounted_kx_distribution(cpu, room)
    )
    unknown_population = opponent_count + unknown_deck_count
    if unknown_population <= 0:
        return 0.0
    return unaccounted_kx * opponent_count / unknown_population


def platinum_expected_unknown_deck_kx(cpu: CpuPlayer, room) -> float:
    """Expected K/X contribution still hidden in the draw pile."""
    unaccounted_kx, opponent_count, unknown_deck_count = (
        platinum_unaccounted_kx_distribution(cpu, room)
    )
    unknown_population = opponent_count + unknown_deck_count
    if unknown_population <= 0:
        return 0.0
    return unaccounted_kx * unknown_deck_count / unknown_population


def platinum_interference_breaks_plan(
    cpu: CpuPlayer,
    candidate: dict,
    active_plan: Optional[dict],
) -> bool:
    if not active_plan:
        return False
    steps = active_plan.get("steps", [])
    index = int(getattr(cpu, "gold_plan_step_index", 0))
    if index >= len(steps):
        return False
    return candidate_fingerprint(candidate) != candidate_fingerprint(steps[index])


def platinum_candidate_uses_last_absolute_trump(
    cpu: CpuPlayer,
    room,
    candidate: dict,
    validator: NumberValidator,
) -> bool:
    if not platinum_candidate_is_absolute(candidate, cpu):
        return False
    remaining = remaining_cards(cpu.hand, candidate_consumed_cards(candidate))
    child = temporary_cpu_with_hand(cpu, remaining)
    empty_room = room_without_field(room)
    max_count = min(9, len(remaining))
    candidates = gold_plan_candidates(child, empty_room, range(1, max_count + 1), validator)
    for count in range(1, max_count + 1):
        candidates.extend(joker_prime_candidates_for_count(child, empty_room, count, validator))
    return not any(
        platinum_candidate_is_absolute(other, child)
        for other in dedupe_candidates(candidates)
    )


def platinum_candidate_is_trump(candidate: dict, cpu: CpuPlayer) -> bool:
    return (
        candidate.get("kind") == "joker_cut"
        or candidate.get("number") in {"X", 57}
        or platinum_candidate_is_absolute(candidate, cpu)
        or platinum_candidate_token(candidate) in PLATINUM_SMALL_TRUMP_TOKENS
    )


def platinum_candidate_trump_strength(candidate: dict) -> float:
    if candidate.get("kind") == "joker_cut" or candidate.get("number") == "X":
        return 100.0
    step = dict(candidate)
    step["role"] = f"rally-{len(candidate.get('cards', []))}"
    return gold_plan_trump_strength_score(
        {"steps": [step]},
        gold_plan_evaluation_config(),
    )


def platinum_candidate_is_response_trump(candidate: dict, cpu: CpuPlayer) -> bool:
    return (
        platinum_candidate_is_trump(candidate, cpu)
        or platinum_candidate_trump_strength(candidate) >= PLATINUM_MIN_TRUMP_STRENGTH
    )


def platinum_unplanned_joker_play_allowed(candidate: dict, cpu: CpuPlayer) -> bool:
    return (
        not step_uses_joker(candidate)
        or platinum_candidate_is_response_trump(candidate, cpu)
    )


def platinum_available_trump_candidates(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
) -> list[dict]:
    """Enumerate currently realizable trumps for interference preservation."""
    empty_room = room_without_field(room)
    max_count = min(9, len(cpu.hand))
    if max_count <= 0:
        return []
    candidates = gold_plan_candidates(cpu, empty_room, range(1, max_count + 1), validator)
    for count in range(1, max_count + 1):
        candidates.extend(joker_prime_candidates_for_count(
            cpu,
            empty_room,
            count,
            validator,
        ))
    candidates.extend(gold_special_cut_candidates(cpu, empty_room))
    return [
        candidate
        for candidate in dedupe_candidates(candidates)
        if platinum_candidate_is_response_trump(candidate, cpu)
    ]


def choose_platinum_strong_plan(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
) -> Optional[dict]:
    candidates = []
    if getattr(cpu, "platinum_opening_phase", True) and len(cpu.hand) == 11:
        dual = build_platinum_dual_wield_plan(cpu, room, validator)
        if dual is not None:
            cpu.platinum_last_strategy_score = platinum_plan_score(dual)
            return dual
    candidates.extend(build_platinum_plans(cpu, room, validator))
    candidates = [
        plan for plan in candidates
        if is_executable_gold_plan(plan, cpu)
        and platinum_opening_multi_play_is_sound(cpu, plan)
    ]
    if not candidates:
        cpu.platinum_last_strategy_score = 0.0
        return None
    if platinum_opponent_hand_is_expanded(cpu, room):
        absolute = [plan for plan in candidates if platinum_plan_has_absolute_trump(plan, cpu)]
        if absolute:
            candidates = absolute
    best = max(candidates, key=platinum_plan_sort_key)
    cpu.platinum_last_strategy_score = platinum_plan_score(best)
    return best if platinum_plan_is_strong(best, cpu, room) else None


def build_platinum_plans(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
) -> list[dict]:
    """Search Gold-depth routes, stopping once an 80+ route is established."""
    non_joker_count = len([card for card in cpu.hand if not is_joker(card)])
    plans = []
    for rally_count in platinum_rally_count_order(cpu, non_joker_count):
        plans.extend(search_same_count_gold_plans(
            cpu,
            room,
            rally_count,
            PLATINUM_PLAN_MAX_STEPS,
            validator,
        ))
        strong = [
            plan for plan in plans
            if is_executable_gold_plan(plan, cpu)
            and platinum_plan_is_strong(plan, cpu, room)
            and platinum_opening_multi_play_is_sound(cpu, plan)
        ]
        if strong:
            strong.sort(key=platinum_plan_sort_key, reverse=True)
            return strong[:GOLD_PLAN_MAX_ALTERNATIVES]
    plans = [plan for plan in plans if platinum_opening_multi_play_is_sound(cpu, plan)]
    plans.sort(key=platinum_plan_sort_key, reverse=True)
    return plans[:GOLD_PLAN_MAX_ALTERNATIVES]


def platinum_rally_count_order(cpu: CpuPlayer, non_joker_count: int) -> tuple[int, ...]:
    upper = min(9, non_joker_count)
    if len(cpu.hand) < PLATINUM_COMPRESSION_MIN_HAND_SIZE or upper < 5:
        return tuple(range(1, upper + 1))
    return (
        *range(upper, 4, -1),
        *range(min(4, upper), 0, -1),
    )


def platinum_opening_multi_play_is_sound(cpu: CpuPlayer, plan: dict) -> bool:
    if not getattr(cpu, "platinum_opening_phase", True):
        return True
    steps = plan.get("steps", [])
    if not steps:
        return False

    remaining = cpu.hand[:]
    for step in steps:
        consumed = candidate_consumed_cards(step)
        if not temporary_cpu_with_hand(cpu, remaining).has_cards(consumed):
            return False
        remaining = remaining_cards(remaining, consumed)
    if remaining:
        return False

    if len(steps) == 1:
        return True
    if steps[-1].get("role") != "finish":
        return False
    if len(steps) == 2:
        return True

    rally_steps = steps[:-1]
    rally_count = len(rally_steps[0].get("cards", []))
    return (
        rally_count > 0
        and all(
            str(step.get("role", "")).startswith("rally-")
            and len(step.get("cards", [])) == rally_count
            for step in rally_steps
        )
        and gold_plan_trump_step_index(plan) == len(steps) - 2
    )


def platinum_plan_sort_key(plan: dict) -> tuple:
    return (
        1 if plan.get("dual_wield") else 0,
        platinum_plan_score(plan),
        gold_plan_score(plan),
    )


def platinum_plan_score(plan: Optional[dict]) -> float:
    if not plan:
        return 0.0
    return float(plan.get("dual_wield_score", plan.get("evaluation", {}).get("score", 0.0)))


def platinum_plan_is_strong(plan: dict, cpu: CpuPlayer, room) -> bool:
    return (
        platinum_plan_score(plan) >= platinum_required_trump_strength(cpu)
        or platinum_plan_has_absolute_trump(plan, cpu)
    )


def platinum_plan_has_absolute_trump(plan: dict, cpu: CpuPlayer) -> bool:
    trump_index = gold_plan_trump_step_index(plan)
    if trump_index is None:
        return False
    steps = plan.get("steps", [])
    return trump_index < len(steps) and platinum_candidate_is_absolute(steps[trump_index], cpu)


def platinum_candidate_is_absolute(candidate: dict, cpu: CpuPlayer) -> bool:
    token = platinum_candidate_token(candidate)
    if token in PLATINUM_ABSOLUTE_ALWAYS:
        return True
    kx_count = sum(
        1
        for card in cpu.hand
        if is_joker(card) or int(card.get("rank", 0)) == 13
    )
    return kx_count >= 4 and token in PLATINUM_ABSOLUTE_KX4


def platinum_candidate_token(candidate: dict) -> str:
    ranks = candidate.get("ranks") or ()
    if ranks:
        return "".join(platinum_rank_token(int(rank)) for rank in ranks)
    assigned = iter(candidate.get("assigned_numbers", []) or [])
    parts = []
    for card in candidate.get("cards", []):
        if is_joker(card):
            assigned_value = next(assigned, None)
            if assigned_value is None:
                parts.append("x")
                continue
            rank = int(assigned_value)
        else:
            rank = int(card.get("rank", 0))
        parts.append(platinum_rank_token(rank))
    return "".join(parts)


def platinum_rank_token(rank: int) -> str:
    return {10: "t", 11: "j", 12: "q", 13: "k"}.get(rank, str(rank))


def platinum_token_ranks(token: str) -> tuple[int, ...]:
    return tuple(
        PLATINUM_TOKEN_RANKS[char] if char in PLATINUM_TOKEN_RANKS else int(char)
        for char in token
    )


def platinum_token_value(token: str) -> int:
    return int("".join(str(rank) for rank in platinum_token_ranks(token)))


def build_platinum_dual_wield_plan(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
) -> Optional[dict]:
    """Find the user-approved initial-11 dual-wield partitions."""
    if len(cpu.hand) != 11:
        return None
    plans = []
    for token, (trump_token, score) in PLATINUM_DUAL_WIELD_TEMPLATES.items():
        value = platinum_token_value(token)
        if value not in cpu.registered_primes:
            continue
        fused_ranks = platinum_token_ranks(token)
        fused = cards_for_ranks_with_jokers(cpu.hand, fused_ranks)
        if fused is None:
            continue
        fused_candidate = {
            "kind": "prime",
            "number": value,
            "cards": fused["cards"],
            "assigned_numbers": fused["assigned_numbers"],
            "ranks": fused_ranks,
            "role": "dual-fused-finish",
        }
        opening_cards = remaining_cards(cpu.hand, fused["cards"])
        opening_cpu = temporary_cpu_with_hand(cpu, opening_cards)
        opening = choose_gold_finish_candidate(opening_cpu, room_without_field(room), validator)
        trump_ranks = platinum_token_ranks(trump_token)
        if platinum_token_value(trump_token) not in cpu.registered_primes:
            continue
        if opening is None or len(opening.get("cards", [])) != len(trump_ranks):
            continue
        trump_realization = cards_for_ranks_with_jokers(fused["cards"], trump_ranks)
        if trump_realization is None:
            continue
        trump = {
            "kind": "prime",
            "number": platinum_token_value(trump_token),
            "cards": trump_realization["cards"],
            "assigned_numbers": trump_realization["assigned_numbers"],
            "ranks": trump_ranks,
        }
        finish_cards = remaining_cards(fused["cards"], trump_realization["cards"])
        finish_cpu = temporary_cpu_with_hand(cpu, finish_cards)
        finish = choose_gold_finish_candidate(finish_cpu, room_without_field(room), validator)
        if finish is None:
            continue
        sequence = [dict(opening), trump, dict(finish)]
        sequence[0]["role"] = f"rally-{len(trump_ranks)}"
        sequence[1]["role"] = f"rally-{len(trump_ranks)}"
        sequence[2]["role"] = "finish"
        plan = finalize_gold_plan(cpu, room, sequence, len(trump_ranks))
        if not plan.get("completed"):
            continue
        plan["dual_wield"] = True
        plan["dual_wield_score"] = score
        plan["dual_wield_template"] = token
        plan["dual_wield_fused"] = fused_candidate
        plan["evaluation"] = {**plan.get("evaluation", {}), "score": score}
        plans.append(plan)
    return max(plans, key=platinum_plan_sort_key) if plans else None


def platinum_opponent_hand_is_expanded(cpu: CpuPlayer, room) -> bool:
    opponents = [
        player for player in (getattr(room, "players", []) or [])
        if getattr(player, "id", None) != cpu.id and getattr(player, "status", "playing") != "finished"
    ]
    if not opponents:
        count = getattr(room, "opponent_hand_count", None)
    else:
        count = max((len(getattr(player, "hand", []) or []) for player in opponents), default=None)
    if count is None:
        return False
    initial = int(getattr(cpu, "platinum_initial_hand_size", 11))
    return count > initial and count >= max(1, len(getattr(room, "deck", []) or []) - 2)


def platinum_deck_has_trump_value(room) -> bool:
    known = getattr(room, "public_known_deck_bottom", []) or []
    return any(is_joker(card) or int(card.get("rank", 0)) == 13 for card in known)


def platinum_deck_has_expected_trump_contribution(cpu: CpuPlayer, room) -> bool:
    return (
        platinum_deck_has_trump_value(room)
        or platinum_expected_unknown_deck_kx(cpu, room) > 0.0
    )


def platinum_interference_danger_active(cpu: CpuPlayer, room) -> bool:
    """Whether the opponent's finish risk calls for interference over all-out."""
    score = platinum_opponent_hand_score(cpu, room)
    if platinum_expected_opponent_kx_remaining(cpu, room) <= 0.1:
        score += 20
    return score >= PLATINUM_INTERFERENCE_BORDER


def platinum_failed_composite_all_out_allowed(
    cpu: CpuPlayer,
    room,
    allow_opening: bool,
) -> bool:
    if platinum_all_out_is_suppressed(cpu, room):
        return False
    if not getattr(getattr(room, "rule", None), "allow_composite", False):
        return False
    if allow_opening and getattr(cpu, "platinum_opening_phase", True):
        return True
    if int(getattr(cpu, "platinum_all_out_attempts", 0)) > 0:
        return (
            bool(getattr(room, "deck", []))
            and not platinum_interference_danger_active(cpu, room)
            and platinum_deck_has_expected_trump_contribution(cpu, room)
        )
    return bool(getattr(room, "deck", [])) and platinum_deck_has_trump_value(room)


def platinum_should_all_out(cpu: CpuPlayer, room) -> bool:
    if not cpu.hand or platinum_all_out_is_suppressed(cpu, room):
        return False
    attempts = int(getattr(cpu, "platinum_all_out_attempts", 0))
    if attempts == 0:
        return True
    return (
        bool(getattr(room, "deck", []))
        and not platinum_interference_danger_active(cpu, room)
        and platinum_deck_has_expected_trump_contribution(cpu, room)
    )


def platinum_has_trump_utilization_prospect(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
) -> bool:
    """Cheaply recognize hands worth the full five-step Platinum search."""
    empty_room = room_without_field(room)
    max_count = min(9, len(cpu.hand))
    if max_count <= 0:
        return False
    counts = range(1, max_count + 1)
    candidates = gold_plan_candidates(cpu, empty_room, counts, validator)
    for count in counts:
        candidates.extend(joker_prime_candidates_for_count(
            cpu, empty_room, count, validator
        ))
    threshold = platinum_required_trump_strength(cpu)
    for candidate in dedupe_candidates(candidates):
        if platinum_candidate_is_absolute(candidate, cpu):
            return True
        if len(candidate.get("cards", [])) >= 5:
            return True
        if platinum_candidate_trump_strength(candidate) >= threshold:
            return True
    return False


def platinum_large_hand_requires_compression(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
) -> bool:
    hand_count = len(cpu.hand)
    if hand_count >= PLATINUM_FORCED_COMPRESSION_HAND_SIZE:
        return True
    if hand_count < PLATINUM_COMPRESSION_MIN_HAND_SIZE:
        return False
    return not platinum_has_trump_utilization_prospect(cpu, room, validator)


def choose_platinum_large_hand_action(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
) -> Optional[CpuAction]:
    if not platinum_large_hand_requires_compression(cpu, room, validator):
        return None
    compression = choose_platinum_compression_action(cpu, room, validator)
    if compression is not None:
        return compression
    if len(cpu.hand) < PLATINUM_FORCED_COMPRESSION_HAND_SIZE:
        return None
    return choose_platinum_bounded_legal_action(
        cpu,
        room,
        validator,
        prefer_compression=True,
    )


def choose_platinum_bounded_legal_action(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
    prefer_compression: bool = False,
) -> Optional[CpuAction]:
    """Choose a legal one-ply play without recursively constructing a route."""
    field_count = len(getattr(room, "field", []) or [])
    max_count = min(PLATINUM_MAX_KNOWLEDGE_CARDS, len(cpu.hand))
    counts = (field_count,) if field_count else range(1, max_count + 1)
    candidates = gold_plan_candidates(cpu, room, counts, validator)
    for count in counts:
        if count <= 9:
            candidates.extend(joker_prime_candidates_for_count(
                cpu, room, count, validator
            ))
    candidates.extend(gold_special_cut_candidates(cpu, room))
    candidates = [
        candidate
        for candidate in dedupe_candidates(candidates)
        if candidate_is_playable(candidate, cpu, room)
        and platinum_unplanned_joker_play_allowed(candidate, cpu)
    ]
    if not candidates:
        return None

    def score(candidate: dict) -> tuple:
        consumed = candidate_consumed_cards(candidate)
        protected = sum(
            1 for card in consumed
            if is_joker(card) or int(card.get("rank", 0)) == 13
        )
        compression = 1 if len(consumed) >= PLATINUM_OPENING_MULTI_PLAY_MIN_CARDS else 0
        if prefer_compression:
            return (
                compression,
                -protected,
                len(consumed),
                -candidate_strength(candidate, room),
            )
        return (
            -protected,
            len(consumed),
            -candidate_strength(candidate, room),
        )

    return candidate_to_action(max(candidates, key=score))


def choose_platinum_timeout_action(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
) -> CpuAction:
    """Recover from a search deadline without treating kamatoto as the default."""
    platinum_refresh_trump_strength_requirement(cpu, room)
    large_hand_action = choose_platinum_large_hand_action(cpu, room, validator)
    if large_hand_action is not None:
        return platinum_commit_play(cpu, large_hand_action)

    if getattr(room, "field", []) or []:
        post_all_out_response = choose_platinum_post_all_out_response(cpu, room, validator)
        if post_all_out_response is not None:
            return platinum_commit_play(cpu, post_all_out_response)
        if platinum_failed_composite_all_out_allowed(cpu, room, allow_opening=False):
            payload = build_composite_practice_all_out_payload(
                cpu, room, rng=cpu.rng, require_invalid=True
            )
            if payload is not None:
                cpu.platinum_all_out_attempts += 1
                return platinum_commit_play(cpu, CpuAction("play_composite", payload))
        action = choose_platinum_bounded_legal_action(cpu, room, validator)
        return platinum_commit_play(cpu, action) if action is not None else CpuAction("pass")

    if platinum_should_all_out(cpu, room):
        payload = build_gold_all_out_payload(cpu.hand, force_random=True, rng=cpu.rng)
        if payload is not None:
            cpu.platinum_all_out_attempts += 1
            clear_gold_active_plan(cpu)
            return platinum_commit_play(cpu, CpuAction("play_prime", payload))
    action = choose_platinum_bounded_legal_action(
        cpu,
        room,
        validator,
        prefer_compression=len(cpu.hand) >= PLATINUM_COMPRESSION_MIN_HAND_SIZE,
    )
    return platinum_commit_play(cpu, action) if action is not None else CpuAction("pass")


def choose_platinum_normal_field_action(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
) -> Optional[CpuAction]:
    field_count = len(getattr(room, "field", []) or [])
    candidates = [
        candidate
        for candidate in dedupe_candidates(gold_plan_candidates(cpu, room, (field_count,), validator))
        if candidate_is_playable(candidate, cpu, room)
    ]
    if not candidates:
        return None
    ranked = []
    for candidate in candidates[:GOLD_PLAN_MAX_BRANCH_CANDIDATES]:
        child = temporary_cpu_with_hand(
            cpu, remaining_cards(cpu.hand, candidate_consumed_cards(candidate))
        )
        plan = build_gold_plan(
            child,
            room_without_field(room),
            max_steps=max(1, PLATINUM_PLAN_MAX_STEPS - 1),
            validator=validator,
        )
        ranked.append((
            1 if is_executable_gold_plan(plan, child) else 0,
            platinum_plan_score(plan),
            len(candidate_consumed_cards(candidate)),
            -candidate_strength(candidate, room),
            candidate,
        ))
    return candidate_to_action(max(ranked, key=lambda item: item[:-1])[-1])


def choose_platinum_compression_action(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
) -> Optional[CpuAction]:
    if len(cpu.hand) < PLATINUM_COMPRESSION_MIN_HAND_SIZE:
        return None
    field_count = len(getattr(room, "field", []) or [])
    max_cards = min(PLATINUM_MAX_KNOWLEDGE_CARDS, len(cpu.hand) - 1)
    counts = (field_count,) if field_count else range(1, max_cards + 1)
    candidates = [
        candidate
        for candidate in dedupe_candidates(gold_plan_candidates(cpu, room, counts, validator))
        if candidate_is_playable(candidate, cpu, room)
        and len(candidate_consumed_cards(candidate)) >= PLATINUM_OPENING_MULTI_PLAY_MIN_CARDS
        and len(candidate_consumed_cards(candidate)) < len(cpu.hand)
    ]
    if not candidates:
        return None

    def score(candidate: dict) -> tuple:
        consumed = candidate_consumed_cards(candidate)
        protected = sum(
            1 for card in consumed
            if is_joker(card) or int(card.get("rank", 0)) == 13
        )
        evens = sum(
            1 for card in consumed
            if not is_joker(card) and int(card.get("rank", 0)) % 2 == 0
        )
        return (
            1 if len(consumed) >= 10 else 0,
            -protected,
            len(consumed),
            evens,
            -candidate_strength(candidate, room),
        )

    candidates.sort(key=score, reverse=True)
    if len(cpu.hand) >= PLATINUM_FORCED_COMPRESSION_HAND_SIZE:
        return candidate_to_action(candidates[0])
    for candidate in candidates[:PLATINUM_COMPRESSION_FOLLOWUP_CHECK_CAP]:
        check_cpu_search_deadline(cpu)
        if platinum_compression_followup_available(cpu, room, candidate, validator):
            return candidate_to_action(candidate)
    return None


def platinum_compression_followup_available(
    cpu: CpuPlayer,
    room,
    candidate: dict,
    validator: NumberValidator,
) -> bool:
    remaining = remaining_cards(cpu.hand, candidate_consumed_cards(candidate))
    if not remaining:
        return False
    child = temporary_cpu_with_hand(cpu, remaining)
    empty_room = room_without_field(room)
    small_max = min(4, len(remaining))
    small = gold_plan_candidates(child, empty_room, range(1, small_max + 1), validator)
    for count in range(1, small_max + 1):
        small.extend(joker_prime_candidates_for_count(child, empty_room, count, validator))
    small.extend(gold_special_cut_candidates(child, empty_room))
    if any(
        other.get("number") in {"X", 57}
        or platinum_candidate_is_absolute(other, child)
        or platinum_candidate_token(other) in PLATINUM_SMALL_TRUMP_TOKENS
        for other in dedupe_candidates(small)
    ):
        return True

    large_max = min(PLATINUM_MAX_KNOWLEDGE_CARDS, len(remaining))
    if large_max < PLATINUM_OPENING_MULTI_PLAY_MIN_CARDS:
        return False
    large = gold_plan_candidates(
        child,
        empty_room,
        range(PLATINUM_OPENING_MULTI_PLAY_MIN_CARDS, large_max + 1),
        validator,
    )
    return any(
        len(candidate_consumed_cards(other)) >= PLATINUM_OPENING_MULTI_PLAY_MIN_CARDS
        for other in large
    )


def platinum_commit_play(cpu: CpuPlayer, action: CpuAction) -> CpuAction:
    if action.kind in ("play_prime", "play_composite"):
        cpu.platinum_opening_phase = False
    return action


def choose_silver_planning_cpu_action(
    cpu: CpuPlayer,
    room,
    validator: Optional[NumberValidator] = None,
) -> CpuAction:
    validator = gold_knowledge_number_validator
    if getattr(room, "field", []) or []:
        action = choose_silver_response_action(cpu, room, validator)
    else:
        action = choose_silver_lead_action(cpu, room, validator)
    return action or CpuAction("pass")


def choose_talkative_fish_cpu_action(
    cpu: CpuPlayer,
    room,
    validator: Optional[NumberValidator] = None,
) -> CpuAction:
    validator = gold_knowledge_number_validator
    priority = choose_fish_343_priority_action(cpu, room, validator)
    if priority is not None:
        return priority
    return choose_silver_planning_cpu_action(cpu, room, validator)


def choose_fish_343_priority_action(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
) -> Optional[CpuAction]:
    field_count = len(getattr(room, "field", []) or [])
    max_cards = min(9, len([card for card in cpu.hand if not is_joker(card)]))
    counts = (field_count,) if field_count else range(1, max_cards + 1)
    candidates = [
        candidate
        for candidate in silver_plan_candidates(cpu, room, counts, validator)
        if candidate_is_playable(candidate, cpu, room)
        and fish_candidate_mentions_343(candidate)
    ]
    if not candidates:
        return None

    candidates = dedupe_candidates(candidates)
    best = max(candidates, key=lambda candidate: fish_343_candidate_score(cpu, room, candidate, validator))
    clear_silver_active_plan(cpu)
    return candidate_to_action(best)


def fish_343_candidate_score(
    cpu: CpuPlayer,
    room,
    candidate: dict,
    validator: NumberValidator,
) -> tuple:
    remaining = remaining_cards(cpu.hand, candidate_consumed_cards(candidate))
    has_followup = has_remaining_known_play(cpu, room, remaining, validator)
    if not getattr(room, "field", []) and not remaining:
        has_followup = True
    strength = candidate_strength(candidate, room)
    if getattr(room, "field", []) or []:
        strength_key = -strength
    else:
        strength_key = -abs(strength)
    return (
        1 if has_followup else 0,
        -len(candidate_consumed_cards(candidate)),
        strength_key,
    )


def fish_candidate_mentions_343(candidate: dict) -> bool:
    if "343" in str(candidate.get("number", "")):
        return True
    expression = candidate.get("expression") or ""
    if "343" in str(expression):
        return True
    ranks = "".join(str(rank) for rank in candidate.get("ranks", ()))
    return "343" in ranks


def choose_silver_lead_action(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
) -> Optional[CpuAction]:
    action = play_next_silver_plan_step(cpu, room)
    if action is not None:
        return action

    plan = build_silver_plan(cpu, room_without_field(room), validator=validator)
    if is_executable_silver_plan(plan, cpu):
        set_silver_active_plan(cpu, plan)
        return play_next_silver_plan_step(cpu, room)

    clear_silver_active_plan(cpu)
    if not getattr(room, "has_drawn", False) and getattr(room, "deck", []):
        return CpuAction("draw")

    relief = choose_silver_even_relief_action(cpu, room, validator)
    if relief is not None:
        return relief
    return choose_silver_hnp_action(cpu, room)


def choose_silver_response_action(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
) -> Optional[CpuAction]:
    action = play_next_silver_plan_step(cpu, room)
    if action is not None:
        return action

    clear_silver_active_plan(cpu)
    field_count = len(getattr(room, "field", []) or [])
    if field_count:
        plan = build_silver_plan(
            cpu,
            room,
            counts=(field_count,),
            validator=validator,
            prefer_two_step_over_direct=True,
        )
        if is_executable_silver_plan(plan, cpu):
            set_silver_active_plan(cpu, plan)
            return play_next_silver_plan_step(cpu, room)

    if silver_waiting_to_finish(cpu, room, validator):
        return None

    if not getattr(room, "has_drawn", False) and getattr(room, "deck", []):
        return CpuAction("draw")

    if field_count:
        return choose_silver_even_relief_action(cpu, room, validator, counts=(field_count,))
    return None


def choose_gold_lead_action(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
) -> Optional[CpuAction]:
    action = play_next_gold_plan_step(cpu, room, validator)
    if action is not None:
        return action

    plan = build_gold_plan(cpu, room_without_field(room), max_steps=20, validator=validator)
    if is_executable_gold_plan(plan, cpu):
        set_gold_active_plan(cpu, plan)
        return play_next_gold_plan_step(cpu, room, validator)

    return choose_gold_all_out_or_draw(cpu, room)


def choose_gold_response_action(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
) -> Optional[CpuAction]:
    field_count = len(getattr(room, "field", []) or [])
    if not active_gold_plan_matches_field(cpu, field_count):
        clear_gold_active_plan(cpu)
        return choose_gold_plan_for_field_action(cpu, room, validator)

    action = play_next_gold_plan_step(cpu, room, validator)
    if action is not None:
        return action

    later = playable_later_gold_plan_steps(cpu, room)
    if later:
        step_index, candidate = later[0]
        trump_index = gold_plan_trump_step_index(cpu.gold_active_plan)
        if step_index == trump_index:
            return choose_gold_trump_or_saved_pass(cpu, room, candidate, validator)
        return play_gold_deviation_with_replan(cpu, room, candidate, validator)

    return choose_gold_correction_action(cpu, room, validator)


def choose_gold_plan_for_field_action(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
) -> Optional[CpuAction]:
    field_count = len(getattr(room, "field", []) or [])
    if not field_count:
        return choose_gold_lead_action(cpu, room, validator)

    plan = build_same_count_gold_plan(cpu, room, field_count, max_steps=20, validator=validator)
    if is_executable_gold_plan(plan, cpu):
        set_gold_active_plan(cpu, plan)
        return play_next_gold_plan_step(cpu, room, validator)

    return choose_gold_correction_action(cpu, room, validator)


def choose_gold_correction_action(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
) -> Optional[CpuAction]:
    field_count = len(getattr(room, "field", []) or [])
    if not field_count:
        return choose_gold_lead_action(cpu, room, validator)

    has_saved_plan = bool(getattr(cpu, "gold_active_plan", None))

    candidates = dedupe_candidates(gold_plan_candidates(cpu, room, [field_count], validator))
    candidates = [candidate for candidate in candidates if candidate_is_playable(candidate, cpu, room)]
    special_cuts = [
        candidate
        for candidate in gold_special_cut_candidates(cpu, room)
        if candidate_is_playable(candidate, cpu, room)
    ]
    if not candidates and not special_cuts:
        return choose_gold_no_correction_recovery(cpu, room, validator, has_saved_plan)

    rng = secrets.SystemRandom()
    sampled = rng.sample(candidates, min(3, len(candidates)))
    best = None
    # A one-card X or two-card 57 can reset the field.  They are evaluated in
    # addition to, rather than instead of, the three ordinary correction plays.
    for candidate in dedupe_candidates(sampled + special_cuts):
        remaining = remaining_cards(cpu.hand, candidate_consumed_cards(candidate))
        temp_cpu = temporary_cpu_with_hand(cpu, remaining)
        plan = build_gold_plan(temp_cpu, room_without_field(room), max_steps=20, validator=validator)
        if not is_executable_gold_plan(plan, temp_cpu):
            continue
        key = gold_plan_score(plan)
        if best is None or key > best[0]:
            best = (key, candidate, plan)

    if best is None:
        return choose_gold_no_correction_recovery(cpu, room, validator, has_saved_plan)

    set_gold_active_plan(cpu, best[2])
    return candidate_to_action(best[1])


def choose_gold_no_correction_recovery(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
    has_saved_plan: bool,
) -> Optional[CpuAction]:
    if has_saved_plan:
        return None

    plan = build_gold_plan(cpu, room_without_field(room), max_steps=20, validator=validator)
    if is_executable_gold_plan(plan, cpu):
        set_gold_active_plan(cpu, plan)
        return None

    if not getattr(room, "has_drawn", False) and getattr(room, "deck", []):
        return CpuAction("draw")
    return None


def play_next_gold_plan_step(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
) -> Optional[CpuAction]:
    plan = getattr(cpu, "gold_active_plan", None)
    if not plan:
        return None

    steps = plan.get("steps", [])
    index = getattr(cpu, "gold_plan_step_index", 0)
    while index < len(steps):
        candidate = steps[index]
        if not candidate_cards_available(candidate, cpu):
            clear_gold_active_plan(cpu)
            return None
        if candidate_is_playable(candidate, cpu, room):
            cpu.gold_plan_step_index = index + 1
            return candidate_to_action(candidate)
        break

    if index >= len(steps):
        clear_gold_active_plan(cpu)
    return None


def playable_later_gold_plan_steps(cpu: CpuPlayer, room) -> list[tuple[int, dict]]:
    plan = getattr(cpu, "gold_active_plan", None)
    if not plan:
        return []
    steps = plan.get("steps", [])
    start = getattr(cpu, "gold_plan_step_index", 0) + 1
    later = [
        (index, step)
        for index, step in enumerate(steps[start:], start=start)
        if candidate_cards_available(step, cpu)
        and candidate_is_playable(step, cpu, room)
        and len(step.get("cards", [])) == len(getattr(room, "field", []) or [])
    ]
    return sorted(later, key=lambda item: gold_plan_candidate_score(item[1], room))


def choose_gold_trump_or_saved_pass(
    cpu: CpuPlayer,
    room,
    trump: dict,
    validator: NumberValidator,
) -> CpuAction:
    remaining = remaining_cards(cpu.hand, candidate_consumed_cards(trump))
    temp_cpu = temporary_cpu_with_hand(cpu, remaining)
    tail = choose_gold_finish_tail(temp_cpu, room_without_field(room), validator)
    if tail:
        plan = finalize_gold_plan(temp_cpu, room_without_field(room), tail, len(trump.get("cards", [])))
        set_gold_active_plan(cpu, plan)
        return candidate_to_action(trump)
    return CpuAction("pass")


def play_gold_deviation_with_replan(
    cpu: CpuPlayer,
    room,
    candidate: dict,
    validator: NumberValidator,
) -> CpuAction:
    remaining = remaining_cards(cpu.hand, candidate_consumed_cards(candidate))
    temp_cpu = temporary_cpu_with_hand(cpu, remaining)
    plan = build_gold_plan(temp_cpu, room_without_field(room), max_steps=20, validator=validator)
    if is_executable_gold_plan(plan, temp_cpu):
        set_gold_active_plan(cpu, plan)
    else:
        clear_gold_active_plan(cpu)
    return candidate_to_action(candidate)


def set_gold_active_plan(cpu: CpuPlayer, plan: dict) -> None:
    cpu.gold_active_plan = plan
    cpu.gold_plan_step_index = 0


def clear_gold_active_plan(cpu: CpuPlayer) -> None:
    cpu.gold_active_plan = None
    cpu.gold_plan_step_index = 0


def set_silver_active_plan(cpu: CpuPlayer, plan: dict) -> None:
    cpu.silver_active_plan = plan
    cpu.silver_plan_step_index = 0


def clear_silver_active_plan(cpu: CpuPlayer) -> None:
    cpu.silver_active_plan = None
    cpu.silver_plan_step_index = 0


def play_next_silver_plan_step(cpu: CpuPlayer, room) -> Optional[CpuAction]:
    plan = getattr(cpu, "silver_active_plan", None)
    if not plan:
        return None

    steps = plan.get("steps", [])
    index = getattr(cpu, "silver_plan_step_index", 0)
    while index < len(steps):
        candidate = steps[index]
        if not candidate_cards_available(candidate, cpu):
            clear_silver_active_plan(cpu)
            return None
        if candidate_is_playable(candidate, cpu, room):
            cpu.silver_plan_step_index = index + 1
            return candidate_to_action(candidate)
        break

    if index >= len(steps):
        clear_silver_active_plan(cpu)
    return None


def is_executable_gold_plan(plan: dict, cpu: CpuPlayer) -> bool:
    return bool(plan.get("steps")) and bool(plan.get("completed")) and all(
        candidate_cards_available(step, cpu)
        for step in plan.get("steps", [])
    )


def is_executable_silver_plan(plan: dict, cpu: CpuPlayer) -> bool:
    return bool(plan.get("steps")) and bool(plan.get("completed")) and all(
        candidate_cards_available(step, cpu)
        for step in plan.get("steps", [])
    )


def active_gold_plan_matches_field(cpu: CpuPlayer, field_count: int) -> bool:
    plan = getattr(cpu, "gold_active_plan", None)
    if not plan or not field_count:
        return bool(plan)
    steps = plan.get("steps", [])
    index = getattr(cpu, "gold_plan_step_index", 0)
    if index >= len(steps):
        return False
    return len(steps[index].get("cards", [])) == field_count


def candidate_cards_available(candidate: dict, cpu: CpuPlayer) -> bool:
    return cpu.has_cards(candidate_consumed_cards(candidate))


def candidate_is_playable(candidate: dict, cpu: CpuPlayer, room) -> bool:
    if not candidate_cards_available(candidate, cpu):
        return False
    number = candidate.get("number")
    if number == "X":
        return len(getattr(room, "field", []) or []) <= 1
    try:
        value = int(number)
    except (TypeError, ValueError):
        return False
    return beats_field(value, len(candidate.get("cards", [])), room)


def gold_plan_trump_step_index(plan: Optional[dict]) -> Optional[int]:
    if not plan:
        return None
    return max(
        (
            index
            for index, step in enumerate(plan.get("steps", []))
            if str(step.get("role", "")).startswith("rally-")
        ),
        default=None,
    )


def choose_gold_all_out_or_draw(cpu: CpuPlayer, room) -> Optional[CpuAction]:
    if not getattr(room, "has_drawn", False) and getattr(room, "deck", []):
        clear_gold_active_plan(cpu)
        return CpuAction("draw")
    relief = choose_gold_even_relief_action(cpu, room)
    if relief is not None:
        clear_gold_active_plan(cpu)
        return relief
    forced = build_gold_all_out_payload(cpu.hand, force_random=True)
    if forced is None:
        return None
    clear_gold_active_plan(cpu)
    return CpuAction("play_prime", forced)


def choose_gold_even_relief_action(cpu: CpuPlayer, room) -> Optional[CpuAction]:
    if len(cpu.hand) < 18 or getattr(room, "field", []):
        return None

    max_cards = min(9, len([card for card in cpu.hand if not is_joker(card)]))
    candidates = gold_plan_candidates(cpu, room, range(1, max_cards + 1), gold_knowledge_number_validator)
    candidates = [
        candidate
        for candidate in dedupe_candidates(candidates)
        if not any(
            is_joker(card) or int(card.get("rank", 0)) in {11, 13}
            for card in candidate_consumed_cards(candidate)
        )
    ]
    if not candidates:
        return None

    hand_ratio = gold_even_card_ratio(cpu.hand)
    ratios = [gold_even_card_ratio(candidate_consumed_cards(candidate)) for candidate in candidates]
    best_ratio = max(ratios)
    if best_ratio < hand_ratio:
        return None

    best = [candidate for candidate, ratio in zip(candidates, ratios) if ratio == best_ratio]
    return candidate_to_action(secrets.SystemRandom().choice(best))


def gold_even_card_ratio(cards: Iterable[Card]) -> float:
    cards = list(cards)
    if not cards:
        return 0.0
    even_ranks = {2, 4, 5, 6, 8, 10, 12}
    return sum(
        1
        for card in cards
        if not is_joker(card) and int(card.get("rank", 0)) in even_ranks
    ) / len(cards)


def choose_silver_even_relief_action(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
    counts: Optional[Iterable[int]] = None,
) -> Optional[CpuAction]:
    if counts is None:
        max_cards = min(9, len([card for card in cpu.hand if not is_joker(card)]))
        counts = range(1, max_cards + 1)

    candidates = silver_plan_candidates(cpu, room, counts, validator)
    candidates = [
        candidate
        for candidate in dedupe_candidates(candidates)
        if candidate_is_playable(candidate, cpu, room)
        and not silver_candidate_uses_joker(candidate)
    ]
    if not candidates:
        return None

    low_count_candidates = [
        candidate for candidate in candidates if len(candidate.get("cards", [])) in SILVER_RALLY_COUNTS
    ]
    protected_tiers = {
        count: max(silver_trump_tier(candidate) for candidate in by_count)
        for count in SILVER_RALLY_COUNTS
        for by_count in [[
            candidate for candidate in low_count_candidates
            if len(candidate.get("cards", [])) == count
        ]]
        if by_count
    }
    filtered = []
    for candidate in candidates:
        count = len(candidate.get("cards", []))
        if (
            count in protected_tiers
            and not silver_preserves_trump_tier_after_play(cpu, room, candidate, protected_tiers[count], validator)
        ):
            continue
        filtered.append(candidate)
    candidates = filtered
    if not candidates:
        return None

    before_ratio = silver_even_card_ratio(cpu.hand)
    best = None
    for candidate in candidates:
        after_cards = remaining_cards(cpu.hand, candidate_consumed_cards(candidate))
        after_ratio = silver_even_card_ratio(after_cards)
        if after_ratio > before_ratio + SILVER_EVEN_RELIEF_MAX_RATIO_INCREASE:
            continue
        consumed = candidate_consumed_cards(candidate)
        key = (
            before_ratio - after_ratio,
            silver_even_card_ratio(consumed),
            len(consumed),
            candidate_strength(candidate, room),
        )
        if best is None or key > best[0]:
            best = (key, candidate)

    if best is None:
        return None
    return candidate_to_action(best[1])


def silver_preserves_trump_tier_after_play(
    cpu: CpuPlayer,
    room,
    candidate: dict,
    protected_tier: int,
    validator: NumberValidator,
) -> bool:
    count = len(candidate.get("cards", []))
    if count not in SILVER_RALLY_COUNTS:
        return True
    remaining = remaining_cards(cpu.hand, candidate_consumed_cards(candidate))
    temp_cpu = temporary_cpu_with_hand(cpu, remaining)
    remaining_candidates = silver_plan_candidates(temp_cpu, room, (count,), validator)
    return any(
        silver_trump_tier(remaining_candidate) >= protected_tier
        for remaining_candidate in remaining_candidates
    )


def silver_even_card_ratio(cards: Iterable[Card]) -> float:
    cards = list(cards)
    if not cards:
        return 0.0
    return sum(
        1
        for card in cards
        if not is_joker(card) and int(card.get("rank", 0)) in SILVER_EVEN_RANKS
    ) / len(cards)


def silver_candidate_uses_joker(candidate: dict) -> bool:
    return any(is_joker(card) for card in candidate_consumed_cards(candidate))


def choose_silver_hnp_action(cpu: CpuPlayer, room) -> Optional[CpuAction]:
    if getattr(room, "field", []) or []:
        return None
    if len(cpu.hand) <= 9:
        payload = build_gold_all_out_payload(cpu.hand, force_random=True)
        return CpuAction("play_prime", payload) if payload is not None else None

    payload = build_silver_hnp_payload(cpu.hand)
    return CpuAction("play_prime", payload) if payload is not None else None


def build_silver_hnp_payload(hand: List[Card]) -> Optional[dict]:
    non_jokers = [card for card in hand if not is_joker(card)]
    if len(non_jokers) < 5:
        return None

    rng = secrets.SystemRandom()
    nucleus = choose_silver_hnp_nucleus(non_jokers, rng)
    if nucleus is None:
        return None

    remaining = [card for card in non_jokers if card is not nucleus]
    evens = [card for card in remaining if int(card.get("rank", 0)) in SILVER_EVEN_RANKS]
    rng.shuffle(evens)
    odds = silver_hnp_odd_pool(remaining, rng)
    others = [
        card
        for card in remaining
        if card not in evens and card not in odds
    ]
    rng.shuffle(others)

    selected = [nucleus]
    snapshots = []
    hand_ratio = silver_even_card_ratio(non_jokers)
    for card in evens + odds + others:
        selected.append(card)
        if len(selected) >= 5:
            snapshots.append(selected[:])

    if not snapshots:
        return None

    ratio_ok = [
        cards for cards in snapshots
        if silver_even_card_ratio(cards) >= hand_ratio
    ]
    for cards in ratio_ok:
        if hand_rank_sum(cards, joker_value=None) % 3 != 0:
            return silver_hnp_payload_from_cards(cards, nucleus, rng)
    if ratio_ok:
        return silver_hnp_payload_from_cards(ratio_ok[0], nucleus, rng)

    for cards in snapshots:
        if hand_rank_sum(cards, joker_value=None) % 3 != 0:
            return silver_hnp_payload_from_cards(cards, nucleus, rng)
    return silver_hnp_payload_from_cards(snapshots[0], nucleus, rng)


def choose_silver_hnp_nucleus(cards: List[Card], rng) -> Optional[Card]:
    for ranks in ({1, 3, 7, 9}, {11}, {13}, SILVER_EVEN_RANKS):
        candidates = [card for card in cards if int(card.get("rank", 0)) in ranks]
        if candidates:
            return rng.choice(candidates)
    return rng.choice(cards) if cards else None


def silver_hnp_odd_pool(cards: List[Card], rng) -> list[Card]:
    pool = []
    for ranks in ({1, 3, 7, 9}, {11}, {13}):
        candidates = [card for card in cards if int(card.get("rank", 0)) in ranks]
        rng.shuffle(candidates)
        pool.extend(candidates)
    return pool


def silver_hnp_payload_from_cards(cards: List[Card], nucleus: Card, rng) -> dict:
    others = [card for card in cards if card is not nucleus]
    rng.shuffle(others)
    return {
        "cards": others + [nucleus],
        "assigned_numbers": [],
    }


def build_gold_all_out_payload(
    hand: List[Card],
    force_random: bool,
    rng=None,
) -> Optional[dict]:
    if not hand:
        return None

    rng = rng or secrets.SystemRandom()
    cards = hand[:]
    jokers = [card for card in cards if is_joker(card)]
    assigned_by_id = {}

    if jokers:
        choices = [1, 3, 7, 9]
        valid = [
            values for values in product(choices, repeat=len(jokers))
            if (
                sum(int(card.get("rank", 0)) for card in cards if not is_joker(card))
                + sum(values)
            ) % 3 != 0
        ]
        if valid:
            assigned_values = rng.choice(valid)
        elif not force_random:
            return None
        else:
            assigned_values = rng.choice(list(product(choices, repeat=len(jokers))))
        assigned_by_id = {
            joker.get("card_id"): str(value)
            for joker, value in zip(jokers, assigned_values)
        }
    elif hand_rank_sum(cards, joker_value=None) % 3 == 0 and not force_random:
        return None

    bottom = choose_gold_all_out_bottom_card(cards, assigned_by_id, rng)
    if bottom is None and not force_random:
        return None

    remaining = cards[:]
    if bottom is not None:
        remaining.remove(bottom)
    rng.shuffle(remaining)
    ordered = remaining + ([bottom] if bottom is not None else [])
    return {
        "cards": ordered,
        "assigned_numbers": [
            assigned_by_id[card.get("card_id")]
            for card in ordered
            if card.get("card_id") in assigned_by_id
        ],
    }


def hand_rank_sum(hand: List[Card], joker_value: Optional[int]) -> int:
    total = 0
    for card in hand:
        if is_joker(card):
            total += joker_value or 0
        else:
            total += int(card.get("rank", 0))
    return total


def choose_gold_all_out_bottom_card(cards: List[Card], assigned_by_id: dict, rng) -> Optional[Card]:
    odd_ranks = {1, 3, 7, 9, 11, 13}
    candidates = [
        card for card in cards
        if (
            int(assigned_by_id.get(card.get("card_id"), card.get("rank", 0))) in odd_ranks
            if is_joker(card)
            else int(card.get("rank", 0)) in odd_ranks
        )
    ]
    if not candidates:
        return None
    return rng.choice(candidates)


def choose_gold_play(
    cpu: CpuPlayer,
    room,
    validator: Optional[NumberValidator] = None,
) -> Optional[dict]:
    validator = validator or gold_knowledge_number_validator
    if not (getattr(room, "field", []) or []):
        plan = build_gold_plan(cpu, room, max_steps=20, validator=validator)
        if plan["steps"]:
            return plan["steps"][0]

    field_count = len(getattr(room, "field", []) or [])
    max_cards = min(9, len([card for card in cpu.hand if not is_joker(card)]))
    counts = [field_count] if field_count else list(range(1, max_cards + 1))
    candidates = knowledge_prime_candidates(cpu, room, validator, counts)
    candidates.extend(knowledge_composite_candidates(cpu, room, counts))
    if not candidates:
        return None

    trumps = strongest_trumps_by_count(cpu, room, validator)
    best = max(
        candidates,
        key=lambda candidate: gold_candidate_score(cpu, room, candidate, trumps, validator),
    )
    return best


def choose_gold_prime_play(
    cpu: CpuPlayer,
    room,
    validator: Optional[NumberValidator] = None,
) -> Optional[dict]:
    candidate = choose_gold_play(cpu, room, validator=validator)
    if candidate is None or candidate.get("kind") != "prime":
        return None
    return {
        "cards": candidate["cards"],
        "assigned_numbers": candidate["assigned_numbers"],
    }


def gold_knowledge_number_validator(number: int, cpu: CpuPlayer, rule) -> bool:
    prime_rule = getattr(rule, "prime_rule", PrimeRule.NORMAL)
    if prime_rule in (PrimeRule.NORMAL, PrimeRule.REGISTERED):
        return cpu.can_use_registered_prime(number)
    return default_number_validator(number, cpu, rule)


def build_gold_plan(
    cpu: CpuPlayer,
    room,
    max_steps: int = 20,
    validator: Optional[NumberValidator] = None,
) -> dict:
    validator = validator or gold_knowledge_number_validator
    plans = build_gold_plans(cpu, room, max_steps=max_steps, validator=validator)
    if plans:
        best = plans[0]
        best["alternatives"] = plans[1:GOLD_PLAN_MAX_ALTERNATIVES]
        return best

    fallback = build_same_count_gold_plan(cpu, room, 1, max_steps, validator)
    fallback["alternatives"] = []
    return fallback


def build_gold_plans(
    cpu: CpuPlayer,
    room,
    max_steps: int = 20,
    validator: Optional[NumberValidator] = None,
) -> list[dict]:
    validator = validator or gold_knowledge_number_validator
    non_joker_count = len([card for card in cpu.hand if not is_joker(card)])
    rally_counts = range(1, min(9, non_joker_count) + 1)
    plans = [
        plan
        for rally_count in rally_counts
        for plan in search_same_count_gold_plans(cpu, room, rally_count, max_steps, validator)
    ]
    plans.sort(key=gold_plan_score, reverse=True)
    return plans[:GOLD_PLAN_MAX_ALTERNATIVES]


def build_same_count_gold_plan(
    cpu: CpuPlayer,
    room,
    rally_count: int,
    max_steps: int,
    validator: NumberValidator,
) -> dict:
    searched = search_same_count_gold_plans(cpu, room, rally_count, max_steps, validator)
    if searched:
        return searched[0]

    temp_cpu = temporary_cpu_with_hand(cpu, cpu.hand[:])
    steps = []
    for _ in range(min(max_steps, GOLD_PLAN_MAX_RALLY_STEPS)):
        candidate = choose_gold_rally_candidate(temp_cpu, room, rally_count, validator)
        if candidate is None:
            break
        append_gold_plan_step(steps, temp_cpu, candidate, role=f"rally-{rally_count}")
        temp_cpu.hand = remaining_cards(temp_cpu.hand, candidate_consumed_cards(candidate))

    plan = {
        "steps": steps,
        "remaining": temp_cpu.hand,
        "completed": not temp_cpu.hand,
        "rally_count": rally_count,
        "last_rally_strength": gold_plan_last_rally_strength(steps, room),
    }
    plan["evaluation"] = evaluate_gold_plan(plan)
    return plan


def search_same_count_gold_plans(
    cpu: CpuPlayer,
    room,
    rally_count: int,
    max_steps: int,
    validator: NumberValidator,
) -> list[dict]:
    check_cpu_search_deadline(cpu)
    direct_tail = choose_gold_finish_tail(cpu, room, validator)
    if direct_tail:
        return [
            finalize_gold_plan(cpu, room, direct_tail, rally_count)
        ]

    results = []
    seen_plans = set()
    for joker_trump in (False, True):
        check_cpu_search_deadline(cpu)
        last_candidates = gold_last_rally_candidates(
            cpu,
            room,
            rally_count,
            validator,
            joker_trump=joker_trump,
        )
        for last in last_candidates:
            check_cpu_search_deadline(cpu)
            last = dict(last)
            last["joker_trump"] = joker_trump
            reserved_hand = remaining_cards(cpu.hand, candidate_consumed_cards(last))
            reserved_cpu = temporary_cpu_with_hand(cpu, reserved_hand)
            last_strength = candidate_strength(last, room)

            def visit(current_cpu: CpuPlayer, bound_strength: int, selected_desc: list[dict]) -> None:
                check_cpu_search_deadline(current_cpu)
                if len(results) >= GOLD_PLAN_MAX_RESULTS_PER_COUNT * 2:
                    return
                tail = choose_gold_finish_tail(current_cpu, room, validator)
                if tail:
                    sequence = list(reversed(selected_desc)) + [last] + tail
                    plan_key = tuple(candidate_fingerprint(candidate) for candidate in sequence)
                    if plan_key not in seen_plans:
                        seen_plans.add(plan_key)
                        plan = finalize_gold_plan(cpu, room, sequence, rally_count)
                        plan["joker_trump"] = joker_trump
                        results.append(plan)
                    return
                if (
                    len(selected_desc) >= GOLD_PLAN_MAX_RALLY_PREFIX_STEPS
                    or len(selected_desc) + 2 >= max_steps
                ):
                    return

                split_plans = gold_large_finish_split_candidates(
                    current_cpu,
                    room,
                    rally_count,
                    bound_strength,
                    validator,
                )
                for rally, finish_tail in split_plans:
                    check_cpu_search_deadline(current_cpu)
                    sequence = list(reversed(selected_desc + [rally])) + [last] + finish_tail
                    if len(sequence) > max_steps:
                        continue
                    plan_key = tuple(candidate_fingerprint(candidate) for candidate in sequence)
                    if plan_key in seen_plans:
                        continue
                    seen_plans.add(plan_key)
                    plan = finalize_gold_plan(cpu, room, sequence, rally_count)
                    plan["joker_trump"] = joker_trump
                    results.append(plan)
                    if len(results) >= GOLD_PLAN_MAX_RESULTS_PER_COUNT * 2:
                        return

                branch_candidates = gold_plan_candidates(current_cpu, room, [rally_count], validator)
                branch_candidates = [
                    candidate for candidate in branch_candidates
                    if len(candidate.get("cards", [])) == rally_count
                    and len(candidate_consumed_cards(candidate)) < len(current_cpu.hand)
                    and candidate_strength(candidate, room) < bound_strength
                ]
                branch_candidates = sorted(
                    dedupe_candidates(branch_candidates),
                    key=lambda candidate: gold_plan_candidate_score(candidate, room),
                    reverse=True,
                )[:gold_branch_candidate_cap(current_cpu)]

                for candidate in branch_candidates:
                    check_cpu_search_deadline(current_cpu)
                    next_hand = remaining_cards(current_cpu.hand, candidate_consumed_cards(candidate))
                    next_cpu = temporary_cpu_with_hand(current_cpu, next_hand)
                    visit(next_cpu, candidate_strength(candidate, room), selected_desc + [candidate])
                    if len(results) >= GOLD_PLAN_MAX_RESULTS_PER_COUNT * 2:
                        return

            visit(reserved_cpu, last_strength, [])
            if len(results) >= GOLD_PLAN_MAX_RESULTS_PER_COUNT * 2:
                break

    results.sort(key=gold_plan_score, reverse=True)
    return results[:GOLD_PLAN_MAX_RESULTS_PER_COUNT]


def build_silver_plan(
    cpu: CpuPlayer,
    room,
    counts: Iterable[int] = SILVER_RALLY_COUNTS,
    validator: Optional[NumberValidator] = None,
    prefer_two_step_over_direct: bool = False,
) -> dict:
    validator = validator or gold_knowledge_number_validator
    count_tuple = tuple(count for count in counts if count in SILVER_RALLY_COUNTS)

    direct_tail = choose_gold_finish_tail(cpu, room, validator)
    if direct_tail and not prefer_two_step_over_direct:
        return finalize_silver_plan(cpu, room, direct_tail, 0)

    plans = [
        plan
        for rally_count in count_tuple
        for plan in search_same_count_silver_plans(cpu, room, rally_count, validator)
    ]
    if plans:
        plans.sort(key=silver_plan_score, reverse=True)
        return plans[0]
    if direct_tail:
        return finalize_silver_plan(cpu, room, direct_tail, 0)
    return {
        "steps": [],
        "remaining": cpu.hand[:],
        "completed": False,
        "rally_count": 0,
        "evaluation": {"score": 0},
    }


def search_same_count_silver_plans(
    cpu: CpuPlayer,
    room,
    rally_count: int,
    validator: NumberValidator,
) -> list[dict]:
    check_cpu_search_deadline(cpu)
    results = []
    seen_plans = set()
    last_candidates = silver_last_rally_candidates(cpu, room, rally_count, validator)

    for last in last_candidates:
        check_cpu_search_deadline(cpu)
        if len(results) >= SILVER_PLAN_SEARCH_RESULT_CAP:
            break
        reserved_hand = remaining_cards(cpu.hand, candidate_consumed_cards(last))
        reserved_cpu = temporary_cpu_with_hand(cpu, reserved_hand)
        last_strength = candidate_strength(last, room)

        def visit(current_cpu: CpuPlayer, bound_strength: int, selected_desc: list[dict]) -> None:
            check_cpu_search_deadline(current_cpu)
            if len(results) >= SILVER_PLAN_SEARCH_RESULT_CAP:
                return
            tail = choose_gold_finish_tail(current_cpu, room_without_field(room), validator)
            if tail:
                sequence = list(reversed(selected_desc)) + [last] + tail
                if silver_sequence_rally_step_count(sequence) <= SILVER_PLAN_MAX_RALLY_STEPS:
                    key = tuple(candidate_fingerprint(candidate) for candidate in sequence)
                    if key not in seen_plans:
                        seen_plans.add(key)
                        results.append(finalize_silver_plan(cpu, room, sequence, rally_count))
                return

            if len(selected_desc) >= SILVER_PLAN_MAX_RALLY_STEPS - 1:
                return

            branch_candidates = silver_plan_candidates(current_cpu, room_without_field(room), (rally_count,), validator)
            branch_candidates = [
                candidate
                for candidate in branch_candidates
                if len(candidate.get("cards", [])) == rally_count
                and len(candidate_consumed_cards(candidate)) < len(current_cpu.hand)
                and candidate_strength(candidate, room) < bound_strength
                and not silver_candidate_uses_joker(candidate)
            ]
            branch_candidates = sorted(
                dedupe_candidates(branch_candidates),
                key=lambda candidate: silver_candidate_score(candidate, room),
                reverse=True,
            )[:GOLD_PLAN_MAX_BRANCH_CANDIDATES]

            for candidate in branch_candidates:
                check_cpu_search_deadline(current_cpu)
                if len(results) >= SILVER_PLAN_SEARCH_RESULT_CAP:
                    return
                next_hand = remaining_cards(current_cpu.hand, candidate_consumed_cards(candidate))
                next_cpu = temporary_cpu_with_hand(current_cpu, next_hand)
                visit(next_cpu, candidate_strength(candidate, room), selected_desc + [candidate])

        visit(reserved_cpu, last_strength, [])

    results.sort(key=silver_plan_score, reverse=True)
    return results[:GOLD_PLAN_MAX_RESULTS_PER_COUNT]


def silver_last_rally_candidates(
    cpu: CpuPlayer,
    room,
    rally_count: int,
    validator: NumberValidator,
) -> list[dict]:
    candidates = silver_plan_candidates(cpu, room, (rally_count,), validator)
    if rally_count == 1:
        candidates.extend(silver_single_joker_candidates(cpu, room))
    candidates = [
        candidate
        for candidate in candidates
        if len(candidate.get("cards", [])) == rally_count
        and len(candidate_consumed_cards(candidate)) < len(cpu.hand)
    ]
    return sorted(
        dedupe_candidates(candidates),
        key=lambda candidate: silver_candidate_score(candidate, room),
        reverse=True,
    )[:GOLD_PLAN_MAX_LAST_CANDIDATES]


def silver_plan_candidates(
    cpu: CpuPlayer,
    room,
    counts: Iterable[int],
    validator: NumberValidator,
) -> list[dict]:
    count_tuple = tuple(counts)
    candidates = gold_plan_candidates(cpu, room, count_tuple, validator)
    return dedupe_candidates(candidates)


def silver_single_joker_candidates(cpu: CpuPlayer, room) -> list[dict]:
    joker = single_joker(cpu.hand)
    if joker is None:
        return []
    if len(cpu.hand) <= 1:
        return []
    if len(getattr(room, "field", []) or []) > 1:
        return []
    return [{
        "kind": "prime",
        "number": "X",
        "cards": [joker],
        "assigned_numbers": [],
        "ranks": (),
    }]


def finalize_silver_plan(
    cpu: CpuPlayer,
    room,
    sequence: list[dict],
    rally_count: int,
) -> dict:
    temp_cpu = temporary_cpu_with_hand(cpu, cpu.hand[:])
    steps = []
    for candidate in sequence:
        role = candidate.get("role", f"rally-{rally_count}" if rally_count else "finish")
        append_gold_plan_step(steps, temp_cpu, candidate, role=role)
        temp_cpu.hand = remaining_cards(temp_cpu.hand, candidate_consumed_cards(candidate))
    plan = {
        "steps": steps,
        "remaining": temp_cpu.hand,
        "completed": not temp_cpu.hand,
        "rally_count": rally_count,
        "last_rally_strength": gold_plan_last_rally_strength(steps, room),
    }
    plan["evaluation"] = evaluate_silver_plan(plan, room)
    return plan


def silver_sequence_rally_step_count(sequence: list[dict]) -> int:
    return sum(1 for candidate in sequence if str(candidate.get("role", "rally")).startswith("rally"))


def evaluate_silver_plan(plan: dict, room) -> dict:
    steps = [step for step in plan.get("steps", []) if step.get("role") != "cut"]
    step_count = len(steps)
    trump = next(
        (step for step in reversed(steps) if str(step.get("role", "")).startswith("rally-")),
        steps[-1] if steps else None,
    )
    tier = silver_trump_tier(trump) if trump else 0
    strength = candidate_strength(trump, room) if trump else -1
    score_tuple = (-step_count, tier, strength)
    return {
        "score": step_count * -1000000 + tier * 1000 + min(strength, 999),
        "score_tuple": score_tuple,
        "step_count": step_count,
        "trump_tier": tier,
        "trump_strength": strength,
    }


def silver_plan_score(plan: dict) -> tuple:
    evaluation = plan.get("evaluation", {})
    score_tuple = evaluation.get("score_tuple")
    if score_tuple is None:
        score_tuple = (-len(plan.get("steps", [])), 0, -1)
    return (
        1 if plan.get("completed") else 0,
        *score_tuple,
    )


def silver_candidate_score(candidate: dict, room) -> tuple:
    return (
        silver_trump_tier(candidate),
        candidate_strength(candidate, room),
        -len(candidate_consumed_cards(candidate)),
    )


def silver_trump_tier(candidate: Optional[dict]) -> int:
    if not candidate:
        return 0
    if candidate.get("number") == "X" and len(candidate.get("cards", [])) == 1:
        return 12
    try:
        number = int(candidate.get("number"))
    except (TypeError, ValueError):
        return 0
    count = len(candidate.get("cards", []))
    thresholds = (
        (2, 1313, 11),
        (3, 131311, 10),
        (4, 13111211, 9),
        (2, 1213, 8),
        (3, 131011, 7),
        (4, 13101211, 6),
        (3, 61211, 5),
        (4, 8101211, 4),
    )
    for threshold_count, threshold, tier in thresholds:
        if count == threshold_count and number >= threshold:
            return tier
    general = {1: 3, 2: 2, 3: 1, 4: 0}
    return general.get(count, 0)


def silver_waiting_to_finish(cpu: CpuPlayer, room, validator: NumberValidator) -> bool:
    return bool(choose_gold_finish_tail(cpu, room_without_field(room), validator))


def gold_last_rally_candidates(
    cpu: CpuPlayer,
    room,
    rally_count: int,
    validator: NumberValidator,
    joker_trump: bool,
) -> list[dict]:
    if joker_trump:
        candidates = [
            candidate for candidate in joker_prime_candidates_for_count(cpu, room, rally_count, validator)
            if any(is_joker(card) for card in candidate.get("cards", []))
        ]
    else:
        candidates = gold_plan_candidates(cpu, room, [rally_count], validator)
    candidates = [
        candidate for candidate in candidates
        if len(candidate.get("cards", [])) == rally_count
        and len(candidate_consumed_cards(candidate)) < len(cpu.hand)
    ]
    return sorted(
        dedupe_candidates(candidates),
        key=lambda candidate: gold_plan_candidate_score(candidate, room),
        reverse=True,
    )[:gold_last_candidate_cap(cpu)]


def choose_gold_finish_candidate(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
) -> Optional[dict]:
    finish_candidates = direct_gold_finish_candidates(cpu, room, validator)
    field_count = len(getattr(room, "field", []) or [])
    if len(cpu.hand) == 1 and is_joker(cpu.hand[0]) and field_count <= 1:
        finish_candidates.append({
            "kind": "prime",
            "number": "X",
            "cards": cpu.hand[:],
            "assigned_numbers": [],
            "ranks": (),
        })
    finish_candidates.extend(joker_prime_finish_candidates(cpu, room, validator))
    if not finish_candidates:
        return None
    return max(finish_candidates, key=lambda candidate: gold_plan_candidate_score(candidate, room))


def direct_gold_finish_candidates(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
) -> list[dict]:
    candidates = direct_prime_finish_candidates(cpu, room, validator)
    candidates.extend(direct_composite_finish_candidates(cpu, room))
    return dedupe_candidates(candidates)


def direct_prime_finish_candidates(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
) -> list[dict]:
    non_joker_ranks = [int(card.get("rank", 0)) for card in cpu.hand if not is_joker(card)]
    joker_count = len(cpu.hand) - len(non_joker_ranks)
    max_cards = int(getattr(getattr(room, "rule", None), "normal_finish_max_hand_size", 0) or 9)
    templates = registered_prime_templates_for_hand(
        cpu.registered_primes,
        non_joker_ranks,
        joker_count=joker_count,
        max_cards=max_cards,
    )
    hand_ids = {card.get("card_id") for card in cpu.hand}
    candidates = []
    seen_numbers = set()
    for number, ranks in templates:
        if number in seen_numbers:
            continue
        if not validator(number, cpu, getattr(room, "rule", None)):
            continue
        if joker_count:
            realization = cards_for_ranks_with_jokers(cpu.hand, ranks)
            if realization is None:
                continue
            cards = realization["cards"]
            assigned_numbers = realization["assigned_numbers"]
        else:
            cards = cards_for_ranks(cpu.hand, ranks)
            if cards is None:
                continue
            assigned_numbers = []
        if {card.get("card_id") for card in cards} != hand_ids:
            continue
        if not beats_field(number, len(cards), room):
            continue
        seen_numbers.add(number)
        candidates.append({
            "kind": "prime",
            "number": number,
            "cards": cards,
            "assigned_numbers": assigned_numbers,
            "ranks": ranks,
        })
    return candidates


@lru_cache(maxsize=32)
def direct_composite_templates(entries: tuple, values: tuple[int, ...]) -> dict[tuple[int, ...], tuple[tuple[int, tuple[int, ...], object], ...]]:
    by_value = {}
    for entry in entries:
        by_value.setdefault(entry.value, []).append(entry)

    by_signature = {}
    for value in values:
        for visible_ranks in registered_value_encodings(value, max_cards=4):
            if not 2 <= len(visible_ranks) <= 4:
                continue
            for entry in by_value.get(value, []):
                material_ranks = tuple(
                    rank
                    for token in entry.expression_tokens
                    if token.kind == "cards"
                    for rank in token.ranks
                )
                signature = tuple(sorted(visible_ranks + material_ranks))
                by_signature.setdefault(signature, []).append((value, visible_ranks, entry))
    return {
        signature: tuple(dict.fromkeys(templates))
        for signature, templates in by_signature.items()
    }


def direct_composite_finish_candidates(cpu: CpuPlayer, room) -> list[dict]:
    if not getattr(getattr(room, "rule", None), "allow_composite", False):
        return []
    if any(is_joker(card) for card in cpu.hand):
        return []

    signature = tuple(sorted(int(card.get("rank", 0)) for card in cpu.hand))
    templates = direct_composite_templates(
        tuple(cpu.registered_composite_entries),
        tuple(sorted(cpu.registered_composites)),
    ).get(signature, ())
    hand_ids = {card.get("card_id") for card in cpu.hand}
    candidates = []
    for value, visible_ranks, entry in templates:
        visible_cards = cards_for_ranks(cpu.hand, visible_ranks)
        if visible_cards is None:
            continue
        material = material_for_composite_entry(cpu.hand, entry, visible_cards)
        if material is None:
            continue
        candidate = {
            "kind": "composite",
            "number": value,
            "cards": visible_cards,
            "assigned_numbers": [],
            "consume_cards": material["cards"],
            "composite_tokens": material["tokens"],
            "composite_assigned_numbers": [],
            "expression": material.get("expression", ""),
            "expression_source": material.get("source", "registered"),
            "ranks": visible_ranks,
        }
        if {card.get("card_id") for card in candidate_consumed_cards(candidate)} != hand_ids:
            continue
        if beats_field(value, len(visible_cards), room):
            candidates.append(candidate)
    return candidates


def choose_gold_finish_tail(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
) -> list[dict]:
    finish = choose_gold_finish_candidate(cpu, room, validator)
    if finish is not None:
        finish["role"] = "finish"
        return [finish]

    for cut in gold_special_cut_candidates(cpu, room):
        after_cut = temporary_cpu_with_hand(cpu, remaining_cards(cpu.hand, candidate_consumed_cards(cut)))
        finish = choose_gold_finish_candidate(after_cut, room, validator)
        if finish is not None:
            cut["role"] = "cut"
            finish["role"] = "finish"
            return [cut, finish]
    return []


def choose_gold_cut_candidate(cpu: CpuPlayer, room) -> Optional[dict]:
    candidates = gold_special_cut_candidates(cpu, room)
    return candidates[0] if candidates else None


def gold_special_cut_candidates(cpu: CpuPlayer, room) -> list[dict]:
    candidates = []
    cut = choose_57_cut(cpu.hand, room)
    if cut is not None and len(cut["cards"]) < len(cpu.hand):
        candidates.append({
            "kind": "prime",
            "number": 57,
            "cards": cut["cards"],
            "assigned_numbers": cut.get("assigned_numbers", []),
            "ranks": (5, 7),
        })

    joker = single_joker(cpu.hand)
    field_count = len(getattr(room, "field", []) or [])
    if joker is not None and field_count <= 1 and len(cpu.hand) > 1:
        candidates.append({
            "kind": "prime",
            "number": "X",
            "cards": [joker],
            "assigned_numbers": [],
            "ranks": (),
        })
    return candidates


def gold_large_finish_split_candidates(
    cpu: CpuPlayer,
    room,
    rally_count: int,
    bound_strength: int,
    validator: NumberValidator,
) -> list[tuple[dict, list[dict]]]:
    """Find a rally-sized opening whose complement is a larger finishing tail."""
    finish_count = len(cpu.hand) - rally_count
    if finish_count <= rally_count:
        return []

    results = []
    for finish_tail in gold_finish_tails_for_consumed_count(cpu, room, finish_count, validator):
        consumed = [card for step in finish_tail for card in candidate_consumed_cards(step)]
        remaining = remaining_cards(cpu.hand, consumed)
        if len(remaining) != rally_count:
            continue

        rally_cpu = temporary_cpu_with_hand(cpu, remaining)
        rallies = gold_plan_candidates(rally_cpu, room, [rally_count], validator)
        rallies = [
            candidate
            for candidate in rallies
            if len(candidate_consumed_cards(candidate)) == rally_count
            and candidate_strength(candidate, room) < bound_strength
        ]
        for rally in sorted(
            dedupe_candidates(rallies),
            key=lambda candidate: gold_plan_candidate_score(candidate, room),
            reverse=True,
        ):
            results.append((rally, finish_tail))
            if len(results) >= gold_branch_candidate_cap(cpu):
                return results
    return results


def gold_finish_tails_for_consumed_count(
    cpu: CpuPlayer,
    room,
    target_count: int,
    validator: NumberValidator,
) -> list[list[dict]]:
    if target_count < 1 or target_count >= len(cpu.hand):
        return []

    tails = []
    for finish in gold_finish_candidates(cpu, room, validator):
        if len(candidate_consumed_cards(finish)) != target_count:
            continue
        finish = dict(finish)
        finish["role"] = "finish"
        tails.append([finish])

    for cut in gold_special_cut_candidates(cpu, room):
        cut_cards = candidate_consumed_cards(cut)
        after_cut = temporary_cpu_with_hand(cpu, remaining_cards(cpu.hand, cut_cards))
        for finish in gold_finish_candidates(after_cut, room, validator):
            if len(cut_cards) + len(candidate_consumed_cards(finish)) != target_count:
                continue
            cut = dict(cut)
            cut["role"] = "cut"
            finish = dict(finish)
            finish["role"] = "finish"
            tails.append([cut, finish])

    seen = set()
    unique_tails = []
    for tail in tails:
        key = tuple(candidate_fingerprint(candidate) for candidate in tail)
        if key in seen:
            continue
        seen.add(key)
        unique_tails.append(tail)
    return unique_tails


def gold_finish_candidates(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
) -> list[dict]:
    max_cards = min(9, len([card for card in cpu.hand if not is_joker(card)]))
    candidates = gold_plan_candidates(cpu, room, range(1, max_cards + 1), validator)
    candidates.extend(
        candidate
        for count in range(1, max_cards + 1)
        for candidate in joker_prime_candidates_for_count(cpu, room, count, validator)
    )
    field_count = len(getattr(room, "field", []) or [])
    if len(cpu.hand) == 1 and is_joker(cpu.hand[0]) and field_count <= 1:
        candidates.append({
            "kind": "prime",
            "number": "X",
            "cards": cpu.hand[:],
            "assigned_numbers": [],
            "ranks": (),
        })
    return dedupe_candidates(candidates)


def joker_prime_finish_candidates(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
) -> list[dict]:
    if not any(is_joker(card) for card in cpu.hand):
        return []
    if len(cpu.hand) > 9:
        return []

    candidates = []
    for number in sorted(cpu.registered_primes):
        if not validator(number, cpu, getattr(room, "rule", None)):
            continue
        for ranks in registered_value_encodings(number, max_cards=9):
            if len(ranks) != len(cpu.hand):
                continue
            realization = cards_for_ranks_with_jokers(cpu.hand, ranks)
            if realization is None:
                continue
            if {card.get("card_id") for card in realization["cards"]} != {card.get("card_id") for card in cpu.hand}:
                continue
            if not beats_field(number, len(realization["cards"]), room):
                continue
            candidates.append({
                "kind": "prime",
                "number": number,
                "cards": realization["cards"],
                "assigned_numbers": realization["assigned_numbers"],
                "ranks": ranks,
            })
            break
    return candidates


def joker_prime_candidates_for_count(
    cpu: CpuPlayer,
    room,
    count: int,
    validator: NumberValidator,
) -> list[dict]:
    if not any(is_joker(card) for card in cpu.hand):
        return []
    if count < 1 or count > 9:
        return []

    candidates = []
    for number in sorted(cpu.registered_primes):
        if not validator(number, cpu, getattr(room, "rule", None)):
            continue
        for ranks in registered_value_encodings(number, max_cards=9):
            if len(ranks) != count:
                continue
            realization = cards_for_ranks_with_jokers(cpu.hand, ranks)
            if realization is None:
                continue
            if not any(is_joker(card) for card in realization["cards"]):
                continue
            if not beats_field(number, len(realization["cards"]), room):
                continue
            candidates.append({
                "kind": "prime",
                "number": number,
                "cards": realization["cards"],
                "assigned_numbers": realization["assigned_numbers"],
                "ranks": ranks,
            })
            break
    return candidates


def choose_gold_rally_candidate(
    cpu: CpuPlayer,
    room,
    rally_count: int,
    validator: NumberValidator,
) -> Optional[dict]:
    candidates = gold_plan_candidates(cpu, room, [rally_count], validator)
    candidates = [
        candidate for candidate in candidates
        if len(candidate.get("cards", [])) == rally_count
        and len(candidate_consumed_cards(candidate)) < len(cpu.hand)
    ]
    if not candidates:
        return None

    finishable = []
    for candidate in candidates:
        temp_cpu = temporary_cpu_with_hand(cpu, remaining_cards(cpu.hand, candidate_consumed_cards(candidate)))
        if choose_gold_finish_candidate(temp_cpu, room, validator) is not None:
            finishable.append(candidate)
    pool = finishable or candidates
    return min(pool, key=lambda candidate: gold_plan_candidate_score(candidate, room))


def gold_plan_candidates(
    cpu: CpuPlayer,
    room,
    counts: Iterable[int],
    validator: NumberValidator,
) -> List[dict]:
    candidates = knowledge_prime_candidates(cpu, room, validator, counts)
    candidates.extend(knowledge_composite_candidates(cpu, room, counts))
    return candidates


def gold_plan_candidate_score(candidate: dict, room) -> tuple:
    if candidate.get("number") == "X":
        return (10**100, 1)
    return (
        candidate_strength(candidate, room),
        -len(candidate_consumed_cards(candidate)),
    )


def append_gold_plan_step(steps: list[dict], cpu: CpuPlayer, candidate: dict, role: str) -> None:
    consume_cards = candidate_consumed_cards(candidate)
    step = dict(candidate)
    step["role"] = role
    step["remaining_before"] = len(cpu.hand)
    step["remaining_after"] = len(cpu.hand) - len(consume_cards)
    step["visible_count"] = len(candidate.get("cards", []))
    steps.append(step)


def finalize_gold_plan(
    cpu: CpuPlayer,
    room,
    sequence: list[dict],
    rally_count: int,
) -> dict:
    temp_cpu = temporary_cpu_with_hand(cpu, cpu.hand[:])
    steps = []
    for candidate in sequence:
        role = candidate.get("role", f"rally-{rally_count}")
        append_gold_plan_step(steps, temp_cpu, candidate, role=role)
        temp_cpu.hand = remaining_cards(temp_cpu.hand, candidate_consumed_cards(candidate))
    plan = {
        "steps": steps,
        "remaining": temp_cpu.hand,
        "completed": not temp_cpu.hand,
        "rally_count": rally_count,
        "last_rally_strength": gold_plan_last_rally_strength(steps, room),
    }
    plan["evaluation"] = evaluate_gold_plan(plan)
    return plan


def dedupe_candidates(candidates: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for candidate in candidates:
        key = candidate_fingerprint(candidate)
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out


def candidate_fingerprint(candidate: dict) -> tuple:
    return (
        candidate.get("kind"),
        candidate.get("number"),
        tuple(card.get("card_id") for card in candidate.get("cards", [])),
        tuple(card.get("card_id") for card in candidate.get("consume_cards", [])),
    )


def gold_plan_last_rally_strength(steps: list[dict], room) -> int:
    rally_steps = [step for step in steps if str(step.get("role", "")).startswith("rally-")]
    if not rally_steps:
        return -1
    return candidate_strength(rally_steps[-1], room)


def gold_plan_score(plan: dict) -> tuple:
    remaining_count = len(plan["remaining"])
    rally_steps = sum(1 for step in plan["steps"] if str(step.get("role", "")).startswith("rally-"))
    cut_steps = sum(1 for step in plan["steps"] if step.get("role") == "cut")
    evaluation_score = plan.get("evaluation", {}).get("score", 0)
    return (
        evaluation_score,
        1 if plan["completed"] else 0,
        -remaining_count,
        plan.get("last_rally_strength", -1),
        rally_steps,
        cut_steps,
        -len(plan["steps"]),
        plan["rally_count"],
    )


_GOLD_PLAN_EVALUATION_CONFIG = None


def gold_plan_evaluation_config() -> dict:
    global _GOLD_PLAN_EVALUATION_CONFIG
    if _GOLD_PLAN_EVALUATION_CONFIG is None:
        try:
            _GOLD_PLAN_EVALUATION_CONFIG = json.loads(GOLD_PLAN_EVALUATION_JSON.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            _GOLD_PLAN_EVALUATION_CONFIG = {
                "immediate_win_trump_strength": 100.0,
                "trump_strength": {},
                "resource_index": {},
            }
    return _GOLD_PLAN_EVALUATION_CONFIG


def evaluate_gold_plan(plan: dict) -> dict:
    config = gold_plan_evaluation_config()
    category = gold_plan_step_category(plan)
    x_role = gold_plan_x_role(plan)
    trump_strength = gold_plan_trump_strength_score(plan, config)
    resource_index = (
        config.get("resource_index", {})
        .get(category, {})
        .get(x_role)
    )
    if resource_index is None:
        resource_index = (
            config.get("resource_index", {})
            .get(category, {})
            .get("no_x", 1.0)
        )
    score = 100 - (100 - trump_strength) * float(resource_index)
    return {
        "score": round(score, 4),
        "trump_strength": trump_strength,
        "resource_index": resource_index,
        "step_category": category,
        "x_role": x_role,
    }


def gold_plan_step_category(plan: dict) -> str:
    steps = [
        step for step in plan.get("steps", [])
        if step.get("role") != "cut"
    ]
    if len(steps) == 1 and steps[0].get("role") == "finish":
        return "immediate"
    if len(steps) == 2 and steps[-1].get("role") == "finish":
        return "trump_finish"
    if len(steps) == 3:
        return "three_steps"
    if len(steps) == 4:
        return "four_steps"
    if len(steps) == 5:
        return "five_steps"
    if len(steps) == 6:
        return "six_steps"
    return "seven_or_more"


def gold_plan_x_role(plan: dict) -> str:
    steps = plan.get("steps", [])
    last_rally_index = max(
        (index for index, step in enumerate(steps) if str(step.get("role", "")).startswith("rally-")),
        default=None,
    )
    x_step_indices = [
        index for index, step in enumerate(steps)
        if step_uses_joker(step)
    ]
    if not x_step_indices:
        return "x_single_saved" if any(is_joker(card) for card in plan.get("remaining", [])) else "no_x"

    index = max(x_step_indices)
    role = steps[index].get("role")
    if role == "finish":
        return "x_finish"
    if role == "cut":
        return "x_single_saved"
    if last_rally_index is not None and index == last_rally_index:
        return "x_trump"
    if last_rally_index is not None and index == last_rally_index - 1:
        return "x_before_trump"
    return "x_early"


def step_uses_joker(step: dict) -> bool:
    return any(
        is_joker(card)
        for card in step.get("cards", []) + step.get("consume_cards", [])
    )


def gold_plan_trump_strength_score(plan: dict, config: dict) -> float:
    steps = plan.get("steps", [])
    if gold_plan_step_category(plan) == "immediate":
        return float(config.get("immediate_win_trump_strength", 100.0))
    trump_step = next(
        (step for step in reversed(steps) if str(step.get("role", "")).startswith("rally-")),
        None,
    )
    if trump_step is None:
        trump_step = next((step for step in reversed(steps) if step.get("role") == "finish"), None)
    if trump_step is None:
        return 0.0
    number = trump_step.get("number")
    if number == "X":
        return 100.0
    try:
        value = int(number)
    except (TypeError, ValueError):
        return 0.0
    count = str(len(trump_step.get("cards", [])))
    table = config.get("trump_strength", {}).get(count)
    if not table:
        return 0.0
    score = float(table.get("default", 0.0))
    for threshold in table.get("thresholds", []):
        if value >= int(threshold.get("value", 0)):
            score = float(threshold.get("score", score))
    return score


def knowledge_prime_candidates(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
    counts: Iterable[int],
) -> List[dict]:
    candidates = []
    count_set = {count for count in counts if count > 0}
    values = tuple(sorted(cpu.registered_primes))
    max_cards = (
        PLATINUM_MAX_KNOWLEDGE_CARDS
        if getattr(cpu, "cpu_key", "") == "platinum_planner"
        else 9
    )
    if (
        max_cards == PLATINUM_MAX_KNOWLEDGE_CARDS
        and getattr(cpu, "prime_template_index_values", ()) == values
    ):
        index = cpu.prime_template_index
    else:
        index = registered_prime_template_index(values, max_cards=max_cards)
    for count in sorted(count_set):
        for number, ranks in index.templates_by_card_count.get(count, ()):
            if not validator(number, cpu, getattr(room, "rule", None)):
                continue
            cards = cards_for_ranks(cpu.hand, ranks)
            if cards is None:
                continue
            if not beats_field(number, len(cards), room):
                continue
            candidates.append({
                "kind": "prime",
                "number": number,
                "cards": cards,
                "assigned_numbers": [],
                "ranks": ranks,
            })
    return candidates


def knowledge_composite_candidates(
    cpu: CpuPlayer,
    room,
    counts: Iterable[int],
) -> List[dict]:
    if not getattr(getattr(room, "rule", None), "allow_composite", False):
        return []
    max_visible_cards = (
        9
        if getattr(getattr(room, "rule", None), "key", None) == "composite-practice-11-n"
        else 4
    )
    count_set = {count for count in counts if 1 <= count <= max_visible_cards}
    if not count_set:
        return []

    entries_by_value: dict[int, list] = {}
    for entry in cpu.registered_composite_entries:
        entries_by_value.setdefault(entry.value, []).append(entry)

    candidates = []
    for value in sorted(set(cpu.registered_composites) | set(entries_by_value)):
        for visible_ranks in registered_value_encodings(value, max_cards=max_visible_cards):
            if len(visible_ranks) not in count_set:
                continue
            visible_cards = cards_for_ranks(cpu.hand, visible_ranks)
            if visible_cards is None:
                continue
            material = material_for_composite_entries(
                cpu.hand,
                entries_by_value.get(value, []),
                visible_cards,
            )
            if material is None:
                continue
            if not beats_field(value, len(visible_cards), room):
                continue
            candidates.append({
                "kind": "composite",
                "number": value,
                "cards": visible_cards,
                "assigned_numbers": [],
                "consume_cards": material["cards"],
                "composite_tokens": material["tokens"],
                "composite_assigned_numbers": [],
                "expression": material.get("expression", ""),
                "expression_source": material.get("source", "registered"),
                "ranks": visible_ranks,
            })
            break
    return candidates


def strongest_candidates_by_count(candidates: List[dict], room) -> dict[int, dict]:
    trumps = {}
    for candidate in candidates:
        count = len(candidate["cards"])
        current = trumps.get(count)
        if current is None or candidate_strength(candidate, room) > candidate_strength(current, room):
            trumps[count] = candidate
    return trumps


def strongest_trumps_by_count(cpu: CpuPlayer, room, validator: NumberValidator) -> dict[int, dict]:
    max_cards = min(9, len([card for card in cpu.hand if not is_joker(card)]))
    counts = range(1, max_cards + 1)
    candidates = knowledge_prime_candidates(cpu, room_without_field(room), validator, counts)
    candidates.extend(knowledge_composite_candidates(cpu, room_without_field(room), range(2, 5)))
    joker = single_joker(cpu.hand)
    if joker is not None:
        candidates.append({
            "kind": "joker_cut",
            "number": float("inf"),
            "cards": [joker],
            "assigned_numbers": [],
            "ranks": (),
        })
    return strongest_candidates_by_count(candidates, room)


def gold_candidate_score(
    cpu: CpuPlayer,
    room,
    candidate: dict,
    trumps: dict[int, dict],
    validator: NumberValidator,
) -> tuple:
    count = len(candidate["cards"])
    is_trump = same_candidate(trumps.get(count), candidate)
    remaining = remaining_cards(cpu.hand, candidate_consumed_cards(candidate))
    has_followup = has_remaining_known_play(cpu, room, remaining, validator)
    return (
        1 if has_followup else 0,
        0 if is_trump else 1,
        1 if candidate.get("kind") == "prime" else 0,
        count,
        candidate_strength(candidate, room),
    )


def candidate_strength(candidate: dict, room) -> int:
    if candidate.get("kind") == "joker_cut":
        return 10**100
    if candidate.get("number") == "X":
        return 10**100
    number = int(candidate["number"])
    return -number if getattr(room, "reverse_order", False) else number


def same_candidate(left: Optional[dict], right: Optional[dict]) -> bool:
    if left is None or right is None:
        return False
    return (
        left.get("kind") == right.get("kind")
        and left.get("number") == right.get("number")
        and {card.get("card_id") for card in left.get("cards", [])}
        == {card.get("card_id") for card in right.get("cards", [])}
    )


def has_remaining_known_play(
    cpu: CpuPlayer,
    room,
    remaining: List[Card],
    validator: NumberValidator,
) -> bool:
    temp_cpu = temporary_cpu_with_hand(cpu, remaining)
    empty_room = room_without_field(room)
    max_cards = min(9, len([card for card in remaining if not is_joker(card)]))
    counts = range(1, max_cards + 1)
    return bool(
        knowledge_prime_candidates(temp_cpu, empty_room, validator, counts)
        or knowledge_composite_candidates(temp_cpu, empty_room, counts)
    )


def candidate_to_action(candidate: dict) -> CpuAction:
    if candidate.get("kind") == "composite":
        return CpuAction("play_composite", {
            "selected": {
                "cards": candidate["cards"],
                "assigned_numbers": candidate.get("assigned_numbers", []),
            },
            "consume": {
                "cards": candidate.get("consume_cards", []),
            },
            "composite": {
                "tokens": candidate.get("composite_tokens", []),
                "assigned_numbers": candidate.get("composite_assigned_numbers", []),
            },
        })
    return CpuAction("play_prime", {
        "cards": candidate["cards"],
        "assigned_numbers": candidate.get("assigned_numbers", []),
    })


def candidate_consumed_cards(candidate: dict) -> List[Card]:
    return list({
        card.get("card_id"): card
        for card in candidate.get("cards", []) + candidate.get("consume_cards", [])
    }.values())


def material_for_composite_entry(
    hand: List[Card],
    entry,
    visible_cards: List[Card],
) -> Optional[dict]:
    excluded_ids = {card.get("card_id") for card in visible_cards}
    used_ids = set(excluded_ids)
    cards = []
    tokens = []
    for expression_token in entry.expression_tokens:
        if expression_token.kind == "op":
            tokens.append({
                "kind": "op",
                "op": "\u00d7" if expression_token.op == "*" else expression_token.op,
            })
            continue
        if expression_token.kind != "cards":
            return None
        for rank in expression_token.ranks:
            card = next(
                (
                    card for card in hand
                    if not is_joker(card)
                    and card.get("rank") == rank
                    and card.get("card_id") not in used_ids
                ),
                None,
            )
            if card is None:
                return None
            used_ids.add(card.get("card_id"))
            cards.append(card)
            tokens.append({"kind": "card", "card_id": card.get("card_id")})
    return {
        "cards": cards,
        "tokens": tokens,
        "expression": getattr(entry, "expression", ""),
        "source": "registered",
    }


def material_for_composite_entries(
    hand: List[Card],
    entries: Iterable,
    visible_cards: List[Card],
) -> Optional[dict]:
    for entry in entries:
        material = material_for_composite_entry(hand, entry, visible_cards)
        if material is not None:
            return material
    return None


def cards_for_ranks(hand: List[Card], ranks: tuple[int, ...]) -> Optional[List[Card]]:
    available = [card for card in hand if not is_joker(card)]
    selected = []
    used_ids = set()
    for rank in ranks:
        card = next(
            (
                card for card in available
                if card.get("rank") == rank and card.get("card_id") not in used_ids
            ),
            None,
        )
        if card is None:
            return None
        selected.append(card)
        used_ids.add(card.get("card_id"))
    return selected


def cards_for_ranks_with_jokers(hand: List[Card], ranks: tuple[int, ...]) -> Optional[dict]:
    selected = []
    assigned_by_card_id = {}
    used_ids = set()
    jokers = [card for card in hand if is_joker(card)]

    for rank in ranks:
        card = next(
            (
                card for card in hand
                if not is_joker(card)
                and card.get("rank") == rank
                and card.get("card_id") not in used_ids
            ),
            None,
        )
        if card is None:
            card = next(
                (
                    joker for joker in jokers
                    if joker.get("card_id") not in used_ids
                ),
                None,
            )
            if card is None:
                return None
            assigned_by_card_id[card.get("card_id")] = str(rank)
        selected.append(card)
        used_ids.add(card.get("card_id"))

    return {
        "cards": selected,
        "assigned_numbers": [
            assigned_by_card_id[card.get("card_id")]
            for card in selected
            if is_joker(card)
        ],
    }


def remaining_cards(hand: List[Card], used_cards: List[Card]) -> List[Card]:
    remaining = hand[:]
    for card in used_cards:
        if card in remaining:
            remaining.remove(card)
    return remaining


def temporary_cpu_with_hand(cpu: CpuPlayer, hand: List[Card]) -> CpuPlayer:
    temp = CpuPlayer(name=cpu.name, player_id=cpu.id, cpu_key=cpu.cpu_key)
    temp.hand = hand
    temp.registered_primes = cpu.registered_primes
    temp.registered_composites = cpu.registered_composites
    temp.registered_composite_entries = cpu.registered_composite_entries
    temp.small_finish_index = cpu.small_finish_index
    temp.prime_template_index = cpu.prime_template_index
    temp.prime_template_index_values = cpu.prime_template_index_values
    temp.platinum_opening_phase = cpu.platinum_opening_phase
    temp.platinum_all_out_attempts = cpu.platinum_all_out_attempts
    temp.platinum_initial_hand_size = cpu.platinum_initial_hand_size
    temp.platinum_last_strategy_score = cpu.platinum_last_strategy_score
    temp.platinum_last_interference_score = cpu.platinum_last_interference_score
    temp.platinum_relaxed_opponent_min_hand_count = cpu.platinum_relaxed_opponent_min_hand_count
    temp.platinum_all_out_suppressed_opponent_min_hand_count = (
        cpu.platinum_all_out_suppressed_opponent_min_hand_count
    )
    temp.platinum_current_min_trump_strength = cpu.platinum_current_min_trump_strength
    temp.rng = cpu.rng
    temp.decision_time_budget_ms = cpu.decision_time_budget_ms
    temp.decision_deadline = cpu.decision_deadline
    return temp


def room_without_field(room):
    class EmptyFieldRoom:
        pass
    copy = EmptyFieldRoom()
    copy.rule = getattr(room, "rule", None)
    copy.field = []
    copy.last_number = None
    copy.reverse_order = getattr(room, "reverse_order", False)
    return copy


def choose_57_cut(hand: List[Card], room) -> Optional[dict]:
    field_count = len(getattr(room, "field", []) or [])
    if field_count not in (0, 2):
        return None
    cards = cards_for_ranks(hand, (5, 7))
    if cards is None:
        return None
    return {"cards": cards, "assigned_numbers": []}


def choose_prime_play(
    cpu: CpuPlayer,
    room,
    validator: Optional[NumberValidator] = None,
    max_cards: int = 3,
) -> Optional[dict]:
    validator = validator or default_number_validator
    best = None
    for cards in prime_play_candidates(cpu.hand, room, max_cards=max_cards):
        number = cards_number(cards)
        if number is None:
            continue
        if not beats_field(number, len(cards), room):
            continue
        if not validator(number, cpu, getattr(room, "rule", None)):
            continue
        payload = {"cards": cards, "assigned_numbers": [], "number": number}
        if best is None or cpu_candidate_sort_key(payload, room) < cpu_candidate_sort_key(best, room):
            best = payload

    if best is not None:
        return {
            "cards": best["cards"],
            "assigned_numbers": best["assigned_numbers"],
        }

    joker = single_joker(cpu.hand)
    field_count = len(getattr(room, "field", []) or [])
    if joker is not None and field_count <= 1:
        return {"cards": [joker], "assigned_numbers": []}

    return None


def prime_play_candidates(hand: List[Card], room, max_cards: int = 3) -> Iterable[List[Card]]:
    non_jokers = [card for card in hand if not is_joker(card)]
    required_count = len(getattr(room, "field", []) or [])
    if required_count:
        counts = [required_count]
    else:
        counts = range(1, min(max_cards, len(non_jokers)) + 1)

    for count in counts:
        if count < 1 or count > max_cards or count > len(non_jokers):
            continue
        seen_numbers = set()
        for cards_tuple in permutations(non_jokers, count):
            cards = list(cards_tuple)
            number = cards_number(cards)
            if number is None or number in seen_numbers:
                continue
            seen_numbers.add(number)
            yield cards


def cards_number(cards: List[Card]) -> Optional[int]:
    if not cards or any(is_joker(card) for card in cards):
        return None
    text = "".join(str(card.get("rank")) for card in cards)
    if text.startswith("0"):
        return None
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def beats_field(number: int, card_count: int, room) -> bool:
    field = getattr(room, "field", []) or []
    if not field:
        return True
    if card_count != len(field):
        return False

    field_number = getattr(room, "last_number", None)
    if field_number is None:
        return True

    if getattr(room, "reverse_order", False):
        return number < field_number
    return number > field_number


def default_number_validator(number: int, cpu: CpuPlayer, rule) -> bool:
    prime_rule = getattr(rule, "prime_rule", PrimeRule.NORMAL)
    if prime_rule is PrimeRule.REGISTERED:
        return cpu.can_use_registered_prime(number)
    if prime_rule is PrimeRule.TETRAD:
        return is_twin_quadruplet_prime(number)
    if prime_rule is PrimeRule.SEMIPRIME:
        return is_semiprime(number)
    return is_prime(number)


def cpu_candidate_sort_key(payload: dict, room) -> tuple:
    number = payload["number"]
    if getattr(room, "reverse_order", False):
        return (len(payload["cards"]), -number)
    return (len(payload["cards"]), number)


def is_joker(card: Card) -> bool:
    return bool(card.get("is_joker")) or card.get("suit") == "X"


def single_joker(hand: List[Card]) -> Optional[Card]:
    for card in hand:
        if is_joker(card):
            return card
    return None


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


@lru_cache(maxsize=1)
def fish_extra_prime_values() -> tuple[int, ...]:
    material_values = fish_extra_prime_values_from_materials()
    if material_values:
        return material_values

    values = []
    number = 2
    while len(values) < FISH_EXTRA_343_PRIME_COUNT:
        if "343" in str(number) and is_prime(number):
            values.append(number)
        number += 1
    return tuple(values)


def fish_extra_prime_values_from_materials() -> tuple[int, ...]:
    for path in fish_343_material_paths():
        if path.exists():
            values = parse_fish_343_prime_table(path.read_text(encoding="utf-8-sig"))
            if values:
                return values
    return ()


def fish_343_material_paths() -> tuple[Path, ...]:
    server_dir = Path(__file__).resolve().parent
    candidates = [
        server_dir / "data" / "knowledge" / "fish_343_primes.txt",
        server_dir / "materials" / "343primes.txt",
        server_dir / "343primes.txt",
    ]
    for parent in server_dir.parents:
        candidates.append(parent / "materials" / "343primes.txt")
    return tuple(dict.fromkeys(candidates))


def parse_fish_343_prime_table(text: str) -> tuple[int, ...]:
    values = []
    seen = set()
    for raw_line in text.splitlines():
        token = raw_line.split(" ", 1)[0].strip().lower()
        if not token:
            continue
        try:
            value = fish_343_pattern_value(token)
        except ValueError:
            continue
        if "343" not in str(value) or value in seen:
            continue
        seen.add(value)
        values.append(value)
    return tuple(values)


def fish_343_pattern_value(pattern: str) -> int:
    parts = []
    for char in pattern:
        if char in FISH_343_TOKEN_VALUES:
            parts.append(FISH_343_TOKEN_VALUES[char])
        elif char.isdigit():
            parts.append(char)
        else:
            raise ValueError("invalid fish prime token")
    if not parts:
        raise ValueError("empty fish prime token")
    return int("".join(parts))


def is_twin_quadruplet_prime(n: int) -> bool:
    if n in {5, 7, 11, 13}:
        return True
    if not is_prime(n):
        return False
    for start in (n, n - 2, n - 6, n - 8):
        if start >= 2 and n in {start, start + 2, start + 6, start + 8}:
            if all(is_prime(value) for value in (start, start + 2, start + 6, start + 8)):
                return True
    return False


def is_semiprime(n: int) -> bool:
    if n < 4 or is_prime(n):
        return False
    for divisor in range(2, int(n**0.5) + 1):
        if n % divisor == 0:
            return is_prime(divisor) and is_prime(n // divisor)
    return False


CPU_PROFILES = {
    "basic": CpuProfile(
        key="basic",
        label="汎用テストCPU",
        description="弱めですが、通常・四つ子・半素数・登録制限の各ルールで最低限の動作確認に使えるCPUです。",
        knowledge=CpuKnowledgeSpec(source="sample", load_timing="registration"),
    ),
    "gold_planner": CpuProfile(
        key="gold_planner",
        label="ゴールドCPU",
        description="GOLD素数表だけを基本知識として参照し、切り札を温存しながら枚数別に候補を探す試作CPUです。",
        rule_keys=(
            "std-5-1",
            "std-7-1",
            "std-11-f",
            "std-11-f-c",
            "std-11-n-c",
            "std-11-n-no-c",
            "registered-11-n-assist",
            "neo-assist-11-n-unlimited",
            "half-7-1-c-assist",
        ),
        knowledge=CpuKnowledgeSpec(source="gold", load_timing="always"),
        action_selector=choose_gold_planning_cpu_action,
    ),
    "platinum_planner": CpuProfile(
        key="platinum_planner",
        label="プラチナCPU",
        description="プラチナ素数表と5手探索を使い、弱い手札の全出し・既知山札の回収・絶対的切り札への切替を行うCPUです。",
        rule_keys=(
            "std-11-n-c",
            "std-11-n-no-c",
            "registered-11-n-assist",
            "neo-assist-11-n-unlimited",
        ),
        knowledge=CpuKnowledgeSpec(
            source="sample_key",
            load_timing="always",
            sample_key="platinum_prime_table",
        ),
        action_selector=choose_platinum_planning_cpu_action,
    ),
    "silver_planner": CpuProfile(
        key="silver_planner",
        label="シルバーCPU",
        description="シルバー素数表を使い、浅いラリー戦術と偶数消費を優先するCPUです。",
        rule_keys=(
            "std-5-1",
            "std-7-1",
            "std-11-f",
            "std-11-f-c",
            "std-11-n-c",
            "std-11-n-no-c",
            "registered-11-n-assist",
            "neo-assist-11-n-unlimited",
            "half-7-1-c-assist",
        ),
        knowledge=CpuKnowledgeSpec(
            source="sample_key",
            load_timing="always",
            sample_key="silver_prime_table",
        ),
        action_selector=choose_silver_planning_cpu_action,
    ),
    "talkative_fish": CpuProfile(
        key="talkative_fish",
        label="饒舌な魚CPU",
        description="シルバー素数表に343入り素数を足し、刺身チャンスを優先するジョークCPUです。",
        rule_keys=(
            "std-5-1",
            "std-7-1",
            "std-11-f",
            "std-11-f-c",
            "std-11-n-c",
            "std-11-n-no-c",
            "registered-11-n-assist",
            "neo-assist-11-n-unlimited",
        ),
        knowledge=CpuKnowledgeSpec(
            source="fish_silver",
            load_timing="always",
            sample_key="silver_prime_table",
        ),
        action_selector=choose_talkative_fish_cpu_action,
    ),
    "composite_practice": CpuProfile(
        key="composite_practice",
        label="合成数練習CPU",
        description="5手以内の合成数分け切りを優先し、合成数上がり、3枚以下の素数上がり、強札温存、ドロー後のランダム合成数全出しの順で評価します。",
        rule_keys=("composite-practice-11-n",),
        knowledge=CpuKnowledgeSpec(
            source="sample_key",
            load_timing="always",
            sample_key="composite_practice_cpu_ge3",
        ),
        action_selector=choose_composite_practice_cpu_action,
    ),
}

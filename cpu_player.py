from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter
from functools import lru_cache
from itertools import combinations, permutations, product
import json
from math import comb
from pathlib import Path
import random
import secrets
import time
from typing import Callable, Iterable, List, Optional

from registered_primes import (
    generate_composite_expression_entries,
    registered_pattern_cards,
    registered_prime_template_index,
    registered_prime_templates_for_hand,
    registered_value_encodings,
)
from hnp_challenge import build_hnp_tokens, choose_hnp_permutation
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
DIAMOND_MAX_KNOWLEDGE_CARDS = 20
DIAMOND_PREFERRED_RALLY_COUNTS = (4, 6)
DIAMOND_DECISION_BUDGET_MS = 1500
DIAMOND_FOUR_OBAKE_TOKENS = (
    "kkkq", "kkjq", "kkqt", "kjkq", "kqqj", "kkqk",
    "kkkt", "kktj", "kqtj", "kkjk", "kqjj", "kqjk",
)
DIAMOND_FOUR_COUNTER_TOKENS = (
    "kkkq", "kkkt", "kkqk", "kkqt", "kkjk", "kkjq",
)
DIAMOND_FOUR_FIXED_COUNTER_MIN_VALUE = 13131112  # KKJQ
DIAMOND_FOUR_KKTJ_VALUE = 13131011
DIAMOND_FOUR_OTHER_CERTAIN_MAX_OPPONENT_HAND_SIZE = 10
DIAMOND_SIX_CERTAIN_MIN_VALUE = 131312121011  # KKQQTJ
DIAMOND_SIX_SOFT_MIN_VALUE = 91212101011  # 9QQTTJ
DIAMOND_CONDITIONAL_DEFAULT_MAX_RETURN_PROBABILITY = 0.30
DIAMOND_CONDITIONAL_DESPERATE_MAX_RETURN_PROBABILITY = 0.80
DIAMOND_RETURN_PROBABILITY_TRIALS = 768
DIAMOND_RECOVERY_PROBABILITY_TRIALS = 96
DIAMOND_RECOVERY_CERTAIN_MIN_PROBABILITY = 0.20
DIAMOND_REVOLUTION_RETURN_MAX_KJQJ_PROBABILITY = 0.20
DIAMOND_FOUR_PRE_TRUMP_MIN_VALUE = 8121011  # 8QTJ
DIAMOND_SIX_PRE_TRUMP_MIN_VALUE = 91212101011  # 9QQTTJ
DIAMOND_OPPONENT_RALLY_INFERENCE_STREAK = 1
DIAMOND_REVOLUTION_AVOID_COUNTS = frozenset({2, 4})
DIAMOND_OPENING_SECOND_MIN_TRUMP_STRENGTH = 95.0
DIAMOND_POST_ALL_OUT_STRONG_CONDITIONAL_MAX_RETURN_PROBABILITY = 0.30
DIAMOND_POST_ALL_OUT_CONDITIONAL_MIN_TRUMP_STRENGTH = 90.0
DIAMOND_POST_ALL_OUT_SOFT_MIN_TRUMP_STRENGTH = 80.0
DIAMOND_POST_ALL_OUT_MAX_RALLY_STEPS = 4
DIAMOND_POST_ALL_OUT_RESULT_CAP = 6
DIAMOND_POST_ALL_OUT_DRAW_RESPONSE_MIN_HAND_SIZE = 18
DIAMOND_POST_ALL_OUT_SEARCH_RESERVE_SECONDS = 0.50
DIAMOND_OPPONENT_ALL_OUT_DRAW_MIN_PLAN_STEPS = 3
DIAMOND_OPENING_SECOND_HNP_MIN_FIELD_COUNT = 8
DIAMOND_OPPONENT_ALL_OUT_TIER9_MIN_OPPONENT_HAND_SIZE = 18
DIAMOND_KQQJ_CONTEXTUAL_MAX_OPPONENT_HAND_SIZE = 20
DIAMOND_POST_ALL_OUT_KX_POLICY_MIN_HAND_SIZE = 18
DIAMOND_POST_ALL_OUT_CONTEXTS = frozenset({
    "opponent-all-out",
    "post-all-out-lead",
    "post-all-out-response",
    "hand-advantage-lead",
    "hand-advantage-interference",
})
ADVANCED_PLANNING_CPU_KEYS = frozenset({"platinum_planner", "diamond_planner"})
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
DIAMOND_OPENING_AUSO_TRUMP_RANGES = {
    3: (131111, 131311),       # KJJ .. KKJ
    4: (13101211, 13111211),  # KTQJ .. KJQJ
}
DIAMOND_OPENING_AUSO_NO_DRAW_RETURN_VALUES = frozenset({
    13101211,  # KTQJ
    13111013,  # KJTK
})
DIAMOND_OPENING_AUSO_ALL_OUT_TRUMP_TOKENS = frozenset({
    "kjj", "ktqj", "kjtk",
})
DIAMOND_OPENING_AUSO_BASE_SCORE = 95.0
PLATINUM_TOKEN_RANKS = {"t": 10, "j": 11, "q": 12, "k": 13}
PLATINUM_SMALL_TRUMP_TOKENS = frozenset({
    "kk", "kkj", "kqk", "kjj", "kjqj", "kjtk", "ktqj", "qqqj", "qk",
    "kq", "kj", "kt",
})


def gold_branch_candidate_cap(cpu: "CpuPlayer") -> int:
    cpu_key = getattr(cpu, "cpu_key", "")
    if cpu_key == "diamond_planner":
        return 8
    if cpu_key in ADVANCED_PLANNING_CPU_KEYS:
        return 12
    return GOLD_PLAN_MAX_BRANCH_CANDIDATES


def gold_last_candidate_cap(cpu: "CpuPlayer") -> int:
    cpu_key = getattr(cpu, "cpu_key", "")
    if cpu_key == "diamond_planner":
        return 12
    if cpu_key in ADVANCED_PLANNING_CPU_KEYS:
        return 16
    return GOLD_PLAN_MAX_LAST_CANDIDATES


def cpu_max_knowledge_cards(cpu: "CpuPlayer") -> int:
    if getattr(cpu, "cpu_key", "") == "diamond_planner":
        return DIAMOND_MAX_KNOWLEDGE_CARDS
    return PLATINUM_MAX_KNOWLEDGE_CARDS
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
            (), max_cards=cpu_max_knowledge_cards(self)
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
        self.diamond_focus_count: Optional[int] = None
        self.diamond_active_route: Optional[dict] = None
        self.diamond_last_route_kind = ""
        self.diamond_last_certainty = "unclassified"
        self.diamond_recovery_targets: tuple[int, ...] = ()
        self.diamond_last_preferred_counts: tuple[int, ...] = ()
        self.diamond_last_obake_counter_tokens: tuple[str, ...] = ()
        self.diamond_last_return_candidates: tuple[str, ...] = ()
        self.diamond_opponent_rally_count: Optional[int] = None
        self.diamond_opponent_rally_streak = 0
        self.diamond_last_observed_play_key: Optional[tuple] = None
        self.diamond_interference_mode_count: Optional[int] = None
        self.diamond_opponent_rally_threat = "none"
        self.diamond_opponent_estimated_max: Optional[int] = None
        self.diamond_last_estimated_opponent_faces: Optional[float] = None
        self.diamond_last_return_probability = 0.0
        self.diamond_last_recovery_certain_probability = 0.0
        self.diamond_last_recovery_guaranteed_kx = 0
        self.diamond_recovery_cache_key: Optional[tuple] = None
        self.diamond_last_opponent_kjqj_probability = 1.0
        self.diamond_revolution_strategy_active = False
        self.diamond_last_context = "opening-lead"
        self.diamond_last_plan_tier: Optional[int] = None
        self.diamond_last_finish_strength = 0
        self.diamond_opening_was_second: Optional[bool] = None
        self.diamond_opponent_ever_13_plus = False
        self.diamond_seen_cards: dict[str, Card] = {}
        self.diamond_public_response_signature_cache: dict[
            int,
            tuple[tuple[int, tuple[int, ...]], ...],
        ] = {}
        self.diamond_last_conditional_limit = (
            DIAMOND_CONDITIONAL_DEFAULT_MAX_RETURN_PROBABILITY
        )
        self.diamond_pending_full_recovery: Optional[dict] = None
        self.diamond_preserved_closeout_plan: Optional[dict] = None
        self.diamond_last_kx_policy: dict = {}
        self.diamond_pending_initial_all_out = False
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
            max_cards=cpu_max_knowledge_cards(self),
        )
        self.prime_template_index_values = sorted_values
        self.diamond_public_response_signature_cache = {}

    def can_use_registered_prime(self, n: int) -> bool:
        return n in self.registered_primes

    def replace_registered_composites(self, values: set[int], entries=()) -> None:
        self.registered_composites = set(values)
        self.registered_composite_entries = tuple(entries)
        self.diamond_public_response_signature_cache = {}

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
    cpu.diamond_focus_count = None
    cpu.diamond_active_route = None
    cpu.diamond_last_route_kind = ""
    cpu.diamond_last_certainty = "unclassified"
    cpu.diamond_recovery_targets = ()
    cpu.diamond_last_preferred_counts = ()
    cpu.diamond_last_obake_counter_tokens = ()
    cpu.diamond_last_return_candidates = ()
    cpu.diamond_opponent_rally_count = None
    cpu.diamond_opponent_rally_streak = 0
    cpu.diamond_last_observed_play_key = None
    cpu.diamond_interference_mode_count = None
    cpu.diamond_opponent_rally_threat = "none"
    cpu.diamond_opponent_estimated_max = None
    cpu.diamond_last_estimated_opponent_faces = None
    cpu.diamond_last_return_probability = 0.0
    cpu.diamond_last_recovery_certain_probability = 0.0
    cpu.diamond_last_recovery_guaranteed_kx = 0
    cpu.diamond_recovery_cache_key = None
    cpu.diamond_last_opponent_kjqj_probability = 1.0
    cpu.diamond_revolution_strategy_active = False
    cpu.diamond_last_context = "opening-lead"
    cpu.diamond_last_plan_tier = None
    cpu.diamond_last_finish_strength = 0
    cpu.diamond_opening_was_second = None
    cpu.diamond_opponent_ever_13_plus = False
    cpu.diamond_seen_cards = {}
    cpu.diamond_last_conditional_limit = (
        DIAMOND_CONDITIONAL_DEFAULT_MAX_RETURN_PROBABILITY
    )
    cpu.diamond_pending_full_recovery = None
    cpu.diamond_preserved_closeout_plan = None
    cpu.diamond_last_kx_policy = {}
    cpu.diamond_pending_initial_all_out = False


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
    if profile and profile.key in ADVANCED_PLANNING_CPU_KEYS:
        budget_ms = max(budget_ms, 1000)
    if profile and profile.key == "diamond_planner":
        budget_ms = max(budget_ms, DIAMOND_DECISION_BUDGET_MS)
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
        if profile and profile.key in ADVANCED_PLANNING_CPU_KEYS:
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


def diamond_tactical_context(cpu: CpuPlayer, room) -> str:
    """Classify the public hand-size/turn state used by Diamond policy."""
    initial = max(1, int(getattr(cpu, "platinum_initial_hand_size", 11)))
    own_count = len(cpu.hand)
    opponent_count = platinum_opponent_hand_count(cpu, room)
    responding = bool(getattr(room, "field", []) or [])
    own_expanded = (
        own_count > initial + 1
        or int(getattr(cpu, "platinum_all_out_attempts", 0)) > 0
    )
    opponent_expanded = (
        opponent_count is not None and opponent_count > initial + 1
    )

    if not own_expanded and not opponent_expanded:
        return "opening-second" if responding else "opening-lead"
    if not own_expanded and opponent_expanded:
        return "opponent-all-out"
    if own_expanded and opponent_expanded:
        return "post-all-out-response" if responding else "post-all-out-lead"
    if own_expanded and not opponent_expanded:
        return (
            "hand-advantage-interference"
            if responding
            else "hand-advantage-lead"
        )
    return "general-response" if responding else "general-lead"


def diamond_should_resume_opening_auso_after_failed_kamatoto(
    cpu: CpuPlayer,
    room,
    previous_context: str,
    current_context: str,
) -> bool:
    """Keep the saved trump -> finish route after the opener's failed all-out."""
    if (
        previous_context != "opening-lead"
        or current_context != "opponent-all-out"
        or (getattr(room, "field", []) or [])
    ):
        return False
    plan = getattr(cpu, "gold_active_plan", None)
    if not plan or not plan.get("diamond_opening_auso"):
        return False
    steps = list(plan.get("steps", []))
    index = int(getattr(cpu, "gold_plan_step_index", 0))
    if index <= 0 or index >= len(steps):
        return False
    trump = steps[index]
    if diamond_opening_auso_trump_score(trump) is None:
        return False
    return all(candidate_cards_available(step, cpu) for step in steps[index:])


def diamond_opening_auso_dominating_action(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
) -> Optional[CpuAction]:
    """Prefer a proved shorter closeout before resuming the saved Auso trump.

    The failed kamatoto policy is a no-draw policy, not a command to ignore an
    already available win.  Keep the saved trump -> finish route as the
    fallback, but first accept either a one-move finish or a certain complete
    plan that uses no more moves than the saved tail.  A same-length replan is
    only useful when the saved tail itself is not certain.
    """
    saved = getattr(cpu, "gold_active_plan", None)
    if not saved or not saved.get("diamond_opening_auso"):
        return None
    steps = list(saved.get("steps", []))
    index = int(getattr(cpu, "gold_plan_step_index", 0))
    if index < 0 or index >= len(steps):
        return None
    saved_tail_steps = [dict(step) for step in steps[index:]]
    saved_move_count = len(saved_tail_steps)

    finish = platinum_one_move_finish_candidate(cpu, room, validator)
    if finish is not None:
        clear_gold_active_plan(cpu)
        action = platinum_commit_play(cpu, candidate_to_action(finish))
        return diamond_record_action(
            cpu,
            action,
            room,
            "opening-auso-direct-finish",
            finish,
        )

    if saved_move_count <= 1:
        return None

    saved_tail = finalize_gold_plan(
        cpu,
        room_without_field(room),
        saved_tail_steps,
        int(saved.get("rally_count", 0) or 0),
    )
    saved_certainty = diamond_post_all_out_plan_certainty(
        saved_tail,
        cpu,
        room,
    )
    try:
        plans = diamond_post_all_out_plans(cpu, room, validator)
    except CpuSearchDeadline:
        # The saved route is deliberately retained as the timeout fallback.
        return None
    eligible = []
    for plan in plans:
        plan_steps = list(plan.get("steps", []))
        if (
            not plan.get("completed")
            or not plan_steps
            or len(plan_steps) > saved_move_count
            or diamond_post_all_out_plan_certainty(plan, cpu, room) != "certain"
            or not diamond_plan_lead_avoids_opponent_finish(plan, cpu, room)
        ):
            continue
        if len(plan_steps) == saved_move_count and saved_certainty == "certain":
            continue
        eligible.append(plan)
    if not eligible:
        return None

    best = max(
        eligible,
        key=lambda plan: (
            -len(plan.get("steps", [])),
            diamond_post_all_out_plan_sort_key(plan, cpu, room),
        ),
    )
    tier = diamond_post_all_out_plan_tier(best, cpu, room)
    best["diamond_tier"] = tier
    best["diamond_finish_strength"] = diamond_plan_finish_strength(best)
    set_gold_active_plan(cpu, best)
    action = play_next_gold_plan_step(cpu, room, validator)
    if action is None:
        set_gold_active_plan(cpu, saved)
        cpu.gold_plan_step_index = index
        return None
    action = platinum_commit_play(cpu, action)
    cpu.diamond_active_route = best
    cpu.diamond_last_plan_tier = tier
    cpu.diamond_last_finish_strength = best["diamond_finish_strength"]
    recorded = diamond_record_action(
        cpu,
        action,
        room,
        "opening-auso-short-certain-replan",
    )
    cpu.diamond_last_certainty = "certain"
    return recorded


def diamond_should_pass_after_opening_auso_four_trump_return(
    cpu: CpuPlayer,
    room,
) -> bool:
    """Preserve the three-card finish after KTQJ/KJTK is overtrumped once."""
    field = getattr(room, "field", []) or []
    if (
        len(field) != 4
        or str(getattr(room, "last_play_player_id", "")) == str(cpu.id)
        or len(cpu.hand) != 3
    ):
        return False
    plan = getattr(cpu, "gold_active_plan", None)
    if (
        not plan
        or not plan.get("diamond_opening_auso")
        or plan.get("diamond_opening_auso_return_pass_used")
    ):
        return False
    steps = list(plan.get("steps", []))
    index = int(getattr(cpu, "gold_plan_step_index", 0))
    if index <= 0 or index >= len(steps):
        return False
    try:
        previous_value = int(steps[index - 1].get("number"))
    except (TypeError, ValueError):
        return False
    if previous_value not in DIAMOND_OPENING_AUSO_NO_DRAW_RETURN_VALUES:
        return False
    remaining_ids = {
        str(card.get("card_id"))
        for step in steps[index:]
        for card in candidate_consumed_cards(step)
    }
    hand_ids = {str(card.get("card_id")) for card in cpu.hand}
    return remaining_ids == hand_ids


def diamond_update_opening_position_history(cpu: CpuPlayer, room) -> None:
    """Remember whether Diamond started second and whether the opener expanded.

    The large-prime HNP response is an opening-second policy.  Current hand size
    alone is insufficient because a first player can take a penalty above 12
    cards and later shrink below the threshold again.
    """
    if getattr(cpu, "diamond_opening_was_second", None) is None:
        first_player_id = getattr(room, "first_player_id", None)
        if first_player_id is not None:
            cpu.diamond_opening_was_second = str(first_player_id) != str(cpu.id)
        else:
            cpu.diamond_opening_was_second = bool(getattr(room, "field", []) or [])
    if not getattr(cpu, "diamond_opening_was_second", False):
        return
    opponent_count = platinum_opponent_hand_count(cpu, room)
    if opponent_count is not None and opponent_count >= 13:
        cpu.diamond_opponent_ever_13_plus = True


def choose_diamond_planning_cpu_action(
    cpu: CpuPlayer,
    room,
    validator: Optional[NumberValidator] = None,
) -> CpuAction:
    """Choose with Diamond certainty, interference, and recovery policies."""
    validator = gold_knowledge_number_validator
    all_out_attempts_before = int(
        getattr(cpu, "platinum_all_out_attempts", 0)
    )
    opening_phase_before = bool(getattr(cpu, "platinum_opening_phase", False))
    non_joker_count = len([card for card in cpu.hand if not is_joker(card)])
    preferred_counts = diamond_rally_count_order(cpu, non_joker_count)
    cpu.diamond_last_preferred_counts = preferred_counts
    cpu.diamond_active_route = None
    diamond_update_opening_position_history(cpu, room)
    previous_context = getattr(cpu, "diamond_last_context", "opening-lead")
    current_context = diamond_tactical_context(cpu, room)
    resume_opening_auso = diamond_should_resume_opening_auso_after_failed_kamatoto(
        cpu,
        room,
        previous_context,
        current_context,
    )
    if (
        current_context in DIAMOND_POST_ALL_OUT_CONTEXTS
        and previous_context not in DIAMOND_POST_ALL_OUT_CONTEXTS
        and not resume_opening_auso
    ):
        clear_gold_active_plan(cpu)
    cpu.diamond_last_context = current_context
    cpu.diamond_last_plan_tier = None
    cpu.diamond_last_finish_strength = 0
    cpu.diamond_last_kx_policy = {}

    pending_initial_all_out = diamond_pending_initial_all_out_action(cpu, room)
    if pending_initial_all_out is not None:
        return pending_initial_all_out

    if (
        getattr(cpu, "diamond_revolution_strategy_active", False)
        and not getattr(room, "reverse_order", False)
    ):
        cpu.diamond_revolution_strategy_active = False

    diamond_observe_opponent_rally(cpu, room)
    field = getattr(room, "field", []) or []
    field_count = len(field)

    if resume_opening_auso:
        dominating = diamond_opening_auso_dominating_action(
            cpu,
            room,
            validator,
        )
        if dominating is not None:
            return dominating
        candidate = getattr(cpu, "gold_active_plan", {}).get("steps", [])[int(
            getattr(cpu, "gold_plan_step_index", 0)
        )]
        action = play_next_gold_plan_step(cpu, room, validator)
        if action is not None:
            action = platinum_commit_play(cpu, action)
            return diamond_record_action(
                cpu,
                action,
                room,
                "opening-auso-after-failed-kamatoto",
                candidate,
            )

    if diamond_should_pass_after_opening_auso_four_trump_return(cpu, room):
        cpu.gold_active_plan["diamond_opening_auso_return_pass_used"] = True
        cpu.diamond_focus_count = None
        cpu.diamond_last_route_kind = "opening-auso-four-trump-return-pass"
        cpu.diamond_last_certainty = "unclassified"
        return CpuAction("pass")

    revolution_return = diamond_1729_revolution_return_candidate(cpu, room)
    if revolution_return is not None:
        clear_gold_active_plan(cpu)
        cpu.diamond_revolution_strategy_active = False
        action = platinum_commit_play(cpu, candidate_to_action(revolution_return))
        return diamond_record_action(
            cpu,
            action,
            room,
            "revolution-return-1729",
            revolution_return,
        )

    # After our planned 1729, do not enter the two/four-card exchanges that
    # expose us to 57 or another 1729. A direct finish is still always legal.
    if (
        getattr(cpu, "diamond_revolution_strategy_active", False)
        and getattr(room, "reverse_order", False)
        and field_count in DIAMOND_REVOLUTION_AVOID_COUNTS
    ):
        finish = platinum_one_move_finish_candidate(cpu, room, validator)
        if finish is not None:
            action = platinum_commit_play(cpu, candidate_to_action(finish))
            return diamond_record_action(cpu, action, room, "revolution-finish", finish)
        clear_gold_active_plan(cpu)
        cpu.diamond_focus_count = None
        cpu.diamond_last_certainty = "unclassified"
        cpu.diamond_last_route_kind = "revolution-avoid-rally"
        return CpuAction("pass")

    resumed_kx_policy = diamond_resume_post_all_out_kx_state(
        cpu,
        room,
        validator,
    )
    if resumed_kx_policy is not None:
        return resumed_kx_policy

    active = getattr(cpu, "gold_active_plan", None)
    if (
        not field
        and active
        and not diamond_plan_lead_avoids_opponent_finish(
            active,
            cpu,
            room,
            step_index=int(getattr(cpu, "gold_plan_step_index", 0)),
        )
    ):
        clear_gold_active_plan(cpu)
        active = None
    active_is_strong = bool(
        active
        and active_gold_plan_matches_field(cpu, field_count)
        and platinum_plan_is_strong(active, cpu, room)
    )
    if (
        field
        and getattr(cpu, "diamond_interference_mode_count", None) == field_count
        and not active_is_strong
    ):
        finish = platinum_one_move_finish_candidate(cpu, room, validator)
        if finish is not None:
            action = platinum_commit_play(cpu, candidate_to_action(finish))
            return diamond_record_action(cpu, action, room, "interference-finish", finish)
        closing = diamond_threat_closing_response_candidate(
            cpu,
            room,
            validator,
        )
        if closing is not None:
            clear_gold_active_plan(cpu)
            closing_tail = list(closing.pop("_diamond_closing_tail", []) or [])
            if closing_tail:
                child = temporary_cpu_with_hand(
                    cpu,
                    remaining_cards(cpu.hand, candidate_consumed_cards(closing)),
                )
                set_gold_active_plan(
                    cpu,
                    finalize_gold_plan(
                        child,
                        room_without_field(room),
                        closing_tail,
                        0,
                    ),
                )
            cpu.diamond_opponent_rally_streak = 0
            cpu.diamond_interference_mode_count = None
            cpu.diamond_opponent_rally_threat = "interrupted"
            action = platinum_commit_play(cpu, candidate_to_action(closing))
            return diamond_record_action(
                cpu,
                action,
                room,
                "threat-closing-response",
                closing,
            )
        interference = choose_platinum_interference_action(
            cpu,
            room,
            validator,
            active_plan=active,
            force=True,
        )
        if interference is not None:
            clear_gold_active_plan(cpu)
            cpu.diamond_opponent_rally_streak = 0
            cpu.diamond_interference_mode_count = None
            cpu.diamond_opponent_rally_threat = "interrupted"
            action = platinum_commit_play(cpu, interference)
            return diamond_record_action(
                cpu,
                action,
                room,
                "opponent-rally-interference",
            )

    post_all_out_kx = choose_diamond_post_all_out_kx_response_action(
        cpu,
        room,
        validator,
    )
    if post_all_out_kx is not None:
        return post_all_out_kx

    opening_second_hnp = diamond_opening_second_hnp_action(cpu, room, validator)
    if opening_second_hnp is not None:
        return opening_second_hnp

    contextual = choose_diamond_context_action(cpu, room, validator)
    if contextual is not None:
        return contextual

    revolution = diamond_1729_revolution_candidate(cpu, room)
    if (
        revolution is not None
        and not diamond_has_current_certain_trump(cpu, room)
        and diamond_lead_candidate_avoids_opponent_finish(
            revolution,
            cpu,
            room,
        )
    ):
        clear_gold_active_plan(cpu)
        cpu.diamond_revolution_strategy_active = True
        action = platinum_commit_play(cpu, candidate_to_action(revolution))
        return diamond_record_action(cpu, action, room, "revolution-1729", revolution)

    action = choose_platinum_planning_cpu_action(cpu, room, validator)
    avoided_opponent_finish_count = False
    forced_opponent_finish_count = False
    if not diamond_action_lead_avoids_opponent_finish(cpu, room, action):
        blocked_plan = getattr(cpu, "gold_active_plan", None)
        blocked_step_index = int(getattr(cpu, "gold_plan_step_index", 0))
        clear_gold_active_plan(cpu)
        cpu.platinum_all_out_attempts = all_out_attempts_before
        cpu.platinum_opening_phase = opening_phase_before
        if not getattr(room, "has_drawn", False) and getattr(room, "deck", []):
            cpu.diamond_last_route_kind = "opponent-finish-count-avoidance-draw"
            cpu.diamond_last_certainty = "unclassified"
            return CpuAction("draw")
        alternate = diamond_safe_alternate_lead_action(
            cpu,
            room,
            validator,
        )
        if alternate is not None:
            action = platinum_commit_play(cpu, alternate)
            avoided_opponent_finish_count = True
        else:
            cpu.gold_active_plan = blocked_plan
            cpu.gold_plan_step_index = blocked_step_index
            forced_opponent_finish_count = True
    active_plan = getattr(cpu, "gold_active_plan", None)
    focus_count = None
    if active_plan:
        rally_count = active_plan.get("rally_count")
        if rally_count in DIAMOND_PREFERRED_RALLY_COUNTS:
            focus_count = int(rally_count)
            cpu.diamond_active_route = active_plan
    if focus_count is None and action.kind in {"play_prime", "play_composite"}:
        cards = action.payload.get("cards") or action.payload.get("selected", {}).get("cards") or []
        if len(cards) in DIAMOND_PREFERRED_RALLY_COUNTS:
            focus_count = len(cards)

    cpu.diamond_focus_count = focus_count
    if active_plan:
        cpu.diamond_last_certainty = (
            diamond_post_all_out_plan_certainty(active_plan, cpu, room)
            if active_plan.get("diamond_tier") is not None
            else diamond_plan_certainty(active_plan, cpu, room)
        )
    else:
        candidate = diamond_candidate_from_action(action)
        cpu.diamond_last_certainty = (
            diamond_candidate_certainty(candidate, cpu, room)
            if candidate is not None
            else "unclassified"
        )
    opening_route_kind = None
    if active_plan:
        if active_plan.get("dual_wield_generalized"):
            opening_route_kind = "opening-dual-wield-generalized"
        elif active_plan.get("dual_wield"):
            opening_route_kind = "opening-dual-wield-fixed"
        elif active_plan.get("diamond_opening_forced_pass"):
            opening_route_kind = "opening-forced-pass"
        elif active_plan.get("diamond_opening_immediate_trump"):
            opening_route_kind = "opening-immediate-trump"
        elif active_plan.get("diamond_opening_auso"):
            opening_route_kind = "opening-auso"
    elif getattr(cpu, "diamond_pending_initial_all_out", False):
        opening_route_kind = "opening-auso-policy-all-out-draw"
    cpu.diamond_last_route_kind = (
        "opponent-finish-count-avoidance"
        if avoided_opponent_finish_count
        else "opponent-finish-count-plan-fallback"
        if forced_opponent_finish_count
        else (
            f"post-all-out-tier-{active_plan['diamond_tier']}"
            if active_plan and active_plan.get("diamond_tier") is not None
            else (
                opening_route_kind
                if opening_route_kind is not None
                else f"preferred-{focus_count}"
                if focus_count is not None
                else "platinum-fallback"
            )
        )
    )
    if active_plan and active_plan.get("diamond_tier") is not None:
        cpu.diamond_last_plan_tier = int(active_plan["diamond_tier"])
        cpu.diamond_last_finish_strength = int(
            active_plan.get("diamond_finish_strength", 0)
        )
    if (
        action.kind in {"play_prime", "play_composite"}
        and int(getattr(cpu, "platinum_all_out_attempts", 0))
        == all_out_attempts_before
    ):
        diamond_remember_cards(cpu, diamond_action_consumed_cards(action))
    return action


def diamond_record_action(
    cpu: CpuPlayer,
    action: CpuAction,
    room,
    route_kind: str,
    candidate: Optional[dict] = None,
) -> CpuAction:
    diamond_remember_cards(cpu, diamond_action_consumed_cards(action))
    cards = diamond_action_cards(action)
    cpu.diamond_focus_count = (
        len(cards) if len(cards) in DIAMOND_PREFERRED_RALLY_COUNTS else None
    )
    cpu.diamond_last_route_kind = route_kind
    if candidate is None:
        candidate = diamond_candidate_from_action(action)
    cpu.diamond_last_certainty = (
        diamond_candidate_certainty(candidate, cpu, room)
        if candidate is not None
        else "unclassified"
    )
    return action


def diamond_action_cards(action: CpuAction) -> list[Card]:
    if action.kind == "play_composite":
        return list(action.payload.get("selected", {}).get("cards", []) or [])
    return list(action.payload.get("cards", []) or [])


def diamond_action_consumed_cards(action: CpuAction) -> list[Card]:
    cards = diamond_action_cards(action)
    if action.kind == "play_composite":
        cards.extend(action.payload.get("consume", {}).get("cards", []) or [])
    return list({card.get("card_id"): card for card in cards}.values())


def diamond_remember_cards(cpu: CpuPlayer, cards: Iterable[Card]) -> None:
    seen = getattr(cpu, "diamond_seen_cards", None)
    if seen is None:
        seen = {}
        cpu.diamond_seen_cards = seen
    for card in cards:
        card_id = card.get("card_id")
        if card_id:
            seen[str(card_id)] = card


def diamond_candidate_from_action(action: CpuAction) -> Optional[dict]:
    if action.kind not in {"play_prime", "play_composite"}:
        return None
    cards = diamond_action_cards(action)
    if not cards:
        return None
    if action.kind == "play_composite":
        assigned_numbers = list(
            action.payload.get("selected", {}).get("assigned_numbers", []) or []
        )
    else:
        assigned_numbers = list(action.payload.get("assigned_numbers", []) or [])
    assigned = iter(assigned_numbers)
    ranks = []
    for card in cards:
        if is_joker(card):
            value = next(assigned, None)
            if value is None or value == "inf":
                return None
            ranks.append(int(value))
        else:
            ranks.append(int(card.get("rank", 0)))
    candidate = {
        "kind": "composite" if action.kind == "play_composite" else "prime",
        "number": int("".join(str(rank) for rank in ranks)),
        "cards": cards,
        "assigned_numbers": assigned_numbers,
        "ranks": tuple(ranks),
    }
    if action.kind == "play_composite":
        candidate.update({
            "consume_cards": list(
                action.payload.get("consume", {}).get("cards", []) or []
            ),
            "composite_tokens": list(
                action.payload.get("composite", {}).get("tokens", []) or []
            ),
            "composite_assigned_numbers": list(
                action.payload.get("composite", {}).get("assigned_numbers", []) or []
            ),
        })
    return candidate


def diamond_action_source_candidate(
    cpu: CpuPlayer,
    action: CpuAction,
) -> tuple[Optional[dict], bool]:
    """Recover the candidate just selected and whether it came from a known plan."""
    action_cards = diamond_action_cards(action)
    action_ids = {card.get("card_id") for card in action_cards}
    plan = getattr(cpu, "gold_active_plan", None)
    index = int(getattr(cpu, "gold_plan_step_index", 0)) - 1
    steps = list(plan.get("steps", [])) if plan else []
    if 0 <= index < len(steps):
        step = steps[index]
        step_ids = {card.get("card_id") for card in step.get("cards", []) or []}
        if step_ids == action_ids:
            return step, True
    return diamond_candidate_from_action(action), False


def diamond_action_lead_avoids_opponent_finish(
    cpu: CpuPlayer,
    room,
    action: CpuAction,
) -> bool:
    if action.kind not in {"play_prime", "play_composite"}:
        return True
    if getattr(room, "field", []) or []:
        return True
    opponent_count = platinum_opponent_hand_count(cpu, room)
    cards = diamond_action_cards(action)
    if opponent_count is None or len(cards) != opponent_count:
        return True
    candidate, from_plan = diamond_action_source_candidate(cpu, action)
    if (
        from_plan
        and candidate is not None
        and len(candidate_consumed_cards(candidate)) == len(cpu.hand)
    ):
        return True
    return diamond_lead_candidate_is_certain(candidate, cpu, room)


def diamond_safe_alternate_lead_action(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
) -> Optional[CpuAction]:
    """Choose a legal different-count lead after blocking a lethal count match."""
    if getattr(room, "field", []) or []:
        return None
    opponent_count = platinum_opponent_hand_count(cpu, room)
    non_joker_count = len([card for card in cpu.hand if not is_joker(card)])
    upper = min(cpu_max_knowledge_cards(cpu), non_joker_count)
    counts = tuple(range(1, upper + 1))
    if not counts:
        return None
    empty_room = room_without_field(room)
    candidates = gold_plan_candidates(cpu, empty_room, counts, validator)
    for count in counts:
        if count <= 9:
            candidates.extend(joker_prime_candidates_for_count(
                cpu,
                empty_room,
                count,
                validator,
            ))
    candidates.extend(gold_special_cut_candidates(cpu, empty_room))
    candidates = [
        candidate
        for candidate in dedupe_candidates(candidates)
        if candidate_is_playable(candidate, cpu, empty_room)
        and diamond_lead_candidate_avoids_opponent_finish(
            candidate,
            cpu,
            room,
        )
    ]
    if not candidates:
        return None
    best = max(
        candidates,
        key=lambda candidate: (
            1 if diamond_lead_candidate_is_certain(candidate, cpu, room) else 0,
            1 if diamond_lead_candidate_has_face_count_lock(candidate, cpu, room) else 0,
            1 if len(candidate.get("cards", []) or []) != opponent_count else 0,
            platinum_candidate_trump_strength(candidate),
            len(candidate_consumed_cards(candidate)),
            0 if step_uses_joker(candidate) else 1,
            candidate_strength(candidate, empty_room),
        ),
    )
    clear_gold_active_plan(cpu)
    return candidate_to_action(best)


def diamond_observe_opponent_rally(cpu: CpuPlayer, room) -> None:
    field = list(getattr(room, "field", []) or [])
    player_id = getattr(room, "last_play_player_id", None)
    if field:
        diamond_remember_cards(cpu, field)
    if not field or player_id is None or player_id == cpu.id:
        return
    play_key = (
        player_id,
        getattr(room, "last_play_hand_before", None),
        getattr(room, "last_number", None),
        tuple(card.get("card_id") for card in field),
    )
    if play_key == getattr(cpu, "diamond_last_observed_play_key", None):
        return
    cpu.diamond_last_observed_play_key = play_key
    count = len(field)
    if count == getattr(cpu, "diamond_opponent_rally_count", None):
        cpu.diamond_opponent_rally_streak += 1
    else:
        cpu.diamond_opponent_rally_count = count
        cpu.diamond_opponent_rally_streak = 1
    threat, estimated_max = diamond_infer_opponent_rally_threat(cpu, room, count)
    cpu.diamond_opponent_rally_threat = threat
    cpu.diamond_opponent_estimated_max = estimated_max
    cpu.diamond_interference_mode_count = (
        count
        if (
            cpu.diamond_opponent_rally_streak >= DIAMOND_OPPONENT_RALLY_INFERENCE_STREAK
            and threat == "certain-likely"
        )
        else None
    )


def diamond_infer_opponent_rally_threat(
    cpu: CpuPlayer,
    room,
    count: int,
) -> tuple[str, Optional[int]]:
    """Infer a same-count finishing threat after a single public play."""
    try:
        field_value = int(getattr(room, "last_number", 0) or 0)
    except (TypeError, ValueError):
        field_value = 0

    if count == 4 and field_value >= DIAMOND_FOUR_PRE_TRUMP_MIN_VALUE:
        feasible = [
            token for token in DIAMOND_FOUR_OBAKE_TOKENS
            if diamond_counter_token_is_physically_possible(cpu, room, token)
        ]
        if feasible:
            return (
                "certain-likely",
                max(platinum_token_value(token) for token in feasible),
            )

    if count == 6 and field_value >= DIAMOND_SIX_PRE_TRUMP_MIN_VALUE:
        available, joker_count = diamond_publicly_unaccounted_rank_counts(cpu, room)
        signature = platinum_token_ranks("kkqqtj")
        opponent_count = platinum_opponent_hand_count(cpu, room)
        capacity = (
            opponent_count + (1 if getattr(room, "deck", []) else 0)
            if opponent_count is not None
            else len(signature)
        )
        if (
            capacity >= len(signature)
            and diamond_requirement_is_physically_possible(
                available,
                joker_count,
                signature,
            )
        ):
            return "certain-likely", DIAMOND_SIX_CERTAIN_MIN_VALUE

    opponent_count = platinum_opponent_hand_count(cpu, room)
    estimated_max = diamond_estimated_opponent_max_value(cpu, room, count)
    if estimated_max is not None and opponent_count is not None and opponent_count <= 5:
        estimated_candidate = {
            "kind": "prime",
            "number": estimated_max,
            "cards": [{}] * count,
            "ranks": (),
        }
        if (
            platinum_candidate_trump_strength(estimated_candidate)
            >= DIAMOND_POST_ALL_OUT_SOFT_MIN_TRUMP_STRENGTH
        ):
            return "certain-likely", estimated_max

    return "ride-or-recover", field_value or None


def diamond_has_aaax(cpu: CpuPlayer) -> bool:
    ace_count = sum(
        1 for card in cpu.hand
        if not is_joker(card) and int(card.get("rank", 0)) == 1
    )
    joker_count = sum(1 for card in cpu.hand if is_joker(card))
    return ace_count >= 3 and joker_count >= 1


def diamond_exact_1729_candidate(cpu: CpuPlayer) -> Optional[dict]:
    cards = cards_for_ranks(cpu.hand, (1, 7, 2, 9))
    if cards is None:
        return None
    return {
        "kind": "prime",
        "number": 1729,
        "cards": cards,
        "assigned_numbers": [],
        "ranks": (1, 7, 2, 9),
    }


def diamond_1729_revolution_candidate(cpu: CpuPlayer, room) -> Optional[dict]:
    if getattr(room, "field", []) or getattr(room, "reverse_order", False):
        return None
    if getattr(getattr(room, "rule", None), "special_numbers_composite_only", False):
        return None
    if not diamond_has_aaax(cpu):
        return None
    return diamond_exact_1729_candidate(cpu)


def diamond_opponent_kjqj_probability(cpu: CpuPlayer, room) -> float:
    opponent_count = platinum_opponent_hand_count(cpu, room)
    if opponent_count is None:
        probability = 1.0
    elif opponent_count < 4:
        probability = 0.0
    else:
        probability = diamond_requirement_union_probability(
            cpu,
            room,
            (platinum_token_ranks("kjqj"),),
            sample_size=opponent_count,
            seed_value=13111211,
        )
    cpu.diamond_last_opponent_kjqj_probability = probability
    return probability


def diamond_1729_revolution_return_candidate(
    cpu: CpuPlayer,
    room,
) -> Optional[dict]:
    if (
        not getattr(cpu, "diamond_revolution_strategy_active", False)
        or not getattr(room, "reverse_order", False)
        or getattr(getattr(room, "rule", None), "special_numbers_composite_only", False)
        or diamond_has_aaax(cpu)
    ):
        return None
    candidate = diamond_exact_1729_candidate(cpu)
    if candidate is None or not candidate_is_playable(candidate, cpu, room):
        return None
    if len(candidate_consumed_cards(candidate)) == len(cpu.hand):
        return candidate
    if (
        diamond_opponent_kjqj_probability(cpu, room)
        >= DIAMOND_REVOLUTION_RETURN_MAX_KJQJ_PROBABILITY
    ):
        return None
    return candidate


def diamond_has_promising_normal_tactic(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
) -> bool:
    """R22 compatibility helper: only a current certain blocks initial 1729."""
    return diamond_has_current_certain_trump(cpu, room)


def choose_platinum_lead_action(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
) -> Optional[CpuAction]:
    active = getattr(cpu, "gold_active_plan", None)
    if (
        active
        and active.get("dual_wield")
        and getattr(cpu, "gold_plan_step_index", 0) == 1
    ):
        pass_tail = list(active.get("dual_wield_pass_tail") or [])
        if not pass_tail:
            fused = active.get("dual_wield_fused")
            if fused:
                pass_tail = [fused]
        if pass_tail:
            pass_plan = finalize_gold_plan(
                cpu,
                room_without_field(room),
                pass_tail,
                0,
            )
            if is_executable_gold_plan(pass_plan, cpu):
                pass_plan["dual_wield_pass_branch"] = True
                set_gold_active_plan(cpu, pass_plan)
                action = play_next_gold_plan_step(cpu, room, validator)
                if action is not None:
                    return platinum_commit_play(cpu, action)

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
    if (
        getattr(cpu, "diamond_revolution_strategy_active", False)
        and field_count in DIAMOND_REVOLUTION_AVOID_COUNTS
    ):
        return []
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
    force: bool = False,
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
    if not force and best_score < PLATINUM_INTERFERENCE_BORDER:
        return None
    eligible = (
        scored
        if force
        else [item for item in scored if item[0] >= PLATINUM_INTERFERENCE_BORDER]
    )
    held_trumps = platinum_available_trump_candidates(cpu, room, validator)
    if force and getattr(cpu, "cpu_key", "") == "diamond_planner":
        empty_room = room_without_field(room)
        held_certain = [
            candidate for candidate in held_trumps
            if diamond_candidate_certainty(candidate, cpu, empty_room) == "certain"
        ]

        def forced_interference_choice_key(item: tuple) -> tuple:
            candidate = item[-1]
            return (
                1 if platinum_candidate_preserves_any(
                    cpu, candidate, held_certain
                ) else 0,
                1 if platinum_candidate_preserves_any(
                    cpu, candidate, held_trumps
                ) else 0,
                1 if candidate.get("kind") == "prime" else 0,
                0 if step_uses_joker(candidate) else 1,
                item[1],
                item[0],
                -item[2],
            )

        best = max(eligible, key=forced_interference_choice_key)
    elif held_trumps:
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
    inferred_diamond_plan = (
        getattr(cpu, "cpu_key", "") == "diamond_planner"
        and getattr(cpu, "diamond_opponent_rally_threat", "none")
        == "certain-likely"
    )
    cpu.platinum_all_out_suppressed_opponent_min_hand_count = (
        count
        if not inferred_diamond_plan or count <= 5
        else None
    )
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


def diamond_opponent_max_kx_after_draw(cpu: CpuPlayer, room) -> int:
    """Conservative maximum K/X the opponent can use on its next response."""
    unaccounted_kx, opponent_count, unknown_deck_count = (
        platinum_unaccounted_kx_distribution(cpu, room)
    )
    draw_available = bool(getattr(room, "deck", []) or [])
    if unknown_deck_count > 0:
        capacity = opponent_count + (1 if draw_available else 0)
        return min(unaccounted_kx, capacity)

    current = min(unaccounted_kx, opponent_count)
    if not draw_available:
        return current
    known_bottom = list(getattr(room, "public_known_deck_bottom", []) or [])
    if not known_bottom:
        return current
    top = known_bottom[0]
    return current + int(
        is_joker(top) or int(top.get("rank", 0)) == 13
    )


def diamond_cpu_monopolizes_kx(cpu: CpuPlayer) -> bool:
    return sum(
        1 for card in cpu.hand
        if is_joker(card) or int(card.get("rank", 0)) == 13
    ) >= 6


def diamond_estimated_opponent_face_count_after_draw(
    cpu: CpuPlayer,
    room,
) -> float:
    """Estimate opponent T/J/Q/K/X after the one draw allowed before replying."""
    opponent_count = platinum_opponent_hand_count(cpu, room)
    if opponent_count is None:
        return float("inf")

    def card_key(card: Card):
        return card.get("card_id") or id(card)

    own_ids = {card_key(card) for card in cpu.hand}
    own_faces = sum(
        1 for card in cpu.hand
        if is_joker(card) or int(card.get("rank", 0)) >= 10
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
    public_faces = sum(
        1 for card in public_cards.values()
        if is_joker(card) or int(card.get("rank", 0)) >= 10
    )
    unaccounted_faces = max(0, 18 - own_faces - public_faces)
    if hasattr(room, "public_unknown_deck_count"):
        unknown_deck_count = max(0, int(room.public_unknown_deck_count))
    else:
        unknown_deck_count = max(
            0,
            len(getattr(room, "deck", []) or [])
            - len(getattr(room, "public_known_deck_bottom", []) or []),
        )

    if unknown_deck_count > 0:
        population = opponent_count + unknown_deck_count
        sample_size = opponent_count + (
            1 if getattr(room, "deck", []) else 0
        )
        estimate = (
            unaccounted_faces * min(sample_size, population) / population
            if population > 0
            else 0.0
        )
    else:
        estimate = float(min(unaccounted_faces, opponent_count))
        known_bottom = list(getattr(room, "public_known_deck_bottom", []) or [])
        if getattr(room, "deck", []) and known_bottom:
            top = known_bottom[0]
            estimate += float(
                is_joker(top) or int(top.get("rank", 0)) >= 10
            )
    cpu.diamond_last_estimated_opponent_faces = estimate
    return estimate


def diamond_candidate_display_face_count(candidate: Optional[dict]) -> int:
    if not candidate:
        return 0
    ranks = tuple(candidate.get("ranks") or ())
    if not ranks:
        token = platinum_candidate_token(candidate)
        if "x" in token:
            return token.count("x")
        try:
            ranks = platinum_token_ranks(token)
        except (KeyError, ValueError):
            return 0
    return sum(1 for rank in ranks if int(rank) >= 10)


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
    diamond = getattr(cpu, "cpu_key", "") == "diamond_planner"
    opening_plans = []
    fixed_dual_found = False
    if getattr(cpu, "platinum_opening_phase", True) and len(cpu.hand) == 11:
        dual = build_platinum_dual_wield_plan(cpu, room, validator)
        if dual is not None and not diamond:
            cpu.platinum_last_strategy_score = platinum_plan_score(dual)
            return dual
        if diamond:
            fixed_dual_found = dual is not None
            opening_plans = build_diamond_opening_tactical_plans(
                cpu,
                room,
                validator,
                fixed_dual=dual,
            )
    if diamond:
        fast = (
            None
            if fixed_dual_found
            else diamond_fast_complete_plan(cpu, room, validator)
        )
        if (
            fast is not None
            and diamond_plan_lead_avoids_opponent_finish(fast, cpu, room)
        ):
            opening_plans.append(fast)
        opening_plans = [
            plan
            for plan in opening_plans
            if is_executable_gold_plan(plan, cpu)
            and platinum_opening_multi_play_is_sound(cpu, plan)
            and diamond_plan_lead_avoids_opponent_finish(plan, cpu, room)
        ]
        if opening_plans:
            best_opening = max(
                opening_plans,
                key=lambda plan: diamond_opening_tactical_plan_sort_key(
                    plan,
                    cpu,
                    room,
                ),
            )
            if diamond_opening_auso_converts_to_initial_all_out(
                best_opening,
                cpu,
            ):
                clear_gold_active_plan(cpu)
                cpu.diamond_pending_initial_all_out = True
                cpu.platinum_last_strategy_score = 0.0
                return None
            cpu.platinum_last_strategy_score = platinum_plan_score(best_opening)
            return best_opening
    candidates.extend(build_platinum_plans(cpu, room, validator))
    candidates = [
        plan for plan in candidates
        if is_executable_gold_plan(plan, cpu)
        and platinum_opening_multi_play_is_sound(cpu, plan)
    ]
    if diamond:
        candidates = [
            plan
            for plan in candidates
            if diamond_plan_lead_avoids_opponent_finish(plan, cpu, room)
        ]
    if not candidates:
        cpu.platinum_last_strategy_score = 0.0
        return None
    if platinum_opponent_hand_is_expanded(cpu, room):
        absolute = [plan for plan in candidates if platinum_plan_has_absolute_trump(plan, cpu)]
        if absolute:
            candidates = absolute
    best = max(
        candidates,
        key=(
            (lambda plan: diamond_plan_sort_key(plan, cpu, room))
            if diamond
            else platinum_plan_sort_key
        ),
    )
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
            strong.sort(
                key=(
                    (lambda plan: diamond_plan_sort_key(plan, cpu, room))
                    if getattr(cpu, "cpu_key", "") == "diamond_planner"
                    else platinum_plan_sort_key
                ),
                reverse=True,
            )
            return strong[:GOLD_PLAN_MAX_ALTERNATIVES]
    plans = [plan for plan in plans if platinum_opening_multi_play_is_sound(cpu, plan)]
    plans.sort(
        key=(
            (lambda plan: diamond_plan_sort_key(plan, cpu, room))
            if getattr(cpu, "cpu_key", "") == "diamond_planner"
            else platinum_plan_sort_key
        ),
        reverse=True,
    )
    return plans[:GOLD_PLAN_MAX_ALTERNATIVES]


def platinum_rally_count_order(cpu: CpuPlayer, non_joker_count: int) -> tuple[int, ...]:
    upper = min(9, non_joker_count)
    if len(cpu.hand) < PLATINUM_COMPRESSION_MIN_HAND_SIZE or upper < 5:
        base_order = tuple(range(1, upper + 1))
    else:
        base_order = (
            *range(upper, 4, -1),
            *range(min(4, upper), 0, -1),
        )
    if getattr(cpu, "cpu_key", "") == "diamond_planner":
        return diamond_rally_count_order(cpu, non_joker_count)
    return base_order


def diamond_preferred_counts(counts: Iterable[int]) -> tuple[int, ...]:
    base = tuple(dict.fromkeys(int(count) for count in counts if int(count) > 0))
    preferred = tuple(count for count in DIAMOND_PREFERRED_RALLY_COUNTS if count in base)
    return preferred + tuple(count for count in base if count not in preferred)


def diamond_rally_count_order(cpu: CpuPlayer, non_joker_count: int) -> tuple[int, ...]:
    """Return the observable Diamond personality without excluding fallbacks."""
    upper = min(9, non_joker_count)
    if upper <= 0:
        return ()
    if len(cpu.hand) < PLATINUM_COMPRESSION_MIN_HAND_SIZE or upper < 5:
        base_order = tuple(range(1, upper + 1))
    else:
        base_order = (
            *range(upper, 4, -1),
            *range(min(4, upper), 0, -1),
        )
    ordered = diamond_preferred_counts(base_order)
    face_card_count = sum(
        1
        for card in cpu.hand
        if not is_joker(card) and 10 <= int(card.get("rank", 0)) <= 13
    )
    if face_card_count >= 11 and 6 in ordered and 4 in ordered:
        ordered = (6, 4) + tuple(
            count for count in ordered if count not in {4, 6}
        )
    if getattr(cpu, "diamond_revolution_strategy_active", False):
        ordered = tuple(
            count for count in ordered
            if count not in DIAMOND_REVOLUTION_AVOID_COUNTS
        )
    return ordered


def diamond_kk_finish_available(cpu: CpuPlayer) -> bool:
    visible_cards = cards_for_ranks(cpu.hand, (13, 13))
    if visible_cards is None:
        return False
    entries = [
        entry for entry in cpu.registered_composite_entries
        if entry.value == 1313
    ]
    return material_for_composite_entries(
        cpu.hand,
        entries,
        visible_cards,
    ) is not None


def diamond_non_certain_three_rally_supported(cpu: CpuPlayer) -> bool:
    """Require a separate lock resource before trusting three non-certain rallies."""
    joker_count = sum(1 for card in cpu.hand if is_joker(card))
    kx_count = sum(
        1
        for card in cpu.hand
        if is_joker(card) or int(card.get("rank", 0)) == 13
    )
    return (
        joker_count >= 2
        or kx_count >= 4
        or diamond_kk_finish_available(cpu)
    )


def diamond_initial_second_three_rally_restriction_active(
    cpu: CpuPlayer,
    room,
) -> bool:
    """Limit the reviewed three-rally gate to an untouched 11-card reply hand."""
    initial = max(1, int(getattr(cpu, "platinum_initial_hand_size", 11)))
    return (
        len(cpu.hand) == initial
        and diamond_tactical_context(cpu, room) == "opponent-all-out"
    )


def diamond_conditional_return_probability_limit(
    cpu: CpuPlayer,
    *,
    has_alternative: bool,
) -> float:
    limit = (
        DIAMOND_CONDITIONAL_DEFAULT_MAX_RETURN_PROBABILITY
        if has_alternative
        else DIAMOND_CONDITIONAL_DESPERATE_MAX_RETURN_PROBABILITY
    )
    cpu.diamond_last_conditional_limit = limit
    return limit


def diamond_publicly_unaccounted_rank_counts(
    cpu: CpuPlayer,
    room,
) -> tuple[dict[int, int], int]:
    """Conservative standard-deck supply available to an opponent or one draw.

    Cards remembered from earlier tricks are intentionally not subtracted here:
    once their field flows they return to the deck and may subsequently enter the
    opponent's hand.  Current hand/field/reserve are the only unavailable zones.
    """
    seen = set()
    accounted = []
    for card in (
        list(cpu.hand)
        + list(getattr(room, "field", []) or [])
        + list(getattr(room, "reserve", []) or [])
    ):
        key = card.get("card_id") or id(card)
        if key in seen:
            continue
        seen.add(key)
        accounted.append(card)

    rank_counts = {rank: 4 for rank in range(1, 14)}
    joker_count = 2
    for card in accounted:
        if is_joker(card):
            joker_count = max(0, joker_count - 1)
            continue
        rank = int(card.get("rank", 0))
        if rank in rank_counts:
            rank_counts[rank] = max(0, rank_counts[rank] - 1)
    return rank_counts, joker_count


@lru_cache(maxsize=32)
def diamond_counter_material_signatures(token: str) -> tuple[tuple[int, ...], ...]:
    """Return every non-dominated face-display plus material requirement."""
    value = platinum_token_value(token)
    visible = platinum_token_ranks(token)
    entries = generate_composite_expression_entries(
        value,
        token,
        0,
        max_entries=None,
        minimum_jokers_only=False,
    )
    signatures = {
        tuple(sorted(
            visible
            + tuple(
                rank
                for expression_token in entry.expression_tokens
                if expression_token.kind == "cards"
                for rank in expression_token.ranks
            )
        ))
        for entry in entries
    }

    def dominates(left: tuple[int, ...], right: tuple[int, ...]) -> bool:
        if len(left) >= len(right):
            return False
        left_counts = Counter(left)
        right_counts = Counter(right)
        return all(
            left_counts.get(rank, 0) <= right_counts.get(rank, 0)
            for rank in set(left_counts) | set(right_counts)
        )

    non_dominated = [
        signature for signature in signatures
        if not any(
            other != signature and dominates(other, signature)
            for other in signatures
        )
    ]
    return tuple(sorted(non_dominated, key=lambda item: (len(item), item)))


def diamond_requirement_is_physically_possible(
    available: dict[int, int],
    joker_count: int,
    signature: tuple[int, ...],
) -> bool:
    required = Counter(signature)
    deficit = sum(
        max(0, count - available.get(rank, 0))
        for rank, count in required.items()
    )
    return deficit <= joker_count


def diamond_counter_token_is_physically_possible(
    cpu: CpuPlayer,
    room,
    token: str,
) -> bool:
    opponent_count = platinum_opponent_hand_count(cpu, room)
    if opponent_count is None:
        return True
    draw_allowance = 1 if getattr(room, "deck", []) else 0
    available, joker_count = diamond_publicly_unaccounted_rank_counts(cpu, room)
    hand_capacity = opponent_count + draw_allowance
    return any(
        len(signature) <= hand_capacity
        and diamond_requirement_is_physically_possible(
            available,
            joker_count,
            signature,
        )
        for signature in diamond_counter_material_signatures(token)
    )


def diamond_draw_satisfies_requirement(
    drawn: Counter,
    signature: tuple[int, ...],
) -> bool:
    required = Counter(signature)
    joker_count = drawn.get(0, 0)
    deficit = sum(
        max(0, count - drawn.get(rank, 0))
        for rank, count in required.items()
    )
    return deficit <= joker_count


def diamond_requirement_union_probability(
    cpu: CpuPlayer,
    room,
    signatures: Iterable[tuple[int, ...]],
    *,
    seed_value: int,
    sample_size: Optional[int] = None,
) -> float:
    opponent_count = platinum_opponent_hand_count(cpu, room)
    if opponent_count is None:
        return 1.0
    available, joker_count = diamond_publicly_unaccounted_rank_counts(cpu, room)
    pool = [
        rank
        for rank, count in sorted(available.items())
        for _ in range(count)
    ] + [0] * joker_count
    if sample_size is None:
        sample_size = opponent_count + (1 if getattr(room, "deck", []) else 0)
    sample_size = min(len(pool), max(0, int(sample_size)))
    signature_tuple = tuple(dict.fromkeys(
        signature for signature in signatures
        if len(signature) <= sample_size
        and diamond_requirement_is_physically_possible(
            available,
            joker_count,
            signature,
        )
    ))
    if not signature_tuple:
        return 0.0

    seed = int(seed_value) * 131 + sample_size
    for rank in range(14):
        count = joker_count if rank == 0 else available.get(rank, 0)
        seed = (seed * 257 + rank * 17 + count) & ((1 << 64) - 1)
    rng = random.Random(seed)
    successes = 0
    for _ in range(DIAMOND_RETURN_PROBABILITY_TRIALS):
        drawn = Counter(rng.sample(pool, sample_size))
        if any(
            diamond_draw_satisfies_requirement(drawn, signature)
            for signature in signature_tuple
        ):
            successes += 1
    return successes / DIAMOND_RETURN_PROBABILITY_TRIALS


def diamond_counter_return_probability(
    candidate: dict,
    cpu: CpuPlayer,
    room,
) -> float:
    """Deterministic rank-level Monte Carlo over all practical counter forms."""
    token = platinum_candidate_token(candidate)
    try:
        value = int(candidate.get("number", platinum_token_value(token)))
    except (TypeError, ValueError):
        return 1.0
    if not diamond_four_candidate_uses_fixed_counter_catalog(candidate):
        return 0.0
    counter_tokens = tuple(
        counter for counter in DIAMOND_FOUR_COUNTER_TOKENS
        if (
            platinum_token_value(counter) < value
            if getattr(room, "reverse_order", False)
            else platinum_token_value(counter) > value
        )
    )
    feasible_tokens = tuple(
        counter for counter in counter_tokens
        if diamond_counter_token_is_physically_possible(cpu, room, counter)
    )
    cpu.diamond_last_obake_counter_tokens = feasible_tokens
    cpu.diamond_last_return_candidates = feasible_tokens
    if not feasible_tokens:
        return 0.0

    signatures = tuple(dict.fromkeys(
        signature
        for counter in feasible_tokens
        for signature in diamond_counter_material_signatures(counter)
    ))
    return diamond_requirement_union_probability(
        cpu,
        room,
        signatures,
        seed_value=int(candidate.get("number", value)),
    )


def diamond_four_candidate_is_face_obake(candidate: Optional[dict]) -> bool:
    if not candidate:
        return False
    if len(candidate.get("cards", []) or []) != 4 or candidate.get("kind") != "composite":
        return False
    ranks = tuple(candidate.get("ranks") or ())
    if not ranks:
        try:
            ranks = platinum_token_ranks(platinum_candidate_token(candidate))
        except (KeyError, ValueError):
            return False
    return len(ranks) == 4 and all(int(rank) >= 10 for rank in ranks)


def diamond_four_candidate_uses_fixed_counter_catalog(candidate: dict) -> bool:
    if not diamond_four_candidate_is_face_obake(candidate):
        return False
    try:
        value = int(candidate.get("number"))
    except (TypeError, ValueError):
        value = platinum_token_value(platinum_candidate_token(candidate))
    return value >= DIAMOND_FOUR_FIXED_COUNTER_MIN_VALUE


def diamond_general_four_return_probability(
    candidate: dict,
    cpu: CpuPlayer,
    room,
) -> float:
    """Estimate all registered four-card returns below the KKJQ catalog tier."""
    try:
        value = int(candidate.get("number"))
    except (TypeError, ValueError):
        return 1.0
    opponent_count = platinum_opponent_hand_count(cpu, room)
    if opponent_count is None:
        return 1.0
    capacity = opponent_count + (1 if getattr(room, "deck", []) else 0)
    reverse = bool(getattr(room, "reverse_order", False))
    entries = tuple(
        (response_value, signature)
        for response_value, signature in diamond_public_response_signatures(cpu, 4)
        if (response_value < value if reverse else response_value > value)
        and len(signature) <= capacity
    )
    available, joker_count = diamond_publicly_unaccounted_rank_counts(cpu, room)
    feasible = tuple(
        (response_value, signature)
        for response_value, signature in entries
        if diamond_requirement_is_physically_possible(
            available,
            joker_count,
            signature,
        )
    )
    cpu.diamond_last_obake_counter_tokens = tuple(
        str(response_value)
        for response_value in dict.fromkeys(value for value, _ in feasible)
    )
    cpu.diamond_last_return_candidates = cpu.diamond_last_obake_counter_tokens
    if not feasible:
        return 0.0
    return diamond_requirement_union_probability(
        cpu,
        room,
        tuple(dict.fromkeys(signature for _, signature in feasible)),
        seed_value=value,
    )


def diamond_six_return_probability(
    candidate: dict,
    cpu: CpuPlayer,
    room,
) -> float:
    try:
        value = int(candidate.get("number"))
    except (TypeError, ValueError):
        return 1.0
    reverse = getattr(room, "reverse_order", False)
    values = tuple(sorted(cpu.registered_primes))
    index = (
        cpu.prime_template_index
        if getattr(cpu, "prime_template_index_values", ()) == values
        else registered_prime_template_index(values, max_cards=6)
    )
    return_entries = tuple(
        (number, tuple(ranks))
        for number, ranks in index.templates_by_card_count.get(6, ())
        if (number < value if reverse else number > value)
    )
    cpu.diamond_last_return_candidates = tuple(
        str(number) for number in dict.fromkeys(number for number, _ in return_entries)
    )
    signatures = tuple(dict.fromkeys(ranks for _, ranks in return_entries))
    return diamond_requirement_union_probability(
        cpu,
        room,
        signatures,
        seed_value=value,
    )


def diamond_candidate_return_probability(
    candidate: Optional[dict],
    cpu: CpuPlayer,
    room,
) -> float:
    if not candidate:
        return 1.0
    visible_count = len(candidate.get("cards", []) or [])
    if visible_count == 4 and candidate.get("kind") == "composite":
        if diamond_four_candidate_uses_fixed_counter_catalog(candidate):
            return diamond_counter_return_probability(candidate, cpu, room)
        if diamond_four_candidate_is_face_obake(candidate):
            return diamond_general_four_return_probability(candidate, cpu, room)
        return 1.0
    if visible_count == 6 and candidate.get("kind") == "prime":
        return diamond_six_return_probability(candidate, cpu, room)
    return 1.0


def diamond_four_obake_certainty(
    candidate: dict,
    cpu: CpuPlayer,
    room,
) -> str:
    token = platinum_candidate_token(candidate)
    if not diamond_four_candidate_is_face_obake(candidate):
        return "unclassified"
    try:
        value = int(candidate.get("number"))
    except (TypeError, ValueError):
        value = platinum_token_value(token)
    if value >= DIAMOND_FOUR_FIXED_COUNTER_MIN_VALUE:
        probability = diamond_counter_return_probability(candidate, cpu, room)
        cpu.diamond_last_return_probability = probability
        return (
            "conditional"
            if cpu.diamond_last_obake_counter_tokens
            else "certain"
        )

    opponent_count = platinum_opponent_hand_count(cpu, room)
    opponent_max_kx = diamond_opponent_max_kx_after_draw(cpu, room)
    if value == DIAMOND_FOUR_KKTJ_VALUE:
        certain = opponent_max_kx < 2
    else:
        certain = (
            diamond_cpu_monopolizes_kx(cpu)
            or (
                opponent_count is not None
                and opponent_count
                <= DIAMOND_FOUR_OTHER_CERTAIN_MAX_OPPONENT_HAND_SIZE
            )
        )
    if certain:
        cpu.diamond_last_obake_counter_tokens = ()
        cpu.diamond_last_return_candidates = ()
        cpu.diamond_last_return_probability = 0.0
        return "certain"

    probability = diamond_general_four_return_probability(candidate, cpu, room)
    cpu.diamond_last_return_probability = probability
    return "conditional"


def diamond_candidate_certainty(
    candidate: Optional[dict],
    cpu: CpuPlayer,
    room,
) -> str:
    if not candidate:
        return "unclassified"
    visible_count = len(candidate.get("cards", []) or [])
    if platinum_candidate_token(candidate) == "kk":
        return "certain"
    if visible_count == 4 and candidate.get("kind") == "composite":
        if diamond_four_candidate_is_face_obake(candidate):
            return diamond_four_obake_certainty(candidate, cpu, room)
        opponent_count = platinum_opponent_hand_count(cpu, room)
        if (
            opponent_count is not None
            and opponent_count <= 18
            and all(rank >= 10 for rank in candidate.get("ranks", ()))
        ):
            return "soft"
        return "unclassified"
    if visible_count != 6 or candidate.get("kind") != "prime":
        return "unclassified"
    try:
        value = int(candidate.get("number"))
    except (TypeError, ValueError):
        return "unclassified"
    if (
        not getattr(room, "reverse_order", False)
        and value >= DIAMOND_SIX_CERTAIN_MIN_VALUE
        and value in cpu.registered_primes
    ):
        return "certain"
    if value > DIAMOND_SIX_SOFT_MIN_VALUE:
        check_cpu_search_deadline(cpu)
        probability = diamond_six_return_probability(candidate, cpu, room)
        cpu.diamond_last_return_probability = probability
        return (
            "conditional"
            if probability <= DIAMOND_CONDITIONAL_DESPERATE_MAX_RETURN_PROBABILITY
            else "soft"
        )
    if value == DIAMOND_SIX_SOFT_MIN_VALUE:
        return "soft"
    return "unclassified"


def diamond_four_obake_candidates(
    cpu: CpuPlayer,
    room,
    validator: Optional[NumberValidator] = None,
) -> list[dict]:
    validator = validator or gold_knowledge_number_validator
    candidates = knowledge_composite_candidates(cpu, room, (4,))
    candidates = [
        candidate for candidate in candidates
        if diamond_four_candidate_is_face_obake(candidate)
    ]
    return sorted(
        dedupe_candidates(candidates),
        key=lambda candidate: candidate_strength(candidate, room),
        reverse=True,
    )


def diamond_plan_trump_candidate(plan: Optional[dict]) -> Optional[dict]:
    if not plan:
        return None
    index = gold_plan_trump_step_index(plan)
    steps = plan.get("steps", [])
    if index is None or index >= len(steps):
        return None
    return steps[index]


def diamond_plan_certainty(plan: Optional[dict], cpu: CpuPlayer, room) -> str:
    return diamond_candidate_certainty(
        diamond_plan_trump_candidate(plan),
        cpu,
        room,
    )


def diamond_lead_candidate_is_certain(
    candidate: Optional[dict],
    cpu: CpuPlayer,
    room,
) -> bool:
    """Return whether the card being led, rather than its whole plan, is certain."""
    if not candidate:
        return False
    certainty = diamond_candidate_certainty(candidate, cpu, room)
    if diamond_four_candidate_is_face_obake(candidate):
        return certainty == "certain"
    return certainty == "certain" or diamond_candidate_is_publicly_uncounterable(
        candidate,
        cpu,
        room,
    )


def diamond_lead_candidate_has_face_count_lock(
    candidate: Optional[dict],
    cpu: CpuPlayer,
    room,
) -> bool:
    """Allow a lethal-count lead whose displayed faces exceed the reply estimate."""
    face_count = diamond_candidate_display_face_count(candidate)
    if face_count <= 0:
        return False
    return face_count > diamond_estimated_opponent_face_count_after_draw(cpu, room)


def diamond_lead_candidate_avoids_opponent_finish(
    candidate: Optional[dict],
    cpu: CpuPlayer,
    room,
) -> bool:
    """Forbid a non-certain lead matching the opponent's whole hand size."""
    if not candidate or getattr(room, "field", []) or []:
        return True
    opponent_count = platinum_opponent_hand_count(cpu, room)
    if opponent_count is None or opponent_count <= 0:
        return True
    visible_count = len(candidate.get("cards", []) or [])
    if visible_count != opponent_count:
        return True
    if len(candidate_consumed_cards(candidate)) == len(cpu.hand):
        return True
    return (
        diamond_lead_candidate_is_certain(candidate, cpu, room)
        or diamond_lead_candidate_has_face_count_lock(candidate, cpu, room)
    )


def diamond_plan_lead_avoids_opponent_finish(
    plan: Optional[dict],
    cpu: CpuPlayer,
    room,
    *,
    step_index: int = 0,
) -> bool:
    if not plan or getattr(room, "field", []) or []:
        return True
    steps = list(plan.get("steps", []))
    if step_index < 0 or step_index >= len(steps):
        return True
    return diamond_lead_candidate_avoids_opponent_finish(
        steps[step_index],
        cpu,
        room,
    )


def diamond_public_response_signatures(
    cpu: CpuPlayer,
    count: int,
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    """Return known prime/composite response values with physical materials."""
    cache = getattr(cpu, "diamond_public_response_signature_cache", None)
    if cache is None:
        cache = {}
        cpu.diamond_public_response_signature_cache = cache
    if count in cache:
        return cache[count]
    values = tuple(sorted(cpu.registered_primes))
    index = (
        cpu.prime_template_index
        if getattr(cpu, "prime_template_index_values", ()) == values
        else registered_prime_template_index(
            values,
            max_cards=cpu_max_knowledge_cards(cpu),
        )
    )
    signatures = {
        (int(value), tuple(sorted(ranks)))
        for value, ranks in index.templates_by_card_count.get(count, ())
    }
    for entry in cpu.registered_composite_entries:
        try:
            visible = registered_pattern_cards(
                entry.pattern,
                allow_unencoded_value=True,
            )
        except ValueError:
            continue
        if len(visible) != count:
            continue
        materials = tuple(
            rank
            for token in entry.expression_tokens
            if token.kind == "cards"
            for rank in token.ranks
        )
        signatures.add((
            int(entry.value),
            tuple(sorted(tuple(visible) + materials)),
        ))
    result = tuple(sorted(
        signatures,
        key=lambda item: (item[0], len(item[1]), item[1]),
    ))
    cache[count] = result
    return result


def diamond_public_response_values(
    candidate_value: int,
    count: int,
    cpu: CpuPlayer,
    room,
    *,
    opponent_capacity: Optional[int] = None,
) -> tuple[int, ...]:
    opponent_count = platinum_opponent_hand_count(cpu, room)
    if opponent_capacity is None:
        if opponent_count is None:
            return (10**100,)
        opponent_capacity = opponent_count + (
            1 if getattr(room, "deck", []) else 0
        )
    if count > max(0, int(opponent_capacity)):
        return ()

    available, joker_count = diamond_publicly_unaccounted_rank_counts(cpu, room)
    reverse = bool(getattr(room, "reverse_order", False))
    responses = []
    if count == 1 and joker_count > 0 and not reverse:
        responses.append(10**100)
    for value, signature in diamond_public_response_signatures(cpu, count):
        if not (value < candidate_value if reverse else value > candidate_value):
            continue
        if len(signature) > opponent_capacity:
            continue
        if diamond_requirement_is_physically_possible(
            available,
            joker_count,
            signature,
        ):
            responses.append(value)
    return tuple(dict.fromkeys(responses))


def diamond_candidate_is_publicly_uncounterable(
    candidate: dict,
    cpu: CpuPlayer,
    room,
    *,
    opponent_capacity: Optional[int] = None,
) -> bool:
    if candidate.get("kind") == "joker_cut" or candidate.get("number") in {"X", 57}:
        return True
    try:
        value = int(candidate.get("number"))
    except (TypeError, ValueError):
        return False
    count = len(candidate.get("cards", []) or [])
    if count <= 0:
        return False
    return not diamond_public_response_values(
        value,
        count,
        cpu,
        room,
        opponent_capacity=opponent_capacity,
    )


def diamond_estimated_opponent_max_value(
    cpu: CpuPlayer,
    room,
    count: int,
) -> Optional[int]:
    """Estimate the strongest same-count value still physically available."""
    opponent_count = platinum_opponent_hand_count(cpu, room)
    if opponent_count is None:
        return None
    capacity = opponent_count + (1 if getattr(room, "deck", []) else 0)
    available, joker_count = diamond_publicly_unaccounted_rank_counts(cpu, room)
    values = []
    if count == 1 and joker_count > 0 and not getattr(room, "reverse_order", False):
        values.append(10**100)
    for value, signature in diamond_public_response_signatures(cpu, count):
        if len(signature) <= capacity and diamond_requirement_is_physically_possible(
            available,
            joker_count,
            signature,
        ):
            values.append(value)
    try:
        field_value = int(getattr(room, "last_number", 0) or 0)
    except (TypeError, ValueError):
        field_value = 0
    if field_value:
        values.append(field_value)
    if not values:
        return None
    return min(values) if getattr(room, "reverse_order", False) else max(values)


def diamond_face_resource_return_probability(
    candidate: dict,
    cpu: CpuPlayer,
    room,
) -> Optional[float]:
    """Approximate a response by the minimum face resources it consumes."""
    try:
        value = int(candidate.get("number"))
    except (TypeError, ValueError):
        return None
    count = len(candidate.get("cards", []) or [])
    opponent_count = platinum_opponent_hand_count(cpu, room)
    if count <= 0 or opponent_count is None:
        return None
    sample_size = opponent_count + (1 if getattr(room, "deck", []) else 0)
    available, joker_count = diamond_publicly_unaccounted_rank_counts(cpu, room)
    reverse = bool(getattr(room, "reverse_order", False))
    requirements = []
    for response_value, signature in diamond_public_response_signatures(cpu, count):
        if not (response_value < value if reverse else response_value > value):
            continue
        if len(signature) > sample_size:
            continue
        if not diamond_requirement_is_physically_possible(
            available,
            joker_count,
            signature,
        ):
            continue
        requirements.append(sum(1 for rank in signature if rank >= 10))
    if count == 1 and joker_count > 0 and not reverse:
        requirements.append(1)
    if not requirements:
        return 0.0

    minimum_faces = min(requirements)
    face_cards = joker_count + sum(
        available.get(rank, 0) for rank in range(10, 14)
    )
    population = joker_count + sum(available.values())
    sample_size = min(sample_size, population)
    if population <= 0 or sample_size <= 0:
        return 0.0
    expected_faces = face_cards * sample_size / population
    if minimum_faces < expected_faces + 1.0:
        return None
    denominator = comb(population, sample_size)
    if denominator <= 0:
        return None
    lower = max(minimum_faces, 0)
    upper = min(face_cards, sample_size)
    probability = sum(
        comb(face_cards, drawn_faces)
        * comb(population - face_cards, sample_size - drawn_faces)
        for drawn_faces in range(lower, upper + 1)
        if 0 <= sample_size - drawn_faces <= population - face_cards
    ) / denominator
    return probability


def diamond_public_certainty_state_key(cpu: CpuPlayer, room) -> tuple:
    def ids(cards: Iterable[Card]) -> tuple[str, ...]:
        return tuple(sorted(str(card.get("card_id") or id(card)) for card in cards))

    return (
        ids(cpu.hand),
        ids(getattr(room, "field", []) or []),
        ids(getattr(room, "reserve", []) or []),
        ids(getattr(room, "public_known_deck_bottom", []) or []),
        int(getattr(room, "public_unknown_deck_count", 0) or 0),
        bool(getattr(room, "deck", []) or []),
        platinum_opponent_hand_count(cpu, room),
        bool(getattr(room, "reverse_order", False)),
    )


def diamond_post_all_out_candidate_certainty(
    candidate: Optional[dict],
    cpu: CpuPlayer,
    room,
) -> str:
    if not candidate:
        return "unclassified"
    state_key = diamond_public_certainty_state_key(cpu, room)
    cached = candidate.get("_diamond_post_all_out_certainty")
    if cached and candidate.get("_diamond_post_all_out_certainty_key") == state_key:
        return str(cached)

    def remember(result: str) -> str:
        candidate["_diamond_post_all_out_certainty"] = result
        candidate["_diamond_post_all_out_certainty_key"] = state_key
        return result

    certainty = diamond_candidate_certainty(candidate, cpu, room)
    if certainty == "conditional":
        candidate["_diamond_return_probability"] = float(
            getattr(cpu, "diamond_last_return_probability", 1.0)
        )
    if diamond_four_candidate_is_face_obake(candidate):
        return remember(certainty)
    if certainty == "certain" or diamond_candidate_is_publicly_uncounterable(
        candidate,
        cpu,
        room,
    ):
        return remember("certain")
    if certainty in {"conditional", "soft"}:
        return remember(certainty)
    face_probability = diamond_face_resource_return_probability(
        candidate,
        cpu,
        room,
    )
    if face_probability is not None and face_probability <= 0.80:
        candidate["_diamond_return_probability"] = face_probability
        candidate["_diamond_face_lock"] = True
        result = "conditional" if face_probability <= 0.30 else "soft"
        return remember(result)
    strength = platinum_candidate_trump_strength(candidate)
    if strength >= DIAMOND_POST_ALL_OUT_CONDITIONAL_MIN_TRUMP_STRENGTH:
        result = "conditional"
    elif strength >= DIAMOND_POST_ALL_OUT_SOFT_MIN_TRUMP_STRENGTH:
        result = "soft"
    else:
        result = "unclassified"
    return remember(result)


def diamond_post_all_out_plan_certainty(
    plan: dict,
    cpu: CpuPlayer,
    room,
) -> str:
    override = plan.get("diamond_certainty_override")
    if override:
        if str(override) != "certain":
            return str(override)
        opponent_count = platinum_opponent_hand_count(cpu, room)
        if opponent_count is None:
            return "conditional"
        current_cpu = cpu
        draw_increment = 1 if getattr(room, "deck", []) else 0
        for depth, step in enumerate(
            step
            for step in plan.get("steps", [])
            if str(step.get("role", "")).startswith("rally-")
        ):
            if diamond_four_candidate_is_face_obake(step):
                step_is_certain = (
                    diamond_four_obake_certainty(step, current_cpu, room)
                    == "certain"
                )
            else:
                step_is_certain = diamond_candidate_is_publicly_uncounterable(
                    step,
                    current_cpu,
                    room,
                    opponent_capacity=opponent_count + draw_increment * (depth + 1),
                )
            if not step_is_certain:
                return "conditional"
            current_cpu = temporary_cpu_with_hand(
                current_cpu,
                remaining_cards(
                    current_cpu.hand,
                    candidate_consumed_cards(step),
                ),
            )
        return "certain"
    trump = diamond_plan_trump_candidate(plan)
    certainty = diamond_post_all_out_candidate_certainty(trump, cpu, room)
    if certainty != "certain" or trump is None:
        return certainty
    if diamond_candidate_certainty(trump, cpu, room) == "certain":
        return "certain"

    # Public information can prove a single candidate uncounterable, but that
    # does not prove a whole route: an opponent may spend an equal trump on an
    # earlier weak rally (for example 7 -> K while we hold both jokers).
    rally_steps = [
        step
        for step in plan.get("steps", [])
        if str(step.get("role", "")).startswith("rally-")
    ]
    if rally_steps and all(
        diamond_candidate_is_publicly_uncounterable(step, cpu, room)
        for step in rally_steps
    ):
        return "certain"
    return "conditional"


def diamond_plan_rally_step_count(plan: dict) -> int:
    return sum(
        1
        for step in plan.get("steps", [])
        if str(step.get("role", "")).startswith("rally-")
    )


def diamond_plan_finish_strength(plan: dict) -> int:
    """Score the post-trump finish using the reviewed K/X/57/KK formula."""
    steps = list(plan.get("steps", []))
    trump_index = gold_plan_trump_step_index(plan)
    tail_start = 0 if trump_index is None else trump_index + 1
    tail = steps[tail_start:]
    finish = next(
        (step for step in reversed(tail) if step.get("role") == "finish"),
        None,
    )
    finish_count = (
        len(candidate_consumed_cards(finish)) if finish is not None else 0
    )
    resource_cards = {
        card.get("card_id"): card
        for step in tail
        for card in candidate_consumed_cards(step)
    }.values()
    kings = sum(
        1
        for card in resource_cards
        if not is_joker(card) and int(card.get("rank", 0)) == 13
    )
    jokers = sum(1 for card in resource_cards if is_joker(card))
    cut_57_pairs = sum(1 for step in tail if step.get("number") == 57)

    kk_bonus = 0
    kk_indices = [
        index
        for index, step in enumerate(steps)
        if platinum_candidate_token(step) == "kk"
    ]
    for index in kk_indices:
        previous = steps[index - 1] if index > 0 else None
        previous_count = len(previous.get("cards", [])) if previous else None
        if previous_count != 2:
            kk_bonus = 5
            break
    return finish_count + cut_57_pairs + kings * 3 + jokers * 5 + kk_bonus


def diamond_plan_has_protected_x_tail(plan: dict) -> bool:
    """Whether X remains a standalone forced reset after the reserved trump."""
    trump_index = gold_plan_trump_step_index(plan)
    if trump_index is None:
        return False
    return any(
        step.get("role") == "cut" and step.get("number") == "X"
        for step in plan.get("steps", [])[trump_index + 1:]
    )


def diamond_plan_reply_profile(
    plan: dict,
    cpu: CpuPlayer,
    room,
) -> tuple[int, int, int, int]:
    """Score a conditional route after one practical same-count reply.

    A direct conditional trump is fragile when a known higher response remains.
    An earlier same-count rally can still be useful when it invites a response
    below the reserved trump, because the reserved trump can then take the turn.
    This is the reviewed TJJ -> (KQK) -> KKJ shape.
    """
    trump_index = gold_plan_trump_step_index(plan)
    steps = list(plan.get("steps", []))
    if trump_index is None or trump_index >= len(steps):
        return (0, 0, 0, 0)
    trump = steps[trump_index]
    try:
        trump_value = int(trump.get("number"))
    except (TypeError, ValueError):
        return (1, 0, 0, 0)
    count = len(trump.get("cards", []) or [])
    if count <= 0:
        return (0, 0, 0, 0)

    responses = diamond_public_response_values(
        trump_value,
        count,
        cpu,
        room,
    )
    if not responses:
        # Structural no-reply cases are already promoted by the certainty tier.
        # Keep this secondary key neutral so it does not invent a new ordering
        # among approximate conditional face shapes.
        return (0, 0, 0, 0)

    prefix = next(
        (
            step
            for step in reversed(steps[:trump_index])
            if str(step.get("role", "")).startswith("rally-")
            and len(step.get("cards", []) or []) == count
        ),
        None,
    )
    if prefix is None:
        return (0, 0, 0, 0)
    try:
        prefix_value = int(prefix.get("number"))
    except (TypeError, ValueError):
        return (0, 0, 0, 0)

    prefix_responses = diamond_public_response_values(
        prefix_value,
        count,
        cpu,
        room,
    )
    reverse = bool(getattr(room, "reverse_order", False))
    if reverse:
        baitable = tuple(
            value for value in prefix_responses
            if trump_value < value < prefix_value
        )
    else:
        baitable = tuple(
            value for value in prefix_responses
            if prefix_value < value < trump_value
        )
    return (
        0,
        1 if baitable else 0,
        0,
        0,
    )


def diamond_should_draw_before_opponent_all_out_plan(
    plan: dict,
    tier: Optional[int],
    cpu: CpuPlayer,
    room,
) -> bool:
    """Allow one recovery draw only when it can replace a long weak route.

    Two-move KJTK/KKJ-style routes remain immediate.  The draw is restored for
    three-or-more-step Tier 8/9 routes, where the old planner often reduced the
    rally count or upgraded the trump without increasing the eventual hand count.
    Keep the already-reviewed two-rally X-protected closeout immediate.
    """
    protected_short_closeout = (
        diamond_plan_rally_step_count(plan) <= 2
        and diamond_plan_has_protected_x_tail(plan)
    )
    if (
        tier is None
        or tier < 8
        or len(plan.get("steps", []))
        < DIAMOND_OPPONENT_ALL_OUT_DRAW_MIN_PLAN_STEPS
        or protected_short_closeout
        or getattr(room, "has_drawn", False)
        or not getattr(room, "deck", [])
    ):
        return False
    return platinum_deck_has_expected_trump_contribution(cpu, room)


def diamond_initial_opponent_all_out_tier9_blocked(
    cpu: CpuPlayer,
    room,
) -> bool:
    """Reject soft tier 9 only on Diamond's first reply to opening all-out."""
    opponent_count = platinum_opponent_hand_count(cpu, room)
    return (
        getattr(cpu, "diamond_opening_was_second", None) is True
        and getattr(cpu, "platinum_opening_phase", True)
        and diamond_tactical_context(cpu, room) == "opponent-all-out"
        and opponent_count is not None
        and opponent_count
        >= DIAMOND_OPPONENT_ALL_OUT_TIER9_MIN_OPPONENT_HAND_SIZE
    )


def diamond_post_all_out_plan_tier(
    plan: dict,
    cpu: CpuPlayer,
    room,
) -> Optional[int]:
    certainty = diamond_post_all_out_plan_certainty(plan, cpu, room)
    rally_steps = diamond_plan_rally_step_count(plan)
    rally_count = int(plan.get("rally_count", 0) or 0)
    preferred = rally_count in DIAMOND_PREFERRED_RALLY_COUNTS
    if (
        certainty != "certain"
        and rally_steps == 3
        and diamond_initial_second_three_rally_restriction_active(cpu, room)
        and not diamond_non_certain_three_rally_supported(cpu)
    ):
        return None
    if certainty == "certain":
        if rally_steps <= 2:
            return 1 if preferred else 2
        if rally_steps == 3:
            return 3 if preferred else 4
        if rally_steps == 4 and rally_count <= 6:
            return 5
        return None

    trump = diamond_plan_trump_candidate(plan)
    if trump is None or rally_steps > DIAMOND_POST_ALL_OUT_MAX_RALLY_STEPS:
        return None
    strength = platinum_candidate_trump_strength(trump)
    if platinum_candidate_token(trump) == "kqqj":
        opponent_count = platinum_opponent_hand_count(cpu, room)
        if (
            opponent_count is None
            or opponent_count > DIAMOND_KQQJ_CONTEXTUAL_MAX_OPPONENT_HAND_SIZE
            or len(cpu.hand) <= opponent_count
        ):
            return None
        trump["_diamond_contextual_kqqj"] = True
        return 8
    # A face-resource estimate is still conditional rather than certain, but a
    # computed 0% is the reviewed tier-8 signal instead of an ordinary soft.
    if (
        trump.get("_diamond_face_lock")
        and float(trump.get("_diamond_return_probability", 1.0)) <= 0.0
    ):
        return 8
    if certainty == "conditional":
        return_probability = trump.get("_diamond_return_probability")
        if return_probability is None:
            return_probability = diamond_candidate_return_probability(
                trump,
                cpu,
                room,
            )
            trump["_diamond_return_probability"] = return_probability
        return_probability = float(return_probability)
        if (
            return_probability
            <= DIAMOND_POST_ALL_OUT_STRONG_CONDITIONAL_MAX_RETURN_PROBABILITY
        ):
            return 6
        if strength >= DIAMOND_POST_ALL_OUT_CONDITIONAL_MIN_TRUMP_STRENGTH:
            return 8
    if strength >= DIAMOND_POST_ALL_OUT_CONDITIONAL_MIN_TRUMP_STRENGTH:
        return 8
    if strength >= DIAMOND_POST_ALL_OUT_SOFT_MIN_TRUMP_STRENGTH:
        if diamond_initial_opponent_all_out_tier9_blocked(cpu, room):
            return None
        return 9
    return None


def diamond_post_all_out_plan_sort_key(
    plan: dict,
    cpu: CpuPlayer,
    room,
) -> tuple:
    tier = diamond_post_all_out_plan_tier(plan, cpu, room)
    if tier is None:
        return (-99,)
    rally_steps = [
        step
        for step in plan.get("steps", [])
        if str(step.get("role", "")).startswith("rally-")
    ]
    second_strength = (
        candidate_strength(rally_steps[1], room)
        if len(rally_steps) >= 2
        else -1
    )
    trump = diamond_plan_trump_candidate(plan) or {}
    opponent_count = getattr(cpu, "diamond_opponent_rally_count", None)
    avoids_opponent_count = (
        opponent_count is None
        or int(plan.get("rally_count", 0) or 0) != opponent_count
    )
    finish_strength = diamond_plan_finish_strength(plan)
    reply_profile = diamond_plan_reply_profile(plan, cpu, room)
    return (
        -tier,
        1 if avoids_opponent_count else 0,
        reply_profile,
        -diamond_plan_rally_step_count(plan),
        1 if diamond_plan_has_protected_x_tail(plan) else 0,
        second_strength if tier == 3 else 0,
        finish_strength,
        0 if step_uses_joker(trump) else 1,
        candidate_strength(trump, room) if trump else -1,
        platinum_plan_score(plan),
        gold_plan_score(plan),
    )


def diamond_post_all_out_search_has_time(cpu: CpuPlayer) -> bool:
    deadline = getattr(cpu, "decision_deadline", None)
    return (
        deadline is None
        or time.perf_counter() + DIAMOND_POST_ALL_OUT_SEARCH_RESERVE_SECONDS
        < deadline
    )


def diamond_forced_pass_plans(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
) -> list[dict]:
    """Build pass-forcing mixed-count routes, including K -> K -> finish."""
    if getattr(room, "field", []) or []:
        return []
    empty_room = room_without_field(room)
    opponent_count = platinum_opponent_hand_count(cpu, room)
    if opponent_count is None:
        return []
    draw_increment = 1 if getattr(room, "deck", []) else 0
    results = []
    seen = set()

    def visit(current_cpu: CpuPlayer, selected: list[dict], depth: int) -> None:
        if len(results) >= DIAMOND_POST_ALL_OUT_RESULT_CAP:
            return
        if not diamond_post_all_out_search_has_time(current_cpu):
            return
        tail = choose_gold_finish_tail(current_cpu, empty_room, validator)
        if selected and tail:
            sequence = selected + tail
            key = tuple(candidate_fingerprint(step) for step in sequence)
            if key not in seen:
                seen.add(key)
                plan = finalize_gold_plan(
                    cpu,
                    empty_room,
                    sequence,
                    len(selected[0].get("cards", [])),
                )
                plan["diamond_forced_pass"] = True
                plan["diamond_certainty_override"] = "certain"
                results.append(plan)
            return
        if depth >= DIAMOND_POST_ALL_OUT_MAX_RALLY_STEPS:
            return

        non_joker_count = len([
            card for card in current_cpu.hand if not is_joker(card)
        ])
        counts = diamond_rally_count_order(current_cpu, non_joker_count)
        candidates = gold_plan_candidates(current_cpu, empty_room, counts, validator)
        for count in counts:
            if count <= 9:
                candidates.extend(joker_prime_candidates_for_count(
                    current_cpu,
                    empty_room,
                    count,
                    validator,
                ))
        candidates.extend(gold_special_cut_candidates(current_cpu, empty_room))
        capacity = opponent_count + draw_increment * (depth + 1)
        candidates = [
            candidate
            for candidate in dedupe_candidates(candidates)
            if len(candidate_consumed_cards(candidate)) < len(current_cpu.hand)
            and (
                (
                    diamond_four_obake_certainty(
                        candidate,
                        current_cpu,
                        empty_room,
                    )
                    == "certain"
                )
                if diamond_four_candidate_is_face_obake(candidate)
                else diamond_candidate_is_publicly_uncounterable(
                    candidate,
                    current_cpu,
                    empty_room,
                    opponent_capacity=capacity,
                )
            )
        ]
        candidates.sort(key=lambda candidate: (
            1 if len(candidate.get("cards", [])) in DIAMOND_PREFERRED_RALLY_COUNTS else 0,
            len(candidate_consumed_cards(candidate)),
            0 if step_uses_joker(candidate) else 1,
            -candidate_strength(candidate, empty_room),
        ), reverse=True)
        for candidate in candidates[:gold_branch_candidate_cap(current_cpu)]:
            step = dict(candidate)
            step["role"] = f"rally-{len(candidate.get('cards', []))}"
            child = temporary_cpu_with_hand(
                current_cpu,
                remaining_cards(
                    current_cpu.hand,
                    candidate_consumed_cards(candidate),
                ),
            )
            visit(child, selected + [step], depth + 1)

    visit(cpu, [], 0)
    return results


def diamond_reserved_trump_plans(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
    counts: Iterable[int],
    *,
    result_cap: int = DIAMOND_POST_ALL_OUT_RESULT_CAP,
    max_prefix_steps: int = DIAMOND_POST_ALL_OUT_MAX_RALLY_STEPS - 1,
) -> list[dict]:
    """Search backwards from a strong trump before the general five-step tree."""
    if getattr(room, "field", []) or []:
        return []
    empty_room = room_without_field(room)
    results = []
    seen = set()
    for rally_count in counts:
        if not diamond_post_all_out_search_has_time(cpu):
            break
        trump_candidates = gold_plan_candidates(
            cpu,
            empty_room,
            (rally_count,),
            validator,
        )
        if rally_count <= 9:
            trump_candidates.extend(joker_prime_candidates_for_count(
                cpu,
                empty_room,
                rally_count,
                validator,
            ))
        trump_candidates = [
            candidate
            for candidate in dedupe_candidates(trump_candidates)
            if len(candidate_consumed_cards(candidate)) < len(cpu.hand)
            and (
                diamond_post_all_out_candidate_certainty(
                    candidate,
                    cpu,
                    room,
                )
                in {"certain", "conditional"}
                or platinum_candidate_trump_strength(candidate)
                >= DIAMOND_POST_ALL_OUT_SOFT_MIN_TRUMP_STRENGTH
            )
        ]
        trump_candidates.sort(key=lambda candidate: (
            {
                "unclassified": 0,
                "soft": 1,
                "conditional": 2,
                "certain": 3,
            }.get(
                diamond_post_all_out_candidate_certainty(
                    candidate,
                    cpu,
                    room,
                ),
                0,
            ),
            platinum_candidate_trump_strength(candidate),
            0 if step_uses_joker(candidate) else 1,
            -len(candidate_consumed_cards(candidate)),
        ), reverse=True)

        for trump in trump_candidates[:gold_last_candidate_cap(cpu)]:
            if not diamond_post_all_out_search_has_time(cpu):
                break
            reserved = remaining_cards(
                cpu.hand,
                candidate_consumed_cards(trump),
            )
            reserved_cpu = temporary_cpu_with_hand(cpu, reserved)
            trump_strength = candidate_strength(trump, empty_room)

            def visit(
                current_cpu: CpuPlayer,
                bound_strength: int,
                selected_desc: list[dict],
            ) -> None:
                if len(results) >= result_cap:
                    return
                if not diamond_post_all_out_search_has_time(current_cpu):
                    return
                tail = choose_gold_finish_tail(current_cpu, empty_room, validator)
                if tail:
                    sequence = list(reversed(selected_desc)) + [trump] + tail
                    key = tuple(candidate_fingerprint(step) for step in sequence)
                    if key not in seen:
                        seen.add(key)
                        results.append(finalize_gold_plan(
                            cpu,
                            empty_room,
                            sequence,
                            rally_count,
                        ))
                    return
                if len(selected_desc) >= max_prefix_steps:
                    return

                branches = gold_plan_candidates(
                    current_cpu,
                    empty_room,
                    (rally_count,),
                    validator,
                )
                if rally_count <= 9:
                    branches.extend(joker_prime_candidates_for_count(
                        current_cpu,
                        empty_room,
                        rally_count,
                        validator,
                    ))
                branches = [
                    candidate
                    for candidate in dedupe_candidates(branches)
                    if len(candidate_consumed_cards(candidate)) < len(current_cpu.hand)
                    and candidate_strength(candidate, empty_room) < bound_strength
                ]
                branches.sort(key=lambda candidate: (
                    0 if step_uses_joker(candidate) else 1,
                    candidate_strength(candidate, empty_room),
                    -len(candidate_consumed_cards(candidate)),
                ), reverse=True)
                for branch in branches[:gold_branch_candidate_cap(current_cpu)]:
                    child = temporary_cpu_with_hand(
                        current_cpu,
                        remaining_cards(
                            current_cpu.hand,
                            candidate_consumed_cards(branch),
                        ),
                    )
                    visit(
                        child,
                        candidate_strength(branch, empty_room),
                        selected_desc + [branch],
                    )

            visit(reserved_cpu, trump_strength, [])
    return results


def diamond_fast_complete_plan(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
) -> Optional[dict]:
    """Find short 95+ routes before the preferred-count deep search.

    This is intentionally shallow: every useful card count gets a cheap chance,
    while the normal 4/6-first search remains responsible for deeper routes.
    """
    if getattr(room, "field", []) or []:
        return None
    empty_room = room_without_field(room)
    candidates = []

    direct_tail = choose_gold_finish_tail(cpu, empty_room, validator)
    if direct_tail:
        candidates.append(finalize_gold_plan(cpu, empty_room, direct_tail, 0))

    non_joker_count = len([card for card in cpu.hand if not is_joker(card)])
    for rally_count in range(min(9, non_joker_count), 8, -1):
        if not diamond_post_all_out_search_has_time(cpu):
            break
        finish_count = len(cpu.hand) - rally_count
        for finish_tail in gold_finish_tails_for_consumed_count(
            cpu,
            empty_room,
            finish_count,
            validator,
        ):
            consumed = [
                card
                for step in finish_tail
                for card in candidate_consumed_cards(step)
            ]
            remaining = remaining_cards(cpu.hand, consumed)
            if len(remaining) != rally_count:
                continue
            rally_cpu = temporary_cpu_with_hand(cpu, remaining)
            rallies = gold_plan_candidates(
                rally_cpu,
                empty_room,
                (rally_count,),
                validator,
            )
            rallies.extend(joker_prime_candidates_for_count(
                rally_cpu,
                empty_room,
                rally_count,
                validator,
            ))
            for rally in sorted(
                dedupe_candidates(rallies),
                key=lambda candidate: gold_plan_candidate_score(candidate, empty_room),
                reverse=True,
            ):
                if len(candidate_consumed_cards(rally)) != rally_count:
                    continue
                candidates.append(finalize_gold_plan(
                    cpu,
                    empty_room,
                    [rally] + finish_tail,
                    rally_count,
                ))
                if len(candidates) >= GOLD_PLAN_MAX_ALTERNATIVES:
                    break
            if len(candidates) >= GOLD_PLAN_MAX_ALTERNATIVES:
                break

    count_order = tuple(
        count
        for count in (1, 2, 3, 4, 6, 5, 7, 8)
        if count <= min(9, non_joker_count)
    )
    for rally_count in count_order:
        if not diamond_post_all_out_search_has_time(cpu):
            break
        candidates.extend(diamond_reserved_trump_plans(
            cpu,
            empty_room,
            validator,
            (rally_count,),
            result_cap=1,
            max_prefix_steps=1,
        ))

    candidates = [
        plan
        for plan in candidates
        if is_executable_gold_plan(plan, cpu)
        and platinum_opening_multi_play_is_sound(cpu, plan)
        and (
            any(
                str(step.get("role", "")).startswith("rally-")
                and len(step.get("cards", []) or []) >= 9
                for step in plan.get("steps", [])
            )
            or
            platinum_plan_score(plan) >= DIAMOND_OPENING_SECOND_MIN_TRUMP_STRENGTH
            or diamond_plan_certainty(plan, cpu, room) == "certain"
        )
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda plan: (
            1 if any(
                str(step.get("role", "")).startswith("rally-")
                and len(step.get("cards", []) or []) >= 9
                for step in plan.get("steps", [])
            ) else 0,
            diamond_plan_sort_key(plan, cpu, room),
        ),
    )


def diamond_post_all_out_plans(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
) -> list[dict]:
    plans = []
    field_count = len(getattr(room, "field", []) or [])
    non_joker_count = len([card for card in cpu.hand if not is_joker(card)])
    count_order = (
        (field_count,)
        if field_count
        else diamond_rally_count_order(cpu, non_joker_count)
    )
    preferred = tuple(
        count for count in count_order if count in DIAMOND_PREFERRED_RALLY_COUNTS
    )
    fallback = tuple(count for count in count_order if count not in preferred)

    plans.extend(diamond_reserved_trump_plans(
        cpu,
        room,
        validator,
        preferred,
    ))
    if any(diamond_post_all_out_plan_tier(plan, cpu, room) == 1 for plan in plans):
        return plans
    plans.extend(diamond_reserved_trump_plans(
        cpu,
        room,
        validator,
        fallback,
    ))
    if any(
        diamond_post_all_out_plan_tier(plan, cpu, room) in {1, 2}
        for plan in plans
    ):
        return plans
    plans.extend(diamond_forced_pass_plans(cpu, room, validator))
    if any(
        diamond_post_all_out_plan_tier(plan, cpu, room) in {1, 2}
        for plan in plans
    ):
        return plans

    def search(counts: Iterable[int]) -> None:
        for rally_count in counts:
            if not diamond_post_all_out_search_has_time(cpu):
                return
            try:
                found = search_same_count_gold_plans(
                    cpu,
                    room,
                    rally_count,
                    PLATINUM_PLAN_MAX_STEPS,
                    validator,
                )
            except CpuSearchDeadline:
                if plans:
                    return
                raise
            plans.extend(
                plan
                for plan in found
                if is_executable_gold_plan(plan, cpu)
                and diamond_plan_rally_step_count(plan) > 0
            )

    search(preferred)
    if any(diamond_post_all_out_plan_tier(plan, cpu, room) == 1 for plan in plans):
        return plans
    search(fallback)
    unique = {}
    for plan in plans:
        key = tuple(candidate_fingerprint(step) for step in plan.get("steps", []))
        unique[key] = plan
    return list(unique.values())


def diamond_draw_can_create_four_card_overtrump(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
) -> bool:
    if (
        len(cpu.hand) < DIAMOND_POST_ALL_OUT_DRAW_RESPONSE_MIN_HAND_SIZE
        or len(getattr(room, "field", []) or []) != 4
        or getattr(room, "has_drawn", False)
        or not getattr(room, "deck", [])
    ):
        return False
    threshold = max(
        int(getattr(room, "last_number", 0) or 0),
        platinum_token_value("kjqj"),
    )
    available, joker_count = diamond_publicly_unaccounted_rank_counts(cpu, room)
    ranks = [rank for rank, count in available.items() if count > 0]
    if joker_count:
        ranks.append(0)
    for rank in ranks:
        synthetic = {
            "card_id": f"diamond-one-draw-{rank}",
            "suit": "X" if rank == 0 else "?",
            "rank": rank,
            "is_joker": rank == 0,
        }
        child = temporary_cpu_with_hand(cpu, cpu.hand + [synthetic])
        candidates = gold_plan_candidates(child, room, (4,), validator)
        candidates.extend(joker_prime_candidates_for_count(
            child,
            room,
            4,
            validator,
        ))
        if any(
            candidate_is_playable(candidate, child, room)
            and int(candidate.get("number", 0)) > threshold
            for candidate in dedupe_candidates(candidates)
            if candidate.get("number") != "X"
        ):
            return True
    return False


def diamond_threat_closing_response_candidate(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
) -> Optional[dict]:
    field_count = len(getattr(room, "field", []) or [])
    opponent_count = platinum_opponent_hand_count(cpu, room)
    if (
        field_count <= 0
        or opponent_count is None
        or (
            opponent_count > 5
            and getattr(cpu, "diamond_opponent_rally_threat", "none")
            != "certain-likely"
        )
    ):
        return None
    estimated = diamond_estimated_opponent_max_value(cpu, room, field_count)
    if estimated is None:
        return None
    target_strength = (
        -estimated if getattr(room, "reverse_order", False) else estimated
    )
    candidates = platinum_legal_response_candidates(
        cpu,
        room,
        validator,
        allow_non_trump_joker=True,
    )
    if not candidates:
        return None

    exact_blockers = [
        candidate
        for candidate in candidates
        if candidate_strength(candidate, room) >= target_strength
    ]
    pool = exact_blockers or candidates

    def closing_key(candidate: dict) -> tuple:
        remaining = remaining_cards(cpu.hand, candidate_consumed_cards(candidate))
        child = temporary_cpu_with_hand(cpu, remaining)
        tail = choose_gold_finish_tail(child, room_without_field(room), validator)
        candidate["_diamond_closing_tail"] = tail
        tail_finish_strength = (
            diamond_plan_finish_strength(finalize_gold_plan(
                child,
                room_without_field(room),
                tail,
                0,
            ))
            if tail
            else 0
        )
        return (
            1 if tail else 0,
            -len(tail) if tail else 0,
            tail_finish_strength,
            0 if step_uses_joker(candidate) else 1,
            len(candidate_consumed_cards(candidate)),
            (
                -candidate_strength(candidate, room)
                if exact_blockers
                else candidate_strength(candidate, room)
            ),
        )

    return max(pool, key=closing_key)


def diamond_hnp_joker_assignments(cards: list[Card]) -> list[tuple[int, ...]]:
    """Return joker values that do not make every permutation divisible by 3."""
    jokers = [card for card in cards if is_joker(card)]
    fixed_sum = sum(
        int(card.get("rank", 0)) for card in cards if not is_joker(card)
    )
    return [
        values
        for values in product((1, 3, 7, 9), repeat=len(jokers))
        if (fixed_sum + sum(values)) % 3 != 0
    ]


def diamond_opening_second_hnp_action(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
) -> Optional[CpuAction]:
    """Try an 8+-card HNP response while preserving a legal one-move finish.

    This is deliberately narrower than ordinary HNP: Diamond must have started
    second, the first player's hand must never have reached 13 cards, and the
    current field must be a prime play of at least eight cards.  A partition is
    eligible only when every card left out of the HNP forms a known legal finish.
    """
    field = list(getattr(room, "field", []) or [])
    field_count = len(field)
    if (
        field_count < DIAMOND_OPENING_SECOND_HNP_MIN_FIELD_COUNT
        or not getattr(cpu, "diamond_opening_was_second", False)
        or getattr(cpu, "diamond_opponent_ever_13_plus", False)
        or getattr(room, "last_play_kind", None) != "prime"
        or getattr(cpu, "gold_active_plan", None)
        or field_count >= len(cpu.hand)
    ):
        return None

    finish_card_count = len(cpu.hand) - field_count
    empty_room = room_without_field(room)
    options = []
    for finish_cards_tuple in combinations(cpu.hand, finish_card_count):
        check_cpu_search_deadline(cpu)
        finish_cards = list(finish_cards_tuple)
        finish_cpu = temporary_cpu_with_hand(cpu, finish_cards)
        finish = platinum_one_move_finish_candidate(
            finish_cpu,
            empty_room,
            validator,
        )
        if finish is None:
            continue
        if {
            card.get("card_id") for card in candidate_consumed_cards(finish)
        } != {card.get("card_id") for card in finish_cards}:
            continue
        hnp_cards = remaining_cards(cpu.hand, finish_cards)
        assignments = diamond_hnp_joker_assignments(hnp_cards)
        if not assignments:
            continue
        options.append((finish, finish_cards, hnp_cards, assignments))

    if not options:
        return None

    options.sort(
        key=lambda item: (
            gold_plan_candidate_score(item[0], empty_room),
            sum(is_joker(card) for card in item[1]),
            tuple(sorted(str(card.get("card_id")) for card in item[1])),
        ),
        reverse=True,
    )
    field_number = getattr(room, "last_number", None)
    for finish, finish_cards, hnp_cards, assignments in options:
        ordered_assignments = list(assignments)
        cpu.rng.shuffle(ordered_assignments)
        for joker_values in ordered_assignments:
            tokens = build_hnp_tokens(
                hnp_cards,
                [str(value) for value in joker_values],
            )
            permutation = choose_hnp_permutation(
                tokens,
                field_number=int(field_number) if field_number is not None else None,
                reverse_order=bool(getattr(room, "reverse_order", False)),
                randbelow=lambda upper: cpu.rng.randrange(upper),
            )
            if permutation is None:
                continue

            hnp_candidate = {
                "kind": "prime",
                "number": permutation.number,
                "cards": permutation.cards,
                "assigned_numbers": permutation.assigned_numbers,
                "ranks": tuple(int(token.text) for token in permutation.tokens),
                "role": f"rally-{field_count}",
            }
            finish_cpu = temporary_cpu_with_hand(cpu, finish_cards)
            tail_plan = finalize_gold_plan(
                finish_cpu,
                empty_room,
                [dict(finish)],
                0,
            )
            if not tail_plan.get("completed"):
                continue
            tail_plan["diamond_hnp_tail"] = True
            tail_plan["diamond_hnp_field_count"] = field_count
            set_gold_active_plan(cpu, tail_plan)
            action = platinum_commit_play(cpu, candidate_to_action(hnp_candidate))
            result = diamond_record_action(
                cpu,
                action,
                room,
                "opening-second-hnp",
                hnp_candidate,
            )
            cpu.diamond_last_certainty = "conditional"
            return result
    return None


def diamond_kx_card_count(cards: Iterable[Card]) -> int:
    return sum(
        1
        for card in cards
        if is_joker(card) or int(card.get("rank", 0)) == 13
    )


def diamond_kx_policy_band(kx_count: int) -> str:
    if kx_count <= 0:
        return "0"
    if kx_count <= 2:
        return "1-2"
    if kx_count == 3:
        return "3"
    return "4-5"


def diamond_set_kx_policy_trace(
    cpu: CpuPlayer,
    *,
    before_kx: int,
    after_kx: Optional[int] = None,
    reason: str = "",
    retained_certain: Optional[dict] = None,
    predicted_deck: Optional[int] = None,
    recovery_capacity: Optional[int] = None,
    pass_reason: str = "",
) -> None:
    after = before_kx if after_kx is None else int(after_kx)
    cpu.diamond_last_kx_policy = {
        "band": diamond_kx_policy_band(before_kx),
        "before_kx": int(before_kx),
        "after_kx": after,
        "spent_kx": max(0, int(before_kx) - after),
        "reason": str(reason),
        "retained_certain": (
            platinum_candidate_token(retained_certain)
            if retained_certain is not None
            else None
        ),
        "predicted_deck": (
            int(predicted_deck) if predicted_deck is not None else None
        ),
        "recovery_capacity": (
            int(recovery_capacity) if recovery_capacity is not None else None
        ),
        "full_recovery": bool(
            predicted_deck is not None
            and recovery_capacity is not None
            and int(predicted_deck) <= int(recovery_capacity)
        ),
        "pass_reason": str(pass_reason),
    }


def diamond_retained_certain_closeout_plan_is_valid(
    plan: Optional[dict],
    cpu: CpuPlayer,
    room,
    *,
    max_rally_steps: int,
) -> bool:
    """Require a final certain rally followed by exactly one finishing play."""
    if not plan or not is_executable_gold_plan(plan, cpu):
        return False
    steps = list(plan.get("steps", []))
    planned_ids = {
        str(card.get("card_id"))
        for step in steps
        for card in candidate_consumed_cards(step)
    }
    if planned_ids != {str(card.get("card_id")) for card in cpu.hand}:
        return False
    trump_index = gold_plan_trump_step_index(plan)
    if (
        trump_index is None
        or trump_index != len(steps) - 2
        or steps[-1].get("role") != "finish"
        or diamond_plan_rally_step_count(plan) > max_rally_steps
    ):
        return False
    if any(
        diamond_kx_card_count(candidate_consumed_cards(step)) > 0
        for step in steps[:trump_index]
    ):
        return False
    trump = steps[trump_index]
    return diamond_candidate_certainty(trump, cpu, room) == "certain"


def diamond_retained_certain_closeout_plan(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
    *,
    max_rally_steps: int,
    max_prefix_steps: Optional[int] = None,
) -> Optional[dict]:
    if getattr(room, "field", []) or [] or max_rally_steps <= 0:
        return None
    non_joker_count = len([card for card in cpu.hand if not is_joker(card)])
    counts = diamond_rally_count_order(cpu, non_joker_count)
    prefix_limit = (
        max_rally_steps - 1
        if max_prefix_steps is None
        else min(max_rally_steps - 1, max(0, int(max_prefix_steps)))
    )
    plans = diamond_reserved_trump_plans(
        cpu,
        room,
        validator,
        counts,
        result_cap=max(DIAMOND_POST_ALL_OUT_RESULT_CAP, len(counts)),
        max_prefix_steps=prefix_limit,
    )
    eligible = [
        plan
        for plan in plans
        if diamond_retained_certain_closeout_plan_is_valid(
            plan,
            cpu,
            room,
            max_rally_steps=max_rally_steps,
        )
    ]
    if not eligible:
        return None
    best = max(
        eligible,
        key=lambda plan: (
            -diamond_plan_rally_step_count(plan),
            diamond_plan_finish_strength(plan),
            1 if diamond_plan_has_protected_x_tail(plan) else 0,
            platinum_plan_score(plan),
            gold_plan_score(plan),
        ),
    )
    best["diamond_kx_retained_closeout"] = True
    return best


def diamond_kx_active_plan_suffix(cpu: CpuPlayer) -> Optional[dict]:
    plan = getattr(cpu, "gold_active_plan", None)
    if not plan or not plan.get("diamond_kx_response_plan"):
        return None
    return diamond_gold_plan_suffix(cpu)


def diamond_gold_plan_suffix(cpu: CpuPlayer) -> Optional[dict]:
    plan = getattr(cpu, "gold_active_plan", None)
    if not plan:
        return None
    index = int(getattr(cpu, "gold_plan_step_index", 0))
    steps = list(plan.get("steps", []))[index:]
    if not steps:
        return None
    suffix = dict(plan)
    suffix["steps"] = steps
    suffix["remaining"] = []
    suffix["completed"] = True
    return suffix


def diamond_resume_post_all_out_kx_state(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
) -> Optional[CpuAction]:
    """Resume a forced recovery or a separately preserved closeout plan."""
    field = list(getattr(room, "field", []) or [])
    pending = getattr(cpu, "diamond_pending_full_recovery", None)
    if pending and field:
        cpu.diamond_pending_full_recovery = None
    elif pending and not field:
        finish = platinum_one_move_finish_candidate(cpu, room, validator)
        if finish is not None:
            cpu.diamond_pending_full_recovery = None
            cpu.diamond_preserved_closeout_plan = None
            clear_gold_active_plan(cpu)
            before_kx = diamond_kx_card_count(cpu.hand)
            diamond_set_kx_policy_trace(
                cpu,
                before_kx=before_kx,
                after_kx=0,
                reason="pending-recovery-direct-finish",
            )
            action = platinum_commit_play(cpu, candidate_to_action(finish))
            return diamond_record_action(
                cpu,
                action,
                room,
                "post-all-out-kx-direct-finish",
                finish,
            )
        if len(getattr(room, "deck", []) or []) <= len(cpu.hand):
            payload = build_gold_all_out_payload(
                cpu.hand,
                force_random=True,
                rng=cpu.rng,
            )
            if payload is not None:
                before_kx = diamond_kx_card_count(cpu.hand)
                predicted = len(getattr(room, "deck", []) or [])
                cpu.diamond_pending_full_recovery = None
                cpu.diamond_preserved_closeout_plan = None
                clear_gold_active_plan(cpu)
                cpu.platinum_all_out_attempts += 1
                cpu.diamond_last_route_kind = "post-all-out-full-recovery-all-out"
                cpu.diamond_last_certainty = "unclassified"
                diamond_set_kx_policy_trace(
                    cpu,
                    before_kx=before_kx,
                    after_kx=before_kx,
                    reason="full-recovery-all-out",
                    predicted_deck=predicted,
                    recovery_capacity=len(cpu.hand),
                )
                return platinum_commit_play(cpu, CpuAction("play_prime", payload))
        cpu.diamond_pending_full_recovery = None

    active_suffix = diamond_kx_active_plan_suffix(cpu)
    if active_suffix is not None and not diamond_retained_certain_closeout_plan_is_valid(
        active_suffix,
        cpu,
        room_without_field(room),
        max_rally_steps=DIAMOND_POST_ALL_OUT_MAX_RALLY_STEPS,
    ):
        clear_gold_active_plan(cpu)

    preserved = getattr(cpu, "diamond_preserved_closeout_plan", None)
    if not field and preserved:
        finish = platinum_one_move_finish_candidate(cpu, room, validator)
        if finish is not None:
            cpu.diamond_preserved_closeout_plan = None
            clear_gold_active_plan(cpu)
            before_kx = diamond_kx_card_count(cpu.hand)
            diamond_set_kx_policy_trace(
                cpu,
                before_kx=before_kx,
                after_kx=0,
                reason="preserved-plan-direct-finish",
            )
            action = platinum_commit_play(cpu, candidate_to_action(finish))
            return diamond_record_action(
                cpu,
                action,
                room,
                "post-all-out-kx-direct-finish",
                finish,
            )
        if diamond_retained_certain_closeout_plan_is_valid(
            preserved,
            cpu,
            room,
            max_rally_steps=DIAMOND_POST_ALL_OUT_MAX_RALLY_STEPS,
        ):
            preserved["diamond_kx_response_plan"] = True
            set_gold_active_plan(cpu, preserved)
            cpu.diamond_preserved_closeout_plan = None
            next_step = preserved.get("steps", [])[0]
            before_kx = diamond_kx_card_count(cpu.hand)
            after_kx = before_kx - diamond_kx_card_count(
                candidate_consumed_cards(next_step)
            )
            diamond_set_kx_policy_trace(
                cpu,
                before_kx=before_kx,
                after_kx=after_kx,
                reason="resume-preserved-closeout",
                retained_certain=diamond_plan_trump_candidate(preserved),
            )
            action = play_next_gold_plan_step(cpu, room, validator)
            if action is not None:
                action = platinum_commit_play(cpu, action)
                return diamond_record_action(
                    cpu,
                    action,
                    room,
                    "post-all-out-kx-preserved-closeout",
                    next_step,
                )
            clear_gold_active_plan(cpu)
        cpu.diamond_preserved_closeout_plan = None
    return None


def diamond_kx_response_scope(cpu: CpuPlayer, room) -> bool:
    opponent_count = platinum_opponent_hand_count(cpu, room)
    return bool(
        diamond_tactical_context(cpu, room) == "post-all-out-response"
        and getattr(room, "field", [])
        and len(cpu.hand) >= DIAMOND_POST_ALL_OUT_KX_POLICY_MIN_HAND_SIZE
        and opponent_count is not None
        and opponent_count >= DIAMOND_POST_ALL_OUT_KX_POLICY_MIN_HAND_SIZE
        and diamond_kx_card_count(cpu.hand) > 0
    )


def diamond_response_pass_state(
    cpu: CpuPlayer,
    room,
    candidate: dict,
) -> tuple[CpuPlayer, object]:
    consumed = candidate_consumed_cards(candidate)
    child = temporary_cpu_with_hand(
        cpu,
        remaining_cards(cpu.hand, consumed),
    )
    diamond_remember_cards(child, consumed)
    return child, room_without_field(room)


def diamond_kx_response_choice_key(record: dict, room) -> tuple:
    candidate = record["candidate"]
    certainty = str(record.get("certainty", "unclassified"))
    certainty_score = {
        "unclassified": 0,
        "soft": 1,
        "conditional": 2,
        "certain": 3,
    }.get(certainty, 0)
    return (
        certainty_score,
        -int(record["kx_spent"]),
        int(record["after_kx"]),
        candidate_strength(candidate, room),
        -len(candidate_consumed_cards(candidate)),
    )


def diamond_commit_kx_response(
    cpu: CpuPlayer,
    room,
    record: dict,
    *,
    reason: str,
) -> CpuAction:
    candidate = record["candidate"]
    plan = record.get("plan")
    clear_gold_active_plan(cpu)
    cpu.diamond_preserved_closeout_plan = None
    cpu.diamond_pending_full_recovery = None
    if plan is not None:
        plan["diamond_kx_response_plan"] = True
        plan["diamond_kx_reason"] = reason
        set_gold_active_plan(cpu, plan)
    if reason == "full-recovery":
        cpu.diamond_pending_full_recovery = {
            "predicted_deck": int(record["predicted_deck"]),
            "recovery_capacity": int(record["recovery_capacity"]),
            "response": candidate_fingerprint(candidate),
        }
    diamond_set_kx_policy_trace(
        cpu,
        before_kx=int(record["before_kx"]),
        after_kx=int(record["after_kx"]),
        reason=reason,
        retained_certain=(
            diamond_plan_trump_candidate(plan) if plan is not None else None
        ),
        predicted_deck=record.get("predicted_deck"),
        recovery_capacity=record.get("recovery_capacity"),
    )
    action = platinum_commit_play(cpu, candidate_to_action(candidate))
    return diamond_record_action(
        cpu,
        action,
        room,
        f"post-all-out-kx-{reason}",
        candidate,
    )


def diamond_continue_kx_active_plan(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
    active: dict,
    suffix: dict,
    next_step: dict,
    before_kx: int,
) -> Optional[CpuAction]:
    after_kx = before_kx - diamond_kx_card_count(
        candidate_consumed_cards(next_step)
    )
    diamond_set_kx_policy_trace(
        cpu,
        before_kx=before_kx,
        after_kx=after_kx,
        reason=(
            "kx-free-retained-certain"
            if after_kx == before_kx
            else "retained-certain"
        ),
        retained_certain=diamond_plan_trump_candidate(suffix),
    )
    active["diamond_kx_response_plan"] = True
    action = play_next_gold_plan_step(cpu, room, validator)
    if action is None:
        clear_gold_active_plan(cpu)
        return None
    action = platinum_commit_play(cpu, action)
    return diamond_record_action(
        cpu,
        action,
        room,
        "post-all-out-kx-continue-closeout",
        next_step,
    )


def choose_diamond_post_all_out_kx_response_action(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
) -> Optional[CpuAction]:
    """Gate large, balanced post-all-out responses by K/X closeout value."""
    if not diamond_kx_response_scope(cpu, room):
        return None
    if getattr(cpu, "diamond_opponent_rally_threat", "none") == "certain-likely":
        return None

    before_kx = diamond_kx_card_count(cpu.hand)
    band = diamond_kx_policy_band(before_kx)
    finish = platinum_one_move_finish_candidate(cpu, room, validator)
    if finish is not None:
        record = {
            "candidate": finish,
            "before_kx": before_kx,
            "after_kx": 0,
            "kx_spent": before_kx,
        }
        return diamond_commit_kx_response(
            cpu,
            room,
            record,
            reason="direct-finish",
        )

    preserved_fallback = getattr(cpu, "diamond_preserved_closeout_plan", None)
    if not diamond_retained_certain_closeout_plan_is_valid(
        preserved_fallback,
        cpu,
        room_without_field(room),
        max_rally_steps=DIAMOND_POST_ALL_OUT_MAX_RALLY_STEPS,
    ):
        preserved_fallback = diamond_gold_plan_suffix(cpu)
        fallback_trump_index = gold_plan_trump_step_index(preserved_fallback)
        if (
            fallback_trump_index is None
            or fallback_trump_index > 1
            or not diamond_retained_certain_closeout_plan_is_valid(
                preserved_fallback,
                cpu,
                room_without_field(room),
                max_rally_steps=DIAMOND_POST_ALL_OUT_MAX_RALLY_STEPS,
            )
        ):
            preserved_fallback = None

    active = getattr(cpu, "gold_active_plan", None)
    suffix = diamond_gold_plan_suffix(cpu)
    active_continuation = None
    if suffix is not None and diamond_retained_certain_closeout_plan_is_valid(
        suffix,
        cpu,
        room_without_field(room),
        max_rally_steps=DIAMOND_POST_ALL_OUT_MAX_RALLY_STEPS,
    ):
        steps = list(active.get("steps", []))
        index = int(getattr(cpu, "gold_plan_step_index", 0))
        next_step = steps[index] if index < len(steps) else None
        if next_step is not None and candidate_is_playable(next_step, cpu, room):
            active_continuation = (active, suffix, next_step)
        else:
            clear_gold_active_plan(cpu)

    candidates = platinum_legal_response_candidates(
        cpu,
        room,
        validator,
        allow_non_trump_joker=True,
    )
    records = []
    predicted_base = (
        len(getattr(room, "deck", []) or [])
        + len(getattr(room, "reserve", []) or [])
    )
    for candidate in candidates:
        consumed = candidate_consumed_cards(candidate)
        child, empty_room = diamond_response_pass_state(cpu, room, candidate)
        after_kx = diamond_kx_card_count(child.hand)
        child_finish = platinum_one_move_finish_candidate(
            child,
            empty_room,
            validator,
        )
        finish_plan = None
        if child_finish is not None:
            child_finish = dict(child_finish)
            child_finish["role"] = "finish"
            finish_plan = finalize_gold_plan(
                child,
                empty_room,
                [child_finish],
                0,
            )
            finish_plan["diamond_kx_direct_tail"] = True
        predicted_deck = predicted_base + len(consumed)
        records.append({
            "candidate": candidate,
            "child": child,
            "empty_room": empty_room,
            "before_kx": before_kx,
            "after_kx": after_kx,
            "kx_spent": before_kx - after_kx,
            "certainty": "unclassified",
            "plan": finish_plan,
            "predicted_deck": predicted_deck,
            "recovery_capacity": len(child.hand),
            "full_recovery": predicted_deck <= len(child.hand),
        })

    direct_tails = [record for record in records if record.get("plan") is not None]
    if direct_tails:
        chosen = max(
            direct_tails,
            key=lambda record: diamond_kx_response_choice_key(record, room),
        )
        return diamond_commit_kx_response(
            cpu,
            room,
            chosen,
            reason="next-move-finish",
        )

    recoveries = [record for record in records if record["full_recovery"]]
    if band == "1-2" and recoveries:
        chosen = max(
            recoveries,
            key=lambda record: diamond_kx_response_choice_key(record, room),
        )
        return diamond_commit_kx_response(
            cpu,
            room,
            chosen,
            reason="full-recovery",
        )

    if active_continuation is not None:
        active, suffix, next_step = active_continuation
        continued = diamond_continue_kx_active_plan(
            cpu,
            room,
            validator,
            active,
            suffix,
            next_step,
            before_kx,
        )
        if continued is not None:
            return continued

    retained = []
    for record in sorted(
        records,
        key=lambda item: diamond_kx_response_choice_key(item, room),
        reverse=True,
    ):
        if not diamond_post_all_out_search_has_time(cpu):
            break
        plan = diamond_retained_certain_closeout_plan(
            record["child"],
            record["empty_room"],
            validator,
            max_rally_steps=DIAMOND_POST_ALL_OUT_MAX_RALLY_STEPS - 1,
        )
        if plan is None:
            continue
        record["plan"] = plan
        retained.append(record)
    if retained:
        chosen = max(
            retained,
            key=lambda record: (
                -diamond_plan_rally_step_count(record["plan"]),
                diamond_plan_finish_strength(record["plan"]),
                diamond_kx_response_choice_key(record, room),
            ),
        )
        return diamond_commit_kx_response(
            cpu,
            room,
            chosen,
            reason=(
                "kx-free-retained-certain"
                if chosen["kx_spent"] == 0
                else "retained-certain"
            ),
        )

    if recoveries:
        chosen = max(
            recoveries,
            key=lambda record: diamond_kx_response_choice_key(record, room),
        )
        return diamond_commit_kx_response(
            cpu,
            room,
            chosen,
            reason="full-recovery",
        )

    preserved = preserved_fallback
    if not diamond_retained_certain_closeout_plan_is_valid(
        preserved_fallback,
        cpu,
        room_without_field(room),
        max_rally_steps=DIAMOND_POST_ALL_OUT_MAX_RALLY_STEPS,
    ):
        preserved = diamond_retained_certain_closeout_plan(
            cpu,
            room_without_field(room),
            validator,
            max_rally_steps=DIAMOND_POST_ALL_OUT_MAX_RALLY_STEPS,
            max_prefix_steps=1,
        )
    if preserved is not None:
        clear_gold_active_plan(cpu)
        cpu.diamond_preserved_closeout_plan = preserved
        cpu.diamond_last_route_kind = "post-all-out-kx-preserve-pass"
        cpu.diamond_last_certainty = "certain"
        diamond_set_kx_policy_trace(
            cpu,
            before_kx=before_kx,
            after_kx=before_kx,
            reason="preserve-pass",
            retained_certain=diamond_plan_trump_candidate(preserved),
            pass_reason="zero-or-one-prefix-certain-finish",
        )
        return CpuAction("pass")

    cpu.diamond_preserved_closeout_plan = None
    clear_gold_active_plan(cpu)
    if (
        not getattr(room, "has_drawn", False)
        and bool(getattr(room, "deck", []) or [])
    ):
        cpu.diamond_last_route_kind = "post-all-out-kx-single-draw"
        cpu.diamond_last_certainty = "unclassified"
        diamond_set_kx_policy_trace(
            cpu,
            before_kx=before_kx,
            after_kx=before_kx,
            reason="single-draw",
            pass_reason="no-approved-response-or-preserved-closeout",
        )
        return CpuAction("draw")

    cpu.diamond_last_route_kind = "post-all-out-kx-preserve-pass"
    cpu.diamond_last_certainty = "unclassified"
    diamond_set_kx_policy_trace(
        cpu,
        before_kx=before_kx,
        after_kx=before_kx,
        reason="preserve-pass",
        pass_reason="draw-used-no-approved-response",
    )
    return CpuAction("pass")


def choose_diamond_context_action(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
) -> Optional[CpuAction]:
    context = diamond_tactical_context(cpu, room)
    cpu.diamond_last_context = context
    if context not in DIAMOND_POST_ALL_OUT_CONTEXTS:
        return None
    if getattr(cpu, "gold_active_plan", None):
        return None

    finish = platinum_one_move_finish_candidate(cpu, room, validator)
    if finish is not None:
        action = platinum_commit_play(cpu, candidate_to_action(finish))
        return diamond_record_action(
            cpu,
            action,
            room,
            "context-finish",
            finish,
        )

    has_field = bool(getattr(room, "field", []) or [])
    if has_field:
        closing = diamond_threat_closing_response_candidate(
            cpu,
            room,
            validator,
        )
        if closing is not None:
            clear_gold_active_plan(cpu)
            closing_tail = list(closing.pop("_diamond_closing_tail", []) or [])
            if closing_tail:
                child = temporary_cpu_with_hand(
                    cpu,
                    remaining_cards(cpu.hand, candidate_consumed_cards(closing)),
                )
                set_gold_active_plan(
                    cpu,
                    finalize_gold_plan(
                        child,
                        room_without_field(room),
                        closing_tail,
                        0,
                    ),
                )
            action = platinum_commit_play(cpu, candidate_to_action(closing))
            return diamond_record_action(
                cpu,
                action,
                room,
                "threat-closing-response",
                closing,
            )

    plans = diamond_post_all_out_plans(cpu, room, validator)
    tiered = [
        plan
        for plan in plans
        if diamond_post_all_out_plan_tier(plan, cpu, room) is not None
    ]
    eligible = [
        plan
        for plan in tiered
        if diamond_plan_lead_avoids_opponent_finish(plan, cpu, room)
    ]
    if not eligible:
        if (
            not has_field
            and tiered
            and not getattr(room, "has_drawn", False)
            and getattr(room, "deck", [])
        ):
            clear_gold_active_plan(cpu)
            cpu.diamond_last_route_kind = "opponent-finish-count-avoidance-draw"
            cpu.diamond_last_certainty = "unclassified"
            return CpuAction("draw")
        if has_field and diamond_draw_can_create_four_card_overtrump(
            cpu,
            room,
            validator,
        ):
            clear_gold_active_plan(cpu)
            cpu.diamond_last_route_kind = "four-card-overtrump-draw"
            cpu.diamond_last_certainty = "conditional"
            return CpuAction("draw")
        return None
    best = max(
        eligible,
        key=lambda plan: diamond_post_all_out_plan_sort_key(plan, cpu, room),
    )
    tier = diamond_post_all_out_plan_tier(best, cpu, room)

    # A current certain/strong complete response takes precedence over drawing
    # for a possible four-card overtrump.  This preserves the turn when the
    # opponent is already close to finishing.
    if (
        has_field
        and tier is not None
        and tier > 6
        and diamond_draw_can_create_four_card_overtrump(cpu, room, validator)
    ):
        clear_gold_active_plan(cpu)
        cpu.diamond_last_route_kind = "four-card-overtrump-draw"
        cpu.diamond_last_certainty = "conditional"
        return CpuAction("draw")

    if (
        tier is not None
        and tier >= 8
        and (
            context != "opponent-all-out"
            or diamond_should_draw_before_opponent_all_out_plan(
                best,
                tier,
                cpu,
                room,
            )
        )
        and not getattr(room, "has_drawn", False)
        and getattr(room, "deck", [])
        and platinum_deck_has_expected_trump_contribution(cpu, room)
    ):
        clear_gold_active_plan(cpu)
        cpu.diamond_last_route_kind = "post-all-out-recovery-draw"
        cpu.diamond_last_certainty = diamond_post_all_out_plan_certainty(
            best,
            cpu,
            room,
        )
        return CpuAction("draw")

    best["diamond_tier"] = tier
    best["diamond_finish_strength"] = diamond_plan_finish_strength(best)
    set_gold_active_plan(cpu, best)
    action = play_next_gold_plan_step(cpu, room, validator)
    if action is None:
        clear_gold_active_plan(cpu)
        return None
    action = platinum_commit_play(cpu, action)
    cpu.diamond_active_route = best
    cpu.diamond_last_plan_tier = tier
    cpu.diamond_last_finish_strength = best["diamond_finish_strength"]
    recorded = diamond_record_action(
        cpu,
        action,
        room,
        f"post-all-out-tier-{tier}",
    )
    cpu.diamond_last_certainty = diamond_post_all_out_plan_certainty(
        best,
        cpu,
        room,
    )
    return recorded


def diamond_recovery_draw_count(cpu: CpuPlayer, room) -> int:
    if hasattr(room, "public_unknown_deck_count"):
        deck_count = max(0, int(room.public_unknown_deck_count)) + len(
            getattr(room, "public_known_deck_bottom", []) or []
        )
    else:
        deck_count = len(getattr(room, "deck", []) or [])
    if deck_count <= 0:
        return 0
    penalty_rule = getattr(getattr(room, "rule", None), "penalty_rule", None)
    requested = 1 if getattr(penalty_rule, "name", "") == "ALWAYS_1" else len(cpu.hand)
    return min(max(0, requested), deck_count)


def diamond_recovery_draw_model(
    cpu: CpuPlayer,
    room,
) -> tuple[list[int], list[Card], int]:
    """Return unknown-rank pool, guaranteed bottom cards reached, and draw count."""
    draw_count = diamond_recovery_draw_count(cpu, room)
    if draw_count <= 0:
        return [], [], 0
    known_bottom = list(getattr(room, "public_known_deck_bottom", []) or [])
    if hasattr(room, "public_unknown_deck_count"):
        unknown_deck_count = max(0, int(room.public_unknown_deck_count))
    else:
        unknown_deck_count = max(
            0,
            len(getattr(room, "deck", []) or []) - len(known_bottom),
        )
    unknown_draw_count = min(draw_count, unknown_deck_count)
    known_draw_count = min(
        len(known_bottom),
        max(0, draw_count - unknown_deck_count),
    )
    guaranteed = known_bottom[:known_draw_count]

    available, joker_count = diamond_publicly_unaccounted_rank_counts(cpu, room)
    for card in known_bottom:
        if is_joker(card):
            joker_count = max(0, joker_count - 1)
        else:
            rank = int(card.get("rank", 0))
            if rank in available:
                available[rank] = max(0, available[rank] - 1)
    pool = [
        rank
        for rank, count in sorted(available.items())
        for _ in range(count)
    ] + [0] * joker_count
    return pool, guaranteed, min(unknown_draw_count, len(pool))


def diamond_recovery_certain_probability(cpu: CpuPlayer, room) -> float:
    """Estimate P(no certain -> certain) from a failed all-out recovery."""
    if diamond_has_current_certain_trump(cpu, room):
        cpu.diamond_last_recovery_certain_probability = 0.0
        cpu.diamond_last_recovery_guaranteed_kx = 0
        cpu.diamond_recovery_targets = ()
        cpu.diamond_recovery_cache_key = None
        return 0.0

    def public_card_key(card: Card) -> tuple:
        return (
            str(card.get("card_id", "")),
            0 if is_joker(card) else int(card.get("rank", 0)),
        )

    cache_key = (
        tuple(sorted(public_card_key(card) for card in cpu.hand)),
        tuple(public_card_key(card) for card in getattr(room, "field", []) or []),
        tuple(public_card_key(card) for card in getattr(room, "reserve", []) or []),
        tuple(
            public_card_key(card)
            for card in getattr(room, "public_known_deck_bottom", []) or []
        ),
        int(getattr(room, "public_unknown_deck_count", 0) or 0),
        len(getattr(room, "deck", []) or []),
        platinum_opponent_hand_count(cpu, room),
        bool(getattr(room, "reverse_order", False)),
    )
    if cache_key == getattr(cpu, "diamond_recovery_cache_key", None):
        return float(cpu.diamond_last_recovery_certain_probability)
    cpu.diamond_recovery_cache_key = cache_key

    pool, guaranteed, unknown_draw_count = diamond_recovery_draw_model(cpu, room)
    guaranteed_kx = sum(
        1
        for card in guaranteed
        if is_joker(card) or int(card.get("rank", 0)) == 13
    )
    non_kx_pool = sum(1 for rank in pool if rank not in {0, 13})
    guaranteed_kx += max(0, unknown_draw_count - non_kx_pool)
    cpu.diamond_last_recovery_guaranteed_kx = guaranteed_kx

    if unknown_draw_count <= 0:
        probability = 1.0 if (
            guaranteed
            and diamond_has_current_certain_trump(
                temporary_cpu_with_hand(cpu, cpu.hand + guaranteed),
                room,
            )
        ) else 0.0
        targets = {
            0 if is_joker(card) else int(card.get("rank", 0))
            for card in guaranteed
            if is_joker(card) or int(card.get("rank", 0)) == 13
        }
        cpu.diamond_recovery_targets = tuple(sorted(targets))
        cpu.diamond_last_recovery_certain_probability = probability
        return probability

    seed = len(cpu.hand) * 1009 + unknown_draw_count * 9176 + len(pool)
    for card in cpu.hand:
        rank = 0 if is_joker(card) else int(card.get("rank", 0))
        seed = (seed * 257 + rank + 1) & ((1 << 64) - 1)
    rng = random.Random(seed)
    successes = 0
    success_ranks = Counter()
    all_ranks = Counter()
    for trial in range(DIAMOND_RECOVERY_PROBABILITY_TRIALS):
        sampled = rng.sample(pool, unknown_draw_count)
        present = set(sampled)
        all_ranks.update(present)
        synthetic = [
            {
                "card_id": f"diamond-recovery-{trial}-{index}",
                "suit": "X" if rank == 0 else "?",
                "rank": rank,
                "is_joker": rank == 0,
            }
            for index, rank in enumerate(sampled)
        ]
        child = temporary_cpu_with_hand(cpu, cpu.hand + guaranteed + synthetic)
        if diamond_has_current_certain_trump(child, room):
            successes += 1
            success_ranks.update(present)
    probability = successes / DIAMOND_RECOVERY_PROBABILITY_TRIALS
    target_scores = []
    if successes:
        for rank in success_ranks:
            lift = (
                success_ranks[rank] / successes
                - all_ranks[rank] / DIAMOND_RECOVERY_PROBABILITY_TRIALS
            )
            if lift > 0:
                target_scores.append((lift, success_ranks[rank], rank))
    targets = {
        0 if is_joker(card) else int(card.get("rank", 0))
        for card in guaranteed
        if is_joker(card) or int(card.get("rank", 0)) == 13
    }
    targets.update(rank for _, _, rank in sorted(target_scores, reverse=True)[:3])
    cpu.diamond_recovery_targets = tuple(sorted(targets))
    cpu.diamond_last_recovery_certain_probability = probability
    return probability


def diamond_recovery_has_certain_prospect(cpu: CpuPlayer, room) -> bool:
    """R20 gate before relaxing a conditional plan from 30% to 80%."""
    probability = diamond_recovery_certain_probability(cpu, room)
    return (
        cpu.diamond_last_recovery_guaranteed_kx >= 1
        or probability >= DIAMOND_RECOVERY_CERTAIN_MIN_PROBABILITY
    )


def diamond_conditional_plan_is_acceptable(
    plan: dict,
    cpu: CpuPlayer,
    room,
) -> bool:
    trump = diamond_plan_trump_candidate(plan)
    if diamond_candidate_certainty(trump, cpu, room) != "conditional":
        return False
    probability = diamond_candidate_return_probability(trump, cpu, room)
    has_alternative = (
        diamond_has_current_certain_trump(cpu, room)
        or diamond_recovery_has_certain_prospect(cpu, room)
    )
    limit = diamond_conditional_return_probability_limit(
        cpu,
        has_alternative=has_alternative,
    )
    return probability <= limit


def diamond_revolution_candidate_tier(candidate: dict, cpu: CpuPlayer) -> int:
    if not candidate:
        return 0
    if candidate.get("kind") == "joker_cut" or candidate.get("number") in {"X", 57}:
        return 5
    try:
        value = int(candidate.get("number"))
    except (TypeError, ValueError):
        return 0
    text = str(value)
    ranks = tuple(int(rank) for rank in (candidate.get("ranks") or ()))
    if value == 10 or text.startswith("101"):
        return 5
    if ranks and ranks[0] == 1:
        return 4
    if ranks and all(1 <= rank <= 9 for rank in ranks):
        return 3
    return 2


def diamond_plan_sort_key(plan: dict, cpu: CpuPlayer, room) -> tuple:
    check_cpu_search_deadline(cpu)
    certainty_tiers = {
        "unclassified": 0,
        "soft": 1,
        "conditional": 2,
        "certain": 3,
    }
    certainty = diamond_plan_certainty(plan, cpu, room)
    preserves_x = not any(step_uses_joker(step) for step in plan.get("steps", []))
    trump = diamond_plan_trump_candidate(plan) or {}
    rally_steps = sum(
        1 for step in plan.get("steps", [])
        if str(step.get("role", "")).startswith("rally-")
    )
    return_probability = (
        diamond_candidate_return_probability(trump, cpu, room)
        if certainty == "conditional"
        else 0.0
    )
    revolution_active = (
        getattr(cpu, "diamond_revolution_strategy_active", False)
        and getattr(room, "reverse_order", False)
    )
    if revolution_active:
        revolution_tier = diamond_revolution_candidate_tier(trump, cpu)
    else:
        revolution_tier = 0
    if revolution_active:
        return (
            revolution_tier,
            -rally_steps,
            1 if preserves_x else 0,
            candidate_strength(trump, room) if trump else -1,
            platinum_plan_score(plan),
            gold_plan_score(plan),
        )
    return (
        certainty_tiers.get(certainty, 0),
        (
            diamond_plan_reply_profile(plan, cpu, room)
            if certainty in {"conditional", "soft", "unclassified"}
            else (1, 0, 0, 0)
        ),
        -rally_steps,
        1 if diamond_plan_has_protected_x_tail(plan) else 0,
        -return_probability,
        1 if preserves_x else 0,
        candidate_strength(trump, room) if trump else -1,
        1 if plan.get("dual_wield") else 0,
        platinum_plan_score(plan),
        gold_plan_score(plan),
    )


def diamond_has_current_certain_trump(cpu: CpuPlayer, room) -> bool:
    empty_room = room_without_field(room)
    candidates = knowledge_prime_candidates(
        cpu,
        empty_room,
        gold_knowledge_number_validator,
        DIAMOND_PREFERRED_RALLY_COUNTS,
    )
    candidates.extend(knowledge_composite_candidates(cpu, empty_room, (2, 4)))
    return any(
        diamond_candidate_is_certain_fast(candidate, cpu, room)
        for candidate in candidates
    )


def diamond_candidate_is_certain_fast(candidate: dict, cpu: CpuPlayer, room) -> bool:
    """Physical certainty check without running return-probability simulations."""
    token = platinum_candidate_token(candidate)
    visible_count = len(candidate.get("cards", []) or [])
    if token == "kk":
        return True
    if visible_count == 4 and candidate.get("kind") == "composite":
        if not diamond_four_candidate_is_face_obake(candidate):
            return False
        try:
            value = int(candidate.get("number"))
        except (TypeError, ValueError):
            value = platinum_token_value(token)
        if value >= DIAMOND_FOUR_FIXED_COUNTER_MIN_VALUE:
            counters = (
                counter
                for counter in DIAMOND_FOUR_COUNTER_TOKENS
                if (
                    platinum_token_value(counter) < value
                    if getattr(room, "reverse_order", False)
                    else platinum_token_value(counter) > value
                )
            )
            return not any(
                diamond_counter_token_is_physically_possible(cpu, room, counter)
                for counter in counters
            )
        opponent_max_kx = diamond_opponent_max_kx_after_draw(cpu, room)
        if value == DIAMOND_FOUR_KKTJ_VALUE:
            return opponent_max_kx < 2
        opponent_count = platinum_opponent_hand_count(cpu, room)
        return (
            diamond_cpu_monopolizes_kx(cpu)
            or (
                opponent_count is not None
                and opponent_count
                <= DIAMOND_FOUR_OTHER_CERTAIN_MAX_OPPONENT_HAND_SIZE
            )
        )
    if visible_count != 6 or candidate.get("kind") != "prime":
        return False
    try:
        value = int(candidate.get("number"))
    except (TypeError, ValueError):
        return False
    return (
        not getattr(room, "reverse_order", False)
        and value >= DIAMOND_SIX_CERTAIN_MIN_VALUE
        and value in cpu.registered_primes
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
    if getattr(cpu, "cpu_key", "") == "diamond_planner":
        trump_index = gold_plan_trump_step_index(plan)
        trump = (
            steps[trump_index]
            if trump_index is not None and trump_index < len(steps)
            else None
        )
        if (
            trump is not None
            and platinum_candidate_token(trump) in DIAMOND_FOUR_OBAKE_TOKENS
        ):
            prefix = steps[:trump_index]
            if not 1 <= len(prefix) <= 3:
                return False
            if any(step.get("kind") != "prime" for step in prefix):
                return False
    if steps[-1].get("role") != "finish":
        return False
    if len(steps) == 2:
        return True

    tail_start = len(steps) - 1
    if (
        getattr(cpu, "cpu_key", "") == "diamond_planner"
        and len(steps) >= 3
        and steps[-2].get("role") == "cut"
        and steps[-2].get("number") in {"X", 57}
    ):
        tail_start -= 1
    rally_steps = steps[:tail_start]
    if not rally_steps:
        return False
    rally_count = len(rally_steps[0].get("cards", []))
    return (
        rally_count > 0
        and all(
            str(step.get("role", "")).startswith("rally-")
            and len(step.get("cards", [])) == rally_count
            for step in rally_steps
        )
        and gold_plan_trump_step_index(plan) == tail_start - 1
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
    if getattr(cpu, "cpu_key", "") == "diamond_planner":
        context = diamond_tactical_context(cpu, room)
        if context in DIAMOND_POST_ALL_OUT_CONTEXTS:
            tier = diamond_post_all_out_plan_tier(plan, cpu, room)
            if tier is None:
                return False
            if (
                context == "opponent-all-out"
                and platinum_plan_score(plan)
                < DIAMOND_OPENING_SECOND_MIN_TRUMP_STRENGTH
            ):
                return False
            return True
        certainty = diamond_plan_certainty(plan, cpu, room)
        if certainty == "certain":
            return True
        if certainty == "conditional":
            return diamond_conditional_plan_is_acceptable(plan, cpu, room)
        if certainty == "soft":
            return False
        if (
            diamond_tactical_context(cpu, room)
            in {"opening-second", "opponent-all-out"}
            and platinum_plan_score(plan)
            < DIAMOND_OPENING_SECOND_MIN_TRUMP_STRENGTH
        ):
            return False
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
    if getattr(cpu, "cpu_key", "") == "diamond_planner":
        if token == "kkkq":
            return True
        try:
            value = int(candidate.get("number"))
        except (TypeError, ValueError):
            value = -1
        if (
            candidate.get("kind") == "prime"
            and len(candidate.get("cards", []) or []) == 6
            and value >= DIAMOND_SIX_CERTAIN_MIN_VALUE
            and value in cpu.registered_primes
        ):
            return True
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


def diamond_opening_auso_trump_score(candidate: dict) -> Optional[float]:
    """Return the opening-only score for the reviewed 3/4-card Auso bands."""
    if candidate.get("kind") != "prime":
        return None
    count = len(candidate.get("cards", []) or [])
    bounds = DIAMOND_OPENING_AUSO_TRUMP_RANGES.get(count)
    if bounds is None:
        return None
    try:
        value = int(candidate.get("number"))
    except (TypeError, ValueError):
        return None
    if not bounds[0] <= value <= bounds[1]:
        return None
    if count == 3:
        if value >= platinum_token_value("kkj"):
            return 97.0
        if value >= platinum_token_value("kqk"):
            return 96.0
        return DIAMOND_OPENING_AUSO_BASE_SCORE
    if value >= platinum_token_value("kjqj"):
        return 99.0
    if value >= platinum_token_value("kjtk"):
        return 97.0
    return DIAMOND_OPENING_AUSO_BASE_SCORE


def diamond_opening_auso_converts_to_initial_all_out(
    plan: dict,
    cpu: CpuPlayer,
) -> bool:
    """Replace only reviewed first-seat non-dual Auso shapes with all-out.

    KQK/KKJ and KJQJ retain their opening routes. A fixed or generalized
    dual-wield plan also remains preferred because its pass branch is the
    tactical proof that the ordinary Auso route lacks.
    """
    if (
        getattr(cpu, "cpu_key", "") != "diamond_planner"
        or getattr(cpu, "diamond_opening_was_second", None) is not False
        or not getattr(cpu, "platinum_opening_phase", True)
        or not plan.get("diamond_opening_auso")
        or plan.get("dual_wield")
    ):
        return False
    trump = diamond_plan_trump_candidate(plan)
    return (
        trump is not None
        and platinum_candidate_token(trump)
        in DIAMOND_OPENING_AUSO_ALL_OUT_TRUMP_TOKENS
    )


def diamond_pending_initial_all_out_action(
    cpu: CpuPlayer,
    room,
) -> Optional[CpuAction]:
    """Complete the reviewed draw -> all-out replacement on the same turn."""
    if not getattr(cpu, "diamond_pending_initial_all_out", False):
        return None
    if getattr(room, "field", []) or []:
        cpu.diamond_pending_initial_all_out = False
        return None
    if not getattr(room, "has_drawn", False) and getattr(room, "deck", []):
        cpu.diamond_last_route_kind = "opening-auso-policy-all-out-draw"
        cpu.diamond_last_certainty = "unclassified"
        return CpuAction("draw")

    payload = build_gold_all_out_payload(
        cpu.hand,
        force_random=True,
        rng=cpu.rng,
    )
    cpu.diamond_pending_initial_all_out = False
    if payload is None:
        return None
    clear_gold_active_plan(cpu)
    cpu.platinum_all_out_attempts += 1
    cpu.diamond_last_route_kind = "opening-auso-policy-all-out"
    cpu.diamond_last_certainty = "unclassified"
    return platinum_commit_play(cpu, CpuAction("play_prime", payload))


def diamond_opening_tactical_plan_sort_key(
    plan: dict,
    cpu: CpuPlayer,
    room,
) -> tuple:
    """Prefer proof, fewer same-count replies, then a strong protected finish."""
    certainty = (
        "certain"
        if plan.get("diamond_certainty_override") == "certain"
        else diamond_plan_certainty(plan, cpu, room)
    )
    certainty_tier = {
        "unclassified": 0,
        "soft": 1,
        "conditional": 2,
        "certain": 3,
    }.get(certainty, 0)
    trump = diamond_plan_trump_candidate(plan) or {}
    trump_index = gold_plan_trump_step_index(plan)
    x_before_trump = any(
        step_uses_joker(step)
        for step in (
            plan.get("steps", [])[:trump_index]
            if trump_index is not None
            else []
        )
    )
    return (
        certainty_tier,
        1 if plan.get("diamond_opening_forced_pass") else 0,
        1 if plan.get("dual_wield") else 0,
        -diamond_plan_rally_step_count(plan),
        1 if diamond_plan_has_protected_x_tail(plan) else 0,
        diamond_plan_finish_strength(plan),
        0 if x_before_trump else 1,
        platinum_plan_score(plan),
        candidate_strength(trump, room) if trump else -1,
        gold_plan_score(plan),
    )


def diamond_opening_finish_candidates(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
) -> list[dict]:
    """Enumerate ways to play every card in a small opening partition."""
    hand_ids = {card.get("card_id") for card in cpu.hand}
    candidates = direct_gold_finish_candidates(cpu, room, validator)
    candidates.extend(joker_prime_finish_candidates(cpu, room, validator))
    return [
        candidate
        for candidate in dedupe_candidates(candidates)
        if {
            card.get("card_id")
            for card in candidate_consumed_cards(candidate)
        }
        == hand_ids
    ]


def build_diamond_opening_tactical_plans(
    cpu: CpuPlayer,
    room,
    validator: NumberValidator,
    *,
    fixed_dual: Optional[dict] = None,
) -> list[dict]:
    """Build short initial-11 proof routes before the general Gold tree.

    Fixed dual-wield templates are inserted first.  The general pass branch is
    then detected while evaluating a 3/4-card Auso opening: after the opening,
    the remaining cards must have a direct or X/57-forced finish that differs
    from the normal ``opening -> trump -> tail`` route.
    """
    if (
        len(cpu.hand) != 11
        or (getattr(room, "field", []) or [])
        or not getattr(cpu, "platinum_opening_phase", True)
    ):
        return []

    empty_room = room_without_field(room)
    plans = []
    seen = set()

    def add_plan(plan: Optional[dict]) -> None:
        if plan is None or not plan.get("completed"):
            return
        key = tuple(
            candidate_fingerprint(step)
            for step in plan.get("steps", [])
        )
        if key in seen:
            return
        seen.add(key)
        plans.append(plan)

    if fixed_dual is not None:
        fused = fixed_dual.get("dual_wield_fused")
        if fused is not None:
            fixed_dual["dual_wield_pass_tail"] = [dict(fused)]
        fixed_dual["diamond_opening_tactical"] = True
        add_plan(fixed_dual)

    # A reviewed fixed shape is already the preferred dual branch and acts as
    # the requested human-like cutoff.  Certain pass chains are still compared
    # below, but the wider Auso partition scan is unnecessary in this case.
    trump_candidates = []
    if fixed_dual is None:
        for count in DIAMOND_OPENING_AUSO_TRUMP_RANGES:
            check_cpu_search_deadline(cpu)
            trump_candidates.extend(gold_plan_candidates(
                cpu,
                empty_room,
                (count,),
                validator,
            ))
            trump_candidates.extend(joker_prime_candidates_for_count(
                cpu,
                empty_room,
                count,
                validator,
            ))
    trump_candidates = [
        candidate
        for candidate in dedupe_candidates(trump_candidates)
        if diamond_opening_auso_trump_score(candidate) is not None
        and len(candidate_consumed_cards(candidate)) < len(cpu.hand)
    ]
    trump_candidates.sort(key=lambda candidate: (
        diamond_opening_auso_trump_score(candidate) or 0.0,
        0 if step_uses_joker(candidate) else 1,
        candidate_strength(candidate, empty_room),
    ), reverse=True)

    for trump in trump_candidates[:gold_last_candidate_cap(cpu)]:
        check_cpu_search_deadline(cpu)
        rally_count = len(trump.get("cards", []) or [])
        score = float(diamond_opening_auso_trump_score(trump) or 0.0)
        reserved_hand = remaining_cards(
            cpu.hand,
            candidate_consumed_cards(trump),
        )
        reserved_cpu = temporary_cpu_with_hand(cpu, reserved_hand)

        # Known opening-only immediate-trump proof: trump, then a forced tail.
        immediate_tail = choose_gold_finish_tail(
            reserved_cpu,
            empty_room,
            validator,
        )
        if immediate_tail:
            immediate = finalize_gold_plan(
                cpu,
                empty_room,
                [trump] + immediate_tail,
                rally_count,
            )
            immediate["diamond_opening_immediate_trump"] = True
            immediate["diamond_opening_tactical"] = True
            immediate["dual_wield_score"] = score
            immediate["evaluation"] = {
                **immediate.get("evaluation", {}),
                "score": max(
                    score,
                    float(immediate.get("evaluation", {}).get("score", 0.0)),
                ),
            }
            add_plan(immediate)

        finish_count = len(reserved_hand) - rally_count
        if finish_count <= 0:
            continue
        finish_tails = gold_finish_tails_for_consumed_count(
            reserved_cpu,
            empty_room,
            finish_count,
            validator,
        )
        finish_tails.sort(key=lambda tail: (
            -len(tail),
            1 if any(step.get("number") == "X" for step in tail) else 0,
            sum(len(candidate_consumed_cards(step)) for step in tail),
        ), reverse=True)
        for finish_tail in finish_tails[:gold_branch_candidate_cap(cpu) * 2]:
            check_cpu_search_deadline(cpu)
            tail_cards = [
                card
                for step in finish_tail
                for card in candidate_consumed_cards(step)
            ]
            opening_cards = remaining_cards(reserved_hand, tail_cards)
            if len(opening_cards) != rally_count:
                continue
            opening_cpu = temporary_cpu_with_hand(cpu, opening_cards)
            openings = diamond_opening_finish_candidates(
                opening_cpu,
                empty_room,
                validator,
            )
            openings = [
                opening
                for opening in openings
                if len(opening.get("cards", []) or []) == rally_count
                and candidate_strength(opening, empty_room)
                < candidate_strength(trump, empty_room)
            ]
            openings.sort(key=lambda opening: (
                0 if step_uses_joker(opening) else 1,
                candidate_strength(opening, empty_room),
            ), reverse=True)
            for opening in openings[:gold_branch_candidate_cap(cpu)]:
                sequence = [dict(opening), dict(trump)] + [
                    dict(step) for step in finish_tail
                ]
                sequence[0]["role"] = f"rally-{rally_count}"
                sequence[1]["role"] = f"rally-{rally_count}"
                plan = finalize_gold_plan(
                    cpu,
                    empty_room,
                    sequence,
                    rally_count,
                )
                plan["diamond_opening_auso"] = True
                plan["diamond_opening_tactical"] = True
                plan["dual_wield_score"] = score
                plan["evaluation"] = {
                    **plan.get("evaluation", {}),
                    "score": max(
                        score,
                        float(plan.get("evaluation", {}).get("score", 0.0)),
                    ),
                }

                post_opening_hand = remaining_cards(
                    cpu.hand,
                    candidate_consumed_cards(opening),
                )
                post_opening_cpu = temporary_cpu_with_hand(
                    cpu,
                    post_opening_hand,
                )
                pass_tail = choose_gold_finish_tail(
                    post_opening_cpu,
                    empty_room,
                    validator,
                )
                if pass_tail:
                    pass_plan = finalize_gold_plan(
                        post_opening_cpu,
                        empty_room,
                        pass_tail,
                        0,
                    )
                    if pass_plan.get("completed"):
                        plan["dual_wield"] = True
                        plan["dual_wield_generalized"] = True
                        plan["dual_wield_pass_tail"] = [
                            dict(step) for step in pass_tail
                        ]
                        plan["dual_wield_score"] = max(score, 95.0)
                add_plan(plan)

    # Initial-only exact pass chains such as K -> K -> finish with both X held.
    kx_count = sum(
        1
        for card in cpu.hand
        if is_joker(card) or int(card.get("rank", 0)) == 13
    )
    king_count = sum(
        1
        for card in cpu.hand
        if not is_joker(card) and int(card.get("rank", 0)) == 13
    )
    joker_count = sum(1 for card in cpu.hand if is_joker(card))
    if kx_count >= 4 or king_count >= 2 or joker_count >= 2:
        for forced in diamond_forced_pass_plans(cpu, empty_room, validator):
            forced["diamond_opening_forced_pass"] = True
            forced["diamond_opening_tactical"] = True
            forced["dual_wield_score"] = 100.0
            forced["evaluation"] = {
                **forced.get("evaluation", {}),
                "score": 100.0,
            }
            add_plan(forced)

    return plans


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
        plan["dual_wield_pass_tail"] = [dict(fused_candidate)]
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
    if (
        getattr(cpu, "cpu_key", "") == "diamond_planner"
        and getattr(cpu, "diamond_opponent_rally_threat", "none")
        == "certain-likely"
    ):
        return True
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
        if getattr(cpu, "cpu_key", "") == "diamond_planner":
            return (
                bool(getattr(room, "deck", []))
                and not platinum_interference_danger_active(cpu, room)
                and not diamond_has_current_certain_trump(cpu, room)
            )
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
    if getattr(cpu, "cpu_key", "") == "diamond_planner":
        return (
            bool(getattr(room, "deck", []))
            and not platinum_interference_danger_active(cpu, room)
            and not diamond_has_current_certain_trump(cpu, room)
        )
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
    include_joker_candidates: bool = True,
) -> Optional[CpuAction]:
    """Choose a legal one-ply play without recursively constructing a route."""
    field_count = len(getattr(room, "field", []) or [])
    max_count = min(cpu_max_knowledge_cards(cpu), len(cpu.hand))
    counts = (field_count,) if field_count else range(1, max_count + 1)
    candidates = gold_plan_candidates(cpu, room, counts, validator)
    if include_joker_candidates:
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
        action = choose_platinum_bounded_legal_action(
            cpu,
            room,
            validator,
            include_joker_candidates=False,
        )
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
        include_joker_candidates=False,
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
    max_cards = min(cpu_max_knowledge_cards(cpu), len(cpu.hand) - 1)
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

    large_max = min(cpu_max_knowledge_cards(cpu), len(remaining))
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

    results.sort(
        key=(
            (lambda plan: diamond_plan_sort_key(plan, cpu, room))
            if getattr(cpu, "cpu_key", "") == "diamond_planner"
            else gold_plan_score
        ),
        reverse=True,
    )
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
        if rally_count <= 9:
            rallies.extend(joker_prime_candidates_for_count(
                rally_cpu,
                room,
                rally_count,
                validator,
            ))
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
    for count in range(1, max_cards + 1):
        check_cpu_search_deadline(cpu)
        candidates.extend(joker_prime_candidates_for_count(
            cpu,
            room,
            count,
            validator,
        ))
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
        check_cpu_search_deadline(cpu)
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
        check_cpu_search_deadline(cpu)
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
    count_tuple = tuple(counts)
    candidates = knowledge_prime_candidates(cpu, room, validator, count_tuple)
    candidates.extend(knowledge_composite_candidates(cpu, room, count_tuple))
    if getattr(cpu, "diamond_revolution_strategy_active", False):
        candidates = [
            candidate for candidate in candidates
            if len(candidate.get("cards", []) or [])
            not in DIAMOND_REVOLUTION_AVOID_COUNTS
        ]
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
    cpu_key = getattr(cpu, "cpu_key", "")
    max_cards = (
        cpu_max_knowledge_cards(cpu)
        if cpu_key in ADVANCED_PLANNING_CPU_KEYS
        else 9
    )
    if (
        cpu_key in ADVANCED_PLANNING_CPU_KEYS
        and getattr(cpu, "prime_template_index_values", ()) == values
    ):
        index = cpu.prime_template_index
    else:
        index = registered_prime_template_index(values, max_cards=max_cards)
    for count in sorted(count_set):
        for template_offset, (number, ranks) in enumerate(
            index.templates_by_card_count.get(count, ())
        ):
            if template_offset % 32 == 0:
                check_cpu_search_deadline(cpu)
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
    if getattr(getattr(room, "rule", None), "key", None) == "composite-practice-11-n":
        max_visible_cards = 9
    else:
        max_visible_cards = 4
    count_set = {count for count in counts if 1 <= count <= max_visible_cards}
    if not count_set:
        return []

    entries_by_value: dict[int, list] = {}
    for entry in cpu.registered_composite_entries:
        entries_by_value.setdefault(entry.value, []).append(entry)

    candidates = []
    for value_offset, value in enumerate(
        sorted(set(cpu.registered_composites) | set(entries_by_value))
    ):
        if value_offset % 16 == 0:
            check_cpu_search_deadline(cpu)
        for visible_ranks in registered_value_encodings(value, max_cards=max_visible_cards):
            if len(visible_ranks) not in count_set:
                continue
            visible = cards_for_ranks_with_jokers(cpu.hand, visible_ranks)
            if visible is None:
                continue
            visible_cards = visible["cards"]
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
                "assigned_numbers": visible["assigned_numbers"],
                "consume_cards": material["cards"],
                "composite_tokens": material["tokens"],
                "composite_assigned_numbers": material.get(
                    "assigned_numbers",
                    [],
                ),
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
    assigned_numbers = []
    jokers = [card for card in hand if is_joker(card)]
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
                card = next(
                    (
                        joker for joker in jokers
                        if joker.get("card_id") not in used_ids
                    ),
                    None,
                )
                if card is None:
                    return None
                assigned_numbers.append(str(rank))
            used_ids.add(card.get("card_id"))
            cards.append(card)
            tokens.append({"kind": "card", "card_id": card.get("card_id")})
    return {
        "cards": cards,
        "tokens": tokens,
        "assigned_numbers": assigned_numbers,
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
    temp.diamond_focus_count = cpu.diamond_focus_count
    temp.diamond_active_route = cpu.diamond_active_route
    temp.diamond_last_route_kind = cpu.diamond_last_route_kind
    temp.diamond_last_certainty = cpu.diamond_last_certainty
    temp.diamond_recovery_targets = cpu.diamond_recovery_targets
    temp.diamond_last_preferred_counts = cpu.diamond_last_preferred_counts
    temp.diamond_last_obake_counter_tokens = cpu.diamond_last_obake_counter_tokens
    temp.diamond_last_return_candidates = cpu.diamond_last_return_candidates
    temp.diamond_opponent_rally_count = cpu.diamond_opponent_rally_count
    temp.diamond_opponent_rally_streak = cpu.diamond_opponent_rally_streak
    temp.diamond_last_observed_play_key = cpu.diamond_last_observed_play_key
    temp.diamond_interference_mode_count = cpu.diamond_interference_mode_count
    temp.diamond_opponent_rally_threat = cpu.diamond_opponent_rally_threat
    temp.diamond_opponent_estimated_max = cpu.diamond_opponent_estimated_max
    temp.diamond_last_estimated_opponent_faces = (
        cpu.diamond_last_estimated_opponent_faces
    )
    temp.diamond_last_return_probability = cpu.diamond_last_return_probability
    temp.diamond_last_recovery_certain_probability = (
        cpu.diamond_last_recovery_certain_probability
    )
    temp.diamond_last_recovery_guaranteed_kx = (
        cpu.diamond_last_recovery_guaranteed_kx
    )
    temp.diamond_recovery_cache_key = cpu.diamond_recovery_cache_key
    temp.diamond_last_opponent_kjqj_probability = (
        cpu.diamond_last_opponent_kjqj_probability
    )
    temp.diamond_revolution_strategy_active = cpu.diamond_revolution_strategy_active
    temp.diamond_last_context = cpu.diamond_last_context
    temp.diamond_last_plan_tier = cpu.diamond_last_plan_tier
    temp.diamond_last_finish_strength = cpu.diamond_last_finish_strength
    temp.diamond_opening_was_second = cpu.diamond_opening_was_second
    temp.diamond_opponent_ever_13_plus = cpu.diamond_opponent_ever_13_plus
    temp.diamond_seen_cards = dict(cpu.diamond_seen_cards)
    temp.diamond_public_response_signature_cache = (
        cpu.diamond_public_response_signature_cache
    )
    temp.diamond_last_conditional_limit = cpu.diamond_last_conditional_limit
    temp.diamond_pending_full_recovery = cpu.diamond_pending_full_recovery
    temp.diamond_preserved_closeout_plan = cpu.diamond_preserved_closeout_plan
    temp.diamond_last_kx_policy = dict(cpu.diamond_last_kx_policy)
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
    copy.reserve = list(getattr(room, "reserve", []) or [])
    copy.public_known_deck_bottom = list(
        getattr(room, "public_known_deck_bottom", []) or []
    )
    copy.public_unknown_deck_count = int(
        getattr(room, "public_unknown_deck_count", 0) or 0
    )
    copy.deck = [None] if getattr(room, "deck", []) else []
    copy.opponent_hand_count = getattr(room, "opponent_hand_count", None)
    copy.players = []
    for player in (getattr(room, "players", []) or []):
        class PublicPlayerCount:
            pass
        public_player = PublicPlayerCount()
        public_player.id = getattr(player, "id", None)
        public_player.status = getattr(player, "status", "playing")
        public_player.hand = [None] * len(getattr(player, "hand", []) or [])
        copy.players.append(public_player)
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
    "diamond_planner": CpuProfile(
        key="diamond_planner",
        label="ダイヤCPU",
        description="ダイヤ素数表を使い、4枚・6枚の組み切りを優先してからプラチナ相当の安全策へ移る開発中CPUです。",
        rule_keys=(
            "std-11-n-c",
            "std-11-n-no-c",
            "registered-11-n-assist",
            "neo-assist-11-n-unlimited",
        ),
        knowledge=CpuKnowledgeSpec(
            source="sample_key",
            load_timing="always",
            sample_key="diamond_prime_table",
        ),
        action_selector=choose_diamond_planning_cpu_action,
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

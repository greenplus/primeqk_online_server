"""Opening Master v0: rank-exact opening plans, with no hidden-state inputs.

Packaged from the league's combined Master release. Knowledge stays shared with
Diamond; accepted plans are handed to Diamond's existing plan executor.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import cached_property, cmp_to_key, lru_cache
from itertools import combinations
from math import comb, fsum
from pathlib import Path
import csv
import time

import cpu_player as diamond
from .master_budget import check as check_master_budget
from registered_primes import registered_value_encodings

ROOT = Path(__file__).resolve().parent / "data"
TABLE = ROOT / "pure_trump_return_probabilities.csv"
ADDENDUM = ROOT / "return_probabilities.csv"
RESPONSE_TABLES = (
    ROOT / "pure_trump_response_details.csv",
    ROOT / "response_details.csv",
)
CAPACITY = (2,) + (4,) * 13
ZERO = (0,) * 14
LABELS = "XA23456789TJQK"
DRAW_MARGIN = 0.05
PACKED_GUARD = sum(16 << (5 * r) for r in range(14))


def packed_counts(signature):
    # Five bits per rank: counts <= 12 leave a guard bit for borrow detection.
    return sum(n << (5 * r) for r, n in enumerate(signature))


@dataclass(frozen=True)
class Config:
    return_threshold: float = 0.45
    draw_margin: float = DRAW_MARGIN
    fallback_value: float = 0.50
    nine_start_weight: float = 0.5
    knowledge_coefficients: tuple[float, ...] = (0.9, 0.7, 0.4, 0.3)
    generalize_dual: bool = True
    fallback_policy: str = 'self'
    audit_diamond: bool = False
    collect_all_candidates: bool = False


def counts(ranks):
    bag = [0] * 14
    for rank in ranks:
        bag[rank] += 1
    return tuple(bag)


def hand_counts(hand):
    return counts(0 if diamond.is_joker(c) else int(c["rank"]) for c in hand)


def subtract(left, right):
    result = []
    for a, b in zip(left, right):
        if b > a:
            return None
        result.append(a - b)
    return tuple(result)


def rank_text(signature):
    return "".join(LABELS[r] * signature[r] for r in (*range(1, 14), 0))


def token_ranks(token):
    ranks = {str(r): r for r in range(1, 10)} | dict(t=10, j=11, q=12, k=13, x=0, a=1)
    return tuple(ranks[c] for c in token.lower())


@lru_cache(maxsize=1)
def return_tables():
    result = {}
    for path in (TABLE, ADDENDUM):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                size = int(row["display_count"])
                if path == TABLE and size < 4:
                    continue
                fixed = row.get("fixed_notation") or row["target_notation"]
                ranks = token_ranks(fixed)
                key = (size, int(row["target_value"]), ranks[:size], counts(ranks))
                result[key] = (float(row["p12"]), float(row["p24"]))
    return result


@lru_cache(maxsize=8192)
def hypergeometric_tail(population, faces, sample, minimum):
    """Exact combinatorial sum (converted to float only at the boundary)."""
    sample = min(sample, population)
    lo = max(0, minimum, sample - (population - faces))
    hi = min(faces, sample)
    return sum(comb(faces, k) * comb(population - faces, sample - k)
               for k in range(lo, hi + 1)) / comb(population, sample)


@dataclass(frozen=True)
class Template:
    kind: str
    value: int  # 0 denotes the unconditional single X cut
    visible: tuple[int, ...]
    material: tuple[int, ...] = ()
    expression: tuple = ()

    @cached_property
    def ranks(self):
        return self.visible + self.material


@dataclass(frozen=True)
class Move:
    template: Template
    physical: tuple[int, ...]
    need: tuple[int, ...]

    @property
    def size(self):
        return len(self.template.visible)

    @property
    def value(self):
        return self.template.value

    @property
    def is_57(self):
        return self.template.kind == "prime" and self.value == 57

    @property
    def cut(self):
        return self.template.kind == "cut" or self.is_57

    @cached_property
    def key(self):
        return (self.template.kind, self.value, self.template.visible,
                self.template.material, self.physical, str(self.template.expression))

    @cached_property
    def used_count(self):
        return sum(self.need)

    @cached_property
    def packed_need(self):
        return packed_counts(self.need)

    def summary(self):
        return dict(kind=self.template.kind, value="X" if not self.value else str(self.value),
                    visible_ranks=list(self.template.visible), physical_ranks=list(self.physical),
                    display_count=self.size, consumed_count=sum(self.need),
                    expression=[list(part) for part in self.template.expression])


@lru_cache(maxsize=8)
def templates(primes, entries, allow_composite):
    result = set()
    for value in primes:
        if value == 1729:  # Revolution-specific openings are outside v0.
            continue
        for ranks in registered_value_encodings(value, max_cards=12):
            result.add(Template("prime", value, ranks))
    # These are legal rule actions already available to Diamond, not new knowledge.
    result.add(Template("cut", 0, (0,)))
    result.add(Template("prime", 57, (5, 7)))
    if allow_composite:
        for entry in entries:
            expression = tuple((t.kind, tuple(t.ranks) if t.kind == "cards" else t.op)
                               for t in entry.expression_tokens)
            material = tuple(r for kind, part in expression if kind == "cards" for r in part)
            for visible in registered_value_encodings(entry.value, max_cards=4):
                # Match Diamond's composite knowledge scope, but keep every encoding/material.
                result.add(Template("composite", int(entry.value), visible, material, expression))
    return tuple(sorted(result, key=lambda t: (t.kind, t.value, t.visible, t.material, str(t.expression))))


@lru_cache(maxsize=8)
def response_signatures(catalog):
    """Proof universe: Diamond knowledge plus the supplied stronger-return catalog.

    A zero sampled probability is never itself a proof. Special cuts/revolution
    and single-X responses are included even when absent from registered primes.
    """
    result = {(len(t.visible), t.value, counts(t.ranks), t.kind == "cut" or t.value == 57)
              for t in catalog}
    for visible in registered_value_encodings(1729, max_cards=4):
        result.add((len(visible), 1729, counts(visible), True))
    for path in RESPONSE_TABLES:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                result.add((int(row["display_count"]), int(row["numeric_value"]),
                            counts(token_ranks(row["response_notation"])), False))
    return tuple(sorted(result))


def physically_possible(requirement, available):
    return requirement[0] + sum(max(0, requirement[r] - available[r])
                                for r in range(1, 14)) <= available[0]


@lru_cache(maxsize=16384)
def prime_upper_bound(available, size):
    """Upper bound including unknown/HNP primes, not merely registered values.

    For each possible odd final card, maximize digit length first, then numeric
    concatenation. Composites are checked separately with their real materials.
    Divisibility filters beyond the final digit are intentionally conservative.
    """
    best = 0
    endings = (2, 3, 5, 7, 11, 13) if size == 1 else (1, 3, 7, 9, 11, 13)
    for end in endings:
        for physical in (end, 0):
            if not available[physical]:
                continue
            pool = list(available)
            pool[physical] -= 1
            faces = [13] * pool[0] + [r for r in (13, 12, 11, 10) for _ in range(pool[r])]
            digits = [r for r in range(9, 0, -1) for _ in range(pool[r])]
            prefix = faces[:size - 1]
            prefix += digits[:size - 1 - len(prefix)]
            if len(prefix) != size - 1:
                continue
            text = sorted(map(str, prefix), key=cmp_to_key(lambda a, b: (b + a > a + b) - (b + a < a + b)))
            best = max(best, int("".join(text) + str(end)))
    return best


def _realizations(template, hand):
    ranks = template.ranks
    if len(ranks) > sum(hand):
        return
    if template.kind == "cut":
        if hand[0]:
            yield Move(template, (0,), (1,) + (0,) * 13)
        return
    requirement = counts(ranks)
    missing = sum(max(0, requirement[r] - hand[r]) for r in range(1, 14))
    if missing > hand[0]:
        return
    # Optional X replacements must also be explored: keeping a natural face
    # card can make the other half of a plan possible. Equal-rank suits are equivalent.
    seen = set()
    for joker_count in range(missing, min(hand[0], len(ranks)) + 1):
        for omitted in combinations(range(len(ranks)), joker_count):
            selected = set(omitted)
            physical = tuple(0 if i in selected else r for i, r in enumerate(ranks))
            need = counts(physical)
            visible_need = counts(physical[:len(template.visible)])
            key = (need, visible_need)
            if key in seen or subtract(hand, need) is None:
                continue
            seen.add(key)
            yield Move(template, physical, need)


@dataclass(frozen=True)
class Plan:
    type: str
    steps: tuple[Move, ...]
    p_return: float
    model: str
    context: int
    pass_finish: Move | None = None
    pass_tail: tuple[Move, ...] = ()

    @property
    def log_type(self):
        if self.type != 'dual_wield':
            return self.type
        return 'dual_wield_immediate' if self.pass_finish else 'dual_wield_certified'

    @property
    def pass_sequence(self):
        return (self.pass_finish,) if self.pass_finish else self.pass_tail

    @property
    def value(self):
        return 1.0 if self.type == "certified_win" else 1.0 - self.p_return

    def summary(self):
        return dict(type=self.type, gate="CERTIFIED_WIN" if self.type == "certified_win" else "MASTER_STRONG",
                    value=self.value, p_return=self.p_return, model=self.model, context=self.context,
                    variant=self.log_type,
                    steps=[move.summary() for move in self.steps],
                    pass_finish=self.pass_finish.summary() if self.pass_finish else None,
                    pass_tail=[m.summary() for m in self.pass_sequence])


@lru_cache(maxsize=8)
def template_metadata(catalog):
    """Immutable lookup metadata shared only by exactly matching knowledge."""
    requirements = {t: counts(t.ranks) for t in catalog}
    relevant = {t: tuple((r, n) for r, n in enumerate(need) if r and n)
                for t, need in requirements.items()}
    return requirements, relevant


class Evaluator:
    def __init__(self, cpu, room, config=Config()):
        self.config = config
        started = time.perf_counter()
        self.profile_ms = defaultdict(float)
        self.profile_counts = Counter()
        self.catalog = templates(tuple(sorted(cpu.registered_primes)),
                                 tuple(cpu.registered_composite_entries), room.rule.allow_composite)
        self.responses = response_signatures(self.catalog)
        self.cache = {}
        self.evaluation_cache = {}
        self.realization_cache = {}
        self.requirements, self.relevant_ranks = template_metadata(self.catalog)
        self.finish_cache = {}
        self.certain_cache = {}
        self.probability_cache = {}
        self.hits = self.misses = 0
        self.opponent_count = diamond.platinum_opponent_hand_count(cpu, room) or 11
        self.profile_ms['cache_build_ms'] += 1000 * (time.perf_counter() - started)

    def moves(self, hand):
        check_master_budget()
        if hand in self.cache:
            self.hits += 1
            return self.cache[hand]
        self.misses += 1
        started = time.perf_counter()
        result = []
        for template in self.catalog:
            check_master_budget()
            # Irrelevant ranks cannot affect the physical representations of a
            # template. Reuse the same representations across virtual draws.
            relevant = (hand[0],) + tuple(min(hand[r], n) for r, n in self.relevant_ranks[template])
            key = (template, relevant)
            if key not in self.realization_cache:
                self.profile_counts['representation_misses'] += 1
                self.realization_cache[key] = tuple(_realizations(template, hand))
            else:
                self.profile_counts['representation_hits'] += 1
            result.extend(self.realization_cache[key])
        result = tuple(result)
        self.profile_ms['physical_representation_ms'] += 1000 * (time.perf_counter() - started)
        self.cache[hand] = result
        return result

    def probability(self, move, hand, context):
        check_master_budget()
        cache_key = (move, hand, context)
        if cache_key not in self.probability_cache:
            self.probability_cache[cache_key] = self._probability(move, hand, context)
        return self.probability_cache[cache_key]

    def _probability(self, move, hand, context):
        key = (move.size, move.value, move.template.visible, counts(move.template.ranks))
        pure = return_tables().get(key)
        if pure is not None:
            return pure[0 if context == 12 else 1], "pure_trump"
        if move.size < 6:
            return 1.0, "unmodeled"
        faces = hand[0] + sum(hand[10:14])
        needed = sum(r == 0 or r >= 10 for r in move.physical[:move.size])
        population = 54 - sum(hand)
        q = hypergeometric_tail(population, 18 - faces, context, needed)
        model = "face_resource"
        if move.template.visible[0] == 9:
            weight = self.config.nine_start_weight
            q = weight * q + (1 - weight) * hypergeometric_tail(population, 18 - faces, context, needed + 1)
            model = "face_resource_9start"
        # Against the expanded 24-card hand, do not discount return capability
        # by the smaller-hand knowledge coefficients.
        if move.size >= 7 and context != 24:
            q *= self.config.knowledge_coefficients[min(move.size - 7, 3)]
        return q, model

    def hard_certain(self, move, hand):
        check_master_budget()
        # 57 cuts only when legally played; it cannot retake a field >= 57.
        if move.is_57:
            return False
        if move.cut:
            return True
        key = (move.size, move.value, hand)
        if key in self.certain_cache:
            return self.certain_cache[key]
        available = subtract(CAPACITY, hand)
        if prime_upper_bound(available, move.size) > move.value:
            self.certain_cache[key] = False
            return False
        for size, value, need, special in self.responses:
            if size != move.size or (value <= move.value and value != 0):
                continue
            if physically_possible(need, available):
                self.certain_cache[key] = False
                return False
        self.certain_cache[key] = True
        return True

    def free_control_certain(self, move, hand):
        """Control from an empty field, where 57 is always legal."""
        return move.cut or self.hard_certain(move, hand)

    def safe_rally(self, lead, trump, hand):
        """All catalog responses must allow T, and none may immediately finish.

        Test both no-draw and one-draw opponent hand sizes. A response ending
        the game, a cut, or a revolution cannot be overridden by T later.
        """
        if trump.is_57:
            return False
        available = subtract(CAPACITY, hand)
        if trump.template.kind != "cut" and prime_upper_bound(available, lead.size) >= trump.value:
            # Equality to T is already too high for a strict response by T.
            return False
        for size, value, need, special in self.responses:
            if size != lead.size or (value <= lead.value and value != 0):
                continue
            if sum(need) > self.opponent_count + 1 or not physically_possible(need, available):
                continue
            if special or (trump.template.kind != "cut" and value >= trump.value):
                return False
            if sum(need) in (self.opponent_count, self.opponent_count + 1):
                return False
        return True

    def evaluate(self, hand):
        check_master_budget()
        started = time.perf_counter()
        cached = self.evaluation_cache.get(hand)
        self.profile_ms['cache_lookup_ms'] += 1000 * (time.perf_counter() - started)
        if cached is not None:
            self.profile_counts['evaluation_hits'] += 1
            best, self.last_plans = cached
            return best
        self.profile_counts['evaluation_misses'] += 1
        best = self._evaluate(hand)
        self.evaluation_cache[hand] = (best, self.last_plans)
        return best

    def _evaluate(self, hand):
        if sum(hand) not in (11, 12):
            raise ValueError("Opening Master requires an 11- or 12-card hand")
        moves = self.moves(hand)
        started = time.perf_counter()
        by_need = defaultdict(list)
        by_size_need = defaultdict(lambda: defaultdict(list))
        for move in moves:
            by_need[move.need].append(move)
            by_size_need[move.size][move.need].append(move)
        for need, candidates in by_need.items():
            if need in self.finish_cache:
                self.profile_counts['finish_cache_hits'] += 1
            else:
                self.profile_counts['finish_cache_misses'] += 1
                self.finish_cache[need] = tuple(candidates)
            by_need[need] = self.finish_cache[need]
        partitions = {size: tuple((need, packed_counts(need), group) for need, group in groups.items())
                      for size, groups in by_size_need.items()}
        self.profile_ms['known_finish_index_ms'] += 1000 * (time.perf_counter() - started)
        plans = []
        started = time.perf_counter()

        # The exact finish index also serves every L/T/F partition; no capped tree.
        for finish in by_need.get(hand, ()):
            plans.append(Plan("certified_win", (finish,), 0.0, "immediate_finish", 0))

        @lru_cache(maxsize=None)
        def control_tail(remaining):
            check_master_budget()
            if remaining == ZERO:
                return ()
            finishes = by_need.get(remaining, ())
            if finishes:
                return (min(finishes, key=lambda m: (m.need[0], m.key)),)
            possibilities = []
            remaining_count = sum(remaining)
            remaining_guarded = packed_counts(remaining) | PACKED_GUARD
            for move in moves:
                check_master_budget()
                if move.used_count > remaining_count:
                    continue
                if (remaining_guarded - move.packed_need) & PACKED_GUARD != PACKED_GUARD:
                    continue
                rest = subtract(remaining, move.need)
                if rest is None or not self.free_control_certain(move, remaining):
                    continue
                suffix = control_tail(rest)
                if suffix is not None:
                    possibilities.append((move,) + suffix)
            return min(possibilities, key=lambda seq: (len(seq), sum(m.need[0] for m in seq),
                                                       tuple(m.key for m in seq))) if possibilities else None

        # Exact control sequences re-account for released cards at each stage.
        certain = control_tail(hand)
        if certain:
            plans.append(Plan("certified_win", certain, 0.0, "public_control_sequence", 0))
        certified_found = bool(plans)
        self.profile_ms['certified_detection_ms'] += 1000 * (time.perf_counter() - started)

        for lead in moves:
            check_master_budget()
            started = time.perf_counter()
            remainder = subtract(hand, lead.need)
            if remainder == ZERO:
                continue
            for finish in by_need.get(remainder, ()):
                if self.free_control_certain(lead, hand):
                    plans.append(Plan("certified_win", (lead, finish), 0.0, "public_control_sequence", 12))
                    certified_found = True
                elif not certified_found or self.config.collect_all_candidates:
                    probability, model = self.probability(lead, hand, 12)
                    if probability < self.config.return_threshold:
                        plans.append(Plan("one_step", (lead, finish), probability, model, 12))
            self.profile_ms['one_step_detection_ms'] += 1000 * (time.perf_counter() - started)
            # At least one physical finish card: 6+6 is never an Auso route.
            if lead.cut or 2 * lead.size >= sum(hand):
                continue
            started = time.perf_counter()
            # A shared physical partition has the same remainder regardless of
            # the known number's ordering. Check it once, then retain every move.
            remainder_guarded = packed_counts(remainder) | PACKED_GUARD
            for need, packed_need, trumps in partitions[lead.size]:
                check_master_budget()
                if (remainder_guarded - packed_need) & PACKED_GUARD != PACKED_GUARD:
                    continue
                tail = subtract(remainder, need)
                if tail is None or tail == ZERO:
                    continue
                finishes = by_need.get(tail, ())
                if not finishes:
                    continue
                pass_finishes = by_need.get(remainder, ())
                for trump in trumps:
                    if trump.template.kind != "cut" and lead.value >= trump.value:
                        continue
                    is_certain = self.hard_certain(trump, remainder) and self.safe_rally(lead, trump, hand)
                    if self.config.collect_all_candidates or (not certified_found and not is_certain):
                        p24, model24 = self.probability(trump, hand, 24)
                        p12, model12 = self.probability(trump, hand, 12)
                    else:
                        # Keep searching all equal-priority Certified plans for
                        # their original tie-break; stop only lower-priority work.
                        p24 = p12 = 1.0
                        model24 = model12 = 'pruned_below_certified'
                        self.profile_counts['lower_priority_branches_skipped'] += 1
                    for finish in finishes:
                        if is_certain:
                            plans.append(Plan("certified_win", (lead, trump, finish), 0.0,
                                              "public_rally_control", 24,
                                              min(pass_finishes, key=lambda m: m.key) if pass_finishes else None))
                            certified_found = True
                        if p24 < self.config.return_threshold:
                            plans.append(Plan("normal_auso", (lead, trump, finish), p24, model24, 24))
                        dual_started = time.perf_counter()
                        if p12 < self.config.return_threshold:
                            if pass_finishes:
                                plans.append(Plan("dual_wield", (lead, trump, finish), p12, model12, 12,
                                                  min(pass_finishes, key=lambda m: m.key)))
                            elif self.config.generalize_dual:
                                # Reuse the existing Certified Win control-sequence
                                # search, with released lead cards back in the
                                # unknown pool. No assumed penalty size/hidden hand.
                                certified_tail = control_tail(remainder)
                                if certified_tail:
                                    plans.append(Plan('dual_wield', (lead, trump, finish), p12, model12, 12,
                                                      pass_tail=certified_tail))
                        self.profile_ms['dual_wield_detection_ms'] += 1000 * (time.perf_counter() - dual_started)
            self.profile_ms['auso_and_dual_partition_ms'] += 1000 * (time.perf_counter() - started)

        retained_cache = {}

        def sort_key(plan):
            # No numerical win-rate bonus and no cross-strength type priority.
            controlling = plan.steps[1] if len(plan.steps) == 3 else plan.steps[0]
            very_strong = controlling.value in (131311, 13111211)
            type_bonus = int(plan.type == ("one_step" if very_strong else "dual_wield"))
            remaining = subtract(hand, plan.steps[0].need)
            if remaining not in retained_cache:
                strengths = sorted((self.probability(m, remaining, 12)[0] for m in moves
                                    if subtract(remaining, m.need) is not None))
                retained_cache[remaining] = (strengths + [1.0, 1.0])[:2]
            retained = retained_cache[remaining]
            return (0 if plan.type == "certified_win" else 1, plan.p_return, -type_bonus,
                    plan.steps[0].need[0], *retained, tuple(m.key for m in plan.steps))

        # Expensive retained-trump tie-breaks are needed only in the best gate
        # and at the best probability; this does not prune any stronger plan.
        started = time.perf_counter()
        finalists = plans
        if plans:
            best_gate = min(p.type != "certified_win" for p in plans)
            finalists = [p for p in plans if (p.type != "certified_win") == best_gate]
            best_probability = min(p.p_return for p in finalists)
            finalists = [p for p in finalists if p.p_return == best_probability]
        best = min(finalists, key=sort_key) if finalists else None
        self.profile_ms['plan_selection_ms'] += 1000 * (time.perf_counter() - started)
        self.last_plans = plans
        return best


def bind_move(move, hand):
    """Bind a rank plan to distinct real cards and canonical server payloads."""
    pools = defaultdict(list)
    for card in sorted(hand, key=lambda c: str(c["card_id"])):
        pools[0 if diamond.is_joker(card) else int(card["rank"])].append(card)
    cards = [pools[rank].pop(0) for rank in move.physical]
    size = move.size
    visible, materials = cards[:size], cards[size:]
    candidate = dict(kind="prime" if move.template.kind == "cut" else move.template.kind,
                     number="X" if move.template.kind == "cut" else move.value,
                     cards=visible, ranks=move.template.visible,
                     assigned_numbers=[] if move.template.kind == "cut" else
                     [str(r) for r, p in zip(move.template.visible, move.physical[:size]) if p == 0])
    if move.template.kind == "composite":
        tokens = []
        offset = 0
        for kind, part in move.template.expression:
            if kind == "op":
                tokens.append(dict(kind="op", op="×" if part == "*" else part))
            else:
                for _ in part:
                    tokens.append(dict(kind="card", card_id=materials[offset]["card_id"]))
                    offset += 1
        candidate.update(consume_cards=materials, composite_tokens=tokens,
                         composite_assigned_numbers=[str(r) for r, p in zip(move.template.material, move.physical[size:]) if p == 0])
    return candidate


def commit_plan(cpu, room, plan):
    remaining = list(cpu.hand)
    steps = []
    for index, move in enumerate(plan.steps):
        candidate = bind_move(move, remaining)
        candidate["role"] = "finish" if index == len(plan.steps) - 1 else ("cut" if move.cut else f"rally-{move.size}")
        steps.append(candidate)
        remaining = diamond.remaining_cards(remaining, diamond.candidate_consumed_cards(candidate))
    active = diamond.finalize_gold_plan(cpu, diamond.room_without_field(room), steps, plan.steps[0].size)
    # Master value ranks openings; it is not a Diamond executor strength score.
    active["master_value"] = plan.value
    active["continuation_authorized"] = True
    active["opening_master_reverse_order"] = bool(room.reverse_order)
    active["opening_master_type"] = plan.type
    active['opening_master_variant'] = plan.log_type
    if plan.pass_sequence:
        after_lead = diamond.remaining_cards(cpu.hand, diamond.candidate_consumed_cards(steps[0]))
        bound_tail = []
        for index, move in enumerate(plan.pass_sequence):
            candidate = bind_move(move, after_lead)
            candidate['role'] = 'finish' if index == len(plan.pass_sequence)-1 else ('cut' if move.cut else f'rally-{move.size}')
            bound_tail.append(candidate)
            after_lead = diamond.remaining_cards(after_lead, diamond.candidate_consumed_cards(candidate))
        active.update(dual_wield=True, dual_wield_pass_tail=bound_tail)
    diamond.set_gold_active_plan(cpu, active)
    cpu.gold_plan_step_index = 1
    diamond.diamond_update_opening_position_history(cpu, room)
    diamond.diamond_remember_cards(cpu, diamond.candidate_consumed_cards(steps[0]))
    cpu.diamond_focus_count = plan.steps[0].size if plan.steps[0].size in diamond.DIAMOND_PREFERRED_RALLY_COUNTS else None
    cpu.diamond_last_route_kind = "opening-master-" + plan.type
    cpu.diamond_last_certainty = "certain" if plan.type == "certified_win" else "conditional"
    cpu.diamond_last_return_probability = plan.p_return
    return diamond.platinum_commit_play(cpu, diamond.candidate_to_action(steps[0]))


def action_summary(action):
    visible = diamond.diamond_action_cards(action)
    if not visible:
        return {"kind": action.kind}
    selected = action.payload.get("selected", {}) if action.kind == "play_composite" else action.payload
    composite = action.payload.get("composite", {})
    return dict(kind=action.kind, visible=[c["card_id"] for c in visible],
                consumed=[c["card_id"] for c in diamond.diamond_action_consumed_cards(action)],
                single_x=len(visible) == 1 and diamond.is_joker(visible[0]) and action.kind == "play_prime",
                assigned_numbers=selected.get("assigned_numbers", []),
                composite_tokens=composite.get("tokens", []),
                composite_assigned_numbers=composite.get("assigned_numbers", []))


def wants_draw(v_draw, v_now, config=Config()):
    # The specification uses strict >; do not turn a floating-point tie into a draw.
    return v_draw - v_now - config.draw_margin > 1e-12


def no_plan_action(cpu, room):
    """Opening-only fallback; uses the existing all-out payload, no Diamond search."""
    diamond.clear_gold_active_plan(cpu)
    cpu.last_decision_timed_out = False
    diamond.diamond_update_opening_position_history(cpu, room)
    if not room.has_drawn and room.deck:
        return diamond.CpuAction('draw')
    payload = diamond.build_gold_all_out_payload(cpu.hand, force_random=True, rng=cpu.rng)
    if payload is None:
        return diamond.CpuAction('pass')
    cpu.platinum_all_out_attempts += 1
    return diamond.platinum_commit_play(cpu, diamond.CpuAction('play_prime', payload))


def choose_opening(cpu, room, diamond_choice, config=Config()):
    """Choose an opening; the optional Diamond comparison uses an isolated clone."""
    cpu.opening_master_trace = None
    phase = getattr(cpu, "opening_master_phase", "initial")
    eligible = (phase in ("initial", "after_draw") and not room.field
                and not room.reverse_order and room.rule.key == "std-11-n-c"
                and len(cpu.hand) == (12 if room.has_drawn else 11)
                and getattr(room, "first_player_id", cpu.id) == cpu.id)
    if not eligible:
        cpu.opening_master_phase = "done"
        return diamond_choice(cpu, room)

    started = time.perf_counter()
    if config.fallback_policy not in ('diamond', 'self'):
        raise ValueError('unknown opening fallback policy')
    existing = getattr(cpu, "opening_master_evaluator", None)
    profile_before = dict(existing.profile_ms) if existing else {}
    counts_before = dict(existing.profile_counts) if existing else {}
    evaluator = existing or Evaluator(cpu, room, config)
    cpu.opening_master_evaluator = evaluator
    hand = hand_counts(cpu.hand)
    before_hits, before_misses = evaluator.hits, evaluator.misses
    evaluated = time.perf_counter()
    best = evaluator.evaluate(hand)
    current_ms = (time.perf_counter() - evaluated) * 1000
    v_now = best.value if best else config.fallback_value
    draw_started = time.perf_counter()
    outcomes = []
    if not room.has_drawn and room.deck:
        for rank in (*range(1, 14), 0):
            remaining = CAPACITY[rank] - hand[rank]
            if not remaining:
                continue
            augmented = list(hand)
            augmented[rank] += 1
            rank_started = time.perf_counter()
            future = evaluator.evaluate(tuple(augmented))
            outcomes.append(dict(rank=LABELS[rank], remaining_count=remaining, probability=remaining / 43,
                                 best_plan_type=future.type if future else "fallback", V_r=future.value if future else config.fallback_value,
                                 best_plan_variant=future.log_type if future else 'fallback',
                                 elapsed_ms=(time.perf_counter() - rank_started) * 1000))
    virtual_ms = (time.perf_counter() - draw_started) * 1000
    v_draw = fsum(row["probability"] * row["V_r"] for row in outcomes) if outcomes else None

    # The clone shares immutable knowledge/indexes, but no mutable planner/RNG state.
    import copy
    shadow = None
    if config.audit_diamond or config.fallback_policy == 'diamond':
        shadow = copy.copy(cpu)
        shadow.__dict__ = {key: copy.deepcopy(value) if key not in {
            "registered_primes", "registered_composites", "registered_composite_entries",
            "prime_template_index", "small_finish_index", "opening_master_evaluator", "ws", "room"
        } else value for key, value in cpu.__dict__.items()}
    shadow_started = time.perf_counter()
    baseline = diamond_choice(shadow, room) if config.audit_diamond or config.fallback_policy == 'diamond' else None
    comparison_reason = ('draw' if baseline.kind=='draw' else 'all_out' if shadow.platinum_all_out_attempts>cpu.platinum_all_out_attempts
                         else 'plan' if shadow.gold_active_plan else 'other') if baseline else None
    shadow_ms = (time.perf_counter() - shadow_started) * 1000
    draw = bool(outcomes and wants_draw(v_draw, v_now, config) and (not best or best.type != "certified_win"))
    used = draw or best is not None
    if draw:
        action = diamond.CpuAction("draw")
    elif best:
        commit_started = time.perf_counter()
        action = commit_plan(cpu, room, best)
        evaluator.profile_ms['plan_construction_ms'] += 1000 * (time.perf_counter() - commit_started)
    elif config.fallback_policy == 'diamond':
        cpu.__dict__.update(shadow.__dict__)
        action = baseline
    else:
        action = no_plan_action(cpu, room)
    cpu.opening_master_phase = "after_draw" if action.kind == "draw" and not room.has_drawn else "done"
    hits, misses = evaluator.hits - before_hits, evaluator.misses - before_misses
    cpu.opening_master_trace = dict(
        seed=getattr(cpu, "opening_master_seed", None), leg=getattr(cpu, "opening_master_leg", None),
        current_hand=rank_text(hand), best_current_plan=best.summary() if best else {"type": "fallback", "value": config.fallback_value},
        V_now=v_now, draws=outcomes, V_draw=v_draw, delta=None if v_draw is None else v_draw - v_now,
        draw_margin=config.draw_margin, drew=action.kind == "draw", diamond_drew=baseline.kind == "draw" if baseline else None,
        diamond_action=action_summary(baseline) if baseline else None, master_action=action_summary(action),
        decision_changed=action_summary(action) != action_summary(baseline) if baseline else None,
        reason="draw" if draw else best.log_type if best else "fallback difference" if config.fallback_policy=='diamond' else 'no_plan_'+action.kind,
        opening_master_used=used, opening_master_fallback_to_diamond=not used and config.fallback_policy=='diamond',
        no_plan_fallback=not used, fallback_policy=config.fallback_policy,
        diamond_comparison_reason=comparison_reason,
        timing=dict(total_ms=(time.perf_counter() - started) * 1000, current_hand_ms=current_ms,
                    virtual_draw_ms=virtual_ms, rank_average_ms=virtual_ms / len(outcomes) if outcomes else 0,
                    diamond_comparison_ms=shadow_ms,
                    internal_ms={k: v - profile_before.get(k, 0) for k, v in evaluator.profile_ms.items()},
                    internal_counts={k: v - counts_before.get(k, 0) for k, v in evaluator.profile_counts.items()},
                    cache_hits=hits, cache_misses=misses,
                    cache_hit_rate=hits / (hits + misses) if hits + misses else 0.0))
    return action

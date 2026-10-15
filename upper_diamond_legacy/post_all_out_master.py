"""Knowledge-relative Tier 0 after all-out, isolated from released Diamond.

Every proof uses the whole standard deck minus the currently retained hand.
No opponent hand/deck contents, probability tables or unknown-prime oracle.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import cpu_player as diamond
from .master_budget import SearchLimit, check as check_master_budget
from registered_primes import registered_value_encodings
from .opening_master import (CAPACITY, ZERO, LABELS, Move, Plan, Template,
                             _realizations, bind_move, counts, hand_counts,
                             physically_possible, subtract, token_ranks)

POLICY_PATH = Path(__file__).with_name('post_all_out_ceiling.json')
ORDER = (9, 8, 10, 4, 6, 7, 3, 2, 1, 5)


@dataclass(frozen=True)
class Config:
    hand_sizes: tuple[int, ...] = tuple(range(18, 31))
    order: tuple[int, ...] = ORDER
    # None is an exhaustive research run. A timeout never certifies a partial proof.
    max_nodes: int | None = 2000000
    budget_ms: int | None = None
    double_ceiling: bool = True
    double_max_nodes: int | None = 2000000
    three_step: bool = True
    three_max_nodes: int | None = None
    draw_57: bool = True
    pre_hnp: bool = True


@lru_cache(maxsize=8)
def catalog(primes, entries, allow_composite, for_responses=False):
    result = {Template('cut', 0, (0,)), Template('prime', 57, (5, 7))}
    for value in primes:
        if value == 1729:
            continue
        for ranks in registered_value_encodings(value, max_cards=10 if for_responses else 25):
            result.add(Template('prime', value, ranks))
    if allow_composite:
        for entry in entries:
            expression = tuple((t.kind, tuple(t.ranks) if t.kind == 'cards' else t.op)
                               for t in entry.expression_tokens)
            material = tuple(r for kind, part in expression if kind == 'cards' for r in part)
            maximum = 10 if for_responses else 25-len(material)
            for visible in registered_value_encodings(entry.value, max_cards=maximum):
                result.add(Template('composite', int(entry.value), visible, material, expression))
    return tuple(sorted(result, key=lambda t: (len(t.ranks), t.kind, t.value,
                                             t.visible, t.material, str(t.expression))))


def knowledge_key(cpu, room):
    return (tuple(sorted(cpu.registered_primes)), tuple(cpu.registered_composite_entries),
            bool(room.rule.allow_composite))


def sequence_key(sequence):
    return (len(sequence), sum(m.need[0] for m in sequence),
            -sequence[-1].used_count if sequence else 0, tuple(m.key for m in sequence))


def move_resource_key(move):
    """X position/formula spelling is immaterial when both card zones agree."""
    return (move.template.kind, move.value, move.size, move.need,
            counts(move.physical[:move.size]))


def leading_k(move):
    visible = move.template.visible
    return next((i for i, rank in enumerate(visible) if rank != 13), len(visible))


def normalize_moves(moves):
    unique = {}
    for move in sorted(moves, key=lambda m: (-leading_k(m), m.key)):
        unique.setdefault(move_resource_key(move), move)
    return tuple(sorted(unique.values(), key=lambda m: (-m.used_count, m.key)))


def resource_fits(need, available, limits=None):
    # Minimal X substitution is sufficient: optional substitutions cannot
    # improve either physical face or K/X caps. Compute it in one pass.
    jokers = need[0]
    faces = kings = 0
    for rank in range(1, 14):
        wanted, held = need[rank], available[rank]
        if wanted > held:
            jokers += wanted-held
            if jokers > available[0]:
                return False
            physical = held
        else:
            physical = wanted
        if rank >= 10:
            faces += physical
            if rank == 13:
                kings = physical
    return jokers <= available[0] and (limits is None or
        jokers+faces <= limits[0] and jokers+kings <= limits[1])


def _physical_needs(need, available):
    """Exact rank-count substitution needs, without positional X duplicates."""
    deficits = [max(0, need[r]-available[r]) for r in range(14)]
    minimum = need[0]+sum(deficits[1:])
    if minimum > available[0]:
        return
    # Extra X can never improve face/KX caps: substitute mandatory shortages only.
    physical = [min(need[r], available[r]) for r in range(14)]
    physical[0] = minimum
    yield tuple(physical)


class Solver:
    def __init__(self, cpu, room, config=Config()):
        self.cpu, self.room, self.config = cpu, room, config
        self.started = time.perf_counter()
        self.nodes = 0
        self.stats = Counter()
        self.timed_out = False
        self.plans = []
        self.three_step_candidates = []
        self.equal_only_candidates = []
        self.knowledge = knowledge_key(cpu, room)
        self.catalog = catalog(*self.knowledge)
        self.policy = json.loads(POLICY_PATH.read_text(encoding='utf-8'))
        self.policy_hash = hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest()
        self.listed = {(len(token_ranks(t)), int(''.join(map(str, token_ranks(t)))))
                       for t in self.policy['trumps']}
        self.responses = defaultdict(list)
        unique = {(len(t.visible), t.value, counts(t.ranks),
                   t.kind == 'cut' or (t.kind == 'prime' and t.value in (57, 1729)))
                  for t in catalog(*self.knowledge, for_responses=True)}
        for ranks in registered_value_encodings(1729, max_cards=4):
            unique.add((len(ranks), 1729, counts(ranks), True))
        for size, value, need, special in sorted(unique):
            if size <= 10:
                self.responses[size].append((value, need, special))
        for rows in self.responses.values():
            rows.sort(key=lambda row: (row[0] if row[0] else 10**100), reverse=True)
        self.opponent_count = diamond.platinum_opponent_hand_count(cpu, room)
        self.proof_cache = {}
        self.maximum_cache = {}
        self.lead_hazard_cache = {}
        self.tail_cache = {}
        self.fit_cache = {}
        self.fit_index = None
        self.partition_cache = {}
        self.by_need = defaultdict(list)
        self.by_size = defaultdict(list)
        self.moves = ()

    def tick(self):
        check_master_budget(self.cpu)
        self.nodes += 1
        if (self.config.max_nodes is not None and self.nodes > self.config.max_nodes
                or self.config.budget_ms is not None
                and (time.perf_counter()-self.started)*1000 >= self.config.budget_ms):
            raise SearchLimit

    def reuse_prepared(self, prepared):
        """Share completed pure results only for the same catalog and full hand.

        Partial prepare() never publishes moves. Proof-node caches remain owned
        by each Double solver; node IDs and swap/preferred state are not shared.
        """
        if (prepared is None or not prepared.moves or self.knowledge != prepared.knowledge
                or hand_counts(self.cpu.hand) != prepared.prepared_hand
                or self.opponent_count != prepared.opponent_count):
            return
        self.moves, self.by_need, self.by_size = prepared.moves, prepared.by_need, prepared.by_size
        self.prepared_hand, self.fit_index = prepared.prepared_hand, prepared.fit_index
        for name in ('proof_cache', 'maximum_cache', 'tail_cache', 'fit_cache', 'partition_cache'):
            setattr(self, name, getattr(prepared, name))
        # safe_lead also depends on whether a draw is possible.
        if bool(self.room.deck) == bool(prepared.room.deck):
            self.lead_hazard_cache = prepared.lead_hazard_cache

    def prepare(self, hand):
        self.prepared_hand = hand
        moves = []
        for template in self.catalog:
            self.tick()
            if physically_possible(counts(template.ranks), hand):
                moves.extend(_realizations(template, hand))
        # Physical needs (including optional X use and expression material) stay distinct.
        self.moves = normalize_moves(moves)
        for move in self.moves:
            self.by_need[move.need].append(move)
            self.by_size[move.size].append(move)
        self.stats['physical_moves'] = len(self.moves)

    def fitting_moves(self, held, size=None, controls_only=False):
        """Exact physical-subset lookup; preserve all values and material zones.

        Intersect per-rank bit sets instead of scanning every known realization
        again for each residual hand. Bit order preserves self.moves ordering.
        """
        key = (held, size, controls_only)
        if key not in self.fit_cache:
            if self.fit_index is None:
                exact = [[0]*(capacity+1) for capacity in CAPACITY]
                sizes = defaultdict(int)
                controls = 0
                for i, move in enumerate(self.moves):
                    self.tick()
                    bit = 1 << i
                    sizes[move.size] |= bit
                    # On a free-field tail, held only shrinks and the opponent
                    # pool only grows. A policy gate absent at preparation
                    # cannot become true later. Final certain() still checks
                    # the actual residual hand and released materials.
                    if move.cut or self.ceiling_reason(move, self.prepared_hand) is not None:
                        controls |= bit
                    for rank, need in enumerate(move.need):
                        exact[rank][need] |= bit
                for row in exact:
                    for n in range(1, len(row)):
                        row[n] |= row[n-1]
                self.fit_index = (exact, sizes, controls)
            allowed, sizes, controls = self.fit_index
            mask = (1 << len(self.moves))-1 if size is None else sizes.get(size, 0)
            if controls_only:
                mask &= controls
            for rank, count in enumerate(held):
                self.tick()
                mask &= allowed[rank][count]
                if not mask:
                    break
            rows = []
            while mask:
                self.tick()
                bit = mask & -mask
                rows.append(self.moves[bit.bit_length()-1])
                mask ^= bit
            self.fit_cache[key] = tuple(rows)
        return self.fit_cache[key]

    def possible_responses(self, move, held, *, inclusive=False, locked=ZERO):
        """Unlimited opponent capacity, with all released cards available again."""
        key = (move.size, move.value, held, inclusive, locked)
        if key not in self.proof_cache:
            available = subtract(subtract(CAPACITY, held), locked)
            found = []
            for value, need, special in self.responses[move.size]:
                self.tick()
                if value != 0 and not (value >= move.value if inclusive else value > move.value):
                    break  # Response rows are sorted by descending value (X first).
                if physically_possible(need, available):
                    found.append((value, need, special))
            self.proof_cache[key] = tuple(found)
        return self.proof_cache[key]

    def certain(self, move, held, *, inclusive=False, locked=ZERO, limits=None):
        """Master patch proofs only; Diamond fallback keeps its own certainty policy."""
        if move.is_57:
            return False
        if move.cut:
            return True
        if move.size > 10:
            return False
        if not inclusive and move.template.material:
            # Once T/control is played, its materials immediately re-enter the
            # deck. They cannot be counted as retained resources against replies.
            material = counts(move.physical[move.size:])
            held = subtract(held, material)
            if limits is not None:
                limits = (limits[0]+material[0]+sum(material[10:14]), limits[1]+material[0]+material[13])
        # The policy is mandatory for every control, including alternative P/Q
        # and Certified tails. Knowledge alone never grants certain status.
        if self.ceiling_reason(move, held, locked=locked, limits=limits) is None:
            return False
        maximum = self.maximum_response(move.size, held, locked=locked, limits=limits)
        return maximum < move.value if inclusive else maximum <= move.value

    def maximum_response(self, size, held, *, locked=ZERO, limits=None):
        key = (size, held, locked, limits)
        if key not in self.maximum_cache:
            available = subtract(subtract(CAPACITY, held), locked)
            maximum = -1
            for value, need, _ in self.responses[size]:
                self.tick()
                if resource_fits(need, available, limits):
                    maximum = value if value else 10**100
                    break
            self.maximum_cache[key] = maximum
        return self.maximum_cache[key]

    def ceiling_reason(self, move, held, *, locked=ZERO, limits=None):
        if move.is_57:
            return None
        available = subtract(subtract(CAPACITY, held), locked)
        visible = move.template.visible
        kx = available[13]+available[0]
        face_count = sum(available[10:14])+available[0]
        if limits is not None:
            face_count, kx = min(face_count, limits[0]), min(kx, limits[1])
        if move.cut:
            return 'rule-cut'
        if (move.size, move.value) in self.listed:
            return 'listed-ceiling'
        if all(rank >= 10 for rank in visible) and kx < leading_k(move):
            return 'k-x-lock'
        if face_count < sum(rank >= 10 for rank in visible):
            return 'face-count-lock'
        return None

    def free_control_certain(self, move, held):
        """57 is a guaranteed cut on an empty field, never a rally ceiling."""
        return move.cut or self.certain(move, held)

    def control_tail(self, held):
        if held in self.tail_cache:
            return self.tail_cache[held]
        self.tick()
        self.stats['tail_states'] += 1
        if held == ZERO:
            return ()
        finishes = self.by_need.get(held)
        if finishes:
            result = (min(finishes, key=lambda m: (m.need[0], m.key)),)
        else:
            result = None
            for move in self.fitting_moves(held, controls_only=True):
                self.tick()
                rest = subtract(held, move.need)
                if rest is None or not self.free_control_certain(move, held):
                    continue
                suffix = self.control_tail(rest)
                if suffix is not None:
                    result = (move,)+suffix
                    break
        self.tail_cache[held] = result
        return result

    def direct_control_tail(self, held):
        """Fast existing Certified gate: a control followed by an exact finish."""
        for finish in self.moves:
            self.tick()
            control_need = subtract(held, finish.need)
            if control_need is None:
                continue
            for control in self.by_need.get(control_need, ()):
                if self.free_control_certain(control, held):
                    return (control, finish)
        return None

    def safe_lead(self, lead, trump, held_after_lead):
        if trump.is_57:
            return False
        trump_value = 10**100 if trump.template.kind == 'cut' else trump.value
        if lead.cut or lead.value >= trump_value or self.opponent_count is None:
            return False
        # A composite L releases materials to the deck immediately.
        draw = bool(getattr(self.room, 'deck', ())) or bool(lead.template.material)
        winning_sizes = {self.opponent_count, self.opponent_count + int(draw)}
        # All lead values sharing the same consumed cards see the same pool.
        # Only the maximum response and maximum cut/immediate-finish hazard
        # matter here, not a repeated list of every prime above each lead.
        maximum = self.maximum_response(lead.size, held_after_lead)
        if maximum >= trump_value:
            return False
        key = (lead.size, held_after_lead, draw)
        if key not in self.lead_hazard_cache:
            available = subtract(CAPACITY, held_after_lead)
            hazard = -1
            for value, need, special in self.responses[lead.size]:
                if special or sum(need) in winning_sizes:
                    self.tick()
                    if physically_possible(need, available):
                        hazard = value if value else 10**100
                        break
            self.lead_hazard_cache[key] = hazard
        return self.lead_hazard_cache[key] <= lead.value

    def partitions(self, remaining, size):
        """L + Certified remainder; enumerate a larger one-move remainder first.

        The second pass covers multi-control remainders and material-heavy L.
        No cap or representative-value pruning changes exhaustive coverage.
        """
        key = (remaining, size)
        if key in self.partition_cache:
            yield from self.partition_cache[key]
            return
        found = []
        leads_by_need = defaultdict(list)
        fitting = self.fitting_moves(remaining)
        for lead in fitting:
            self.tick()
            if lead.size == size and not lead.cut:
                leads_by_need[lead.need].append(lead)
        seen = set()
        for finish in fitting:
            self.tick()
            lead_need = subtract(remaining, finish.need)
            if lead_need is None or finish.used_count < sum(lead_need):
                continue
            for lead in leads_by_need.get(lead_need, ()):
                if lead.key not in seen:
                    seen.add(lead.key)
                    found.append((lead, (finish,)))
                    yield lead, (finish,)
        for need, leads in leads_by_need.items():
            for lead in leads:
                if lead.key not in seen:
                    found.append((lead, None))
                    yield lead, None
        self.partition_cache[key] = tuple(found)

    def save_third_stage(self, trump, remaining):
        # Save a concrete physical L1<L2<T, never a Mate or an executable plan.
        if trump.is_57:
            return
        for first in self.fitting_moves(remaining, trump.size):
            self.tick()
            rest = subtract(remaining, first.need)
            if first.cut or first.value >= trump.value or rest is None:
                continue
            for second in self.fitting_moves(rest, trump.size):
                self.tick()
                tail = subtract(rest, second.need)
                if second.cut or not first.value < second.value < trump.value or tail is None:
                    continue
                self.three_step_candidates.append(dict(
                    type='three_step_candidate', mate=False,
                    steps=[m.summary() for m in (first, second, trump)],
                    remainder=list(tail)))
                return

    def evaluate(self, hand):
        if sum(hand) not in self.config.hand_sizes:
            raise ValueError('Post-all-out Master requires an eligible hand size')
        try:
            self.prepare(hand)
            finishes = self.by_need.get(hand, ())
            if finishes:
                self.plans = [Plan('certified_win', (min(finishes, key=lambda m: m.key),),
                                   0, 'immediate_finish', 0)]
                return self.plans[0]
            existing = self.direct_control_tail(hand)
            if existing:
                self.plans.append(Plan('certified_win', existing, 0, 'public_control_sequence', 0))
            failed = []
            order = tuple(dict.fromkeys((*self.config.order, *range(1, 11))))
            for size in order:
                for trump in self.by_size[size]:
                    self.tick()
                    remaining = subtract(hand, trump.need)
                    if sum(remaining) < size:
                        continue
                    if self.ceiling_reason(trump, hand) is None:
                        continue
                    # Cheap optimistic prefilter; final proof releases L as well.
                    if not self.certain(trump, hand, inclusive=True):
                        continue
                    self.stats['ceiling_candidates'] += 1
                    found = False
                    for lead, tail in self.partitions(remaining, size):
                        if lead.value >= (10**100 if trump.template.kind == 'cut' else trump.value):
                            continue
                        held = subtract(hand, lead.need)
                        if self.ceiling_reason(trump, held) is None:
                            continue
                        if not self.certain(trump, held, inclusive=True):
                            if self.maximum_response(trump.size, held) == trump.value:
                                replies = self.possible_responses(trump, held, inclusive=True)
                                if replies and all(v == trump.value and not special for v, _, special in replies):
                                    self.equal_only_candidates.append(dict(lead=lead.summary(), trump=trump.summary(), mate=False))
                            continue
                        if not self.safe_lead(lead, trump, held):
                            continue
                        if not self.certain(trump, held):
                            continue
                        if tail is None:
                            tail = self.control_tail(subtract(remaining, lead.need))
                        if tail is None:
                            continue
                        pass_tail = self.control_tail(held)
                        if pass_tail is None:
                            continue
                        # `tail` was certified with L and T released, including their materials.
                        plan = Plan('certified_win', (lead, trump)+tail, 0,
                                    'same_size_two_step_mate', 0, pass_tail=pass_tail)
                        self.plans.append(plan)
                        self.stats[self.ceiling_reason(trump, held)] += 1
                        found = True
                    if not found:
                        failed.append((trump, remaining))
            existing = self.control_tail(hand)
            if existing:
                self.plans.append(Plan('certified_win', existing, 0, 'public_control_sequence', 0))
            # Lower-tier work cannot use up the budget before any two-step proof.
            for trump, remaining in failed:
                if sum(remaining) >= 2*trump.size:
                    self.save_third_stage(trump, remaining)
        except SearchLimit:
            self.timed_out = True
        return min(self.plans, key=lambda p: (p.model != 'public_control_sequence',
                   sequence_key(p.steps), sequence_key(p.pass_sequence))) if self.plans else None

    def summary(self, best):
        return dict(policy_version=self.policy['version'], policy_sha256=self.policy_hash,
                    proof_scope='registered-knowledge', elapsed_ms=(time.perf_counter()-self.started)*1000,
                    nodes=self.nodes, timed_out=self.timed_out, exhaustive=not self.timed_out,
                    stats=dict(self.stats), certified_plans=len(self.plans),
                    best=best.summary() if best else None,
                    three_step_candidates=self.three_step_candidates,
                    equal_only_candidates=self.equal_only_candidates)


def play_next(cpu, room, active, sequence):
    move, *rest = sequence
    candidate = bind_move(move, cpu.hand)
    expected = subtract(hand_counts(cpu.hand), move.need)
    active.update(expected=expected, sequence=tuple(rest))
    cpu.post_all_out_master_active = active if rest else None
    cpu.diamond_last_route_kind = 'post-all-out-master-'+active['model']
    cpu.diamond_last_certainty = 'heuristic' if active['model'] == 'three_step' else 'certain'
    cpu.diamond_last_return_probability = None if active['model'] == 'three_step' else 0.0
    diamond.diamond_update_opening_position_history(cpu, room)
    diamond.diamond_remember_cards(cpu, diamond.candidate_consumed_cards(candidate))
    return diamond.platinum_commit_play(cpu, diamond.candidate_to_action(candidate))


def continue_plan(cpu, room):
    active = getattr(cpu, 'post_all_out_master_active', None)
    if not active:
        return None
    if (room.reverse_order or hand_counts(cpu.hand) != active['expected']
            or knowledge_key(cpu, room) != active['knowledge']):
        cpu.post_all_out_master_active = None
        return None
    if active.get('branch_certificate') or active['model'] == 'double_ceiling_mate':
        from .double_ceiling import continue_double
        return continue_double(cpu, room, active)
    if active['model'] == 'three_step':
        from .three_step import continue_three
        return continue_three(cpu, room, active)
    sequence = active['sequence']
    if active.pop('awaiting_lead_reply', False) and not room.field:
        sequence = active['pass_tail']
    if not sequence:
        cpu.post_all_out_master_active = None
        return None
    move = sequence[0]
    if room.field and (len(room.field) != move.size or
                       not move.template.kind == 'cut' and int(room.last_number) >= move.value):
        cpu.post_all_out_master_active = None
        return None
    cpu.post_all_out_master_trace = dict(continuation=True, model=active['model'])
    return play_next(cpu, room, active, sequence)


def evaluate_current_state(cpu, room, config=Config()):
    """Search only the supplied hand; do not commit actions or change CPU state."""
    solver = Solver(cpu, room, config)
    best = solver.evaluate(hand_counts(cpu.hand))
    trace = solver.summary(best)
    trace['stage_results'] = dict(mate=best.summary() if best else None,
        mate_count=len(solver.plans), double=None, three=None)
    owner = solver
    if best is None and config.double_ceiling and sum(
            1 for card in cpu.hand if diamond.is_joker(card) or int(card['rank']) >= 10) >= 10:
        from .double_ceiling import DoubleCeilingSolver, commit_node
        double = DoubleCeilingSolver(cpu, room, config, prepared=solver)
        double_best = double.evaluate_double(hand_counts(cpu.hand))
        solver.double_candidates = double.candidates
        trace['double_ceiling'] = double.summary_double()
        trace['stage_results']['double'] = double_best.summary() if double_best else None
        if double_best is not None:
            best, owner = double_best, double
            trace['best'] = best.summary()
            trace['certified_plans'] += 1
    if best is None and config.three_step:
        from .three_step import ThreeStepSolver
        from .double_ceiling import commit_node
        three = ThreeStepSolver(cpu, room, config, prepared=solver)
        three_best = three.evaluate_three(hand_counts(cpu.hand))
        trace['three_step'] = three.summary_three()
        trace['stage_results']['three'] = three_best.summary() if three_best else None
        if three_best is not None:
            best, owner = three_best, three
            trace['best'] = best.summary()
            if three.certified is not None:
                trace['certified_plans'] += 1
                trace['three_step']['certificate'] = three.certificate(three_best.root)
    return solver, owner, best, trace


def commit_current_plan(cpu, room, owner, best):
    diamond.clear_gold_active_plan(cpu)
    if hasattr(best, 'root'):
        from .double_ceiling import commit_node
        active = dict(model=best.model, knowledge=owner.knowledge, branch_certificate=True)
        return commit_node(cpu, room, active, best.root)
    if best.model == 'three_step':
        active = dict(model=best.model, knowledge=owner.knowledge,
                      response_nodes=best.response_nodes, pass_node=best.pass_node,
                      awaiting_three_reply=True)
        return play_next(cpu, room, active, best.steps)
    active = dict(model=best.model, knowledge=owner.knowledge,
                  awaiting_lead_reply=best.model == 'same_size_two_step_mate',
                  pass_tail=best.pass_sequence)
    return play_next(cpu, room, active, best.steps)


def choose_post_all_out(cpu, room, fallback, config=Config()):
    from .after_all_out_planner import choose_after_all_out
    return choose_after_all_out(cpu, room, fallback, config)

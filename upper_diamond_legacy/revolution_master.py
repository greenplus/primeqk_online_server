"""Reverse-order Master: certified two-step, heuristic three-step, 1729 recovery.

Only currently registered numerical knowledge is used. Zero digits require real
X cards. The recovery stage is the extension point for future recovery policies;
it runs below completed tactics and above Diamond, without changing Diamond.
"""
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache

import cpu_player as diamond
from .opening_master import (CAPACITY, ZERO, Move, Template, _realizations,
                             bind_move, counts, hand_counts, physically_possible, subtract)
from .post_all_out_master import (Config, Solver, knowledge_key,
                                  sequence_key)
from .master_budget import SearchLimit


LISTED = {2: frozenset((10, 11, 12, 13)),
          3: frozenset((101, 103, 104, 105, 106, 107)),
          5: frozenset((10125, 10133, 10139))}


@lru_cache(maxsize=8192)
def encodings(value, max_cards=25):
    """Known values only; zero is an X requirement, never a physical digit card."""
    text = str(value)
    result = []

    def visit(i, ranks, zeros):
        if len(ranks) > max_cards or zeros > 2:
            return
        if i == len(text):
            if physically_possible(counts(ranks), CAPACITY):
                result.append(ranks)
            return
        digit = int(text[i])
        visit(i + 1, ranks + (digit,), zeros + (digit == 0))
        if i + 1 < len(text) and 10 <= int(text[i:i+2]) <= 13:
            visit(i + 2, ranks + (int(text[i:i+2]),), zeros)

    if value > 0:
        visit(0, (), 0)
    return tuple(sorted(set(result), key=lambda r: (len(r), r)))


@lru_cache(maxsize=8)
def catalog(primes, entries, allow_composite, responses=False):
    result = {Template('cut', 0, (0,)), Template('prime', 57, (5, 7))}
    for value in primes:
        if value == 1729:
            continue
        for visible in encodings(value, 10 if responses else 25):
            result.add(Template('prime', value, visible))
    if responses:
        for visible in encodings(1729, 4):
            result.add(Template('prime', 1729, visible))
    if allow_composite:
        for entry in entries:
            expression = tuple((t.kind, tuple(t.ranks) if t.kind == 'cards' else t.op)
                               for t in entry.expression_tokens)
            material = tuple(r for kind, part in expression if kind == 'cards' for r in part)
            for visible in encodings(entry.value, 10 if responses else 25-len(material)):
                result.add(Template('composite', int(entry.value), visible, material, expression))
    return tuple(sorted(result, key=lambda t: (len(t.visible), t.value, t.kind,
                                              t.visible, t.material, str(t.expression))))


def prefix_locked(move, available):
    """A/X alone rules out an equal or smaller leading 0/1 prefix.

    Other digits are unlimited, optimistically for the opponent. All displayed
    tokens must be single digits, including X=0..9 (minimum decimal length).
    """
    visible = move.template.visible
    if not visible or visible[0] != 1 or any(r > 9 for r in visible):
        return False
    prefix = []
    for rank in visible:
        if rank not in (0, 1):
            break
        prefix.append(rank)
    aces, jokers = available[1], available[0]
    lower = []
    for i in range(len(prefix)):
        if i and jokers:
            lower.append(0)
            jokers -= 1
        elif aces:
            lower.append(1)
            aces -= 1
        elif jokers:
            lower.append(1)
            jokers -= 1
        else:
            lower.append(2)
    return tuple(lower) > tuple(prefix)


@dataclass(frozen=True)
class RevolutionPlan:
    steps: tuple
    model: str
    pass_tail: tuple = ()

    @property
    def certified(self):
        return self.model != 'three_step'

    def summary(self):
        return dict(model=self.model, mate=self.certified,
                    certainty='certain' if self.certified else 'heuristic',
                    steps=[m.summary() for m in self.steps],
                    pass_tail=[m.summary() for m in self.pass_tail])


class RevolutionSolver(Solver):
    def __init__(self, cpu, room, config=None):
        super().__init__(cpu, room, config or Config(hand_sizes=tuple(range(1, 31)),
                                                    max_nodes=None))
        self.catalog = catalog(*self.knowledge)
        self.responses = defaultdict(list)
        unique = {(len(t.visible), t.value, counts(t.ranks),
                   t.kind == 'cut' or (t.kind == 'prime' and t.value in (57, 1729)))
                  for t in catalog(*self.knowledge, responses=True)}
        for size, value, need, special in sorted(unique):
            self.responses[size].append((value, need, special))

    def minimum_response(self, size, held, locked=ZERO):
        key = (size, held, locked)
        if key not in self.maximum_cache:
            available = subtract(subtract(CAPACITY, held), locked)
            value = 10**100
            if available is not None:
                for number, need, _ in self.responses[size]:
                    self.tick()
                    if physically_possible(need, available):
                        value = number if number else -1  # single X cuts even in revolution
                        break
            self.maximum_cache[key] = value
        return self.maximum_cache[key]

    def ceiling_reason(self, move, held, *, locked=ZERO, limits=None):
        if move.is_57:
            return None
        if move.template.kind == 'cut':
            return 'rule-cut'
        if move.size > 10:
            return None
        if move.size in LISTED:
            return 'listed-revolution' if move.value in LISTED[move.size] else None
        available = subtract(subtract(CAPACITY, held), locked)
        if available is not None and prefix_locked(move, available):
            return 'a-x-prefix-lock'
        return None

    def certain(self, move, held, *, inclusive=False, locked=ZERO, limits=None):
        if move.template.kind == 'cut':
            return True
        if not inclusive and move.template.material:
            held = subtract(held, counts(move.physical[move.size:]))
        if held is None or self.ceiling_reason(move, held, locked=locked) is None:
            return False
        minimum = self.minimum_response(move.size, held, locked)
        return minimum > move.value if inclusive else minimum >= move.value

    def safe_lead(self, lead, trump, held_after_lead):
        trump_value = -1 if trump.template.kind == 'cut' else trump.value
        if lead.cut or trump.is_57 or lead.value <= trump_value or self.opponent_count is None:
            return False
        if lead.size == 2 and lead.value >= 57:
            return False
        if lead.size == 4 and lead.value >= 1729:
            return False
        if self.minimum_response(lead.size, held_after_lead) <= trump_value:
            return False
        draw = bool(self.room.deck) or bool(lead.template.material)
        winning_sizes = {self.opponent_count, self.opponent_count + int(draw)}
        available = subtract(CAPACITY, held_after_lead)
        for value, need, special in self.responses[lead.size]:
            self.tick()
            if value and value >= lead.value:
                break
            if (special or sum(need) in winning_sizes) and physically_possible(need, available):
                return False
        return True

    def evaluate_revolution(self):
        hand = hand_counts(self.cpu.hand)
        best = None
        try:
            self.prepare(hand)
            if hand in self.by_need:
                return RevolutionPlan((min(self.by_need[hand], key=lambda m: m.key),), 'immediate')
            direct = self.direct_control_tail(hand)
            if direct:
                return RevolutionPlan(direct, 'control_sequence')
            for size in tuple(dict.fromkeys((*self.config.order, *range(1, 11)))):
                for trump in self.by_size[size]:
                    self.tick()
                    if trump.is_57 or not self.certain(trump, hand, inclusive=True):
                        continue
                    remaining = subtract(hand, trump.need)
                    for lead, tail in self.partitions(remaining, size):
                        self.tick()
                        held = subtract(hand, lead.need)
                        if (not self.certain(trump, held, inclusive=True)
                                or not self.safe_lead(lead, trump, held)
                                or not self.certain(trump, held)):
                            continue
                        if tail is None:
                            tail = self.control_tail(subtract(remaining, lead.need))
                        if tail is None:
                            continue
                        passed = self.control_tail(held)
                        if passed is None:
                            continue
                        plan = RevolutionPlan((lead, trump)+tail, 'two_step_mate', passed)
                        if best is None or sequence_key(plan.steps) < sequence_key(best.steps):
                            best = plan
            control = self.control_tail(hand)
            if control:
                return RevolutionPlan(control, 'control_sequence')
            if best:
                return best
            # Lower tier: no frontier, reply enumeration, or Double Ceiling.
            for size in tuple(dict.fromkeys((*self.config.order, *range(1, 11)))):
                if size in (2, 4):
                    continue
                for trump in self.by_size[size]:
                    self.tick()
                    if trump.is_57 or not self.certain(trump, hand, inclusive=True):
                        continue
                    after_t = subtract(hand, trump.need)
                    for middle in self.fitting_moves(after_t, size):
                        if middle.cut or middle.value <= trump.value:
                            continue
                        left = subtract(after_t, middle.need)
                        for lead in self.fitting_moves(left, size):
                            self.tick()
                            if lead.cut or lead.value <= middle.value:
                                continue
                            tail_hand = subtract(left, lead.need)
                            held = subtract(subtract(hand, lead.need), middle.need)
                            if not self.certain(trump, held, inclusive=True) or not self.certain(trump, held):
                                continue
                            tail = self.control_tail(tail_hand)
                            if tail is None:
                                continue
                            plan = RevolutionPlan((lead, middle, trump)+tail, 'three_step')
                            if best is None or sequence_key(plan.steps) < sequence_key(best.steps):
                                best = plan
        except SearchLimit:
            self.timed_out = True
        return best


def legal(move, room):
    if not room.field:
        return True
    if move.template.kind == 'cut':
        return len(room.field) == 1
    return move.size == len(room.field) and move.value < int(room.last_number)


def play(cpu, room, move, model, certified):
    candidate = bind_move(move, cpu.hand)
    diamond.clear_gold_active_plan(cpu)
    diamond.diamond_update_opening_position_history(cpu, room)
    diamond.diamond_remember_cards(cpu, diamond.candidate_consumed_cards(candidate))
    cpu.diamond_last_route_kind = 'revolution-master-'+model
    cpu.diamond_last_certainty = 'certain' if certified else 'heuristic'
    cpu.diamond_last_return_probability = 0.0 if certified else None
    return diamond.platinum_commit_play(cpu, diamond.candidate_to_action(candidate))


def execute_plan(cpu, room, active):
    move, *rest = active['sequence']
    active.update(sequence=tuple(rest), expected=subtract(hand_counts(cpu.hand), move.need))
    cpu.revolution_master_active = active if rest else None
    return play(cpu, room, move, active['model'], active['model'] != 'three_step')


def continue_plan(cpu, room):
    active = getattr(cpu, 'revolution_master_active', None)
    if not active:
        return None
    if (not room.reverse_order or active['expected'] != hand_counts(cpu.hand)
            or active['knowledge'] != knowledge_key(cpu, room)):
        cpu.revolution_master_active = None
        return None
    if active.pop('awaiting_lead_reply', False) and not room.field:
        active['sequence'] = active['pass_tail']
    if not active['sequence'] or not legal(active['sequence'][0], room):
        # Never skip an unplayable middle move and execute the wrong remainder.
        cpu.revolution_master_active = None
        return None
    cpu.revolution_master_trace = dict(continuation=True, model=active['model'])
    return execute_plan(cpu, room, active)


def public_locked(room):
    cards = {str(c['card_id']): c for c in (*room.field, *getattr(room, 'reserve', ())) }
    return hand_counts(cards.values())


def recovery_candidate(solver):
    cpu, room = solver.cpu, solver.room
    context = getattr(room, 'revolution_recovery_context', None)
    if not context or not context['active'] or len(room.field) != 4:
        return None
    reserve = list(getattr(room, 'reserve', ()))
    if not set(context['trigger_ids']) <= {str(c['card_id']) for c in reserve}:
        return None
    held = hand_counts(cpu.hand)
    locked = public_locked(room)
    choices = []
    for template in solver.catalog:
        solver.tick()
        if len(template.visible) != 4 or template.value >= int(room.last_number):
            continue
        if not physically_possible(counts(template.ranks), held):
            continue
        for move in _realizations(template, held):
            solver.tick()
            material = counts(move.physical[move.size:])
            retained = subtract(held, material)
            available = subtract(subtract(CAPACITY, retained), locked)
            if available is None or not prefix_locked(move, available):
                continue
            if solver.minimum_response(4, retained, locked) <= move.value:
                continue
            # With an empty deck the opponent could draw a just-returned
            # material card. Such a route cannot recover every physical card.
            if not room.deck and move.template.material:
                continue
            after_count = len(cpu.hand) - move.used_count
            future_deck = len(room.deck) + len(reserve) + move.used_count
            if after_count <= 0 or future_deck > after_count + 2:
                continue
            choices.append(move)
    return min(choices, key=lambda m: (m.value, m.used_count, m.need[0], m.key)) if choices else None


def recovery_pending(cpu, room):
    pending = getattr(cpu, 'revolution_master_recovery', None)
    if not pending:
        return None
    context = getattr(room, 'revolution_recovery_context', None)
    valid = (room.reverse_order and not room.field and context and not context['active']
             and context['serial'] == pending['serial']
             and context['closing_player'] == cpu.id
             and set(context['closing_ids']) == set(pending['visible_ids'])
             and knowledge_key(cpu, room) == pending['knowledge'])
    current_ids = {str(c['card_id']) for c in cpu.hand}
    expected_ids = set(pending['expected_ids'])
    if pending['phase'] == 'draw':
        valid = valid and room.has_drawn and expected_ids < current_ids and len(current_ids) == len(expected_ids)+1
    else:
        valid = valid and current_ids == expected_ids and not room.has_drawn
    known_ids = {str(c['card_id']) for c in getattr(room, 'public_known_deck_bottom', ())}
    valid = valid and set(pending['target_ids']) <= known_ids | current_ids
    if not valid:
        cpu.revolution_master_recovery = None
        return None
    if pending['phase'] == 'await_flow':
        if not room.deck or len(room.deck) > len(cpu.hand)+2:
            cpu.revolution_master_recovery = None
            return None
        pending['phase'] = 'draw'
        cpu.revolution_master_trace = dict(recovery=True, phase='draw', target_ids=pending['target_ids'])
        cpu.diamond_last_route_kind = 'revolution-master-recovery-draw'
        return diamond.CpuAction('draw')
    cpu.revolution_master_recovery = None
    if len(room.deck) > len(cpu.hand):
        return None
    payload = diamond.build_gold_all_out_payload(cpu.hand, force_random=True, rng=cpu.rng)
    if payload is None:
        return None
    cpu.platinum_all_out_attempts += 1
    diamond.clear_gold_active_plan(cpu)
    cpu.revolution_master_trace = dict(recovery=True, phase='all_out', target_ids=pending['target_ids'])
    cpu.diamond_last_route_kind = 'revolution-master-recovery-all-out'
    cpu.diamond_last_certainty, cpu.diamond_last_return_probability = 'unclassified', None
    return diamond.platinum_commit_play(cpu, diamond.CpuAction('play_prime', payload))


def choose_revolution(cpu, room, fallback):
    cpu.revolution_master_trace = None
    if room.rule.key != 'std-11-n-c' or not room.reverse_order:
        cpu.revolution_master_active = None
        cpu.revolution_master_recovery = None
        return fallback(cpu, room)
    pending = recovery_pending(cpu, room)
    if pending is not None:
        return pending
    continued = continue_plan(cpu, room)
    if continued is not None:
        return continued
    context = getattr(room, 'revolution_recovery_context', None)
    recovery_scope = bool(context and context['active'] and len(room.field) == 4)
    mate_scope = (not room.field and 0 < len(cpu.hand) <= 30
                  and getattr(cpu, 'platinum_all_out_attempts', 0) > 0)
    if not mate_scope and not recovery_scope:
        return fallback(cpu, room)
    solver = RevolutionSolver(cpu, room, getattr(cpu, 'revolution_master_config', None))
    if mate_scope:
        best = solver.evaluate_revolution()
        cpu.revolution_master_trace = dict(nodes=solver.nodes, timed_out=solver.timed_out,
                                           best=best.summary() if best else None)
        if best:
            active = dict(sequence=best.steps, model=best.model, knowledge=knowledge_key(cpu, room),
                          awaiting_lead_reply=best.model == 'two_step_mate', pass_tail=best.pass_tail)
            return execute_plan(cpu, room, active)
    if recovery_scope:
        try:
            action = recovery_action(solver)
        except SearchLimit:
            cpu.revolution_master_trace = dict(recovery=True, timed_out=True, nodes=solver.nodes)
        else:
            if action is not None:
                return action
    return fallback(cpu, room)


def recovery_action(solver):
    cpu, room = solver.cpu, solver.room
    context = room.revolution_recovery_context
    # Immediate known finishes precede any new recovery reservation.
    held = hand_counts(cpu.hand)
    for template in solver.catalog:
        solver.tick()
        if len(template.ranks) != len(cpu.hand) or not physically_possible(counts(template.ranks), held):
            continue
        for move in _realizations(template, held):
            if legal(move, room):
                return play(cpu, room, move, 'immediate', True)
    move = recovery_candidate(solver)
    if move:
        candidate = bind_move(move, cpu.hand)
        consumed = diamond.candidate_consumed_cards(candidate)
        used_ids = {str(c['card_id']) for c in consumed}
        cpu.revolution_master_recovery = dict(
            serial=context['serial'], phase='await_flow', knowledge=knowledge_key(cpu, room),
            expected_ids=tuple(str(c['card_id']) for c in cpu.hand if str(c['card_id']) not in used_ids),
            target_ids=tuple(context['trigger_ids'])+tuple(sorted(used_ids)),
            visible_ids=tuple(str(c['card_id']) for c in candidate['cards']))
        cpu.revolution_master_trace = dict(recovery=True, phase='take_lead', move=move.summary())
        return play(cpu, room, move, 'recovery-take-lead', True)
    if not room.has_drawn and room.deck:
        cpu.revolution_master_trace = dict(recovery=True, phase='try_draw')
        cpu.diamond_last_route_kind = 'revolution-master-recovery-try-draw'
        return diamond.CpuAction('draw')
    return None

"""Master epsilon's public-information tactics and recovery policy."""
from dataclasses import dataclass, replace
from functools import lru_cache
import time

import cpu_player as diamond
from . import master_combined
from .after_all_out_planner import public_room
from .master_budget import CURRENT, MasterDeadline, check
from .move_safety import uses_spare_natural
from .opening_master import CAPACITY, ZERO, Move, Template, bind_move, counts, hand_counts, subtract
from .post_all_out_master import knowledge_key
from .public_control import (
    ControlSearch, choose_public_control, known_draw_finish, legal, numerical_bound,
    realizations, unique_cards,
)
from .revolution_master import LISTED, catalog, encodings


EPSILON_CERTAIN_CANDIDATES = {**LISTED, 5: LISTED[5] | {10125, 10128}}
REVERSE_TRUMPS = frozenset(
    (size, value) for size, values in EPSILON_CERTAIN_CANDIDATES.items() for value in values
)
GENERALIZED_SIZES = (3, 5, 6)


def bounded_search(milliseconds, search, *args):
    budget = CURRENT.get()
    if budget is None:
        return search(*args)
    cap = min(budget.config.search_ms,
              int(1000 * (time.perf_counter() - budget.started)) + milliseconds)
    if cap == budget.config.search_ms:
        return search(*args)
    token = CURRENT.set(replace(budget, config=replace(budget.config, search_ms=cap)))
    try:
        return search(*args)
    except MasterDeadline:
        return None
    finally:
        CURRENT.reset(token)


@lru_cache(maxsize=2048)
def listed_counter(pool, size, value, reverse):
    for count, number in REVERSE_TRUMPS:
        if count != size or not bound_beats(number, value, reverse):
            continue
        for ranks in encodings(number, size):
            if len(ranks) != size:
                continue
            need = counts(ranks)
            missing = sum(max(0, need[r] - pool[r]) for r in range(1, 14))
            if need[0] + missing <= pool[0]:
                return True
    return False


@dataclass(frozen=True)
class Resources:
    pool: tuple
    opponent: int
    queue: tuple
    reserve: tuple

    def after_draw(self):
        if not self.queue:
            return self
        return replace(self, opponent=self.opponent + 1, queue=self.queue[1:])


@dataclass(frozen=True)
class Rally:
    move: Move
    reply: 'Rally | None' = None
    certainty: bool = True
    pass_tail: tuple = ()
    alternate: 'Rally | None' = None


@dataclass(frozen=True)
class GeneralizedPlan:
    steps: tuple[Move, ...]
    finish: Move
    depth: int
    decoy: bool = False


def resources(cpu, room):
    # A known bottom card can become available after later draws. Keep it in
    # the upper bound, but retain its actual queue position for recovery.
    reserve = unique_cards((*getattr(room, 'reserve', ()), *room.field))
    pool = subtract(CAPACITY, hand_counts(unique_cards((*cpu.hand, *reserve))))
    opponent = diamond.platinum_opponent_hand_count(cpu, room)
    if pool is None or opponent is None:
        return None
    queue = tuple(None if card is None else rank(card) for card in public_room(room).draw_queue)
    return Resources(pool, opponent, queue, tuple(map(rank, reserve)))


def rank(card):
    return 0 if diamond.is_joker(card) else int(card['rank'])


def bound_beats(value, limit, reverse, inclusive=False):
    if value is None:
        return False
    if reverse:
        return value <= limit if inclusive else value < limit
    return value >= limit if inclusive else value > limit


def special_reply(move, pool, reverse):
    if move.size == 1 and pool[0]:
        return True
    special = {2: (57, (5, 7)), 4: (1729, (1, 7, 2, 9))}.get(move.size)
    if special is None or not bound_beats(special[0], move.value, reverse):
        return False
    need = counts(special[1])
    return sum(max(0, need[r] - pool[r]) for r in range(1, 14)) <= pool[0]


def revolution_move(move):
    return move.template.kind == 'prime' and move.value == 1729


def expose(state, move):
    visible = move.physical[:move.size]
    material = move.physical[move.size:]
    pool = tuple(n + k for n, k in zip(state.pool, counts(material)))
    return Resources(pool, state.opponent, state.queue + material, state.reserve + visible)


def turn_states(state, move):
    exposed = expose(state, move)
    if move.cut or not exposed.queue:
        return (exposed,)
    return exposed, exposed.after_draw()


def flow(state):
    visible = counts(rank for rank in state.reserve if rank is not None)
    pool = tuple(n + k for n, k in zip(state.pool, visible))
    return Resources(pool, state.opponent, state.queue + state.reserve, ())


def pass_result(state, move):
    return tuple(map(flow, turn_states(state, move)))


@lru_cache(maxsize=32768)
def reply_costs(pool, size, value):
    # Enumerate only A/X consumption, not response values. A numeric bound
    # over all remaining ranks includes unregistered primes and composites.
    costs = []
    for jokers in range(pool[0] + 1):
        for aces in range(pool[1] + 1):
            limited = list(pool)
            limited[0], limited[1] = jokers, aces
            best = numerical_bound(tuple(limited), size, True)
            if best is not None and best < value:
                costs.append((jokers, aces))
    return tuple((x, a) for x, a in costs if not any(
        (u, v) != (x, a) and u <= x and v <= a for u, v in costs
    ))


def can_overtake(move, pool):
    return move.template.kind == 'cut' or not bound_beats(
        numerical_bound(pool, move.size, True), move.value, True, inclusive=True
    )


def can_overtake_reply(move, lead, pool):
    # All ordinary replies already beat the previous lead, so a move that
    # fails to improve it cannot overtake those replies.
    return move.template.kind == 'cut' or (move.value < lead.value and can_overtake(move, pool))


def minimum_ax_replies(after, lead):
    for jokers, aces in reply_costs(after.pool, lead.size, lead.value):
        pool = list(after.pool)
        pool[0] -= jokers
        pool[1] -= aces
        locked = (0,) * jokers + (1,) * aces + (None,) * (lead.size-jokers-aces)
        yield replace(after, pool=tuple(pool), opponent=after.opponent-lead.size,
                      reserve=after.reserve + locked)


class Search:
    def __init__(self, cpu, room):
        self.cpu, self.room = cpu, room
        self.held = hand_counts(cpu.hand)
        self.state = resources(cpu, room)
        self.reverse = room.reverse_order
        self.protected = (0, 1 if self.reverse else 13)
        self.moves = ()
        self.by_size = {}
        self.by_need = {}
        self.tails = {}
        self.fitting = {}
        self.rally_branches = {}
        self.generalized_cache = {}

    def prepare(self):
        if self.moves:
            return
        templates = catalog(*knowledge_key(self.cpu, self.room))
        templates += (Template('prime', 1729, (1, 7, 2, 9)),)
        moves = []
        for template in templates:
            check(self.cpu)
            moves.extend(realizations(template, self.held))
        self.moves = tuple(sorted(moves, key=lambda move: (
            -move.used_count, move.need[0] + move.need[1], move.value, move.key)))
        for move in self.moves:
            self.by_size.setdefault(move.size, []).append(move)
            self.by_need.setdefault(move.need, []).append(move)

    def protected_reply(self, moves):
        return min(
            (move for move in moves if preserves_revolution_cards(move, self.held)),
            key=lambda move: (move.value, -move.size), default=None,
        )

    def finish(self, held):
        return next((move for move in self.by_need.get(held, ())
                     if not uses_spare_natural(move, held)), None)

    def can_take_trump(self, trump, state):
        return all(self.controls(trump, after) for after in turn_states(state, trump))

    def generalized(self, held, state, depth, field=None):
        key = held, state, depth, field
        if key not in self.generalized_cache:
            self.generalized_cache[key] = self.find_generalized(held, state, depth, field)
        return self.generalized_cache[key]

    def find_generalized(self, held, state, depth, field):
        if state is None:
            return None
        sizes = [size for size in GENERALIZED_SIZES
                 if depth * size < sum(held) and (field is None or size == field[0])]
        trumps = {}
        middle = []
        for size in sizes:
            rows = self.by_size.get(size, ())
            trumps[size] = []
            for move in rows:
                check(self.cpu)
                if move.cut or subtract(held, move.need) is None:
                    continue
                if self.can_take_trump(move, state):
                    trumps[size].append(move)
                middle.append(move)
            trumps[size].sort(key=lambda move: (move.value, move.used_count, move.key))
        # The last move before the trump is the most important middle
        # move in both three- and four-step plans.
        middle.sort(key=lambda move: (move.value, move.need[0] + move.need[1], move.key))
        for move in middle:
            check(self.cpu)
            for trump in trumps[move.size]:
                check(self.cpu)
                if trump.value >= move.value:
                    break
                available = subtract(subtract(held, trump.need), move.need)
                if available is None:
                    continue
                for prefix, rest in self.generalized_prefixes(
                    available, move.size, move.value, depth - 2, field
                ):
                    finish = self.finish(rest)
                    if finish is None:
                        continue
                    steps = (*prefix, move, trump)
                    if self.generalized_sequence_valid(held, state, steps):
                        return GeneralizedPlan(steps, finish, depth)
        return None

    def generalized_prefixes(self, held, size, minimum, remaining, field):
        if not remaining:
            yield (), held
            return
        for move in self.by_size[size]:
            check(self.cpu)
            if (move.cut or move.need[0] or move.need[1] or move.value <= minimum
                    or (remaining == 1 and field is not None and move.value >= field[1])):
                continue
            rest = subtract(held, move.need)
            if rest is not None:
                for prefix, tail in self.generalized_prefixes(
                    rest, size, move.value, remaining - 1, field
                ):
                    yield (*prefix, move), tail

    def generalized_sequence_valid(self, held, state, steps):
        for move in steps[:-1]:
            if uses_spare_natural(move, held) or opponent_can_recover(self, move, state):
                return False
            held = subtract(held, move.need)
            # A pass releases the display and lets the opponent draw. Use
            # that larger resource pool when checking the eventual trump.
            state = pass_result(state, move)[-1]
        trump = steps[-1]
        return not uses_spare_natural(trump, held) and self.can_take_trump(trump, state)

    def decoy(self, moves):
        if self.state is None:
            return None
        candidates = sorted(
            (move for move in moves if not (move.cut or revolution_move(move)
                                            or move.need[0] or move.need[1])),
            key=lambda move: (-move.size, sum(move.need[r] for r in range(1, 14, 2)),
                              move.value, move.key),
        )
        for move in candidates:
            check(self.cpu)
            left = subtract(self.held, move.need)
            state = pass_result(self.state, move)[-1]
            for depth in (3, 4):
                plan = self.generalized(left, state, depth)
                if plan:
                    return replace(plan, steps=(move, *plan.steps), decoy=True)
        return None

    def controls(self, move, state):
        if move.cut:
            return True
        if revolution_move(move):
            return False
        if move.size > state.opponent:
            return True
        return not (
            special_reply(move, state.pool, self.reverse)
            or listed_counter(state.pool, move.size, move.value, self.reverse)
            or bound_beats(numerical_bound(state.pool, move.size, self.reverse),
                           move.value, self.reverse)
        )

    def control_flows(self, move, state):
        states = turn_states(state, move)
        if all(self.controls(move, after) for after in states):
            return tuple(dict.fromkeys(map(flow, states)))
        return ()

    def tail(self, held, state):
        check(self.cpu)
        key = (held, state)
        if key in self.tails:
            return self.tails[key]
        finish = self.finish(held)
        if finish:
            self.tails[key] = (finish,)
            return self.tails[key]
        if held not in self.fitting:
            rows = []
            for need, moves in self.by_need.items():
                check(self.cpu)
                left = subtract(held, need)
                if left is not None:
                    rows.extend((move, left) for move in moves)
            self.fitting[held] = sorted(rows, key=lambda row: (
                not row[0].cut, row[0].value, -row[0].used_count, row[0].key
            ))
        for move, left in self.fitting[held]:
            check(self.cpu)
            if left == ZERO or uses_spare_natural(move, held):
                continue
            branches = self.control_flows(move, state)
            if branches:
                tails = [self.tail(left, after) for after in branches]
                if tails[0] and all(tail == tails[0] for tail in tails):
                    self.tails[key] = (move, *tails[0])
                    return self.tails[key]
        self.tails[key] = None
        return None

    def common_tail(self, held, states):
        tails = [self.tail(held, state) for state in states]
        if tails and tails[0] and all(tail == tails[0] for tail in tails):
            return tails[0]
        return None

    def branches(self, lead, states, certify):
        key = lead, states, certify
        if key not in self.rally_branches:
            self.rally_branches[key] = self.evaluate_branches(lead, states, certify)
        return self.rally_branches[key]

    def evaluate_branches(self, lead, states, certify):
        flowed, responded, reply_pools = [], [], set()
        for after in (branch for state in states for branch in turn_states(state, lead)):
            flowed.append(flow(after))
            if after.opponent < lead.size:
                continue
            if special_reply(lead, after.pool, True):
                return None
            replies = tuple(minimum_ax_replies(after, lead))
            if not replies:
                continue
            # We cannot certify against an arbitrary whole-hand expression.
            if after.opponent == lead.size or (certify and self.room.rule.allow_composite
                                               and after.opponent >= lead.size + 2):
                return None
            responded.extend(replies)
            reply_pools.add(after.pool)
        if not responded:
            return None
        return tuple(dict.fromkeys(flowed)), tuple(dict.fromkeys(responded)), reply_pools

    def safe_step(self, lead, held, states, remaining, trump, *, certify=True):
        """One fixed next move must work across every possible ordinary reply."""
        if lead.cut or revolution_move(lead):
            return None
        left = subtract(held, lead.need)
        if left is None or subtract(left, trump.need) is None:
            return None
        branches = self.branches(lead, states, certify)
        if branches is None:
            return None
        flowed, reply_states, reply_pools = branches
        candidates = (trump,) if remaining == 2 else self.by_size[lead.size]
        for middle in candidates:
            check(self.cpu)
            rest = subtract(left, middle.need)
            if (rest is None or middle.is_57 or revolution_move(middle)
                    or uses_spare_natural(middle, left)):
                continue
            if remaining == 3 and (
                not trump.value < middle.value < lead.value
                or subtract(rest, trump.need) is None
            ):
                continue
            overtakes_all = all(can_overtake_reply(middle, lead, pool) for pool in reply_pools)
            if remaining == 2 and not overtakes_all:
                continue
            if not certify and any(
                opponent_can_recover(self, middle, reply)
                for reply in reply_states
            ):
                continue
            if remaining == 2 and overtakes_all and rest != ZERO:
                branches = tuple(self.control_flows(middle, state) for state in reply_states)
                if all(branches):
                    finish = self.common_tail(rest, tuple(after for branch in branches for after in branch))
                    if finish:
                        passed = self.common_tail(left, flowed)
                        if passed is None:
                            return None
                        return Rally(lead, Rally(middle, certainty=certify, pass_tail=finish),
                                     certify, passed)
            if remaining > 2 and middle.value < lead.value:
                route = self.safe_step(middle, left, reply_states, remaining-1, trump, certify=certify)
                if route:
                    passed = self.common_tail(left, flowed)
                    if passed is None:
                        return None
                    if overtakes_all:
                        return Rally(lead, route, certify, passed)
                    if all(can_overtake_reply(trump, lead, pool) for pool in reply_pools):
                        skip_left = subtract(left, trump.need)
                        branches = tuple(self.control_flows(trump, state) for state in reply_states)
                        if not uses_spare_natural(trump, left) and all(branches):
                            finish = self.common_tail(skip_left, tuple(after for branch in branches for after in branch))
                            if finish:
                                alternative = Rally(trump, certainty=certify, pass_tail=finish)
                                return Rally(lead, replace(route, alternate=alternative), certify, passed)
        return None

    def mate(self, legal_moves, *, certify=True):
        if self.state is None:
            return None
        leads_by_size = {}
        for lead in legal_moves:
            if not lead.cut and not revolution_move(lead):
                leads_by_size.setdefault(lead.size, []).append(lead)
        trumps = sorted(
            (move for size in leads_by_size for move in self.by_size[size]
             if not move.is_57 and not revolution_move(move)),
            key=lambda move: (move.value, move.need[0] + move.need[1], move.used_count, move.key),
        )
        for depth in (2, 3):
            for trump in trumps:
                check(self.cpu)
                if (depth - 1) * trump.size + trump.used_count >= sum(self.held):
                    continue
                # In a two-step rally the first reply can only have access
                # to more ranks than this initial public pool. Three-step
                # rallies may deplete A/X, so they need the branch proof.
                if depth == 2 and not can_overtake(trump, self.state.pool):
                    continue
                available = subtract(self.held, trump.need)
                for lead in leads_by_size[trump.size]:
                    check(self.cpu)
                    if (lead.value <= trump.value or subtract(available, lead.need) is None
                            or uses_spare_natural(lead, self.held)):
                        continue
                    route = self.safe_step(lead, self.held, (self.state,), depth, trump, certify=certify)
                    if route:
                        return route
        return None


def play(cpu, move, reason, certain=False):
    candidate = bind_move(move, cpu.hand)
    action = diamond.candidate_to_action(candidate)
    diamond.clear_gold_active_plan(cpu)
    cpu.diamond_last_route_kind = 'master-epsilon-' + reason
    cpu.diamond_last_certainty = 'certain' if certain else 'unclassified'
    cpu.diamond_last_return_probability = 0.0 if certain else None
    diamond.diamond_remember_cards(cpu, diamond.candidate_consumed_cards(candidate))
    return diamond.platinum_commit_play(cpu, action)


def recovery_gain(state, hand_size, protected, already_drawn=False):
    """Compare the recoverable prefix with the A/X (or K/X) already held."""
    if hand_size <= 0 or not state.queue:
        return 0
    reach = hand_size + (0 if already_drawn else 2)
    prefix = state.queue[:reach]
    known = sum(card in protected for card in prefix)
    unknown = sum(card is None for card in prefix)
    unseen = sum(state.pool[r] for r in protected) - sum(
        card in protected for card in state.queue
    )
    return known + min(unknown, max(0, unseen))


def draw_or_all_out(cpu, room, reason):
    if not room.has_drawn and room.deck:
        cpu.diamond_last_route_kind = 'master-epsilon-' + reason + '-draw'
        return diamond.CpuAction('draw')
    if room.field:
        if not room.deck:
            return diamond.CpuAction('pass')
        payload = diamond.build_composite_practice_all_out_payload(
            cpu, room, rng=cpu.rng, require_invalid=True)
    else:
        payload = diamond.build_gold_all_out_payload(cpu.hand, force_random=True, rng=cpu.rng)
    if payload is None:
        return diamond.CpuAction('pass')
    diamond.clear_gold_active_plan(cpu)
    cpu.master_epsilon_rally = None
    cpu.platinum_all_out_attempts += 1
    cpu.diamond_last_route_kind = 'master-epsilon-' + reason + '-all-out'
    return diamond.platinum_commit_play(cpu, diamond.CpuAction(
        'play_composite' if room.field else 'play_prime', payload))


def reserve_recovery(search, move):
    cpu = search.cpu
    candidate = bind_move(move, cpu.hand)
    used = diamond.candidate_consumed_cards(candidate)
    ids = {str(c['card_id']) for c in used}
    cpu.master_epsilon_recovery = (
        search.reverse, knowledge_key(cpu, search.room),
        frozenset(str(c['card_id']) for c in cpu.hand if str(c['card_id']) not in ids),
        frozenset(str(c['card_id']) for c in used if rank(c) in search.protected),
    )
    return play(cpu, move, 'recovery-control')


def can_take_recovery(search, move, branches):
    paid = sum(move.need[r] for r in search.protected)
    remaining = len(search.cpu.hand) - move.used_count
    for branch in branches:
        first_visible = len(branch.queue) - len(search.state.reserve) - move.size
        first_material = first_visible - len(move.template.material)
        if first_material < 0:
            return False
        visible_start = len(branch.queue) - move.size
        own = (*enumerate(move.physical[:move.size], visible_start),
               *enumerate(move.physical[move.size:], first_material))
        if any(rank in search.protected and index >= remaining + 2 for index, rank in own):
            return False
        if recovery_gain(branch, remaining, search.protected) <= paid:
            return False
    return True


def continue_recovery(cpu, room):
    pending = getattr(cpu, 'master_epsilon_recovery', None)
    cpu.master_epsilon_recovery = None
    if pending is None or room.field:
        return None
    reverse, knowledge, expected, targets = pending
    held = {str(c['card_id']) for c in cpu.hand}
    bottom = getattr(room, 'public_known_deck_bottom', ())
    known = {str(c['card_id']) for c in bottom}
    if (reverse != room.reverse_order or knowledge != knowledge_key(cpu, room)
            or not targets <= held | known or not expected <= held
            or len(held - expected) != int(room.has_drawn)):
        return None
    capacity = len(cpu.hand) + (0 if room.has_drawn else 2)
    first = len(room.deck) - len(bottom)
    if any(first + i >= capacity for i, card in enumerate(bottom)
           if str(card['card_id']) in targets):
        return None
    if not room.deck:
        return None
    state = resources(cpu, room)
    protected = (0, 1 if reverse else 13)
    drawn_gain = sum(str(card['card_id']) not in expected and rank(card) in protected
                     for card in cpu.hand)
    if state is None or drawn_gain + recovery_gain(
        state, len(cpu.hand), protected, room.has_drawn
    ) <= len(targets):
        return None
    if not room.has_drawn and room.deck:
        cpu.master_epsilon_recovery = pending
    return draw_or_all_out(cpu, room, 'recovery')


def opponent_can_recover(search, move, state=None):
    state = search.state if state is None else state
    if state is None or move.cut or not any(move.need[r] for r in search.protected):
        return False
    reversed_order = not search.reverse if revolution_move(move) else search.reverse
    for after in turn_states(state, move):
        if after.opponent <= move.size or not (
            listed_counter(after.pool, move.size, move.value, reversed_order)
            or bound_beats(numerical_bound(after.pool, move.size, reversed_order),
                           move.value, reversed_order)
        ):
            continue
        # Use the maximum remaining hand after a legal response. A composite
        # reply would spend additional cards on materials.
        remaining = after.opponent - move.size
        recovery_queue = after.queue + after.reserve + (None,) * move.size
        # Cover the opponent's own trump as well as all of ours. The visible
        # cards of both plays are returned before the opponent's next draw.
        if (len(recovery_queue) <= remaining + 2
                and sum(card in search.protected for card in recovery_queue) > 0):
            return True
    return False


def preserves_revolution_cards(move, held):
    return not (move.need[0] or move.need[1]) and all(
        held[r] - move.need[r] >= min(held[r], 1) for r in (7, 2, 9)
    )


def route_action(search, route):
    cpu, room = search.cpu, search.room
    move = route.move
    cpu.master_epsilon_rally = (
        search.reverse, knowledge_key(cpu, room), subtract(search.held, move.need), route
    ) if route.reply or route.pass_tail else None
    return play(cpu, move, 'mate' if route.certainty else 'rally', certain=route.certainty)


def play_generalized(search, plan):
    move, *rest = plan.steps
    cpu = search.cpu
    cpu.master_epsilon_generalized = (
        knowledge_key(cpu, search.room), subtract(search.held, move.need),
        replace(plan, steps=tuple(rest)),
    )
    reason = 'decoy' if plan.decoy else f'generalized-{plan.depth}'
    return play(cpu, move, reason, certain=not rest)


def continue_generalized(search, pending):
    knowledge, expected, plan = pending
    room = search.room
    if (expected != search.held or knowledge != knowledge_key(search.cpu, room)
            or (plan.decoy and room.field) or search.state is None):
        return None
    if not plan.steps:
        if legal(plan.finish, room):
            return play(search.cpu, plan.finish, 'finish', certain=True)
        return None
    if legal(plan.steps[0], room) and search.generalized_sequence_valid(
        search.held, search.state, plan.steps
    ):
        return play_generalized(search, replace(plan, decoy=False))
    trump = plan.steps[-1]
    if (not legal(trump, room) or uses_spare_natural(trump, search.held)
            or not search.can_take_trump(trump, search.state)):
        return None
    search.prepare()
    finish = search.finish(subtract(search.held, trump.need))
    if finish is not None:
        return play_generalized(search, replace(plan, steps=(trump,), finish=finish, decoy=False))
    return None


def safe_exit(room):
    if not room.has_drawn and room.deck:
        return diamond.CpuAction('draw')
    return diamond.CpuAction('pass')


def select(cpu, room, _fallback):
    generalized = getattr(cpu, 'master_epsilon_generalized', None)
    cpu.master_epsilon_generalized = None
    pending = getattr(cpu, 'master_epsilon_rally', None)
    cpu.master_epsilon_rally = None
    if pending:
        reverse, knowledge, expected, previous = pending
        route = previous.reply if room.field else None
        if route and not legal(route.move, room):
            route = route.alternate
        if not room.field and previous.pass_tail:
            move, *tail = previous.pass_tail
            route = Rally(move, certainty=previous.certainty, pass_tail=tuple(tail))
        if (reverse == room.reverse_order and knowledge == knowledge_key(cpu, room)
                and expected == hand_counts(cpu.hand) and route and legal(route.move, room)):
            search = Search(cpu, room)
            if route.certainty or not opponent_can_recover(search, route.move):
                return route_action(search, route)

    finisher = ControlSearch(cpu, room)
    finish = finisher.finish()
    if finish:
        return play(cpu, finish, 'finish', certain=True)
    action = known_draw_finish(cpu, room)
    if action is not None:
        return action

    search = Search(cpu, room)
    if generalized:
        action = continue_generalized(search, generalized)
        if action is not None:
            return action
    search.prepare()
    moves = [move for move in search.moves if legal(move, room)
             and not uses_spare_natural(move, search.held)]
    cpu.master_epsilon_deadline_move = search.protected_reply(moves)

    if search.reverse:
        route = bounded_search(1200, search.mate, moves)
        if route:
            return route_action(search, route)
    action = bounded_search(1400, choose_public_control, cpu, room)
    if action is not None:
        return action

    recovered = continue_recovery(cpu, room)
    if recovered is not None:
        return recovered
    if search.state is None:
        return safe_exit(room)

    controls = []
    for move in moves:
        check(cpu)
        if not room.field and not search.state.reserve:
            break
        branches = search.control_flows(move, search.state)
        if branches:
            controls.append(move)
            # If the opponent can draw a released material card before
            # passing, that card cannot be promised back to us.
            if (not search.state.queue and len(branches) > 1 and move.template.material
                    and move.physical[move.size] in search.protected):
                continue
            if can_take_recovery(search, move, branches):
                return reserve_recovery(search, move)

    if search.reverse:
        state = search.state
        field = (len(room.field), int(room.last_number)) if room.field else None
        for depth in (3, 4):
            plan = bounded_search(900, search.generalized, search.held, state, depth, field)
            if plan:
                return play_generalized(search, plan)
        plan = bounded_search(850, search.decoy, moves)
        if plan:
            return play_generalized(search, plan)
        unknown = any(card is None for card in state.queue)
        known_recovery = recovery_gain(state, len(cpu.hand), search.protected, room.has_drawn)
        if room.field:
            known_recovery = max(known_recovery, recovery_gain(
                flow(state), len(cpu.hand), search.protected, room.has_drawn
            ))
        if unknown or known_recovery:
            action = draw_or_all_out(cpu, room, 'reshape')
            if action.kind != 'pass':
                return action
        certain = controls or any(
            not revolution_move(move) and search.control_flows(move, state)
            for move in search.moves if not uses_spare_natural(move, search.held)
        )
        if certain:
            candidates = sorted(
                (move for move in moves if not (move.need[0] or move.need[1] or revolution_move(move))),
                key=lambda move: (-move.size, sum(move.need[r] for r in range(1, 14, 2)), move.value),
            )
            for move in candidates:
                check(cpu)
                if all(search.tail(subtract(search.held, move.need), after)
                       for after in pass_result(state, move)):
                    return play(cpu, move, 'setup')
        else:
            if cpu.master_epsilon_deadline_move:
                return play(cpu, cpu.master_epsilon_deadline_move, 'preserve-ax')
        return draw_or_all_out(cpu, room, 'reshape')


def choose_epsilon(cpu, room, diamond_choice):
    if not room.reverse_order:
        cpu.master_epsilon_rally = None
        cpu.master_epsilon_recovery = None
        cpu.master_epsilon_generalized = None
        return master_combined.choose_combined(cpu, room, diamond_choice)

    cpu.master_epsilon_deadline_move = None

    def on_deadline(player, state):
        player.last_decision_timed_out = True
        player.master_epsilon_rally = None
        player.master_epsilon_recovery = None
        player.master_epsilon_generalized = None
        if state.reverse_order and player.master_epsilon_deadline_move:
            return play(player, player.master_epsilon_deadline_move, 'deadline-preserve-ax')
        return safe_exit(state)

    try:
        return master_combined.choose_combined(cpu, room, on_deadline, selector=select)
    finally:
        cpu.master_epsilon_deadline_move = None

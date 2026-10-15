"""Finite, public-information-only search for unanswerable plays then a finish.

Opponent bounds ignore primality. Every ordinary control permits a draw, and
released cards re-enter the public queue at their actual return time. A larger
possible opponent hand dominates a skipped draw; the rank pool keeps all cards
that could have been drawn earlier. No weights or fitted parameters are used.
"""
from functools import cmp_to_key, lru_cache
from copy import copy
from itertools import combinations

import cpu_player as diamond
from .master_budget import check
from .opening_master import CAPACITY, ZERO, Move, Template, _realizations, bind_move, counts, hand_counts, subtract
from .post_all_out_master import knowledge_key
from .revolution_master import catalog
from .after_all_out_planner import public_room, next_known
from .move_safety import special_exposed, uses_spare_natural


@lru_cache(maxsize=8)
def templates_by_size(key):
    """Immutable knowledge index shared by the current and known-draw hand."""
    rows = {}
    templates = catalog(*key) + (Template('prime', 1729, (1, 7, 2, 9)),)
    for template in templates:
        rows.setdefault(len(template.ranks), []).append(template)
    return {size: tuple(group) for size, group in rows.items()}


def realizations(template, hand):
    """Same order and deduplication as _realizations, counting before building.

    Most X-position combinations fail physical capacity. Compute their rank
    counts first; allocate physical tuples only for distinct feasible moves.
    """
    ranks = template.ranks
    if len(ranks) > sum(hand):
        return
    if template.kind == 'cut':
        yield from _realizations(template, hand)
        return
    requirement = counts(ranks)
    missing = sum(max(0, requirement[r]-hand[r]) for r in range(1, 14))
    if missing > hand[0]:
        return
    visible = counts(template.visible)
    size = len(template.visible)
    seen = set()
    for joker_count in range(missing, min(hand[0], len(ranks))+1):
        for omitted in combinations(range(len(ranks)), joker_count):
            need = list(requirement)
            visible_need = list(visible)
            for i in omitted:
                need[ranks[i]] -= 1
                need[0] += 1
                if i < size:
                    visible_need[ranks[i]] -= 1
                    visible_need[0] += 1
            if any(n > h for n, h in zip(need, hand)):
                continue
            need = tuple(need)
            key = (need, tuple(visible_need))
            if key in seen:
                continue
            seen.add(key)
            selected = set(omitted)
            physical = tuple(0 if i in selected else r for i, r in enumerate(ranks))
            yield Move(template, physical, need)


@lru_cache(maxsize=32768)
def numerical_bound(pool, size, reverse=False):
    """Exact extremum of unrestricted concatenations of size physical cards."""
    if size < 1 or sum(pool) < size:
        return None
    if not reverse:
        tokens = [13] * pool[0] + [r for r in range(13, 9, -1) for _ in range(pool[r])]
        tokens += [r for r in range(9, 0, -1) for _ in range(pool[r])]
        parts = sorted(map(str, tokens[:size]), key=cmp_to_key(
            lambda a, b: (b+a > a+b) - (b+a < a+b)))
        return int(''.join(parts))
    best = None
    # A leading X is 1; elsewhere it is 0. Enumerating the first physical
    # rank also covers choosing a natural lead and retaining X for zero.
    for first in range(14):
        if not pool[first]:
            continue
        left = list(pool)
        left[first] -= 1
        tail = [0] * left[0] + [r for r in range(1, 14) for _ in range(left[r])]
        tail = tail[:size-1]  # digit length first, then value
        parts = sorted(map(str, tail), key=cmp_to_key(
            lambda a, b: (a+b > b+a) - (a+b < b+a)))
        value = int(str(first or 1) + ''.join(parts))
        best = value if best is None else min(best, value)
    return best


def legal(move, room):
    if not room.field:
        return True
    if move.size != len(room.field):
        return False
    if move.template.kind == 'cut':
        return True
    return move.value < int(room.last_number) if room.reverse_order else move.value > int(room.last_number)


def unique_cards(cards):
    return tuple({str(c['card_id']): c for c in cards}.values())


def initial_resources(cpu, room):
    state = public_room(room)
    reserve = unique_cards(list(getattr(room, 'reserve', ()) or ()) + list(room.field))
    known = tuple(c for c in state.draw_queue if c is not None)
    excluded = unique_cards(list(cpu.hand) + list(reserve) + list(known))
    pool = subtract(CAPACITY, hand_counts(excluded))
    opponent = diamond.platinum_opponent_hand_count(cpu, room)
    if pool is None or opponent is None or opponent < 1 or sum(pool) < opponent:
        return None
    queue = tuple(None if c is None else (0 if diamond.is_joker(c) else int(c['rank']))
                  for c in state.draw_queue)
    return pool, opponent, queue, tuple(0 if diamond.is_joker(c) else int(c['rank']) for c in reserve)


class ControlSearch:
    def __init__(self, cpu, room):
        self.cpu, self.room = cpu, room
        self.hand = hand_counts(cpu.hand)
        self.reverse = bool(room.reverse_order)
        self.nodes = 0
        self.failed = set()
        self.fit = {}
        self.moves = None
        self.by_need = {}
        self.terminal_realizations = {}
        self.resources = initial_resources(cpu, room)

    def templates(self):
        return catalog(*knowledge_key(self.cpu, self.room)) + (Template('prime', 1729, (1, 7, 2, 9)),)

    def tick(self):
        check(self.cpu)
        self.nodes += 1

    def prepare(self):
        if self.moves is not None:
            return
        moves = []
        # 1729 is a legal terminal special play; the structural guard excludes
        # it from nonterminal chains because it changes the comparison order.
        templates = self.templates()
        hand_size = sum(self.hand)
        for template in templates:
            self.tick()
            if len(template.ranks) <= hand_size:
                if template in self.terminal_realizations:
                    moves.extend(self.terminal_realizations[template])
                else:
                    moves.extend(realizations(template, self.hand))
        # A stable traversal order, not a learned strength score. Stop at the
        # first fully proved path; all branches remain eligible until deadline.
        self.moves = tuple(sorted(moves, key=lambda m: (-m.used_count, m.need[0], m.key)))
        exact = [[0] * (capacity+1) for capacity in CAPACITY]
        self.control_mask = 0
        controls = []
        prime_controls = {}
        for index, move in enumerate(self.moves):
            self.tick()
            self.by_need.setdefault(move.need, []).append(move)
            control = False
            if move.cut or self.resources is None:
                control = True
            elif not self.unanswerable(move, self.resources[0], self.resources[1]):
                pass  # Even before any draw, the opponent can already exceed it.
            elif move.template.kind != 'prime':
                control = True
            else:
                key = (move.size, move.value)
                if key not in prime_controls:
                    prime_controls[key] = self.advance(move, *self.resources) is not None
                # Opponent capacity/pool can only grow along a forced-pass
                # path. A prime already answerable now cannot become a control.
                # Do not apply this pruning to composites: material draw timing
                # depends on the queue at that particular step.
                if prime_controls[key]:
                    control = True
            if not control:
                continue
            bit = 1 << len(controls)
            controls.append(move)
            for rank, need in enumerate(move.need):
                exact[rank][need] |= bit
        for row in exact:
            for n in range(1, len(row)):
                row[n] |= row[n-1]
        self.fit_index = exact
        self.controls = tuple(controls)
        self.control_mask = (1 << len(controls)) - 1

    def fitting(self, held):
        self.prepare()
        if held not in self.fit:
            rows = []
            mask = self.control_mask
            for rank, n in enumerate(held):
                mask &= self.fit_index[rank][n]
            while mask:
                self.tick()
                bit = mask & -mask
                move = self.controls[bit.bit_length()-1]
                if not uses_spare_natural(move, held):
                    rows.append(move)
                mask ^= bit
            self.fit[held] = tuple(rows)
        return self.fit[held]

    def finish(self):
        if self.moves is not None:
            return next((m for m in self.by_need.get(self.hand, ()) if legal(m, self.room)), None)
        # A finish consumes exactly the whole hand. Other template sizes and
        # the nonterminal control index cannot affect this answer.
        finishes = []
        size = sum(self.hand)
        for template in templates_by_size(knowledge_key(self.cpu, self.room)).get(size, ()):
            self.tick()
            if len(template.ranks) == size:
                realized = tuple(realizations(template, self.hand))
                self.terminal_realizations[template] = realized
                for move in realized:
                    self.tick()
                    if move.need == self.hand and legal(move, self.room):
                        finishes.append(move)
        return min(finishes, key=lambda m: (-m.used_count, m.need[0], m.key), default=None)

    def advance(self, move, pool, opponent, queue, reserve):
        """Conservative state after the opponent draws/passes and the field flows."""
        if special_exposed(move, self.reverse):
            return None
        # Materials return before the opponent's draw. Displayed cards remain
        # locked until the field flows. Cuts give no opponent turn or draw.
        queue = queue + tuple(move.physical[move.size:])
        available = list(pool)
        if not move.cut:
            if queue:
                drawn, *rest = queue
                queue = tuple(rest)
                opponent += 1
                if drawn is not None:
                    available[drawn] += 1
            available = tuple(available)
            if not self.unanswerable(move, available, opponent):
                return None
        queue += reserve + tuple(move.physical[:move.size])
        return tuple(available), opponent, queue, ()

    def unanswerable(self, move, pool, opponent):
        if special_exposed(move, self.reverse) or (move.size == 1 and pool[0]):
            return False
        if move.size > opponent:
            return True
        bound = numerical_bound(pool, move.size, self.reverse)
        return bound is not None and (bound >= move.value if self.reverse else bound <= move.value)

    def visit(self, held, resources, first=False):
        self.tick()
        key = (held, resources, first)
        if key in self.failed:
            return None
        self.prepare()
        for move in self.by_need.get(held, ()):
            self.tick()
            if not first or legal(move, self.room):
                return (move,)
        rows = self.fitting(held)
        for move in rows:
            self.tick()
            if first and not legal(move, self.room):
                continue
            after = self.advance(move, *resources)
            if after is None:
                continue
            rest = self.visit(subtract(held, move.need), after)
            if rest is not None:
                return (move,) + rest
        self.failed.add(key)
        return None

    def search(self):
        resources = self.resources
        return None if resources is None else self.visit(self.hand, resources, first=True)


def play(cpu, room, sequence, reason):
    move, *rest = sequence
    candidate = bind_move(move, cpu.hand)
    diamond.clear_gold_active_plan(cpu)
    cpu.post_all_out_master_active = None
    cpu.revolution_master_active = None
    cpu.master_public_control_active = (dict(expected=subtract(hand_counts(cpu.hand), move.need),
        reverse=bool(room.reverse_order), knowledge=knowledge_key(cpu, room), sequence=tuple(rest)) if rest else None)
    cpu.diamond_last_route_kind = 'master-public-control-' + reason
    cpu.diamond_last_certainty = 'certain'
    cpu.diamond_last_return_probability = 0.0
    diamond.diamond_update_opening_position_history(cpu, room)
    diamond.diamond_remember_cards(cpu, diamond.candidate_consumed_cards(candidate))
    return diamond.platinum_commit_play(cpu, diamond.candidate_to_action(candidate))


def known_draw_finish(cpu, room):
    if room.has_drawn or not room.deck:
        return None
    card = next_known(public_room(room))
    if card is None:
        return None
    child = copy(cpu)
    child.hand = list(cpu.hand) + [dict(card)]
    state = copy(room)
    state.has_drawn = True
    search = ControlSearch(child, state)
    move = search.finish()
    if move is None:
        return None
    # Retain the exact known finish so the subsequent decision cannot spend
    # its deadline rediscovering the already proved immediate win.
    cpu.master_public_control_draw_finish = dict(expected=hand_counts(child.hand),
        reverse=bool(room.reverse_order), knowledge=knowledge_key(cpu, room), move=move)
    cpu.master_public_control_trace = dict(reason='known_draw_finish', finish=move.summary())
    cpu.diamond_last_route_kind = 'master-known-draw-finish'
    return diamond.CpuAction('draw')


def choose_public_control(cpu, room):
    if room.rule.key != 'std-11-n-c':
        return None
    pending = getattr(cpu, 'master_public_control_draw_finish', None)
    cpu.master_public_control_draw_finish = None
    if (pending and room.has_drawn and pending['expected'] == hand_counts(cpu.hand)
            and pending['reverse'] == bool(room.reverse_order)
            and pending['knowledge'] == knowledge_key(cpu, room) and legal(pending['move'], room)):
        cpu.master_public_control_trace = dict(reason='known_draw_finish_complete')
        return play(cpu, room, (pending['move'],), 'known-draw-finish')
    active = getattr(cpu, 'master_public_control_active', None)
    cpu.master_public_control_active = None
    if (active and active['expected'] == hand_counts(cpu.hand)
            and active['reverse'] == bool(room.reverse_order)
            and active['knowledge'] == knowledge_key(cpu, room)):
        # Revalidate the saved certificate against the actual public state,
        # without re-enumerating alternatives or disturbing its exact suffix.
        verifier = ControlSearch(cpu, room)
        resources = initial_resources(cpu, room)
        held = hand_counts(cpu.hand)
        valid = resources is not None
        for index, move in enumerate(active['sequence']):
            verifier.tick()
            if not valid or (index == 0 and not legal(move, room)):
                valid = False
                break
            remaining = subtract(held, move.need)
            if remaining is None or uses_spare_natural(move, held):
                valid = False
                break
            if remaining != ZERO:
                resources = verifier.advance(move, *resources)
                valid = resources is not None
            held = remaining
        if valid and held == ZERO:
            cpu.master_public_control_trace = dict(reason='certified_continuation')
            return play(cpu, room, active['sequence'], 'continue')
    search = ControlSearch(cpu, room)
    finish = search.finish()
    if finish:
        cpu.master_public_control_trace = dict(reason='immediate_finish')
        return play(cpu, room, (finish,), 'finish')
    draw = known_draw_finish(cpu, room)
    if draw is not None:
        return draw
    sequence = search.search()
    cpu.master_public_control_trace = dict(reason='proved' if sequence else 'no_plan', nodes=search.nodes,
        opponent_model='unrestricted_numeric_bound', reverse=bool(room.reverse_order),
        steps=[m.summary() for m in sequence] if sequence else [])
    return play(cpu, room, sequence, 'chain') if sequence else None

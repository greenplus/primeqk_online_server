"""Last-resort normal-order plans for Master kappa."""

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from math import log10

import cpu_player as diamond
from hnp_challenge import build_hnp_tokens, choose_hnp_permutation
from .after_all_out_planner import hnp_assignments, public_room
from .master_budget import MasterDeadline, check
from .move_safety import special_exposed, uses_spare_natural
from .opening_master import CAPACITY, bind_move, counts, hand_counts, subtract
from .post_all_out_master import Config, Solver, knowledge_key
from .public_control import legal, realizations, templates_by_size


TIERS = {"A": 1, "C": 2, "E": 2, "I": 2, "F1": 2, "D": 3, "F2": 3,
         "G": 3, "H": 4, "B": 4}
DECOY_STEP = {"C": 0, "D": 0, "E": 1, "G": 0}
STRENGTH_STEP = {**DECOY_STEP, "I": 1}
DECOY_BANDS = {4: (8121011, 4), 5: (910121111, 3), 3: (111211, 2)}


def add(first, second):
    return tuple(a + b for a, b in zip(first, second))


def decoy_strength(move):
    value = move.value
    threshold, band = DECOY_BANDS.get(move.size, (0, 0))
    if band and value >= threshold:
        return band, 0
    return 0, -0.2 * move.size + 0.6 * log10(value)


@dataclass(frozen=True)
class Plan:
    name: str
    steps: tuple
    retained: tuple = ()
    hnp_size: int = 0
    setup: object = None

    def priority(self):
        tier = TIERS[self.name]
        if tier == 4:
            return (-tier, -self.hnp_size)
        strength = decoy_strength(self.steps[STRENGTH_STEP[self.name]]) if self.name in STRENGTH_STEP else (
            (1, 0) if self.name in {"F1", "F2"} else (0, 0)
        )
        held = self.retained
        return (-tier, held[0] + held[13], strength, held[0] + sum(held[10:]),
                sum(held), self.setup is None)


@lru_cache(maxsize=4096)
def safe_hnp_assignments(held):
    if sum(held) < 2:
        return ()
    natural_sum = sum(rank * held[rank] for rank in range(1, 14))
    ends = (1, 3, 7, 9, 11, 13)
    natural_ending = any(held[rank] for rank in ends)
    return tuple(assigned for assigned in hnp_assignments(held)
                 if (natural_sum + sum(assigned)) % 3
                 and (natural_ending or any(rank in ends for rank in assigned)))


def is_kk(move):
    t = move.template
    return (t.kind == "composite" and t.visible == (13, 13)
            and t.expression in ((("cards", (13,)), ("op", "*"), ("cards", (10, 1))),
                                 (("cards", (10, 1)), ("op", "*"), ("cards", (13,)))))


def is_obake(move):
    return move.size == 4 and move.template.kind == "composite" and move.value > 13111211


class Search:
    def __init__(self, cpu, room, held=None, moves=None, *, four_only=False):
        self.cpu, self.room = cpu, room
        self.held = hand_counts(cpu.hand) if held is None else held
        self.moves = () if moves is None else moves
        self.by_need = defaultdict(list)
        self.by_size = defaultdict(list)
        if moves is None:
            key = knowledge_key(cpu, room)
            generated = []
            for templates in templates_by_size(key).values():
                for template in templates:
                    check(cpu)
                    if template.value != 1729 and (not four_only or len(template.visible) == 4):
                        generated.extend(realizations(template, self.held))
            self.moves = tuple(generated)
        for move in self.moves:
            if move.need == self.held or not uses_spare_natural(move, self.held):
                self.by_need[move.need].append(move)
                self.by_size[move.size].append(move)
        self.kk = tuple(m for m in self.by_size[2] if is_kk(m))
        self.best = None

    def finish(self, held):
        return next(iter(self.by_need.get(held, ())), None)

    def consider(self, plan):
        if self.best is None or plan.priority() > self.best.priority():
            lead = plan.steps[0]
            if (plan.setup is not None or plan.name not in DECOY_STEP or
                    diamond.diamond_lead_candidate_avoids_opponent_finish(
                        bind_move(lead, self.cpu.hand), self.cpu, self.room,
                    )):
                self.best = plan

    def long(self, known):
        opponent = diamond.platinum_opponent_hand_count(self.cpu, self.room)
        limit = max(12, opponent - 6) if known else 12
        for size in range(limit, sum(self.held)):
            for lead in self.by_size[size]:
                check(self.cpu)
                left = subtract(self.held, lead.need)
                finish = self.finish(left)
                if known and finish:
                    self.consider(Plan("A", (lead, finish), left))
                elif not known and safe_hnp_assignments(left):
                    self.consider(Plan("B", (lead,), hnp_size=sum(left)))

    def kk_plans(self, names):
        for kk in self.kk:
            rest = subtract(self.held, kk.need)
            for lead in self.moves:
                check(self.cpu)
                if lead.cut or special_exposed(lead):
                    continue
                left = subtract(rest, lead.need)
                if left is None:
                    continue
                finish = self.finish(left)
                if "C" in names and finish:
                    self.consider(Plan("C", (lead, kk, finish), add(kk.need, left)))
                if "D" in names:
                    for two in self.by_size[2]:
                        check(self.cpu)
                        if two.cut or two.value >= kk.value:
                            continue
                        tail = subtract(left, two.need)
                        finish = self.finish(tail)
                        if finish:
                            self.consider(Plan("D", (lead, two, kk, finish), add(kk.need, tail)))
                if "E" in names:
                    for middle in self.by_size[lead.size]:
                        check(self.cpu)
                        if middle.cut or middle.value <= lead.value or special_exposed(middle):
                            continue
                        tail = subtract(left, middle.need)
                        finish = self.finish(tail)
                        if finish:
                            self.consider(Plan("E", (lead, middle, kk, finish), add(kk.need, tail)))

    def quasi(self):
        for size, moves in self.by_size.items():
            for middle in moves:
                check(self.cpu)
                if (middle.cut or (size == 2 and middle.value == 1313)
                        or sum(r == 0 or r >= 10 for r in middle.physical[:size]) < 2):
                    continue
                remaining = subtract(self.held, middle.need)
                for lead in moves:
                    check(self.cpu)
                    if lead.cut or lead.value >= middle.value:
                        continue
                    left = subtract(remaining, lead.need)
                    finish = self.finish(left)
                    if finish:
                        self.consider(Plan("I", (lead, middle, finish), left))

    def obake_plans(self, names, setup=None):
        for obake in self.by_size[4]:
            check(self.cpu)
            if not is_obake(obake):
                continue
            remaining = subtract(self.held, obake.need)
            if "F1" in names:
                for lead in self.by_need.get(remaining, ()):
                    if lead.size == 4 and lead.value < obake.value:
                        self.consider(Plan("F1", (lead, obake), obake.need, setup=setup))
            if not names.intersection({"F2", "G"}):
                continue
            for lead in self.by_size[4]:
                check(self.cpu)
                if lead.value >= obake.value:
                    continue
                left = subtract(remaining, lead.need)
                for prior in self.by_need.get(left, ()):
                    if prior.cut:
                        continue
                    if "F2" in names and prior.size == 4 and prior.value < lead.value:
                        self.consider(Plan("F2", (prior, lead, obake), obake.need, setup=setup))
                    if "G" in names:
                        self.consider(Plan("G", (prior, lead, obake), obake.need, setup=setup))

    def adjust_obake(self, names):
        # Safe, publicly determined setup cards. After a real setup Master
        # re-evaluates the new hand before the comeback search runs again.
        state = public_room(self.room)
        prefix = 4 if names == {"F1"} else 8
        sizes = {sum(self.held) - 2, sum(self.held) + 1}
        eligible_sizes = {
            len(template.ranks) + prefix
            for group in templates_by_size(knowledge_key(self.cpu, self.room)).values()
            for template in group
            if len(template.visible) == 4 and template.kind == "composite"
            and template.value > 13111211 and len(template.ranks) + prefix in sizes
        }
        for cut in self.by_size[2]:
            check(self.cpu)
            if cut.is_57 and sum(self.held) - 2 in eligible_sizes:
                held = subtract(self.held, cut.need)
                study = Search(self.cpu, self.room, held,
                               (m for m in self.by_size[4] if subtract(held, m.need) is not None))
                study.obake_plans(names, setup=cut)
                if study.best:
                    self.consider(study.best)
        if not self.room.has_drawn and state.draw_queue and sum(self.held) + 1 in eligible_sizes:
            next_card = state.draw_queue[0]
            if next_card is not None:
                ranks = [next_card["rank"]]
            else:
                unavailable = add(self.held, hand_counts(
                    card for card in (*state.reserve, *state.draw_queue) if card is not None
                ))
                ranks = [r for r in range(14) if unavailable[r] < CAPACITY[r]]
            for rank in ranks:
                check(self.cpu)
                study = Search(self.cpu, self.room, add(self.held, counts((rank,))), four_only=True)
                study.obake_plans(names, setup="draw")
                if study.best:
                    self.consider(study.best)

    def hnp_back(self):
        state = public_room(self.room)
        reserve = tuple(int(c["rank"]) for c in state.reserve)
        queue = tuple(None if c is None else (0 if diamond.is_joker(c) else int(c["rank"]))
                      for c in state.draw_queue)
        direct = []
        for kk in self.kk:
            check(self.cpu)
            if kk.physical[:2] != (13, 13):
                continue
            left = subtract(self.held, kk.need)
            result = recovery_draw(left, queue + kk.physical[2:] + reserve)
            if result is not None and legal(kk, self.room):
                direct.append(Plan("H", (kk,), hnp_size=sum(left) + result))
            for two in self.by_size[2]:
                check(self.cpu)
                if two.cut or two.value >= kk.value or not legal(two, self.room):
                    continue
                after = subtract(left, two.need)
                if after is None:
                    continue
                # If the opponent passes, the current reserve and our two
                # display cards circulate before KK can be played.
                before_kk = queue + two.physical[2:] + reserve + two.physical[:2]
                result = recovery_draw(after, before_kk + kk.physical[2:])
                if result is not None:
                    self.consider(Plan("H", (two, kk), hnp_size=sum(after) + result))
        if self.best is None:
            for plan in direct:
                self.consider(plan)


def recovery_draw(held, before_kk):
    """The displayed KK is the final two entries until the next own draw."""
    if sum(held) >= len(before_kk) + 2 and safe_hnp_assignments(held):
        return False
    if sum(held) < len(before_kk):
        return None
    rank = before_kk[0] if before_kk else 13
    ranks = range(14) if rank is None else (rank,)
    if any(held[r] < CAPACITY[r] and safe_hnp_assignments(add(held, counts((r,))))
           for r in ranks):
        return True
    return None


def play(cpu, room, move, name):
    candidate = bind_move(move, cpu.hand)
    diamond.clear_gold_active_plan(cpu)
    cpu.diamond_last_route_kind = "master-kappa-comeback-" + name
    cpu.diamond_last_certainty = "conditional"
    cpu.diamond_last_return_probability = None
    diamond.diamond_update_opening_position_history(cpu, room)
    diamond.diamond_remember_cards(cpu, diamond.candidate_consumed_cards(candidate))
    return diamond.platinum_commit_play(cpu, diamond.candidate_to_action(candidate))


def hnp(cpu, room, name):
    if room.field:
        return None
    options = safe_hnp_assignments(hand_counts(cpu.hand))
    if not options:
        return None
    permutation = choose_hnp_permutation(
        build_hnp_tokens(cpu.hand, tuple(map(str, cpu.rng.choice(options)))),
        randbelow=cpu.rng.randrange,
    )
    if permutation is None:
        return None
    cpu.diamond_last_route_kind = "master-kappa-comeback-" + name + "-hnp"
    cpu.diamond_last_certainty = "conditional"
    return diamond.platinum_commit_play(cpu, diamond.CpuAction("play_prime", {
        "cards": permutation.cards, "assigned_numbers": permutation.assigned_numbers,
    }))


def commit(cpu, room, plan, index=0):
    if plan.setup == "draw":
        cpu.diamond_last_route_kind = "master-kappa-comeback-" + plan.name + "-setup-draw"
        return diamond.CpuAction("draw")
    if plan.setup is not None:
        return play(cpu, room, plan.setup, plan.name + "-setup-57")
    move = plan.steps[index]
    expected = subtract(hand_counts(cpu.hand), move.need)
    active = dict(plan=plan, next=index + 1, expected=expected)
    if plan.name == "H" and is_kk(move):
        active["kk_ids"] = tuple(str(c["card_id"]) for c in bind_move(move, cpu.hand)["cards"])
    cpu.master_kappa_comeback = active if any(expected) else None
    return play(cpu, room, move, plan.name)


def continue_plan(cpu, room):
    active = getattr(cpu, "master_kappa_comeback", None)
    if not active:
        return None
    cpu.master_kappa_comeback = None
    held = hand_counts(cpu.hand)
    expected = active["expected"]
    drawn = subtract(held, expected)
    if room.reverse_order or (
        held != expected and not (active.get("awaiting_draw") and room.has_drawn
                                  and drawn is not None and sum(drawn) == 1)
    ):
        return None
    plan, index = active["plan"], active["next"]
    if index - 1 == DECOY_STEP.get(plan.name) and not active.get("passed"):
        if room.field:
            active["passed"] = True
            cpu.master_kappa_comeback = active
            cpu.diamond_last_route_kind = "master-kappa-comeback-" + plan.name + "-decoy-pass"
            return diamond.CpuAction("pass")
    if plan.name == "E" and index == 1 and not room.field:
        return None
    if plan.name in DECOY_STEP and index - 1 == DECOY_STEP[plan.name] and room.field:
        return None
    if plan.name == "H":
        if "kk_ids" not in active:
            search = Search(cpu, room)
            search.hnp_back()
            return commit(cpu, room, search.best) if search.best else None
        if room.field:
            return None
        queue = public_room(room).draw_queue
        targets = set(active["kk_ids"])
        targets.difference_update(str(c["card_id"]) for c in cpu.hand)
        indices = [i for i, card in enumerate(queue) if card and str(card["card_id"]) in targets]
        if len(indices) != len(targets):
            return None
        last = max(indices, default=-1)
        if last < len(cpu.hand) and safe_hnp_assignments(held):
            return hnp(cpu, room, "H")
        next_card = queue[0] if queue else None
        ranks = range(14) if next_card is None else (next_card["rank"],)
        if (not room.has_drawn and queue and last < len(cpu.hand) + 2
                and any(held[r] < CAPACITY[r] and safe_hnp_assignments(add(held, counts((r,))))
                        for r in ranks)):
            active["awaiting_draw"] = True
            cpu.master_kappa_comeback = active
            cpu.diamond_last_route_kind = "master-kappa-comeback-H-recovery-draw"
            return diamond.CpuAction("draw")
        return None
    if index >= len(plan.steps):
        return hnp(cpu, room, "B") if plan.name == "B" else None
    move = plan.steps[index]
    if diamond.candidate_is_playable(bind_move(move, cpu.hand), cpu, room):
        return commit(cpu, room, plan, index)
    return None


def choose_comeback(cpu, room):
    if room.reverse_order or room.field or diamond.platinum_opponent_hand_count(cpu, room) is None:
        return None
    if diamond.opening_master_continuation_allowed(cpu, room):
        return None
    held = hand_counts(cpu.hand)
    solver = Solver(cpu, room, Config(max_nodes=None))
    solver.prepare(held)
    if any(move.size >= 3 and solver.certain(move, held) for move in solver.moves):
        return None
    search = Search(cpu, room)
    try:
        search.long(known=True)
        if search.best is None:
            search.obake_plans({"F1"})
            search.kk_plans({"C", "E"})
            search.quasi()
            search.adjust_obake({"F1"})
        if search.best is None:
            search.obake_plans({"F2", "G"})
            search.kk_plans({"D"})
            search.adjust_obake({"F2"})
        if search.best is None:
            search.hnp_back()
            search.long(known=False)
    except MasterDeadline:
        cpu.last_decision_timed_out = True
        if search.best is None:
            raise
    return commit(cpu, room, search.best) if search.best else None

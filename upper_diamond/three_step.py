"""Non-mate L < M < T selection, below the two certified gates."""
from dataclasses import dataclass
from fractions import Fraction
from itertools import groupby
import time

from .opening_master import ZERO, subtract, hand_counts
from .post_all_out_master import Config, SearchLimit
from .master_budget import check as check_master_budget
from .double_ceiling import DoubleCeilingSolver, DoublePlan, add, visible_need


def choke_tier(size, value):
    if size == 4 and value > 8121011:
        return 1
    if size == 3 and value > 111211:
        return 2
    if size == 5 and value > 910121111:
        return 3
    if size == 4 and value == 8121011:
        return 4
    if size == 5 and value > 910101011:
        return 5
    if size == 3 and value > 71011:
        return 6
    return 7


def strength(size, value, tier):
    return Fraction(value, 10 ** (size - 1)) if tier == 7 else Fraction(value)


@dataclass
class ThreePlan:
    steps: tuple
    frontier: int
    remainder: tuple
    # Strongest unused alternative control after L (excluding nominated M/T).
    alternative: Fraction = Fraction(0)
    alternative_pending: bool = False
    proof: dict | None = None
    response_nodes: tuple = ()
    pass_node: object = None
    model: str = 'three_step'

    @property
    def tier(self):
        return choke_tier(self.steps[1].size, self.frontier)

    @property
    def primary_key(self):
        lead, middle = self.steps[:2]
        return (self.tier, -strength(middle.size, self.frontier, self.tier),
                -strength(middle.size, middle.value, self.tier),
                lead.need[0] + lead.need[13], lead.need[0])

    @property
    def sort_key(self):
        lead, middle, trump, *tail = self.steps
        retained = tuple(a + b for a, b in zip(middle.need, trump.need))
        retained = tuple(a + b for a, b in zip(retained, self.remainder))
        return (*self.primary_key, -self.alternative,
                len(tail), -tail[-1].used_count if tail else 0,
                -retained[0], -min(retained[5], retained[7]),
                tuple(m.key for m in self.steps))

    def summary(self):
        return dict(type='three_step', gate='THREE_STEP', model=self.model, mate=False,
                    effective_frontier=str(self.frontier), choke_tier=self.tier,
                    middle_strength=str(strength(self.steps[1].size, self.steps[1].value, self.tier)),
                    alternative_strength=None if self.alternative_pending else str(self.alternative),
                    steps=[m.summary() for m in self.steps], remainder=list(self.remainder),
                    frontier_proof=self.proof)


class ThreeStepSolver(DoubleCeilingSolver):
    def __init__(self, cpu, room, config=Config(), prepared=None):
        super().__init__(cpu, room, config, prepared)
        self.config = Config(max_nodes=config.three_max_nodes if config.three_max_nodes is not None
                             else config.max_nodes, budget_ms=config.budget_ms)
        self.three_plans = []
        self.certified = None
        self.alternative_cache = {}
        self.alternative_controls = {}
        self.ranked = None
        self.original_hand = hand_counts(cpu.hand)

    def alternative_strength(self, held, middle, trump):
        key = (held, middle.key, trump.key)
        if key not in self.alternative_cache:
            if held not in self.alternative_controls:
                # At most two distinct keys (M/T) are excluded. The strongest
                # three certain keys therefore answer every such query exactly.
                controls = sorted((m for m in self.fitting_moves(held, controls_only=True) if not m.cut),
                    key=lambda m: (-Fraction(m.value, 10 ** (m.size-1)), m.key))
                found = []
                for move in controls:
                    self.tick()
                    if self.certain(move, held):
                        found.append((move.key, Fraction(move.value, 10 ** (move.size-1))))
                        if len(found) == 3:
                            break
                self.alternative_controls[held] = tuple(found)
            self.alternative_cache[key] = next((value for k, value in self.alternative_controls[held]
                if k not in (middle.key, trump.key)), Fraction(0))
        return self.alternative_cache[key]

    def resolve_alternative(self, hand, plan):
        if plan.alternative_pending:
            value = self.alternative_strength(subtract(hand, plan.steps[0].need), *plan.steps[1:3])
            plan.alternative = value
            plan.alternative_pending = False

    def rank_plans(self, hand):
        result = []
        # Only equal primary keys can be reordered by alternative strength.
        for _, items in groupby(sorted(self.three_plans, key=lambda p: p.primary_key), key=lambda p: p.primary_key):
            group = list(items)
            if len(group) > 1:
                try:
                    for plan in group:
                        self.resolve_alternative(hand, plan)
                except SearchLimit:
                    self.timed_out = True
                    # Never compare a missing alternative as zero at timeout.
                    group = [p for p in group if not p.alternative_pending]
                group.sort(key=lambda p: p.sort_key)
            result.extend(group)
        return result

    def prove_frontier(self, hand, plan):
        """All physical classes at a value must pass; the first gap stops growth.

        This uses exact Reply classes, not the broad below/above-M compression.
        No responses at a numeric value means no registered-knowledge obstacle.
        A timeout preserves only fully completed value buckets.
        """
        lead, middle, trump, *tail = plan.steps
        audit = dict(status='unproved', completed_values=[], blocked_value=None)
        plan.proof = audit
        rest = subtract(hand, lead.need)
        locked = visible_need(lead)
        self.swap_keys = {middle.key, trump.key}
        passed = self.free_certificate(rest)
        plan.pass_node = passed
        replies = self.reply_classes(lead, hand, ZERO, (self.opponent_count,))
        replies = sorted(replies, key=lambda r: (r.move.value or 10**100, r.key))
        _, caps = self.pool_after_move(lead, hand, ZERO, None)
        children = []
        all_proved = passed is not None
        for value, bucket in groupby(replies, key=lambda r: r.move.value or 10**100):
            bucket_proved = True
            for reply in bucket:
                self.tick()
                if reply.move.cut or reply.move.value == 1729 or 0 in reply.opponent_counts:
                    child = None
                else:
                    limits = (caps[0] - (reply.visible[0]+sum(reply.visible[10:14])),
                              caps[1] - reply.visible[0]-reply.visible[13])
                    child, _ = self.respond(rest, add(locked, reply.visible), reply.opponent_counts,
                        (reply.move.size, reply.move.value), (middle, trump)+tuple(tail),
                        'repartition', limits=limits)
                if child is None:
                    bucket_proved = False
                    break
                children.append((reply, child))
                plan.response_nodes = tuple(children)
            all_proved &= bucket_proved
            if value >= middle.value:
                if not bucket_proved:
                    audit['blocked_value'] = str(value)
                    break
                plan.frontier = value + 1
                audit['completed_values'].append(str(value))
        if all_proved and replies and passed is not None:
            root = self.new_node(lead, hand, ZERO, (self.opponent_count,), None,
                                 pass_node=passed, replies=tuple(children))
            candidate = dict(id=0, status='certified', mate=True)
            self.certified = DoublePlan(plan.steps, root, candidate)
            audit['status'] = 'certified'

    def evaluate_three(self, hand):
        self.original_hand = hand
        if self.opponent_count is None or self.room.field or self.room.reverse_order:
            return None
        try:
            if not self.moves:
                self.prepare(hand)
            # Give each displayed count a share before using the proof budget.
            total = self.config.max_nodes
            for index, size in enumerate((4, 3, 5, 6, 7, 2, 1, 8)):
                self.config = Config(max_nodes=None if total is None else total*(index+1)//12,
                                     budget_ms=self.config.budget_ms)
                try:
                    for lead, middle, trump, tail, remaining, _ in self.nominal_candidates(hand, size, inclusive=False):
                        held = subtract(subtract(hand, lead.need), middle.need)
                        if not self.certain(trump, held):
                            continue
                        plan = ThreePlan((lead, middle, trump)+tail, middle.value, remaining,
                                         alternative_pending=True)
                        self.three_plans.append(plan)
                    self.completed_sizes.append(size)
                except SearchLimit:
                    self.timed_out = True
            self.config = Config(max_nodes=total, budget_ms=self.config.budget_ms)
            # Start with the best ordinary candidate of each tier, then the rest.
            ranked = self.rank_plans(hand)
            first = {}
            for plan in ranked:
                first.setdefault(plan.tier, plan)
            leaders = list(first.values())
            for plan in leaders + [p for p in ranked if all(p is not q for q in leaders)]:
                check_master_budget(self.cpu)
                self.prove_frontier(hand, plan)
                if self.certified:
                    return self.certified
        except SearchLimit:
            self.timed_out = True
            if self.three_plans and 'plan' in locals() and plan.proof:
                plan.proof['status'] = 'timeout'
        self.ranked = self.rank_plans(hand)
        return self.ranked[0] if self.ranked else None

    def summary_three(self):
        ranked = self.ranked if self.ranked is not None else self.rank_plans(self.original_hand)
        tiers = {}
        for plan in ranked:
            if str(plan.tier) not in tiers:
                tiers[str(plan.tier)] = plan.summary()
        return dict(nodes=self.nodes, timed_out=self.timed_out,
                    exhaustive=not self.timed_out and self.certified is None,
                    completed_sizes=self.completed_sizes, candidates=len(ranked), by_tier=tiers,
                    elapsed_ms=1000*(time.perf_counter()-self.started))


def continue_three(cpu, room, active):
    from .post_all_out_master import play_next
    from .double_ceiling import commit_node
    import cpu_player as diamond
    if active.pop('awaiting_three_reply', False):
        if room.field:
            physical = hand_counts(room.field)
            opponent = diamond.platinum_opponent_hand_count(cpu, room)
            node = next((child for reply, child in active['response_nodes']
                         if reply.matches(int(room.last_number), physical, opponent)), None)
        else:
            node = active['pass_node']
        if node is not None:
            active.update(model='public_control_sequence', branch_certificate=True)
            cpu.post_all_out_master_trace = dict(continuation=True, model=active['model'],
                                                 certificate_node=node.id)
            return commit_node(cpu, room, active, node)
    sequence = active['sequence']
    if not sequence:
        return None
    # A free field permits M directly. Unproved replies over M require replanning.
    if room.field:
        while sequence and (sequence[0].size != len(room.field) or
                            sequence[0].template.kind != 'cut' and sequence[0].value <= int(room.last_number)):
            sequence = sequence[1:]
        # Skipping M changes the remainder. Replan instead of consuming a
        # nominal tail with unaccounted cards or claiming a certified finish.
        if sequence != active['sequence']:
            cpu.post_all_out_master_active = None
            return None
    cpu.post_all_out_master_trace = dict(continuation=True, model='three_step')
    return play_next(cpu, room, active, sequence)

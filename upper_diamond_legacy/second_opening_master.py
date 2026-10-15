"""Second player's first free-field decision after the opener's failed all-out.

Search and execution are shared with Opening Master and Diamond. All structural
candidates are retained for audit, including candidates below a Certified plan.
No virtual draws or hidden-card inputs are used by the selector.
"""
from dataclasses import replace

from . import opening_master as opening

THRESHOLD = 0.50
ALL_OUT_VALUE = 0.45


def controlling(plan):
    return plan.steps[1] if plan.model == 'public_rally_control' or plan.type in ('normal_auso', 'dual_wield') else plan.steps[0]


def notation(move):
    return ''.join(opening.LABELS[r] for r in move.template.visible)


def classify(plan):
    if plan.type == 'certified_win':
        return 'certified', None
    if plan.type == 'normal_auso':
        return 'rejected', 'normal_auso_requires_absolute_certain'
    if plan.type not in ('one_step', 'dual_wield'):
        return 'rejected', 'unsupported_plan_type'
    if plan.model == 'unmodeled':
        return 'rejected', 'p24_unmodeled'
    if plan.p_return < THRESHOLD:
        return 'high_confidence', None
    trump = notation(controlling(plan))
    if trump == 'KJQJ' or (trump == 'KQ' and plan.type == 'one_step'):
        return 'practical_exception', None
    return 'rejected', 'p24_at_least_0.50'


class Evaluator(opening.Evaluator):
    def __init__(self, cpu, room):
        # Retain even p=1/unmodeled candidates, then apply the strict real gate.
        super().__init__(cpu, room, opening.Config(return_threshold=1.01,
            collect_all_candidates=True, generalize_dual=True))

    def probability(self, move, hand, context):
        return super().probability(move, hand, 24)

    def _evaluate(self, hand):
        super()._evaluate(hand)
        self.last_plans = [replace(p, context=24) if p.type != 'certified_win' else p
                           for p in self.last_plans]
        accepted = [p for p in self.last_plans if classify(p)[0] != 'rejected']
        priority = {'certified': 0, 'high_confidence': 1, 'practical_exception': 2}
        return min(accepted, key=lambda p: (priority[classify(p)[0]], p.p_return,
            -p.steps[-1].used_count, sum(m.need[0] for m in p.steps),
            p.log_type, tuple(m.key for m in p.steps),
            tuple(m.key for m in p.pass_sequence))) if accepted else None

    def candidate_rows(self, hand, best):
        rows = []
        for index, plan in enumerate(self.last_plans):
            label, reason = classify(plan)
            trump = controlling(plan)
            p24, model = self.probability(trump, hand, 24)
            rows.append(dict(candidate_id=index, candidate_type=plan.log_type,
                lead=notation(plan.steps[0]), trump=notation(trump), p24=p24,
                probability_model=model, label=label, rejection_reason=reason,
                selected=plan == best, band_45_50=0.45 <= p24 < 0.50,
                plan=plan.summary()))
        return rows


def choose_second_opening(cpu, room, diamond_choice):
    cpu.second_opening_master_trace = None
    phase = getattr(cpu, 'second_opening_master_phase', 'initial')
    # The phase closes at the first ineligible decision, including an ordinary
    # opening response. Later opponent penalties cannot re-open this policy.
    eligible = (phase in ('initial', 'after_draw')
        and getattr(room, 'first_player_id', cpu.id) != cpu.id
        and room.rule.key == 'std-11-n-c' and not room.reverse_order
        and getattr(room, 'opening_all_out_failed', False)
        and not room.field and len(cpu.hand) == (12 if room.has_drawn else 11)
        and (opening.diamond.platinum_opponent_hand_count(cpu, room) or 0) in (22, 24)
        and not getattr(cpu, 'platinum_all_out_attempts', 0))
    if not eligible:
        cpu.second_opening_master_phase = 'done'
        return diamond_choice(cpu, room)
    ev = getattr(cpu, 'second_opening_master_evaluator', None) or Evaluator(cpu, room)
    cpu.second_opening_master_evaluator = ev
    hand = opening.hand_counts(cpu.hand)
    best = ev.evaluate(hand)
    if best is None:
        action = opening.no_plan_action(cpu, room)
        label = 'draw' if action.kind == 'draw' else 'all_out'
    else:
        # Diamond clears pre-expansion plans on a tactical-context transition.
        # This plan was already proved against the expanded opponent: establish
        # that public context before handing it to Diamond's normal executor.
        cpu.diamond_last_context = opening.diamond.diamond_tactical_context(cpu, room)
        action = opening.commit_plan(cpu, room, best)
        label = classify(best)[0]
    cpu.second_opening_master_phase = 'after_draw' if action.kind == 'draw' else 'done'
    cpu.second_opening_master_trace = dict(
        seed=getattr(cpu, 'opening_master_seed', None),
        leg=getattr(cpu, 'opening_master_leg', None),
        hand=opening.rank_text(hand), hand_size=sum(hand),
        threshold=THRESHOLD, all_out_value=ALL_OUT_VALUE,
        label=label, action=opening.action_summary(action),
        best=best.summary() if best else None,
        candidates=ev.candidate_rows(hand, best))
    return action

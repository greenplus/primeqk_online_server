"""Public-information draw/57 planning and conditional U -> C -> R HNP."""
from collections import Counter
from copy import copy, deepcopy
from dataclasses import dataclass, replace
from functools import cmp_to_key
from types import SimpleNamespace
import time

import cpu_player as diamond
from .master_budget import check as check_master_budget, MasterDeadline
from hnp_challenge import build_hnp_tokens, choose_hnp_permutation
from .opening_master import ZERO, Move, Template, bind_move, counts, hand_counts, subtract
from .post_all_out_master import (Solver, SearchLimit, knowledge_key, evaluate_current_state,
                                  commit_current_plan, continue_plan, play_next)


def public_room(room):
    """Never copy or inspect hidden deck cards, even in an authoritative room."""
    known = list(getattr(room, 'public_known_deck_bottom', ()) or ())
    total = len(room.deck)
    unknown = max(0, int(getattr(room, 'public_unknown_deck_count', total-len(known))))
    queue = [None]*min(unknown, total) + known
    queue = (queue + [None]*total)[:total]
    return SimpleNamespace(rule=room.rule, field=list(room.field),
        reserve=list(getattr(room, 'reserve', ()) or ()), reverse_order=room.reverse_order,
        last_number=getattr(room, 'last_number', None), has_drawn=room.has_drawn,
        opponent_hand_count=getattr(room, 'opponent_hand_count', None),
        deck=[None]*total, public_unknown_deck_count=unknown,
        public_known_deck_bottom=known, draw_queue=queue)


def next_known(room):
    queue = room.draw_queue
    return queue[0] if queue else None


def simulated_draw(cpu, room):
    card = room.draw_queue.pop(0)
    assert card is not None
    cpu.hand = cpu.hand + [dict(card, card_id=f'lookahead-{len(cpu.hand)}-{len(room.deck)}')]
    room.deck.pop()
    room.has_drawn = True


def simulated_57(cpu, room):
    candidate = bind_move(Move(Template('prime', 57, (5, 7)), (5, 7), counts((5, 7))), cpu.hand)
    used = candidate['cards']
    ids = {c['card_id'] for c in used}
    cpu.hand = [c for c in cpu.hand if c['card_id'] not in ids]
    returned = room.reserve + used
    room.draw_queue.extend(returned)
    room.deck.extend([None]*len(returned))
    room.reserve = []
    room.has_drawn = False


def fresh_solver(cpu, room, config, prepared=None):
    s = Solver(cpu, room, replace(config, hand_sizes=(len(cpu.hand),)))
    s.reuse_prepared(prepared)
    if not s.moves:
        s.prepare(hand_counts(cpu.hand))
    return s


def certain_resources(cpu, room, config, prepared=None):
    """Protect every current certain realization by rank-count maxima, not IDs."""
    try:
        s = fresh_solver(cpu, room, config, prepared)
        held = hand_counts(cpu.hand)
        controls = []
        for m in s.moves:
            s.tick()
            if s.certain(m, held):
                controls.append(m)
        reserved = tuple(max((m.need[r] for m in controls), default=0) for r in range(14))
        free = subtract(held, reserved)
        pairs = min(free[5], free[7])
        reason = 'NO_SAFE_57'
        if not held[5] or not held[7]:
            reason = '57_RESOURCE_EXHAUSTED'
        elif not pairs:
            reason = '57_RESERVED_BY_CERTAIN'
        else:
            # Released cards must not invalidate resource-dependent ceilings.
            after = subtract(held, counts((5, 7)))
            if not all(s.certain(m, after) for m in controls):
                pairs, reason = 0, '57_INVALIDATES_CERTAIN'
        return dict(complete=True, reserved=list(reserved), free_5=free[5], free_7=free[7],
                    safe_57_count=pairs, reason=reason,
                    certain=[m.summary() for m in controls])
    except SearchLimit:
        return dict(complete=False, certain=[], safe_57_count=0, reason='CERTAIN_SEARCH_LIMIT')


def tactics_summary(trace):
    stages = trace.get('stage_results', {})
    return dict(mate=stages.get('mate'), mate_count=stages.get('mate_count', 0),
        double=trace.get('double_ceiling'), three=trace.get('three_step'),
        selected=trace.get('best'), timed_out=any(trace.get(k, {}).get('timed_out', False)
            for k in ('double_ceiling', 'three_step')) or trace.get('timed_out', False))


def draw_path(cpu, room, config, prepared=None):
    """Read an entire legal known sequence; reaching an unknown draw is enough."""
    player, state = copy(cpu), public_room(room)
    player.hand = list(cpu.hand)
    path, audits = [], []
    while True:
        try:
            check_master_budget(cpu)
        except MasterDeadline:
            return [], dict(reason='KNOWN_DRAW_UNPROVED', search_complete=False,
                            known_draw_lookahead_length=path.count('draw'), steps=audits)
        if not state.has_drawn and state.deck:
            card = next_known(state)
            if card is None:
                return path + ['draw'], dict(reason='UNKNOWN_DRAW' if not path else 'KNOWN_PATH_TO_UNKNOWN',
                    known_draw_lookahead_length=path.count('draw'), steps=audits)
            simulated_draw(player, state)
            path.append('draw')
            base, _, best, trace = evaluate_current_state(player, state,
                replace(config, hand_sizes=(len(player.hand),)))
            resources = certain_resources(player, state, config, base) if best is None else None
            audits.append(dict(action='draw', card=dict(card), hand=list(hand_counts(player.hand)),
                               tactics=tactics_summary(trace), certain=resources))
            if best is not None:
                return path, dict(reason='KNOWN_DRAW_PLAN_FOUND',
                    known_draw_lookahead_length=path.count('draw'), steps=audits)
            prepared = base
        else:
            resources = certain_resources(player, state, config, prepared)
            if not resources['safe_57_count']:
                complete = resources['complete'] and not any(r.get('tactics', {}).get('timed_out') for r in audits)
                return [], dict(reason='KNOWN_DRAW_NO_NEW_TACTIC' if complete else 'KNOWN_DRAW_UNPROVED',
                    stop_reason=resources['reason'],
                    known_draw_lookahead_length=path.count('draw'), steps=audits,
                    search_complete=complete)
            simulated_57(player, state)
            path.append('57')
            audits.append(dict(action='57', physical_ranks=[5, 7], certain=resources))
            prepared = None
            # Intentionally no search here: the next operation is the new draw.


def hnp_assignments(held):
    cards = [dict(rank=r, is_joker=r == 0) for r, n in enumerate(held) for _ in range(n)]
    if not held[0] and not any(held[r] for r in (1, 3, 7, 9, 11, 13)):
        return ()
    return tuple(diamond.diamond_hnp_joker_assignments(cards))


def min_hnp_value(held, assignments):
    """A lower bound for every permutation; no primality or factor oracle."""
    texts = [str(r) for r, n in enumerate(held) if r for _ in range(n)] + list(map(str, assignments))
    texts.sort(key=cmp_to_key(lambda a, b: (a+b > b+a)-(a+b < b+a)))
    return int(''.join(texts))


@dataclass
class HnpPlan:
    u: tuple
    c: Move
    r: Move
    assignments: tuple

    @property
    def key(self):
        return (sum(self.u), self.u[0]+sum(self.u[10:]), self.u, self.c.key, self.r.key)

    def summary(self):
        return dict(model='pre_hnp', gate='CONDITIONAL_HNP', certainty='conditional',
                    U=list(self.u), U_size=sum(self.u), face_count=self.u[0]+sum(self.u[10:]),
                    C=self.c.summary(), R=self.r.summary(), C_size=self.c.size,
                    one_move_remainder=True, failure_penalty=sum(self.u),
                    obvious_failure=False, conditional_certificate=True,
                    proof_scope='existing Master registered-response catalog; all randomized U orders')


def certified_hnp(s, held, u, c, r, assignments):
    remaining = subtract(held, u)
    if remaining is None or subtract(remaining, c.need) != r.need:
        return False
    if not s.certain(c, remaining, inclusive=True) or not s.certain(c, remaining):
        return False
    for assignment in assignments:
        # A possible 1729 play changes the rules, even though it bypasses primality.
        visible = counts([rank for rank, n in enumerate(u) if rank for _ in range(n)] + list(assignment))
        if sum(u) == 4 and visible == counts((1, 7, 2, 9)):
            return False
        lower = min_hnp_value(u, assignment)
        lead = Move(Template('prime', lower, (1,)*sum(u)), (), u)
        if not s.safe_lead(lead, c, remaining):
            return False
    return True


def hnp_audit(audit, **fields):
    # dataclasses.asdict rebuilds Counter from (key,value) pairs, producing
    # tuple keys. Store plain mappings in game records instead.
    return dict(audit, rejected=dict(audit['rejected']), **fields)


def search_pre_hnp(cpu, room, config, prepared=None):
    audit = dict(considered=True, deck_remaining=len(room.deck), rejected=Counter())
    if not room.deck:
        return None, hnp_audit(audit, reason='PRE_HNP_EMPTY_DECK')
    best = None
    try:
        s = fresh_solver(cpu, room, config, prepared)
        held = hand_counts(cpu.hand)
        # C and exact R first. U is solely their physical complement.
        for c in sorted(s.moves, key=lambda m: (m.size, m.key)):
            s.tick()
            if c.size < 4:
                audit['rejected']['PRE_HNP_TOO_SMALL'] += 1
                continue
            if best is not None and c.size > sum(best.u):
                break
            if c.size < len(room.deck):
                audit['rejected']['PRE_HNP_PENALTY_TOO_SMALL'] += 1
                continue
            if not s.certain(c, held):
                audit['rejected']['PRE_HNP_NO_CERTAIN'] += 1
                continue
            after_c = subtract(held, c.need)
            for r in s.fitting_moves(after_c):
                s.tick()
                u = subtract(after_c, r.need)
                if sum(u) != c.size:
                    audit['rejected']['PRE_HNP_SIZE_MISMATCH'] += 1
                    continue
                assignments = hnp_assignments(u)
                if not assignments:
                    audit['rejected']['PRE_HNP_OBVIOUS_COMPOSITE'] += 1
                    continue
                if not certified_hnp(s, held, u, c, r, assignments):
                    audit['rejected']['PRE_HNP_NO_CERTIFIED_SEQUENCE'] += 1
                    continue
                candidate = HnpPlan(u, c, r, assignments)
                if best is None or candidate.key < best.key:
                    best = candidate
    except SearchLimit:
        # Minimum U / minimum faces cannot be guaranteed from a partial scan.
        return None, hnp_audit(audit, reason='PRE_HNP_SEARCH_LIMIT', timed_out=True)
    return best, hnp_audit(audit, reason='PRE_HNP_FOUND' if best else 'PRE_HNP_NO_CERTIFIED_SEQUENCE',
                      selected=best.summary() if best else None)


def commit_hnp(cpu, room, plan, audit):
    # U is already unique. Only its assignment/permutation is randomized here.
    selected = []
    for rank, n in enumerate(plan.u):
        selected.extend(sorted((c for c in cpu.hand if (0 if diamond.is_joker(c) else int(c['rank'])) == rank),
                               key=lambda c: str(c['card_id']))[:n])
    assignment = cpu.rng.choice(plan.assignments)
    permutation = choose_hnp_permutation(build_hnp_tokens(selected, list(map(str, assignment))),
                                         randbelow=cpu.rng.randrange)
    assert permutation is not None
    candidate = dict(kind='prime', number=permutation.number, cards=permutation.cards,
                     assigned_numbers=permutation.assigned_numbers)
    diamond.clear_gold_active_plan(cpu)
    cpu.post_all_out_master_active = dict(model='pre_hnp', knowledge=knowledge_key(cpu, room),
        expected=subtract(hand_counts(cpu.hand), plan.u), sequence=(plan.c, plan.r))
    audit.update(executed_value=str(permutation.number), success=None, game_result=None)
    cpu.post_all_out_hnp_audit = audit
    diamond.diamond_update_opening_position_history(cpu, room)
    diamond.diamond_remember_cards(cpu, selected)
    cpu.diamond_last_route_kind = 'post-all-out-master-pre-hnp'
    cpu.diamond_last_certainty = 'conditional'
    cpu.diamond_last_return_probability = None
    return diamond.platinum_commit_play(cpu, diamond.candidate_to_action(candidate))


def choose_after_all_out(cpu, room, fallback, config):
    cpu.post_all_out_master_trace = None
    action = continue_plan(cpu, room)
    if action is not None:
        return action
    phase = getattr(cpu, 'post_all_out_draw_phase', None)
    if phase and (phase['knowledge'] != knowledge_key(cpu, room) or room.field or room.reverse_order
                  or room.rule.key != 'std-11-n-c'):
        phase = cpu.post_all_out_draw_phase = None
    if phase and phase.get('pending') == '57':
        if hand_counts(cpu.hand) == phase['expected'] and not room.has_drawn:
            # The promised immediate draw precedes any Master detector.
            phase['pending'] = 'draw'
            phase['before_draw'] = hand_counts(cpu.hand)
            phase['log']['draw_count'] += 1
            phase['log']['actions'].append(dict(action='draw', after_57=True))
            cpu.diamond_last_route_kind = 'post-all-out-master-draw-after-57'
            cpu.post_all_out_master_trace = dict(planner=deepcopy(phase['log']), reason='DRAW_AFTER_57')
            return diamond.CpuAction('draw')
        phase = cpu.post_all_out_draw_phase = None
    if phase and phase.get('pending') == 'draw':
        delta = subtract(hand_counts(cpu.hand), phase['before_draw'])
        if delta is None or sum(delta) != 1 or not room.has_drawn:
            phase = cpu.post_all_out_draw_phase = None
    eligible = (not room.field and not room.reverse_order and room.rule.key == 'std-11-n-c'
        and (len(cpu.hand) in config.hand_sizes or phase is not None)
        and getattr(cpu, 'platinum_all_out_attempts', 0) > 0)
    if not eligible:
        return fallback(cpu, room)
    current_config = replace(config, hand_sizes=(len(cpu.hand),))
    base, owner, best, trace = evaluate_current_state(cpu, room, current_config)
    cpu.post_all_out_master_trace = trace
    cpu.post_all_out_master_three_step_cache = base.three_step_candidates
    cpu.post_all_out_master_double_cache = getattr(base, 'double_candidates', [])
    cpu.last_decision_timed_out = tactics_summary(trace)['timed_out']
    if best is not None:
        trace['reason'] = 'DIRECT_MASTER_PLAN_FOUND' if phase is None else 'NEW_MASTER_PLAN_FOUND'
        if phase:
            phase['log']['new_tactic'] = best.summary()
            phase['log']['after_draw'].append(dict(tactics=tactics_summary(trace),
                certain=certain_resources(cpu, room, config, base)))
            trace['planner'] = deepcopy(phase['log'])
        cpu.post_all_out_draw_phase = None
        return commit_current_plan(cpu, room, owner, best)
    if phase is None:
        phase = dict(knowledge=knowledge_key(cpu, room), log=dict(initial_hand=list(hand_counts(cpu.hand)),
            initial_tactics=tactics_summary(trace), initial_certain=None,
            draw_count=0, uses_57=0, actions=[], after_draw=[], fallback=False))
    resources = certain_resources(cpu, room, config, base) if config.draw_57 else None
    if phase['log']['initial_certain'] is None:
        phase['log']['initial_certain'] = resources
    if phase.get('pending') == 'draw':
        phase['log']['after_draw'].append(dict(tactics=tactics_summary(trace), certain=resources))
    path, look = draw_path(cpu, room, config, base) if config.draw_57 else ([], dict(reason='DRAW_57_DISABLED'))
    cpu.last_decision_timed_out |= look.get('search_complete') is False
    phase['log'].update(next_card_known=bool(next_known(public_room(room))), lookahead=look)
    if path:
        diamond.clear_gold_active_plan(cpu)
        cpu.post_all_out_draw_phase = phase
        phase['pending'] = path[0]
        trace['reason'] = look['reason']
        if path[0] == 'draw':
            phase['before_draw'] = hand_counts(cpu.hand)
            phase['log']['draw_count'] += 1
            phase['log']['actions'].append(dict(action='draw', after_57=False))
            action = diamond.CpuAction('draw')
        else:
            move = Move(Template('prime', 57, (5, 7)), (5, 7), counts((5, 7)))
            candidate = bind_move(move, cpu.hand)
            phase['expected'] = subtract(hand_counts(cpu.hand), move.need)
            phase['log']['uses_57'] += 1
            phase['log']['actions'].append(dict(action='57', cards=deepcopy(candidate['cards'])))
            trace['reason'] = 'PLAY_57'
            diamond.diamond_update_opening_position_history(cpu, room)
            diamond.diamond_remember_cards(cpu, candidate['cards'])
            action = diamond.platinum_commit_play(cpu, diamond.candidate_to_action(candidate))
        cpu.diamond_last_route_kind = 'post-all-out-master-'+trace['reason'].lower()
        cpu.diamond_last_certainty, cpu.diamond_last_return_probability = None, None
        trace['planner'] = deepcopy(phase['log'])
        return action
    cpu.post_all_out_draw_phase = None
    hnp, audit = search_pre_hnp(cpu, room, config, base) if config.pre_hnp else (None, dict(considered=False))
    trace['pre_hnp'] = audit
    cpu.last_decision_timed_out |= audit.get('timed_out', False)
    if hnp:
        trace['reason'] = 'PRE_HNP_FOUND'
        trace['planner'] = deepcopy(phase['log'])
        return commit_hnp(cpu, room, hnp, audit)
    started = time.perf_counter()
    # A rejected known series must not be drawn anyway by the fallback.
    fallback_room = copy(room) if config.draw_57 else room
    if config.draw_57:
        fallback_room.has_drawn = True
    action = fallback(cpu, fallback_room)
    if config.draw_57 and action.kind == 'draw':
        action = diamond.CpuAction('pass')
    phase['log']['fallback'] = True
    trace.update(reason='FALLBACK', planner=deepcopy(phase['log']), fallback_reason='no-certified-plan',
        fallback_budget_ms=cpu.decision_time_budget_ms, fallback_elapsed_ms=(time.perf_counter()-started)*1000,
        fallback_action=action.kind)
    return action


def record_outcome(cpu, trace, outcome):
    """Called after real game adjudication; never feeds a primality oracle to search."""
    audit = (trace.get('post_all_out_master') or {}).get('pre_hnp')
    if audit and 'executed_value' in audit:
        audit['success'] = outcome['result'] == 'success'
        audit['outcome'] = outcome['result']
        if not audit['success']:
            cpu.post_all_out_master_active = None

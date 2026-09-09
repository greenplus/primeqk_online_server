"""Compose the released opening and post-all-out patches over unchanged Diamond."""
from .opening_master import Config as OpeningConfig, choose_opening
from .post_all_out_master import Config as MateConfig, choose_post_all_out
from .second_opening_master import choose_second_opening
from .master_budget import Budget, CURRENT, MasterDeadline, TimingConfig
import time
from copy import deepcopy
import cpu_player as diamond


def _choose_combined(cpu, room, diamond_choice):
    cpu.opening_master_trace = None
    cpu.post_all_out_master_trace = None
    cpu.second_opening_master_trace = None

    def after_opening(player, state):
        def after_second(subject, public):
            return choose_post_all_out(subject, public, diamond_choice,
                                       getattr(subject, 'post_all_out_master_config',
                                               MateConfig(max_nodes=None, double_max_nodes=None)))
        return choose_second_opening(player, state, after_second)

    # choose_opening already restricts initial searches to the first player.
    # Its committed plans continue through the original Diamond executor.
    return choose_opening(cpu, room, after_opening,
                          getattr(cpu, 'opening_master_config', OpeningConfig()))


def clear_interrupted_plan(cpu):
    diamond.clear_gold_active_plan(cpu)
    diamond.clear_silver_active_plan(cpu)
    cpu.post_all_out_master_active = None
    cpu.post_all_out_draw_phase = None
    # An interrupted evaluator may have partial diagnostic state, but is never
    # allowed to publish it as a complete cached decision on the next turn.
    cpu.opening_master_evaluator = None
    cpu.second_opening_master_evaluator = None


def decision_snapshot(cpu):
    # Knowledge/indexes and keyed calculation caches do not represent executed
    # actions. Copy mutable play/history state, not these large shared objects.
    shared = {'ws', 'room', 'rng', 'registered_primes', 'registered_composites',
              'registered_composite_entries', 'small_finish_index', 'prime_template_index',
              'opening_master_evaluator', 'second_opening_master_evaluator'}
    saved = {k: (v if k in shared or 'cache' in k or k.endswith('_trace') else deepcopy(v))
             for k,v in cpu.__dict__.items()}
    try:
        rng_state = cpu.rng.getstate()
    except (AttributeError, NotImplementedError):
        rng_state = None
    return saved, rng_state


def choose_combined(cpu, room, diamond_choice):
    """Default production budget; master_timing_config=None opts into research."""
    config = getattr(cpu, 'master_timing_config', TimingConfig())
    cpu.master_budget_trace = None
    if config is None:
        return _choose_combined(cpu, room, diamond_choice)
    budget = Budget(time.perf_counter(), config)
    saved, rng_state = decision_snapshot(cpu)
    token = CURRENT.set(budget)
    fallback_used = emergency = False
    old_hard = getattr(cpu, 'decision_hard_deadline', None)
    old_soft = getattr(cpu, 'decision_soft_deadline', None)

    def fallback(player, state):
        nonlocal fallback_used
        fallback_used = True
        now = time.perf_counter()
        if budget.fallback_deadline is None:
            budget.fallback_deadline = min(now+config.fallback_ms/1000, budget.total_deadline)
        # Give Diamond's normal timeout recovery the final response margin.
        # An ordinary 1500 ms search timeout must not become an emergency pass.
        margin = min(500, config.total_ms-config.search_ms-config.fallback_ms)/1000
        deadline = min(budget.fallback_deadline+margin, budget.total_deadline)
        if old_hard is not None:
            deadline = min(deadline, old_hard)
        player.decision_hard_deadline = deadline
        player.decision_soft_deadline = min(budget.fallback_deadline, old_soft) if old_soft is not None else budget.fallback_deadline
        if now >= budget.fallback_deadline or now >= deadline:
            raise diamond.CpuDecisionDeadline
        # Diamond uses its own clock; do not charge its checkpoints against
        # Master's six seconds or restart its shared fallback allowance.
        suspended = CURRENT.set(None)
        try:
            return diamond_choice(player, state)
        finally:
            CURRENT.reset(suspended)
            player.decision_hard_deadline = old_hard
            player.decision_soft_deadline = old_soft

    try:
        try:
            action = _choose_combined(cpu, room, fallback)
        except MasterDeadline:
            budget.expired = True
            clear_interrupted_plan(cpu)
            action = fallback(cpu, room)
        except diamond.CpuDecisionDeadline:
            raise
        if time.perf_counter() >= budget.total_deadline:
            raise diamond.CpuDecisionDeadline
        return action
    except diamond.CpuDecisionDeadline:
        emergency = True
        traces = {name: getattr(cpu,name,None) for name in
                  ('post_all_out_master_trace','opening_master_trace','second_opening_master_trace')}
        cpu.__dict__.clear()
        cpu.__dict__.update(saved)
        cpu.__dict__.update(traces)
        if rng_state is not None:
            cpu.rng.setstate(rng_state)
        clear_interrupted_plan(cpu)
        post = getattr(cpu, 'post_all_out_master_trace', None)
        if post is not None:
            post['deadline_discarded_best'] = post.get('best')
            post.update(best=None, reason='MASTER_DEADLINE_PASS')
        opening = getattr(cpu, 'opening_master_trace', None)
        if opening is not None:
            opening.update(opening_master_used=False, reason='MASTER_DEADLINE_PASS')
        second = getattr(cpu, 'second_opening_master_trace', None)
        if second is not None:
            second.update(label='deadline_pass', best=None)
        # Passing is legal on both free and occupied fields. Do not introduce
        # an unproved draw after a known draw/57 path was rejected.
        cpu.diamond_last_route_kind = 'master-deadline-pass'
        cpu.diamond_last_certainty = None
        cpu.diamond_last_return_probability = None
        return diamond.CpuAction('pass')
    finally:
        cpu.decision_hard_deadline = old_hard
        cpu.decision_soft_deadline = old_soft
        CURRENT.reset(token)
        cpu.last_decision_timed_out = bool(getattr(cpu, 'last_decision_timed_out', False)
                                           or budget.expired or emergency)
        cpu.master_budget_trace = dict(search_ms=config.search_ms, fallback_ms=config.fallback_ms,
            total_ms=config.total_ms, elapsed_ms=1000*(time.perf_counter()-budget.started),
            master_expired=budget.expired, fallback_used=fallback_used, emergency_pass=emergency)

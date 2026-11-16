"""Public-information recovery decisions for normal-order Master kappa."""

import cpu_player as diamond

from .after_all_out_planner import public_room
from .master_budget import check
from .move_safety import uses_spare_natural
from .opening_master import bind_move
from .post_all_out_master import knowledge_key
from .public_control import ControlSearch, realizations, templates_by_size


def card_ids(cards):
    return frozenset(str(card["card_id"]) for card in cards)


def is_kx(card):
    return diamond.is_joker(card) or card["rank"] == 13


def note(cpu, room, kind, status, **details):
    if getattr(cpu, "master_kappa_recovery_trace", None) is None:
        cpu.master_kappa_recovery_trace = []
    cpu.master_kappa_recovery_trace.append(dict(
        kind=kind, status=status, hand=len(cpu.hand), deck=len(room.deck),
        field=len(room.field), opponent=diamond.platinum_opponent_hand_count(cpu, room),
        **details,
    ))


def danger(cpu, room, kind):
    threat = (diamond.diamond_infer_opponent_rally_threat(cpu, room, len(room.field))[0]
              if room.field else "none")
    if not diamond.platinum_interference_danger_active(cpu, room, rally_threat=threat):
        return False
    note(cpu, room, kind, "danger", rally_threat=threat,
         opponent_kx=diamond.platinum_expected_opponent_kx_remaining(cpu, room))
    return True


def commit(cpu, room, kind, phase, action, *, candidate=None, **details):
    diamond.clear_gold_active_plan(cpu)
    cpu.master_kappa_comeback = None
    cpu.post_all_out_draw_phase = None
    cpu.diamond_last_route_kind = f"master-kappa-recovery-{kind}-{phase}"
    cpu.diamond_last_certainty = "certain" if phase == "control" else "unclassified"
    cpu.diamond_last_return_probability = None
    note(cpu, room, kind, phase, **details)
    if action.kind == "draw":
        return action
    diamond.diamond_update_opening_position_history(cpu, room)
    if candidate is not None:
        diamond.diamond_remember_cards(cpu, diamond.candidate_consumed_cards(candidate))
    else:
        cpu.platinum_all_out_attempts += 1
    return diamond.platinum_commit_play(cpu, action)


def all_out(cpu, room, kind, **details):
    if room.field:
        payload = diamond.build_composite_practice_all_out_payload(
            cpu, room, rng=cpu.rng, require_invalid=True,
        )
        action = "play_composite"
    else:
        payload = diamond.build_gold_all_out_payload(cpu.hand, force_random=True, rng=cpu.rng)
        action = "play_prime"
    if payload is None:
        note(cpu, room, kind, "no-payload", **details)
        return None
    return commit(cpu, room, kind, "all-out", diamond.CpuAction(action, payload), **details)


def choose_full_recovery(cpu, room):
    if not room.field or getattr(room, "last_play_player_id", None) in (None, cpu.id):
        return None
    public = public_room(room)
    remaining_public = card_ids((*room.field, *(c for c in public.draw_queue if c is not None)))
    opponent_play = (*room.field, *getattr(room, "last_play_materials", ()))
    kings = {str(card["card_id"]) for card in opponent_play
             if card["rank"] == 13 and str(card["card_id"]) in remaining_public}
    if not kings or danger(cpu, room, "full"):
        return None

    search = ControlSearch(cpu, room)
    if search.resources is None:
        return None
    max_use = (len(cpu.hand) + 2 - len(public.draw_queue) - len(search.resources[3])) // 2
    if max_use < len(room.field):
        return None
    size = len(room.field)
    best = None
    for templates in templates_by_size(knowledge_key(cpu, room)).values():
        for template in templates:
            check(cpu)
            if len(template.visible) != size or len(template.ranks) > max_use:
                continue
            for move in realizations(template, search.hand):
                check(cpu)
                if move.used_count == len(cpu.hand) or uses_spare_natural(move, search.hand):
                    continue
                candidate = bind_move(move, cpu.hand)
                if not diamond.candidate_is_playable(candidate, cpu, room):
                    continue
                if search.advance(move, *search.resources) is None:
                    continue
                spent = diamond.candidate_consumed_cards(candidate)
                own_ids = card_ids(spent)
                materials = candidate.get("consume_cards", [])
                first = (public.draw_queue + list(materials) or [None])[0]
                # The other player may draw before passing. Display cards
                # are still locked, but already returned materials are not.
                if not move.cut and first is not None and str(first["card_id"]) in kings | own_ids:
                    continue
                score = (-move.used_count, -sum(is_kx(c) for c in spent), -move.value)
                if best is None or score > best[0]:
                    best = (score, candidate, own_ids)
    if best is None:
        return None
    _, candidate, own_ids = best
    cpu.master_kappa_recovery = dict(
        hand=card_ids(cpu.hand) - own_ids, targets=kings | own_ids,
        opponent_kings=kings, awaiting_draw=False,
    )
    return commit(cpu, room, "full", "control", diamond.candidate_to_action(candidate),
                  candidate=candidate, opponent_kings=sorted(kings),
                  targets=sorted(kings | own_ids))


def continue_recovery(cpu, room):
    pending = getattr(cpu, "master_kappa_recovery", None)
    cpu.master_kappa_recovery = None
    if pending is None:
        return None
    ids = card_ids(cpu.hand)
    expected = pending["hand"]
    valid = (expected < ids and len(ids) == len(expected) + 1 and room.has_drawn
             if pending["awaiting_draw"] else ids == expected and not room.has_drawn)
    known = card_ids(getattr(room, "public_known_deck_bottom", ()))
    if (room.reverse_order or room.field or not valid
            or not pending["targets"] <= known | ids):
        note(cpu, room, "full", "cancelled")
        return None
    if danger(cpu, room, "full"):
        return None
    details = dict(opponent_kings=sorted(pending["opponent_kings"]), targets=sorted(pending["targets"]))
    if room.deck and len(room.deck) <= len(cpu.hand):
        return all_out(cpu, room, "full", **details)
    if room.deck and not room.has_drawn and len(room.deck) <= len(cpu.hand) + 2:
        pending["awaiting_draw"] = True
        cpu.master_kappa_recovery = pending
        return commit(cpu, room, "full", "draw", diamond.CpuAction("draw"), **details)
    note(cpu, room, "full", "out-of-reach", **details)
    return None


def choose_gain_recovery(cpu, room):
    if not room.deck:
        return None
    public = public_room(room)
    draw = not room.has_drawn
    capacity = len(cpu.hand) + int(draw)
    if room.field and capacity < len(room.field) + 2:
        capacity = 0  # cannot build a whole-hand composite on this field
    prefix = public.draw_queue[:capacity + int(draw)]
    unknown_kx, _, _ = diamond.platinum_unaccounted_kx_distribution(cpu, room)
    known_kx = [str(card["card_id"]) for card in prefix if card is not None and is_kx(card)]
    unknown = any(card is None for card in prefix) and unknown_kx > 0
    if not (known_kx or unknown) or danger(cpu, room, "kx"):
        return None
    details = dict(known_kx=known_kx, unknown_kx_possible=unknown, reach=len(prefix))
    if draw:
        return commit(cpu, room, "kx", "draw", diamond.CpuAction("draw"), **details)
    return all_out(cpu, room, "kx", **details)


def choose_before_diamond(cpu, room):
    from .kappa_comeback import choose_comeback

    if (room.reverse_order or room.rule.key != "std-11-n-c"
            or diamond.opening_master_continuation_allowed(cpu, room)):
        return None
    for choose in (choose_full_recovery, choose_gain_recovery, choose_comeback):
        action = choose(cpu, room)
        if action is not None:
            return action
    return diamond.diamond_choose_normal_revolution(cpu, room, route="master-kappa-revolution-1729")

"""Shared structural guards; no strength weights or probability thresholds."""


def special_exposed(move, reverse=False):
    """Reject nonterminal rallies in the cut/revolution band, except own cuts."""
    if move.cut:
        return False
    limit = {2: 57, 4: 1729}.get(move.size)
    return limit is not None and (move.value >= limit if reverse else move.value <= limit)


def uses_spare_natural(move, held):
    """Count both display and material, so one natural card cannot be used twice."""
    if move.template.kind == 'cut':
        return False
    return any(physical == 0 and rank != 0 and held[rank] > move.need[rank]
               for rank, physical in zip(move.template.ranks, move.physical))


def natural_sequence(held, sequence):
    for move in sequence:
        if uses_spare_natural(move, held):
            return False
        held = tuple(a-b for a, b in zip(held, move.need))
        if min(held) < 0:
            return False
    return True


def naturalize_action(cpu, action):
    """Apply the same physical-card rule to the legacy Diamond fallback."""
    from copy import deepcopy
    from . import master_combined
    diamond = master_combined.diamond
    candidate = diamond.diamond_candidate_from_action(action)
    if candidate is None:
        return action
    candidate = deepcopy(candidate)
    used = {c['card_id'] for c in diamond.candidate_consumed_cards(candidate)}
    spare = [c for c in cpu.hand if c['card_id'] not in used and not diamond.is_joker(c)]
    replacements = {}

    def replace(cards, assignments):
        values = iter(assignments)
        result, remaining = [], []
        for card in cards:
            if diamond.is_joker(card):
                value = next(values)
                natural = next((c for c in spare if str(c['rank']) == str(value)), None)
                if natural is not None:
                    spare.remove(natural)
                    replacements[card['card_id']] = natural
                    card = natural
                else:
                    remaining.append(value)
            result.append(card)
        return result, remaining

    candidate['cards'], candidate['assigned_numbers'] = replace(
        candidate['cards'], candidate['assigned_numbers'])
    if candidate['kind'] == 'composite':
        material = {c['card_id']: c for c in candidate['consume_cards']}
        ordered = [material[t['card_id']] for t in candidate['composite_tokens'] if t['kind'] == 'card']
        _, candidate['composite_assigned_numbers'] = replace(ordered, candidate['composite_assigned_numbers'])
        candidate['consume_cards'] = [replacements.get(c['card_id'], c) for c in candidate['consume_cards']]
        for token in candidate['composite_tokens']:
            if token.get('card_id') in replacements:
                token['card_id'] = replacements[token['card_id']]['card_id']
    if not replacements:
        return action
    # Legacy cached physical IDs no longer describe the remaining hand.
    master_combined.clear_interrupted_plan(cpu)
    cpu.master_natural_cards_trace = dict(replaced_x_ids=list(replacements),
                                         natural_ids=[c['card_id'] for c in replacements.values()])
    return diamond.candidate_to_action(candidate)

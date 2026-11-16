"""Normal-order rally selection for Master kappa."""

import cpu_player as diamond


def catalog_return_risk(trump, holder, room, draw_allowance):
    cards = trump.get("cards", [])
    value = int(trump["number"])
    signatures = [
        signature
        for reply, signature in diamond.diamond_public_response_signatures(holder, len(cards))
        if reply > value
    ]
    if len(cards) == 4:
        signatures.extend(
            signature
            for token in diamond.DIAMOND_FOUR_OBAKE_TOKENS
            if diamond.platinum_token_value(token) > value
            for signature in diamond.diamond_counter_material_signatures(token)
        )
    return diamond.diamond_requirement_union_probability(
        holder, room, signatures, seed_value=0,
        sample_size=diamond.platinum_opponent_hand_count(holder, room) + draw_allowance,
    )


def trump_position(plan, cpu):
    index = diamond.gold_plan_trump_step_index(plan)
    steps = plan["steps"]
    trump = steps[index]
    # Earlier plays and materials may circulate before the opponent responds.
    released = [card for step in steps[:index] for card in diamond.candidate_consumed_cards(step)]
    display = {card["card_id"] for card in trump.get("cards", [])}
    released.extend(card for card in diamond.candidate_consumed_cards(trump)
                    if card["card_id"] not in display)
    holder = diamond.temporary_cpu_with_hand(cpu, diamond.remaining_cards(cpu.hand, released))
    return trump, holder, index, len(released)


def choose_rally_switch(plan, tier, eligible, cpu, room):
    """Switch away from a vulnerable four-card route at an empty field."""
    if tier not in (8, 9) or plan.get("rally_count") != 4:
        return None
    held_kx = sum(diamond.is_joker(card) or card["rank"] == 13 for card in cpu.hand)
    if held_kx > 2 or diamond.platinum_opponent_hand_count(cpu, room) is None:
        return None

    trump = diamond.diamond_plan_trump_candidate(plan)
    if trump is None or not any(
        diamond.platinum_token_value(token) > int(trump["number"])
        and diamond.diamond_counter_token_is_physically_possible(cpu, room, token)
        for token in diamond.DIAMOND_FOUR_OBAKE_TOKENS
    ):
        return None
    if diamond.diamond_has_current_certain_trump(cpu, room):
        return None

    protected_tail = diamond.diamond_plan_has_protected_x_tail(plan)
    rally_steps = diamond.diamond_plan_rally_step_count(plan)
    alternatives = [
        other for other in eligible
        if other.get("rally_count") in (3, 5, 6, 7, 8, 9)
        and diamond.diamond_plan_rally_step_count(other) <= rally_steps
        and (not protected_tail or diamond.diamond_plan_has_protected_x_tail(other))
    ]
    if not alternatives:
        return None
    safer = []
    original_risk = None
    for other in alternatives:
        if not diamond.diamond_post_all_out_search_has_time(cpu):
            break
        candidate, holder, index, released = trump_position(other, cpu)
        draws = min(index + 1, len(room.deck) + released)
        risk = catalog_return_risk(candidate, holder, room, draws)
        if original_risk is None:
            original, original_holder, original_index, original_released = trump_position(plan, cpu)
            original_draws = min(original_index + 1, len(room.deck) + original_released)
            original_risk = catalog_return_risk(original, original_holder, room, original_draws)
        if risk <= diamond.DIAMOND_CONDITIONAL_DEFAULT_MAX_RETURN_PROBABILITY and original_risk - risk >= 0.20:
            safer.append((risk, other))
    best = min(safer, default=None, key=lambda entry: entry[0])
    return best[1] if best else None

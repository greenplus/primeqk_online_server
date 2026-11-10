"""Knowledge-specific normal-order trump policy for Master CPU alpha."""
from collections import defaultdict
from functools import lru_cache

from registered_primes import (
    RegisteredCompositeEntry,
    RegisteredCompositeExpressionToken,
    registered_prime_pattern_value,
    tokenize_registered_prime_pattern,
)

CPU_KEY = "master_alpha_planner"

TRUMP_TOKENS = (
    "kkk kkq kkj "
    "kkkq kkkt kkqk kkqj kkqt kkjq "
    "kkkqj kkktt kkqkq kkqkj "
    "kkkktj kkkqqj kkkttj kkqqqj kkqqtj kkqjjj kkqtjj "
    "kkkkqtj kkkqqqj kkkqqjj kkkqqtk kkkjjjj kkkjtqj kkktttj kkqqjjj kkqjqtj kkjjqjj kkjtjtj "
    "kkkkqqtj kkkqjtqj kkkqtqtj kkkjqjtj kkkjjttj kkktjttj kkqqtqqk kkqqtqqj kkqqtqtk "
    "kkkkqtqtj kkkkjtqtj kkkqqttqj kkkqtttjj kkkqtjqjj kkqqtjtjj kkkqttjjj kkktqtqtj "
    "kkktqqjqj kkqqqtttj kkqtqqtjj kkqjjqqjj kkqqjqtqj "
    "9kkkqqtttj 9kkkqqqtjj 9kkqqttjjj 9kkqqttjqj 9kkkqttqqj 9kkktqjttj 9kkktjqtjj 9kkkqtjjqj"
).split()
TRUMP_VALUES = frozenset((len(token), registered_prime_pattern_value(token)) for token in TRUMP_TOKENS)

# The ordinary factor parser cannot encode the zero in 305377. Rank 0 in this
# trusted entry requires a physical joker assigned to zero, in the materials.
KKQJ_ENTRY = RegisteredCompositeEntry(
    source_line=0,
    pattern="kkqj",
    value=13131211,
    expression="43*3x5377",
    expression_tokens=(
        RegisteredCompositeExpressionToken("cards", ranks=(4, 3), text="43"),
        RegisteredCompositeExpressionToken("op", op="*", text="*"),
        RegisteredCompositeExpressionToken("cards", ranks=(3, 0, 5, 3, 7, 7), text="3x5377"),
    ),
)


@lru_cache(maxsize=8)
def trump_replies(primes, entries, allow_composite):
    """Group the specified replies by displayed count, with all material ranks."""
    material_by_value = defaultdict(set)
    if allow_composite:
        for entry in entries:
            material_by_value[entry.value].add(tuple(
                rank for token in entry.expression_tokens for rank in token.ranks
            ))
    groups = defaultdict(set)
    for token in TRUMP_TOKENS:
        value = registered_prime_pattern_value(token)
        ranks = tokenize_registered_prime_pattern(token)
        materials = ((),) if value in primes else material_by_value[value]
        for material in materials:
            combined = ranks + material
            groups[len(ranks)].add((value, tuple(combined.count(rank) for rank in range(14))))
    return {size: tuple(sorted(rows, reverse=True)) for size, rows in groups.items()}

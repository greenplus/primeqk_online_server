from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from math import factorial
import secrets
from typing import Callable, Iterable, Optional, Sequence


Card = dict
RandBelow = Callable[[int], int]
SAFE_LAST_DIGITS = frozenset("1379")


@dataclass(frozen=True)
class HnpToken:
    card: Card
    text: str
    assigned_number: Optional[str] = None

    @property
    def is_joker(self) -> bool:
        return self.assigned_number is not None

    @property
    def has_safe_last_digit(self) -> bool:
        return bool(self.text) and self.text[-1] in SAFE_LAST_DIGITS


@dataclass(frozen=True)
class HnpPermutation:
    tokens: tuple[HnpToken, ...]
    number: int

    @property
    def cards(self) -> list[Card]:
        return [token.card for token in self.tokens]

    @property
    def assigned_numbers(self) -> list[str]:
        return [
            token.assigned_number
            for token in self.tokens
            if token.assigned_number is not None
        ]


def is_joker_card(card: Card) -> bool:
    return bool(card.get("is_joker")) or card.get("suit") == "X"


def build_hnp_tokens(cards: Sequence[Card], assigned_numbers: Sequence[str]) -> tuple[HnpToken, ...]:
    tokens = []
    joker_index = 0
    for card in cards:
        if is_joker_card(card):
            if joker_index >= len(assigned_numbers):
                raise ValueError("joker assignment is missing")
            assigned = str(assigned_numbers[joker_index])
            joker_index += 1
            tokens.append(HnpToken(card=card, text=assigned, assigned_number=assigned))
        else:
            tokens.append(HnpToken(card=card, text=str(card.get("rank"))))
    if joker_index != len(assigned_numbers):
        raise ValueError("too many joker assignments")
    return tuple(tokens)


def hnp_tokens_number(tokens: Iterable[HnpToken]) -> int:
    text = "".join(token.text for token in tokens)
    if not text or text.startswith("0"):
        raise ValueError("HNP number must not start with zero")
    return int(text)


def _comparison_after_text(relation: int, position: int, text: str, field_text: str) -> int:
    if relation:
        return relation
    expected = field_text[position:position + len(text)]
    if text < expected:
        return -1
    if text > expected:
        return 1
    return 0


def _initial_relation(total_digits: int, field_text: Optional[str]) -> int:
    if field_text is None:
        return 1
    if total_digits < len(field_text):
        return -1
    if total_digits > len(field_text):
        return 1
    return 0


def _is_favorable(relation: int, reverse_order: bool) -> bool:
    return relation < 0 if reverse_order else relation > 0


def _counter_tuple(texts: tuple[str, ...], counts: Counter[str]) -> tuple[int, ...]:
    return tuple(counts[text] for text in texts)


def _make_prefix_counter(
    token_texts: Sequence[str],
    terminal_text: str,
    field_text: Optional[str],
    reverse_order: bool,
):
    texts = tuple(sorted(set(token_texts)))
    total_digits = sum(len(text) for text in token_texts) + len(terminal_text)
    favorable_relation = -1 if reverse_order else 1
    initial_relation = (
        favorable_relation
        if field_text is None
        else _initial_relation(total_digits, field_text)
    )

    @lru_cache(maxsize=None)
    def count(counts_tuple: tuple[int, ...], position: int, relation: int, first: bool) -> int:
        remaining_count = sum(counts_tuple)
        if relation == -favorable_relation:
            return 0
        if relation == favorable_relation:
            if remaining_count == 0:
                return 0 if first and terminal_text.startswith("0") else 1
            if first:
                nonzero_count = sum(
                    counts_tuple[index]
                    for index, text in enumerate(texts)
                    if not text.startswith("0")
                )
                return nonzero_count * factorial(remaining_count - 1)
            return factorial(remaining_count)

        if remaining_count == 0:
            if first and terminal_text.startswith("0"):
                return 0
            terminal_relation = _comparison_after_text(
                relation,
                position,
                terminal_text,
                field_text or "",
            )
            return int(_is_favorable(terminal_relation, reverse_order))

        total = 0
        for index, text in enumerate(texts):
            physical_count = counts_tuple[index]
            if physical_count == 0 or (first and text.startswith("0")):
                continue
            next_counts = list(counts_tuple)
            next_counts[index] -= 1
            next_relation = _comparison_after_text(
                relation,
                position,
                text,
                field_text or "",
            )
            total += physical_count * count(
                tuple(next_counts),
                position + len(text),
                next_relation,
                False,
            )
        return total

    return texts, initial_relation, count


def _terminal_options(
    tokens: Sequence[HnpToken],
    field_number: Optional[int],
    reverse_order: bool,
    safe_terminal_only: bool,
):
    pools: dict[str, list[HnpToken]] = defaultdict(list)
    for token in tokens:
        pools[token.text].append(token)
    field_text = str(field_number) if field_number is not None else None
    options = []
    for terminal_text, terminal_pool in pools.items():
        if safe_terminal_only and terminal_text[-1] not in SAFE_LAST_DIGITS:
            continue
        remaining_texts = [token.text for token in tokens]
        remaining_texts.remove(terminal_text)
        remaining_counts = Counter(remaining_texts)
        texts, initial_relation, counter = _make_prefix_counter(
            remaining_texts,
            terminal_text,
            field_text,
            reverse_order,
        )
        prefix_count = counter(
            _counter_tuple(texts, remaining_counts),
            0,
            initial_relation,
            True,
        )
        if prefix_count:
            options.append({
                "terminal_text": terminal_text,
                "terminal_pool": terminal_pool,
                "remaining_counts": remaining_counts,
                "texts": texts,
                "initial_relation": initial_relation,
                "counter": counter,
                "prefix_count": prefix_count,
                "weight": len(terminal_pool) * prefix_count,
            })
    return options


def count_legal_hnp_permutations(
    tokens: Sequence[HnpToken],
    field_number: Optional[int] = None,
    reverse_order: bool = False,
    safe_terminal_only: bool = False,
) -> int:
    return sum(
        option["weight"]
        for option in _terminal_options(
            tokens,
            field_number,
            reverse_order,
            safe_terminal_only,
        )
    )


def _weighted_choice(options: Sequence[dict], randbelow: RandBelow) -> dict:
    total = sum(option["weight"] for option in options)
    pick = randbelow(total)
    for option in options:
        if pick < option["weight"]:
            return option
        pick -= option["weight"]
    raise RuntimeError("weighted HNP choice fell through")


def choose_hnp_permutation(
    tokens: Sequence[HnpToken],
    field_number: Optional[int] = None,
    reverse_order: bool = False,
    randbelow: RandBelow = secrets.randbelow,
) -> Optional[HnpPermutation]:
    if len(tokens) < 2:
        return None

    options = _terminal_options(tokens, field_number, reverse_order, safe_terminal_only=True)
    if not options:
        options = _terminal_options(tokens, field_number, reverse_order, safe_terminal_only=False)
    if not options:
        return None

    terminal_option = _weighted_choice(options, randbelow)
    terminal_pool = terminal_option["terminal_pool"]
    terminal = terminal_pool[randbelow(len(terminal_pool))]

    pools: dict[str, list[HnpToken]] = defaultdict(list)
    terminal_removed = False
    for token in tokens:
        if token is terminal and not terminal_removed:
            terminal_removed = True
            continue
        pools[token.text].append(token)

    texts = terminal_option["texts"]
    counts = terminal_option["remaining_counts"].copy()
    counter = terminal_option["counter"]
    relation = terminal_option["initial_relation"]
    position = 0
    first = True
    ordered = []

    while sum(counts.values()):
        branches = []
        counts_tuple = _counter_tuple(texts, counts)
        for index, text in enumerate(texts):
            physical_count = counts_tuple[index]
            if physical_count == 0 or (first and text.startswith("0")):
                continue
            next_counts = list(counts_tuple)
            next_counts[index] -= 1
            next_relation = _comparison_after_text(
                relation,
                position,
                text,
                str(field_number) if field_number is not None else "",
            )
            completion_count = counter(
                tuple(next_counts),
                position + len(text),
                next_relation,
                False,
            )
            if completion_count:
                branches.append({
                    "text": text,
                    "next_relation": next_relation,
                    "completion_count": completion_count,
                    "weight": physical_count * completion_count,
                })
        branch = _weighted_choice(branches, randbelow)
        text = branch["text"]
        pool = pools[text]
        token = pool.pop(randbelow(len(pool)))
        ordered.append(token)
        counts[text] -= 1
        relation = branch["next_relation"]
        position += len(text)
        first = False

    ordered.append(terminal)
    ordered_tuple = tuple(ordered)
    return HnpPermutation(tokens=ordered_tuple, number=hnp_tokens_number(ordered_tuple))

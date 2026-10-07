# rules.py
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict

class DeckRule(Enum):
    DEFAULT = auto()       # 通常の54枚デッキ
    EVEN_HALVED = auto()   # 偶数カードを半分に間引く
    EVEN_HALVED_WITH_CHEFS = auto()
    RANK_MULTIPLICITY = auto()  # A～Kはランクnごとにn枚、ジョーカー2枚の93枚
    REVERSE_RANK_MULTIPLICITY = auto()  # A13枚～K1枚、X2枚

class PenaltyRule(Enum):
    ALWAYS_1 = auto()      # 必ず1枚
    FIELD_COUNT = auto()   # 場の枚数
    NORMAL = auto()        # 通常（合成数では材料札も含む）

class PrimeRule(Enum):
    NORMAL = auto()       # 通常の素数
    TETRAD = auto()       # 四つ子素数
    SEMIPRIME = auto()    # 半素数
    REGISTERED = auto()   # プレイヤーごとの登録済み素数

class MovePolicy(Enum):
    STANDARD = auto()
    COMPOSITE_ONLY_WITH_SMALL_HAND_FINISH = auto()

@dataclass(frozen=True)
class RulePreset:
    key: str
    label: str
    deck_rule: DeckRule
    hand_size: int
    penalty_rule: PenaltyRule
    allow_composite: bool = False
    start_revolution: bool = False
    prime_rule: PrimeRule = PrimeRule.NORMAL
    assist_enabled: bool = False
    registration_enabled: bool = False
    hnp_challenge_enabled: bool = False
    registered_number_limit: int | None = None
    move_policy: MovePolicy = MovePolicy.STANDARD
    normal_finish_max_hand_size: int = 0
    special_numbers_composite_only: bool = False
    cpu_profile_keys: tuple[str, ...] = ()
    burst_hand_size: int | None = None
    second_joker_guarantee: bool = False
    match_digits: bool = False
    kjqj_conversion: bool = False
    hyakki: bool = False
    description: str = ""

PRESETS: Dict[str, RulePreset] = {
    "hyakki-second-x": RulePreset(
        key="hyakki-second-x", label="後手のみX1枚保証",
        deck_rule=DeckRule.DEFAULT, hand_size=11, penalty_rule=PenaltyRule.NORMAL,
        allow_composite=True, second_joker_guarantee=True, hyakki=True,
        description="後手に10枚配り、Xがなければ残り山札からXを1枚、あれば山札の先頭を1枚追加。その後、先手に11枚配ります。通常54枚・通常ペナルティ・合成数あり。",
    ),
    "hyakki-reverse-n": RulePreset(
        key="hyakki-reverse-n", label="nが(14-n)枚ある",
        deck_rule=DeckRule.REVERSE_RANK_MULTIPLICITY, hand_size=11,
        penalty_rule=PenaltyRule.NORMAL, allow_composite=True, burst_hand_size=32, hyakki=True,
        description="1が13枚～13が1枚、X2枚の計93枚。初期11枚・通常ペナルティ・合成数あり。手札32枚でバースト負け。",
    ),
    "hyakki-same-digits": RulePreset(
        key="hyakki-same-digits", label="同じ桁数を出す",
        deck_rule=DeckRule.DEFAULT, hand_size=11, penalty_rule=PenaltyRule.NORMAL,
        allow_composite=True, match_digits=True, hyakki=True, cpu_profile_keys=("basic",),
        description="枚数ではなく同じ桁数で大小を比較します。57は2桁、1729は4桁。単独Xは空場か1桁の場のみ。通常54枚・初期11枚・通常ペナルティ・合成数あり。",
    ),
    "hyakki-kjqj": RulePreset(
        key="hyakki-kjqj", label="4枚出しをKJQJに変換",
        deck_rule=DeckRule.DEFAULT, hand_size=11, penalty_rule=PenaltyRule.NORMAL,
        allow_composite=True, kjqj_conversion=True, hyakki=True, cpu_profile_keys=("basic",),
        description="出す直前に自分12枚以下・相手13枚以上なら、任意の4枚の素数出しをKJQJに変換できます。何度でも使用可。元の数が素数でなければペナルティ。合成数出しは対象外。流れた後は元の札に戻ります。",
    ),
    "hyakki-yagyo-11-n-c-b32": RulePreset(
        key="hyakki-yagyo-11-n-c-b32",
        label="素数大富豪百鬼夜行: 11枚 / nがn枚 / 32枚バースト",
        deck_rule=DeckRule.RANK_MULTIPLICITY,
        hand_size=11,
        penalty_rule=PenaltyRule.NORMAL,
        allow_composite=True,
        burst_hand_size=32,
        hyakki=True,
    ),
    "std-5-1": RulePreset(
        key="std-5-1",
        label="5枚 / ペナ1",
        deck_rule=DeckRule.DEFAULT,
        hand_size=5,
        penalty_rule=PenaltyRule.ALWAYS_1,
        allow_composite=True,
    ),
    "std-7-1": RulePreset(
        key="std-7-1",
        label="7枚 / ペナ1",
        deck_rule=DeckRule.DEFAULT,
        hand_size=7,
        penalty_rule=PenaltyRule.ALWAYS_1,
        allow_composite=True,
    ),
    "std-11-f": RulePreset(
        key="std-11-f",
        label="標準: 11枚 / 場の枚数",
        deck_rule=DeckRule.DEFAULT,
        hand_size=11,
        penalty_rule=PenaltyRule.FIELD_COUNT,
        allow_composite=False,
    ),
    "std-11-f-c": RulePreset(
        key="std-11-f-c",
        label="11枚 / ペナ場の枚数",
        deck_rule=DeckRule.DEFAULT,
        hand_size=11,
        penalty_rule=PenaltyRule.FIELD_COUNT,
        allow_composite=True,
    ),
    "std-11-n-c": RulePreset(
        key="std-11-n-c",
        label="11枚 / 通常",
        deck_rule=DeckRule.DEFAULT,
        hand_size=11,
        penalty_rule=PenaltyRule.NORMAL,
        allow_composite=True,
    ),
    "std-11-n-no-c": RulePreset(
        key="std-11-n-no-c",
        label="11枚 / 合成数なし",
        deck_rule=DeckRule.DEFAULT,
        hand_size=11,
        penalty_rule=PenaltyRule.NORMAL,
        allow_composite=False,
    ),
    "std-11-n-c-rev": RulePreset(
        key="std-11-n-c-rev",
        label="初期革命: 11枚 / 通常",
        deck_rule=DeckRule.DEFAULT,
        hand_size=11,
        penalty_rule=PenaltyRule.NORMAL,
        allow_composite=True,
        start_revolution=True,
    ),
    "half-5-f": RulePreset(
        key="half-5-f",
        label="偶数半減: 5枚 / 場の枚数",
        deck_rule=DeckRule.EVEN_HALVED,
        hand_size=5,
        penalty_rule=PenaltyRule.FIELD_COUNT,
        allow_composite=False,
    ),
    "half-7-1-c": RulePreset(
        key="half-7-1-c",
        label="初級: 7枚 / 偶数半減 / ペナ1",
        deck_rule=DeckRule.EVEN_HALVED,
        hand_size=7,
        penalty_rule=PenaltyRule.ALWAYS_1,
        allow_composite=True,
    ),
    "half-7-1-c-assist": RulePreset(
        key="half-7-1-c-assist",
        label="初級: 7枚 / 偶数半減 / ペナルティ1枚 / 登録制限",
        deck_rule=DeckRule.EVEN_HALVED,
        hand_size=7,
        penalty_rule=PenaltyRule.ALWAYS_1,
        allow_composite=True,
        prime_rule=PrimeRule.REGISTERED,
        assist_enabled=True,
        registration_enabled=True,
        hnp_challenge_enabled=True,
        registered_number_limit=500,
    ),
    "event-chef-11-1-c": RulePreset(
        hyakki=True,
        key="event-chef-11-1-c",
        label="偶数の半分がコックさんに / 11枚 / ペナ1",
        deck_rule=DeckRule.EVEN_HALVED_WITH_CHEFS,
        hand_size=11,
        penalty_rule=PenaltyRule.ALWAYS_1,
        allow_composite=True,
    ),
    "tetrad-11-n": RulePreset(
        key="tetrad-11-n",
        label="四つ子素数: 11枚 / 通常",
        deck_rule=DeckRule.DEFAULT,
        hand_size=11,
        penalty_rule=PenaltyRule.NORMAL,
        allow_composite=False,
        prime_rule=PrimeRule.TETRAD,
    ),
    "tetrad-11-n-c": RulePreset(
        key="tetrad-11-n-c",
        label="四つ子素数: 11枚 / 通常",
        deck_rule=DeckRule.DEFAULT,
        hand_size=11,
        penalty_rule=PenaltyRule.NORMAL,
        allow_composite=True,
        prime_rule=PrimeRule.TETRAD,
    ),
    "semiprime-11-n": RulePreset(
        key="semiprime-11-n",
        label="半素数: 11枚 / 通常",
        deck_rule=DeckRule.DEFAULT,
        hand_size=11,
        penalty_rule=PenaltyRule.NORMAL,
        allow_composite=False,
        prime_rule=PrimeRule.SEMIPRIME,
    ),
    "semiprime-11-n-c": RulePreset(
        key="semiprime-11-n-c",
        label="半素数: 11枚 / 通常",
        deck_rule=DeckRule.DEFAULT,
        hand_size=11,
        penalty_rule=PenaltyRule.NORMAL,
        allow_composite=True,
        prime_rule=PrimeRule.SEMIPRIME,
    ),
    "semiprime-11-1-c": RulePreset(
        key="semiprime-11-1-c",
        label="半素数: 11枚 / ペナルティ1枚",
        deck_rule=DeckRule.DEFAULT,
        hand_size=11,
        penalty_rule=PenaltyRule.ALWAYS_1,
        allow_composite=True,
        prime_rule=PrimeRule.SEMIPRIME,
    ),
    "registered-11-n": RulePreset(
        key="registered-11-n",
        label="登録制限: 11枚 / 通常 / アシストなし",
        deck_rule=DeckRule.DEFAULT,
        hand_size=11,
        penalty_rule=PenaltyRule.NORMAL,
        allow_composite=True,
        prime_rule=PrimeRule.REGISTERED,
        registration_enabled=True,
        hnp_challenge_enabled=True,
    ),
    "registered-11-n-assist": RulePreset(
        key="registered-11-n-assist",
        label="登録制限: 11枚 / 通常 / アシストあり",
        deck_rule=DeckRule.DEFAULT,
        hand_size=11,
        penalty_rule=PenaltyRule.NORMAL,
        allow_composite=True,
        prime_rule=PrimeRule.REGISTERED,
        assist_enabled=True,
        registration_enabled=True,
        hnp_challenge_enabled=True,
    ),
    "neo-assist-11-n-unlimited": RulePreset(
        key="neo-assist-11-n-unlimited",
        label="登録アシスト: 11枚 / 通常 / 制限なし",
        deck_rule=DeckRule.DEFAULT,
        hand_size=11,
        penalty_rule=PenaltyRule.NORMAL,
        allow_composite=True,
        assist_enabled=True,
        registration_enabled=True,
    ),
    "composite-practice-11-n": RulePreset(
        key="composite-practice-11-n",
        label="合成数練習: 11枚 / 通常",
        deck_rule=DeckRule.DEFAULT,
        hand_size=11,
        penalty_rule=PenaltyRule.NORMAL,
        allow_composite=True,
        assist_enabled=True,
        registration_enabled=True,
        move_policy=MovePolicy.COMPOSITE_ONLY_WITH_SMALL_HAND_FINISH,
        normal_finish_max_hand_size=3,
        special_numbers_composite_only=True,
        cpu_profile_keys=("composite_practice",),
    ),
}

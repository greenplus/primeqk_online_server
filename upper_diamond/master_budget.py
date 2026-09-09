"""One cooperative deadline across every Master stage and virtual draw."""
from contextvars import ContextVar
from dataclasses import dataclass
import time


class SearchLimit(Exception):
    pass


class MasterDeadline(SearchLimit):
    pass


@dataclass(frozen=True)
class TimingConfig:
    search_ms: int = 6000
    fallback_ms: int = 1500
    total_ms: int = 8000

    def __post_init__(self):
        if min(self.search_ms, self.fallback_ms) < 0 or self.total_ms <= 0:
            raise ValueError('Master time budgets must be nonnegative with a positive total')
        if self.search_ms + self.fallback_ms > self.total_ms:
            raise ValueError('Master and fallback budgets exceed the total')


@dataclass
class Budget:
    started: float
    config: TimingConfig
    expired: bool = False
    fallback_deadline: float | None = None

    @property
    def search_deadline(self):
        return self.started + self.config.search_ms / 1000

    @property
    def total_deadline(self):
        return self.started + self.config.total_ms / 1000


CURRENT = ContextVar('master_budget', default=None)


def check(cpu=None):
    budget = CURRENT.get()
    deadline = budget.search_deadline if budget else getattr(cpu, 'master_search_deadline', None)
    if deadline is not None and time.perf_counter() >= deadline:
        if budget:
            budget.expired = True
        raise MasterDeadline

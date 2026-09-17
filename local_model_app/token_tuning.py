from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable


@dataclass(frozen=True)
class BudgetTrial:
    budget: int
    attempts: int
    passes: int
    required_passes: int

    @property
    def passed(self) -> bool:
        return self.passes >= self.required_passes


@dataclass(frozen=True)
class BudgetSearchResult:
    candidates: list[int]
    selected_budget: int | None
    trials: list[BudgetTrial]

    def as_dict(self) -> dict:
        return {
            "candidates": self.candidates,
            "selected_budget": self.selected_budget,
            "trials": [asdict(trial) | {"passed": trial.passed} for trial in self.trials],
        }


def bracketed_budget_search(
    candidates: list[int],
    evaluate: Callable[[int], bool],
    *,
    repeats: int = 3,
    required_passes: int | None = None,
) -> BudgetSearchResult:
    """Find a conservative minimum passing token ceiling on an ordered candidate grid.

    The search assumes quality is broadly monotonic, then confirms the selected boundary
    and its immediate neighbors. Results are cached so expensive model evaluations are
    never repeated for the same candidate within one search.
    """
    if not candidates:
        raise ValueError("At least one budget candidate is required.")
    if candidates != sorted(set(candidates)) or any(value < 1 for value in candidates):
        raise ValueError("Budget candidates must be unique positive integers in ascending order.")
    if repeats < 1:
        raise ValueError("repeats must be at least 1.")
    threshold = repeats if required_passes is None else required_passes
    if not 1 <= threshold <= repeats:
        raise ValueError("required_passes must be between 1 and repeats.")

    observed: dict[int, BudgetTrial] = {}

    def trial(index: int) -> BudgetTrial:
        budget = candidates[index]
        if budget not in observed:
            passes = sum(bool(evaluate(budget)) for _ in range(repeats))
            observed[budget] = BudgetTrial(
                budget=budget,
                attempts=repeats,
                passes=passes,
                required_passes=threshold,
            )
        return observed[budget]

    highest = len(candidates) - 1
    if not trial(highest).passed:
        return BudgetSearchResult(
            candidates=list(candidates),
            selected_budget=None,
            trials=sorted(observed.values(), key=lambda item: item.budget),
        )

    low = 0
    high = highest
    while low < high:
        middle = (low + high) // 2
        if trial(middle).passed:
            high = middle
        else:
            low = middle + 1

    selected_index = low
    # Confirm both sides of the boundary. If the lower neighbor also passes, walk
    # downward until the first observed failure so a coarse initial bracket remains safe.
    if selected_index < highest:
        trial(selected_index + 1)
    while selected_index > 0 and trial(selected_index - 1).passed:
        selected_index -= 1
    if selected_index > 0:
        trial(selected_index - 1)

    return BudgetSearchResult(
        candidates=list(candidates),
        selected_budget=candidates[selected_index],
        trials=sorted(observed.values(), key=lambda item: item.budget),
    )

"""Objective-agnostic feasibility checker and objective evaluator.

This is the single source of truth for what a *valid* schedule is and what its
makespan / idle-time values are. Both the exact solver and every heuristic are
scored through here, so the comparison is apples-to-apples regardless of how a
heuristic represents its internal state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .model import Instance, Solution

EPS = 1e-6


@dataclass
class EvalResult:
    feasible: bool
    complete: bool                 # every ship placed exactly once
    errors: List[str] = field(default_factory=list)
    makespan: float = float("inf")
    idle: float = 0.0
    placed: int = 0

    def objective(self, which: str) -> float:
        if which == "makespan":
            return self.makespan
        if which == "idle":
            return self.idle
        raise ValueError(f"unknown objective {which!r}")

    @property
    def ok(self) -> bool:
        """Usable for scoring: feasible AND all ships placed."""
        return self.feasible and self.complete and not self.errors


def _in_some_tide(instance: Instance, t: float) -> bool:
    return any(td.s - EPS <= t <= td.e + EPS for td in instance.tides)


def evaluate(instance: Instance, solution: Solution) -> EvalResult:
    """Validate a Solution against the BAPT constraints and compute objectives.

    Constraints checked (mirroring the MILP):
      * each ship placed exactly once;
      * entry l and exit r each fall inside some tide window;
      * r >= l + a  (attention fits);
      * no two ships on the same berth overlap in [l, r).
    """
    errors: List[str] = []
    by_ship = solution.by_ship()

    placed = len(by_ship)
    if placed != len(solution.assignments):
        errors.append("a ship is assigned more than once")

    complete = set(by_ship) == set(instance.ship_ids)
    missing = set(instance.ship_ids) - set(by_ship)
    if missing:
        errors.append(f"ships not placed: {sorted(missing)}")

    idle = 0.0
    makespan = 0.0
    for sid, asg in by_ship.items():
        a = instance.a(sid)
        if asg.berth < 0 or asg.berth >= instance.berths:
            errors.append(f"ship {sid}: berth {asg.berth} out of range")
        if not _in_some_tide(instance, asg.l):
            errors.append(f"ship {sid}: entry {asg.l:.3f} not within any tide")
        if not _in_some_tide(instance, asg.r):
            errors.append(f"ship {sid}: exit {asg.r:.3f} not within any tide")
        if asg.r < asg.l + a - EPS:
            errors.append(f"ship {sid}: attention {a} does not fit in [{asg.l:.3f},{asg.r:.3f}]")
        idle += max(0.0, asg.r - asg.l - a)
        makespan = max(makespan, asg.r)

    # No overlap per berth.
    for k in range(instance.berths):
        onk = sorted((a for a in by_ship.values() if a.berth == k), key=lambda x: x.l)
        for u, v in zip(onk, onk[1:]):
            if v.l < u.r - EPS:
                errors.append(f"berth {k}: overlap between ships {u.ship_id} and {v.ship_id}")

    feasible = not errors
    return EvalResult(
        feasible=feasible,
        complete=complete,
        errors=errors,
        makespan=makespan if by_ship else float("inf"),
        idle=idle,
        placed=placed,
    )

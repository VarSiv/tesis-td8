"""Data model for BAPT instances and solutions, plus .dat I/O.

The .dat format matches what the ZIMPL model (`tides.zpl`) and the Benders Java
solver read:

    ships.dat            tides.dat
    -----------          ----------------
    <berths>             <t> <start> <end>
    <id> <attention>     <t> <start> <end>
    <id> <attention>     ...
    ...

Ship ids and tide ids are 1-based and contiguous (1..n, 1..T), matching the
generator and the solvers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class Tide:
    """A high-tide window [s, e]; ships may only enter/leave within a window."""
    id: int
    s: float
    e: float


@dataclass
class Instance:
    """A BAPT instance.

    Attributes:
        berths:     number of berths (amarraderos), m.
        attention:  attention[i] is the service time a_i of ship i+1 (0-based list,
                    ship ids are 1..n).
        tides:      list of Tide windows, ids 1..T.
        name:       optional label for reporting.
    """
    berths: int
    attention: List[float]
    tides: List[Tide]
    name: str = ""

    # --- convenient views -------------------------------------------------
    @property
    def n_ships(self) -> int:
        return len(self.attention)

    @property
    def n_tides(self) -> int:
        return len(self.tides)

    @property
    def ship_ids(self) -> List[int]:
        return list(range(1, self.n_ships + 1))

    def a(self, ship_id: int) -> float:
        return self.attention[ship_id - 1]

    def tide(self, tide_id: int) -> Tide:
        return self.tides[tide_id - 1]

    @property
    def horizon(self) -> float:
        """mu: latest tide end (upper bound / big-M horizon)."""
        return max(t.e for t in self.tides)

    def summary(self) -> str:
        return (f"n={self.n_ships} m={self.berths} T={self.n_tides} "
                f"mu={self.horizon:.2f}")

    # --- .dat I/O ---------------------------------------------------------
    def write_dat(self, ships_path: str, tides_path: str) -> None:
        with open(ships_path, "w") as f:
            f.write(f"{self.berths}\n")
            for i, a in enumerate(self.attention, start=1):
                f.write(f"{i}\t{_fmt(a)}\n")
        with open(tides_path, "w") as f:
            for t in self.tides:
                f.write(f"{t.id}\t{_fmt(t.s)}\t{_fmt(t.e)}\n")

    @staticmethod
    def read_dat(ships_path: str, tides_path: str, name: str = "") -> "Instance":
        with open(ships_path) as f:
            rows = [ln.split() for ln in f if ln.strip()]
        berths = int(float(rows[0][0]))
        attention = [float(r[1]) for r in rows[1:]]

        with open(tides_path) as f:
            tides = [Tide(int(float(r[0])), float(r[1]), float(r[2]))
                     for r in (ln.split() for ln in f if ln.strip())]
        tides.sort(key=lambda t: t.s)
        return Instance(berths, attention, tides, name=name or ships_path)


def _fmt(x: float) -> str:
    """Write integers without a trailing .0 (keeps .dat files tidy)."""
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return repr(x)


@dataclass(frozen=True)
class Assignment:
    """One ship's placement in a schedule."""
    ship_id: int
    berth: int          # 0-based berth index
    l: float            # entry instant (must fall inside a tide window)
    r: float            # exit instant  (must fall inside a tide window)
    tide_in: Optional[int] = None
    tide_out: Optional[int] = None

    @property
    def idle(self) -> float:
        # idle contribution needs the attention time; computed in checker.
        raise NotImplementedError("use EvalResult.idle")


@dataclass
class Solution:
    """A candidate schedule produced by a heuristic (or the exact solver).

    `assignments` need not cover every ship: a heuristic may fail to place some
    (an infeasible/partial solution). The checker reports completeness.
    """
    assignments: List[Assignment] = field(default_factory=list)
    meta: dict = field(default_factory=dict)   # free-form (e.g. iterations)

    def by_ship(self) -> dict:
        return {a.ship_id: a for a in self.assignments}

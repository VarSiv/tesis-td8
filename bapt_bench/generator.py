"""Instance generator.

Pure-Python reimplementation of the group's `generator.cpp` semantics:

    tides:  current = 0
            for each tide t:
                current += U(0, 10)                 # gap before the window
                length  = factor * U(0, 1)          # window width
                tide    = [current, current + length]
                current += length
    ships:  attention_i = randint(5, 24)

It is *seeded and reproducible* but not bit-identical to the C++ `rand()`
stream (a different RNG). That is fine for benchmarking: what matters is a
controlled, reproducible distribution of instances.
"""

from __future__ import annotations

import random
from typing import List

from .model import Instance, Tide


def generate(berths: int, ships: int, tides: int, factor: float = 20.0,
             seed: int = 0, name: str = "") -> Instance:
    """Generate one instance.

    Args:
        berths:  number of berths (m).
        ships:   number of ships (n).
        tides:   number of tide windows (T).
        factor:  scales tide-window width; larger => wider high-tide windows
                 (easier to fit ships with little/no idle).
        seed:    RNG seed for reproducibility.
        name:    optional label.
    """
    rng = random.Random(seed)

    tide_list: List[Tide] = []
    current = 0.0
    for i in range(1, tides + 1):
        current += rng.uniform(0.0, 10.0)
        length = factor * rng.uniform(0.0, 1.0)
        tide_list.append(Tide(i, round(current, 4), round(current + length, 4)))
        current += length

    attention = [float(rng.randint(5, 24)) for _ in range(ships)]

    label = name or f"gen_b{berths}_n{ships}_t{tides}_f{factor:g}_s{seed}"
    return Instance(berths, attention, tide_list, name=label)


def generate_many(specs, base_seed: int = 0) -> List[Instance]:
    """Generate a batch of instances.

    `specs` is an iterable of dicts, each with keys accepted by `generate`
    (berths, ships, tides, factor, ...) except `seed`, which is derived from
    `base_seed + index` unless the spec provides its own `seed`.

    Example:
        generate_many([
            dict(berths=2, ships=5, tides=4, factor=20),
            dict(berths=3, ships=8, tides=5, factor=15),
        ], base_seed=100)
    """
    out = []
    for idx, spec in enumerate(specs):
        spec = dict(spec)
        spec.setdefault("seed", base_seed + idx)
        out.append(generate(**spec))
    return out

"""CLI for the BAPT benchmark harness.

Examples:
    # Benchmark all registered heuristics on a generated batch, makespan objective
    python -m bapt_bench --objective makespan --n 5 --m 2 --t 4 --count 10

    # Fair comparison of the idle-minimising heuristic against the idle optimum
    python -m bapt_bench --objective idle --count 20 --csv results_idle.csv

    # Sweep sizes and pick heuristics explicitly
    python -m bapt_bench --sizes 5x2x4,8x3x5,10x3x6 --count 5 --heuristics greedy,grasp

    # Score a single existing .dat pair
    python -m bapt_bench --ships ../ships.dat --tides ../tides.dat --objective makespan
"""

from __future__ import annotations

import argparse
import sys
from typing import List

from .model import Instance
from .generator import generate
from .heuristics import all_heuristics
from .benchmark import run_benchmark, format_table, format_summary, write_csv


def _parse_sizes(spec: str):
    """'5x2x4,8x3x5' -> [(n,m,t), ...] as ships x berths x tides."""
    out = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        n, m, t = (int(v) for v in chunk.lower().split("x"))
        out.append((n, m, t))
    return out


def build_instances(args) -> List[Instance]:
    if args.ships and args.tides:
        return [Instance.read_dat(args.ships, args.tides, name=args.ships)]

    sizes = _parse_sizes(args.sizes) if args.sizes else [(args.n, args.m, args.t)]
    instances = []
    seed = args.seed
    for (n, m, t) in sizes:
        for i in range(args.count):
            instances.append(generate(berths=m, ships=n, tides=t,
                                       factor=args.factor, seed=seed))
            seed += 1
    return instances


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="bapt_bench", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--objective", choices=["makespan", "idle"], default="makespan",
                   help="objective for BOTH exact solver and scoring (default: makespan)")
    p.add_argument("--heuristics", default="",
                   help="comma-separated heuristic names (default: all registered)")

    # single existing instance
    p.add_argument("--ships", help="path to a ships.dat to score directly")
    p.add_argument("--tides", help="path to a tides.dat to score directly")

    # generated batch
    p.add_argument("--n", type=int, default=5, help="ships (default 5)")
    p.add_argument("--m", type=int, default=2, help="berths (default 2)")
    p.add_argument("--t", type=int, default=4, help="tides (default 4)")
    p.add_argument("--sizes", help="e.g. 5x2x4,8x3x5 (ships x berths x tides); overrides --n/--m/--t")
    p.add_argument("--count", type=int, default=5, help="instances per size (default 5)")
    p.add_argument("--factor", type=float, default=20.0, help="tide-width factor (default 20)")
    p.add_argument("--seed", type=int, default=0, help="base RNG seed")

    p.add_argument("--time-limit", type=float, default=120.0, help="exact solver time limit (s)")
    p.add_argument("--csv", help="write per-row results to this CSV path")
    p.add_argument("--list", action="store_true", help="list registered heuristics and exit")
    args = p.parse_args(argv)

    if args.list:
        print("Registered heuristics:", ", ".join(sorted(all_heuristics())))
        return 0

    heuristics = [h.strip() for h in args.heuristics.split(",") if h.strip()] or None
    instances = build_instances(args)

    rows = run_benchmark(instances, objective=args.objective,
                         heuristics=heuristics, time_limit=args.time_limit)

    print("\n" + format_table(rows))
    print(format_summary(rows))

    if args.csv:
        write_csv(rows, args.csv)
        print(f"\nWrote {len(rows)} rows to {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

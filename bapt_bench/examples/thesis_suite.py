"""Larger benchmark suite for the thesis write-up.

Generates a varied set of instances (sizes x tide-width factors x seeds), runs
every heuristic against the exact optimum for BOTH objectives (makespan and
idle), writes per-instance CSVs, and prints breakdowns:
  * overall per-heuristic summary,
  * per-(size,factor) group breakdown,
  * a compact per-instance table.

Run:  python -m bapt_bench.examples.thesis_suite [out_dir]
"""

from __future__ import annotations

import sys
from collections import defaultdict

import bapt_bench as bb
import bapt_bench.heuristica_var        # noqa: F401  registers golosa_var
from bapt_bench.benchmark import (format_table, format_summary, write_csv,
                                  aggregate)

HEURISTICS = ["golosa_var", "greedy", "grasp"]

# --- instance variety -----------------------------------------------------
SIZES = [   # (ships, berths, tides)
    (5, 2, 4),
    (6, 2, 4),
    (7, 2, 5),
    (8, 3, 5),
    (9, 3, 6),
    (10, 3, 6),
]
FACTORS = [12, 20, 30]     # tide-window width: tighter -> harder, more idle
SEEDS = 3                  # instances per (size, factor)
TIME_LIMIT = 20.0


def build_instances():
    specs = []
    for (n, m, t) in SIZES:
        for f in FACTORS:
            for _ in range(SEEDS):
                specs.append(dict(berths=m, ships=n, tides=t, factor=f))
    return bb.generate_many(specs, base_seed=0)


def group_key(row):
    return f"n{row.n}_m{row.m}_t{row.t}"


def factor_of(name):          # names look like gen_b2_n5_t4_f20_s0
    for part in name.split("_"):
        if part.startswith("f"):
            return part[1:]
    return "?"


def per_group_breakdown(rows):
    """mean gap% and feasible% per (size x factor) group, per heuristic."""
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (f"n{r.n}/m{r.m}/t{r.t}", f"f{factor_of(r.instance)}")
        groups[key][r.heuristic].append(r)

    lines = ["\n--- per (size, factor) breakdown ---",
             f"{'size':<12}{'factor':<7}{'heur':<12}{'feas%':>6}{'scored':>7}{'gap%':>8}{'opt%':>6}"]
    for gkey in sorted(groups):
        size, factor = gkey
        for h in HEURISTICS:
            rs = groups[gkey].get(h, [])
            if not rs:
                continue
            feas = [r for r in rs if r.feasible and r.complete]
            scored = [r for r in rs if r.gap_pct is not None and r.exact_status == "optimal"]
            gaps = [r.gap_pct for r in scored]
            mg = f"{sum(gaps)/len(gaps):.2f}" if gaps else "-"
            optp = (f"{100*sum(1 for r in scored if r.is_optimal)/len(scored):.0f}"
                    if scored else "-")
            lines.append(f"{size:<12}{factor:<7}{h:<12}"
                         f"{100*len(feas)/len(rs):>6.0f}{len(scored):>7}{mg:>8}{optp:>6}")
        lines.append("")
    return "\n".join(lines)


def main(out_dir="."):
    instances = build_instances()
    print(f"Generated {len(instances)} instances "
          f"({len(SIZES)} sizes x {len(FACTORS)} factors x {SEEDS} seeds)\n")

    for objective in ("makespan", "idle"):
        print("=" * 78)
        print(f"OBJECTIVE: {objective}")
        print("=" * 78)
        rows = bb.run_benchmark(instances, objective=objective,
                                heuristics=HEURISTICS, time_limit=TIME_LIMIT,
                                verbose=False)
        csv_path = f"{out_dir}/results_{objective}.csv"
        write_csv(rows, csv_path)

        print(format_summary(rows))
        print(per_group_breakdown(rows))
        print(f"[wrote per-instance rows -> {csv_path}]\n")

    print("Done. Per-instance detail is in results_makespan.csv / results_idle.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")

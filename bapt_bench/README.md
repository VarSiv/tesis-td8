# bapt_bench — Benchmark harness for the Berth Allocation Problem with Tides

Generates instances, solves them **exactly** with SCIP, runs your **heuristics**,
and reports an **accuracy score** (gap vs. the exact optimum). Built to make it
cheap to test many heuristic approaches under identical, fair conditions.

## The one thing to understand first: objectives

There are **two different objectives** floating around this project:

| Objective  | Minimises                          | Used by |
|------------|------------------------------------|---------|
| `makespan` | max exit time `max_i r_i`          | the paper, `tides.zpl`, the Benders Java solver |
| `idle`     | total idle `Σ (r_i − l_i − a_i)`   | the group's greedy + GRASP heuristics |

These are **not the same optimum**. Comparing the idle-minimising heuristic
against the makespan MILP is apples-to-oranges. So the harness fixes **one**
objective for *both* the exact solver and the score:

```bash
python -m bapt_bench --objective idle      # fair test of the greedy/GRASP idea
python -m bapt_bench --objective makespan   # test vs. the paper's objective
```

## Requirements

- Python 3 (standard library only — no numpy/pandas/pyscipopt needed).
- The SCIP binary shipped in the repo at `../scipoptsuite-10.0.2/bin/scip`.
  Override with `BAPT_SCIP_BIN=/path/to/scip` if needed.

## CLI

```bash
# All registered heuristics, 10 instances of size 5 ships / 2 berths / 4 tides
python -m bapt_bench --n 5 --m 2 --t 4 --count 10 --objective makespan

# Sweep several sizes (ships x berths x tides), write per-row CSV
python -m bapt_bench --sizes 5x2x4,8x3x5,10x3x6 --count 5 --csv results.csv

# Score a single existing .dat pair
python -m bapt_bench --ships ../ships.dat --tides ../tides.dat --objective makespan

# List heuristics
python -m bapt_bench --list
```

Key flags: `--objective {makespan,idle}`, `--heuristics greedy,grasp`,
`--factor` (tide width; **lower = harder**, and low enough can make instances
infeasible), `--seed`, `--time-limit` (exact solver, seconds), `--csv`.

## Scoring

For a fixed objective (minimisation):

- `gap%  = 100·(heur − opt) / opt`  — 0 is optimal, higher is worse
- `ratio = opt / heur`              — 1.0 is optimal

Only instances the exact solver proves **optimal** count toward gap statistics.
Heuristic solutions are validated by a shared checker (`checker.evaluate`);
infeasible or incomplete schedules are reported and excluded from the gap.

## Adding a new heuristic (the reusable part)

A heuristic is any callable `Instance -> Solution`:

```python
from bapt_bench import register, Instance, Solution, Assignment

@register("my_method")
def my_method(instance: Instance) -> Solution:
    asg = []
    for sid in instance.ship_ids:
        # ... decide berth, entry l, exit r for this ship ...
        asg.append(Assignment(ship_id=sid, berth=k, l=l, r=r))
    return Solution(assignments=asg)
```

Import the module that defines it, then it shows up in `--heuristics` and in
`run_benchmark`. See `examples/custom_heuristic.py` for a runnable example
(`python -m bapt_bench.examples.custom_heuristic`).

## Programmatic use

```python
import bapt_bench as bb
insts = bb.generate_many([dict(berths=2, ships=6, tides=5, factor=15)], base_seed=0)
rows  = bb.run_benchmark(insts, objective="makespan", heuristics=["greedy", "grasp"])
from bapt_bench.benchmark import format_summary, write_csv
print(format_summary(rows)); write_csv(rows, "out.csv")
```

## Modules

- `model.py`      — `Instance`, `Tide`, `Assignment`, `Solution`; `.dat` I/O.
- `generator.py`  — reproducible instance generator (Python port of `generator.cpp`).
- `exact.py`      — exact MILP via a generated ZIMPL model + SCIP, per objective.
- `checker.py`    — feasibility validation + makespan/idle evaluation (the referee).
- `heuristics.py` — registry + adapters wrapping the team's `prueba1.py` greedy and GRASP.
- `benchmark.py`  — orchestration, scoring, tables, CSV.

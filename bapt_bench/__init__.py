"""
bapt_bench — Reusable benchmarking harness for the Berth Allocation Problem
with Tides (BAPT / "amarraderos con mareas").

Pipeline:  generate instances  ->  solve exactly (SCIP)  ->  run heuristics
           ->  score heuristics against the exact optimum.

The harness is objective-aware. The reference paper / ZIMPL model minimises the
**makespan** (max exit time), while the group's constructive heuristic minimises
**idle time** (sum of r_i - l_i - a_i). Those are different objectives, so a fair
comparison must fix one objective for BOTH the exact solver and the heuristic.
Use `objective="makespan"` or `objective="idle"` throughout.

Public API:
    Instance, Assignment, Solution          (model)
    generate, generate_many                 (generator)
    solve_exact, ExactResult                (exact)
    evaluate, EvalResult                    (checker)
    register, get_heuristic, all_heuristics (heuristics)
    run_benchmark                           (benchmark)
"""

from .model import Instance, Assignment, Solution, Tide
from .generator import generate, generate_many
from .exact import solve_exact, ExactResult
from .checker import evaluate, EvalResult
from .heuristics import register, get_heuristic, all_heuristics
from .benchmark import run_benchmark, BenchmarkRow

__all__ = [
    "Instance", "Assignment", "Solution", "Tide",
    "generate", "generate_many",
    "solve_exact", "ExactResult",
    "evaluate", "EvalResult",
    "register", "get_heuristic", "all_heuristics",
    "run_benchmark", "BenchmarkRow",
]

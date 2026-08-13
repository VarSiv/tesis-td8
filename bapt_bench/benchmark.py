"""Benchmark orchestrator: generate/solve exactly, run heuristics, score gaps.

For a fixed `objective` (makespan or idle):
  * solve each instance exactly once (cached);
  * run each selected heuristic;
  * score the heuristic against the exact optimum.

Scoring (minimisation):
    gap%   = 100 * (heur - opt) / opt        (0 = optimal, higher = worse)
    ratio  = opt / heur                       (1.0 = optimal, in (0,1])
A heuristic that returns an infeasible or incomplete schedule is recorded as
such and excluded from gap statistics.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .model import Instance
from .checker import evaluate
from .exact import solve_exact, ExactResult
from .heuristics import all_heuristics, get_heuristic

EPS = 1e-6


@dataclass
class BenchmarkRow:
    instance: str
    heuristic: str
    objective: str
    exact_status: str
    optimum: float
    heur_value: float
    feasible: bool
    complete: bool
    is_optimal: bool          # heur value equals exact optimum (within EPS)
    gap_pct: Optional[float]  # None when not scorable
    ratio: Optional[float]
    heur_time: float
    exact_time: float
    n: int
    m: int
    t: int
    errors: List[str] = field(default_factory=list)


def _score(opt: float, heur: float):
    if opt == float("inf") or heur == float("inf"):
        return None, None, False
    if abs(opt) < EPS:                       # optimum is exactly 0 (e.g. idle=0)
        is_opt = abs(heur) < EPS
        gap = 0.0 if is_opt else float(heur) * 100.0  # absolute-ish fallback
        ratio = 1.0 if is_opt else 0.0
        return gap, ratio, is_opt
    gap = 100.0 * (heur - opt) / opt
    ratio = opt / heur if heur > 0 else 0.0
    return gap, ratio, abs(heur - opt) < EPS


def run_benchmark(instances: Sequence[Instance],
                  objective: str = "makespan",
                  heuristics: Optional[Sequence[str]] = None,
                  time_limit: float = 120.0,
                  verbose: bool = True) -> List[BenchmarkRow]:
    """Run all (instance x heuristic) pairs for `objective` and return rows."""
    names = list(heuristics) if heuristics is not None else sorted(all_heuristics())
    funcs = {nm: get_heuristic(nm) for nm in names}

    rows: List[BenchmarkRow] = []
    for inst in instances:
        if verbose:
            print(f"[exact] {inst.name}  ({inst.summary()})  obj={objective} ...",
                  flush=True)
        ex: ExactResult = solve_exact(inst, objective=objective, time_limit=time_limit)
        if verbose:
            print(f"        exact: status={ex.status} opt={_fmt(ex.optimum)} "
                  f"({ex.solve_time:.2f}s)", flush=True)

        for nm in names:
            t0 = time.perf_counter()
            try:
                sol = funcs[nm](inst)
                err = None
            except Exception as exc:              # a broken heuristic shouldn't kill the run
                sol, err = None, exc
            dt = time.perf_counter() - t0

            if sol is None:
                rows.append(_bad_row(inst, nm, objective, ex, dt,
                                     [f"heuristic raised: {err}"]))
                continue

            ev = evaluate(inst, sol)
            heur_val = ev.objective(objective) if ev.ok else float("inf")
            gap, ratio, is_opt = _score(ex.optimum, heur_val) if ex.optimum != float("inf") else (None, None, False)
            rows.append(BenchmarkRow(
                instance=inst.name, heuristic=nm, objective=objective,
                exact_status=ex.status, optimum=ex.optimum, heur_value=heur_val,
                feasible=ev.feasible, complete=ev.complete,
                is_optimal=bool(is_opt) if ev.ok else False,
                gap_pct=gap if ev.ok else None,
                ratio=ratio if ev.ok else None,
                heur_time=dt, exact_time=ex.solve_time,
                n=inst.n_ships, m=inst.berths, t=inst.n_tides,
                errors=ev.errors,
            ))
    return rows


def _bad_row(inst, nm, objective, ex, dt, errors):
    return BenchmarkRow(
        instance=inst.name, heuristic=nm, objective=objective,
        exact_status=ex.status, optimum=ex.optimum, heur_value=float("inf"),
        feasible=False, complete=False, is_optimal=False, gap_pct=None, ratio=None,
        heur_time=dt, exact_time=ex.solve_time,
        n=inst.n_ships, m=inst.berths, t=inst.n_tides, errors=errors)


# --- reporting ------------------------------------------------------------
def _fmt(x) -> str:
    if x is None:
        return "-"
    if x == float("inf"):
        return "inf"
    return f"{x:.3f}"


def format_table(rows: Sequence[BenchmarkRow]) -> str:
    hdr = ["instance", "heur", "n/m/t", "status", "opt", "heur", "gap%",
           "feas", "opt?", "h_s", "e_s"]
    lines = ["  ".join(h.ljust(w) for h, w in zip(
        hdr, [22, 8, 9, 9, 8, 8, 8, 4, 4, 6, 6]))]
    for r in rows:
        lines.append("  ".join(str(c).ljust(w) for c, w in zip([
            r.instance[:22], r.heuristic[:8], f"{r.n}/{r.m}/{r.t}",
            r.exact_status[:9], _fmt(r.optimum), _fmt(r.heur_value),
            _fmt(r.gap_pct), "Y" if r.feasible and r.complete else "N",
            "Y" if r.is_optimal else "N", f"{r.heur_time:.2f}", f"{r.exact_time:.2f}",
        ], [22, 8, 9, 9, 8, 8, 8, 4, 4, 6, 6])))
    return "\n".join(lines)


def aggregate(rows: Sequence[BenchmarkRow]) -> Dict[str, dict]:
    """Per-heuristic summary over the rows (only exact-optimal instances count
    toward gap statistics, so the reference is a true optimum)."""
    out: Dict[str, dict] = {}
    heur_names = sorted({r.heuristic for r in rows})
    for nm in heur_names:
        rs = [r for r in rows if r.heuristic == nm]
        scorable = [r for r in rs if r.gap_pct is not None and r.exact_status == "optimal"]
        feas = [r for r in rs if r.feasible and r.complete]
        gaps = [r.gap_pct for r in scorable]
        out[nm] = {
            "instances": len(rs),
            "feasible_rate": len(feas) / len(rs) if rs else 0.0,
            "scored": len(scorable),
            "mean_gap_pct": sum(gaps) / len(gaps) if gaps else None,
            "max_gap_pct": max(gaps) if gaps else None,
            "optimal_rate": (sum(1 for r in scorable if r.is_optimal) / len(scorable)
                             if scorable else None),
            "mean_heur_time": sum(r.heur_time for r in rs) / len(rs) if rs else 0.0,
        }
    return out


def format_summary(rows: Sequence[BenchmarkRow]) -> str:
    agg = aggregate(rows)
    obj = rows[0].objective if rows else "?"
    lines = [f"\n=== SUMMARY (objective = {obj}) ===",
             "heuristic     feas%   scored  mean_gap%  max_gap%  optimal%  avg_time"]
    for nm, a in agg.items():
        lines.append(
            f"{nm:<12}  {100*a['feasible_rate']:5.1f}  {a['scored']:6d}  "
            f"{_pct(a['mean_gap_pct']):>9}  {_pct(a['max_gap_pct']):>8}  "
            f"{_pct(100*a['optimal_rate'] if a['optimal_rate'] is not None else None):>8}  "
            f"{a['mean_heur_time']:7.3f}s")
    return "\n".join(lines)


def _pct(x) -> str:
    return "-" if x is None else f"{x:.2f}"


def write_csv(rows: Sequence[BenchmarkRow], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance", "heuristic", "objective", "n", "m", "t",
                    "exact_status", "optimum", "heur_value", "gap_pct", "ratio",
                    "feasible", "complete", "is_optimal", "heur_time", "exact_time",
                    "errors"])
        for r in rows:
            w.writerow([r.instance, r.heuristic, r.objective, r.n, r.m, r.t,
                        r.exact_status, r.optimum, r.heur_value,
                        "" if r.gap_pct is None else f"{r.gap_pct:.6f}",
                        "" if r.ratio is None else f"{r.ratio:.6f}",
                        int(r.feasible), int(r.complete), int(r.is_optimal),
                        f"{r.heur_time:.6f}", f"{r.exact_time:.6f}",
                        "; ".join(r.errors)])

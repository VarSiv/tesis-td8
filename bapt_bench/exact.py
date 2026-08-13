"""Exact MILP solver for BAPT, driven by the SCIP binary via a generated ZIMPL model.

Why a generated model instead of the existing `tides.zpl`?
  * We need to switch the objective between `makespan` and `idle` so the
    exact optimum matches whatever a given heuristic optimises (fair gap).
  * `tides.zpl` requires a `reinf` string parameter and hard-codes the .dat
    filenames; generating the model keeps the harness self-contained.

The formulation is the base BAPT MILP (same constraints as `tides.zpl`, minus
the optional reinforcement/valid-inequality families, which do not change the
optimum).
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .model import Instance, Assignment, Solution
from .checker import evaluate, EvalResult

# --- locating the SCIP binary shipped with the repo -----------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]      # .../tesis
_SCIP_HOME = _REPO_ROOT / "scipoptsuite-10.0.2"


def _scip_bin() -> str:
    return os.environ.get("BAPT_SCIP_BIN", str(_SCIP_HOME / "bin" / "scip"))


def _scip_env() -> dict:
    env = dict(os.environ)
    libs = [str(_SCIP_HOME / "lib"), str(_SCIP_HOME / "lib64")]
    prev = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join([p for p in libs if p] + ([prev] if prev else []))
    return env


@dataclass
class ExactResult:
    status: str                      # optimal | feasible | infeasible | error | timeout
    objective: str                   # makespan | idle
    optimum: float = float("inf")    # objective value of the returned solution
    makespan: float = float("inf")
    idle: float = float("inf")
    solution: Optional[Solution] = None
    solve_time: float = 0.0
    eval: Optional[EvalResult] = field(default=None)
    raw_status: str = ""

    @property
    def is_optimal(self) -> bool:
        return self.status == "optimal"


def _zimpl_model(ships_dat: str, tides_dat: str, objective: str) -> str:
    """Build the ZIMPL model string for the requested objective."""
    if objective == "makespan":
        obj_line = "minimize obj: z;"
    elif objective == "idle":
        obj_line = "minimize obj: sum <i> in N: (r[i] - l[i] - a[i]);"
    else:
        raise ValueError(f"unknown objective {objective!r}")

    # Absolute paths avoid any cwd / base-directory ambiguity in ZIMPL.
    s = ships_dat.replace("\\", "/")
    t = tides_dat.replace("\\", "/")
    return f'''
param barcos := "{s}";
param mareas := "{t}";

param m := read barcos as "1n" use 1;

set N := {{ read barcos as "<1n>" skip 1 }};
set T := {{ read mareas as "<1n>" }};
set M := {{ 1 .. m }};

param a[N] := read barcos as "<1n> 2n" skip 1;
param s[T] := read mareas as "<1n> 2n";
param e[T] := read mareas as "<1n> 3n";

param mu := max <t> in T: e[t];

var x[N*M] binary;
var y[N*N] binary;
var wr[N*T] binary;
var wl[N*T] binary;
var l[N] >= 0;
var r[N] >= 0;
var z >= 0;

{obj_line}

subto unberth: forall <i> in N:
    sum <k> in M: x[i,k] == 1;
subto unainicial: forall <i> in N:
    sum <t> in T: wl[i,t] == 1;
subto unafinal: forall <i> in N:
    sum <t> in T: wr[i,t] == 1;
subto tiempo_atencion: forall <i> in N:
    r[i] >= l[i] + a[i];
subto no_superposicion: forall <i,j> in N*N with i != j:
    r[i] <= l[j] + mu * (1 - y[i,j]);
subto define_makespan: forall <i> in N:
    r[i] <= z;
subto def_precedencia: forall <i,j,k> in N*N*M with i != j:
    x[i,k] + x[j,k] - 1 <= y[i,j] + y[j,i];
subto inicial_izquierdo: forall <i,t> in N*T:
    l[i] >= s[t] - mu * (1 - wl[i,t]);
subto inicial_derecho: forall <i,t> in N*T:
    l[i] <= e[t] + mu * (1 - wl[i,t]);
subto final_izquierdo: forall <i,t> in N*T:
    r[i] >= s[t] - mu * (1 - wr[i,t]);
subto final_derecho: forall <i,t> in N*T:
    r[i] <= e[t] + mu * (1 - wr[i,t]);
'''


_SOL_LINE = re.compile(r"^(\S+)\s+([-+0-9.eE]+)")


def _parse_sol(sol_text: str) -> tuple[str, Dict[str, float]]:
    status = ""
    vals: Dict[str, float] = {}
    for line in sol_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("solution status:"):
            status = line.split(":", 1)[1].strip()
            continue
        if line.startswith("objective value:"):
            continue
        m = _SOL_LINE.match(line)
        if m:
            try:
                vals[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
    return status, vals


def _solution_from_vals(instance: Instance, vals: Dict[str, float]) -> Solution:
    asg: List[Assignment] = []
    for sid in instance.ship_ids:
        berth = 0
        for k in range(1, instance.berths + 1):
            if vals.get(f"x#{sid}#{k}", 0.0) > 0.5:
                berth = k - 1
                break
        tide_in = next((t.id for t in instance.tides
                        if vals.get(f"wl#{sid}#{t.id}", 0.0) > 0.5), None)
        tide_out = next((t.id for t in instance.tides
                         if vals.get(f"wr#{sid}#{t.id}", 0.0) > 0.5), None)
        asg.append(Assignment(
            ship_id=sid,
            berth=berth,
            l=vals.get(f"l#{sid}", 0.0),
            r=vals.get(f"r#{sid}", 0.0),
            tide_in=tide_in,
            tide_out=tide_out,
        ))
    return Solution(assignments=asg, meta={"source": "exact"})


def solve_exact(instance: Instance, objective: str = "makespan",
                time_limit: float = 120.0) -> ExactResult:
    """Solve `instance` to optimality (or the time limit) for `objective`.

    Returns an ExactResult. The reported makespan/idle are recomputed by the
    shared checker from the solution's (l, r) values, so they are directly
    comparable with heuristic solutions.
    """
    with tempfile.TemporaryDirectory(prefix="bapt_") as d:
        ships = os.path.join(d, "ships.dat")
        tides = os.path.join(d, "tides.dat")
        model = os.path.join(d, "model.zpl")
        solf = os.path.join(d, "out.sol")
        instance.write_dat(ships, tides)
        with open(model, "w") as f:
            f.write(_zimpl_model(ships, tides, objective))

        cmd = [
            _scip_bin(),
            "-c", f"set limits time {time_limit}",
            "-c", f"read {model}",
            "-c", "optimize",
            "-c", f"write solution {solf}",
            "-c", "quit",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  env=_scip_env(), timeout=time_limit + 60)
        except (subprocess.TimeoutExpired, FileNotFoundError) as ex:
            return ExactResult(status="error", objective=objective,
                               raw_status=str(ex))

        stdout = proc.stdout
        solve_time = _parse_time(stdout)
        raw = _parse_scip_status(stdout)

        if not os.path.exists(solf):
            status = "infeasible" if "infeasible" in raw else "error"
            return ExactResult(status=status, objective=objective,
                               solve_time=solve_time, raw_status=raw)

        with open(solf) as f:
            sol_status, vals = _parse_sol(f.read())

    if "no solution" in sol_status.lower() or not vals:
        status = "infeasible" if "infeasible" in raw else "error"
        return ExactResult(status=status, objective=objective,
                           solve_time=solve_time, raw_status=raw or sol_status)

    solution = _solution_from_vals(instance, vals)
    ev = evaluate(instance, solution)
    if "optimal" in raw:
        status = "optimal"
    elif "time limit" in raw:
        status = "timeout"      # a feasible incumbent, not proven optimal
    else:
        status = "feasible"

    return ExactResult(
        status=status,
        objective=objective,
        optimum=ev.objective(objective),
        makespan=ev.makespan,
        idle=ev.idle,
        solution=solution,
        solve_time=solve_time,
        eval=ev,
        raw_status=raw or sol_status,
    )


def _parse_scip_status(stdout: str) -> str:
    m = re.search(r"SCIP Status\s*:\s*(.+)", stdout)
    return m.group(1).strip().lower() if m else ""


def _parse_time(stdout: str) -> float:
    m = re.search(r"Solving Time \(sec\)\s*:\s*([0-9.]+)", stdout)
    return float(m.group(1)) if m else 0.0

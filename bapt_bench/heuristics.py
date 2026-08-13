"""Heuristic registry + adapters for the team's existing heuristics.

Add a new approach in one of two ways:

    from bapt_bench.heuristics import register
    from bapt_bench.model import Solution, Assignment

    @register("my_method")
    def my_method(instance):
        ...
        return Solution(assignments=[Assignment(ship_id, berth, l, r), ...])

A heuristic is any callable `Instance -> Solution`. The benchmark validates and
scores whatever Solution you return via the shared checker, so heuristics never
need to know how they will be graded.

The two adapters below wrap the group's real code (`prueba1.py` greedy and the
notebook's GRASP), so we are benchmarking their actual algorithms, not a
reimplementation.
"""

from __future__ import annotations

import os
import random
import sys
from typing import Callable, Dict

from .model import Instance, Solution, Assignment

# --- make the team's prueba1.py importable --------------------------------
_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # tesis-td8/
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

import prueba1 as _p  # noqa: E402  (Barco, Marea, Asignacion, goloso, ubicar, ...)


# --- registry -------------------------------------------------------------
_REGISTRY: Dict[str, Callable[[Instance], Solution]] = {}


def register(name: str):
    """Decorator registering a heuristic under `name`."""
    def deco(fn: Callable[[Instance], Solution]) -> Callable[[Instance], Solution]:
        if name in _REGISTRY:
            raise ValueError(f"heuristic {name!r} already registered")
        _REGISTRY[name] = fn
        fn.heuristic_name = name  # type: ignore[attr-defined]
        return fn
    return deco


def get_heuristic(name: str) -> Callable[[Instance], Solution]:
    if name not in _REGISTRY:
        raise KeyError(f"no heuristic {name!r}; have {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def all_heuristics() -> Dict[str, Callable[[Instance], Solution]]:
    return dict(_REGISTRY)


# --- conversion helpers ---------------------------------------------------
def _to_prueba1(instance: Instance):
    m = instance.berths
    barcos = [_p.Barco(i, a) for i, a in enumerate(instance.attention, start=1)]
    mareas = [_p.Marea(t.id, t.s, t.e) for t in instance.tides]
    return m, barcos, mareas


def _from_asignaciones(asignaciones) -> Solution:
    asg = [Assignment(ship_id=x.barco, berth=x.amarradero, l=x.l, r=x.r,
                      tide_in=x.marea_in, tide_out=x.marea_out)
           for x in asignaciones]
    return Solution(assignments=asg)


# --- GRASP (ported from the notebook, reusing prueba1 primitives) ---------
def _construir_grasp(m, barcos, mareas, rng, rcl_frac=0.4):
    rangos = _p.rangos_por_par(mareas)
    barcos_copia = [_p.Barco(b.id, b.a) for b in barcos]
    _p.marcar_barcos(barcos_copia, rangos)
    por_id = {t.id: t for t in mareas}
    libres = [0.0] * m
    asignaciones = []
    pendientes = list(barcos_copia)
    orden = lambda b: (b.opciones, -b.a, b.id)

    cambio = True
    while cambio:
        cambio = False
        candidatos = []
        for b in pendientes:
            for par, (dmin, dmax) in rangos.items():
                if not (dmin - _p.EPS <= b.a <= dmax + _p.EPS):
                    continue
                t1, t2 = por_id[par[0]], por_id[par[1]]
                asig = _p.ubicar(b, t1, t2, libres)
                if asig is not None and asig.ocio <= _p.EPS:
                    candidatos.append((b, asig))
                    break
        if not candidatos:
            break
        candidatos.sort(key=lambda x: orden(x[0]))
        k = max(1, int(len(candidatos) * rcl_frac))
        rcl = candidatos[:k]
        b, asig = rng.choice(rcl)
        asignaciones.append(asig)
        libres[asig.amarradero] = asig.r
        pendientes.remove(b)
        cambio = True

    for b in sorted(pendientes[:], key=orden):
        opciones_pares = _p.pares_del_barco(b, rangos) or list(rangos)
        candidatos = []
        for par in opciones_pares:
            asig = _p.ubicar(b, por_id[par[0]], por_id[par[1]], libres)
            if asig is not None:
                candidatos.append(asig)
        if not candidatos:
            continue
        candidatos.sort(key=lambda a: (a.ocio, a.r))
        k = max(1, int(len(candidatos) * rcl_frac))
        rcl = candidatos[:k]
        asig = rng.choice(rcl)
        asignaciones.append(asig)
        libres[asig.amarradero] = asig.r
        pendientes.remove(b)

    asignaciones.sort(key=lambda x: (x.amarradero, x.l))
    return asignaciones, pendientes


def _grasp(m, barcos, mareas, n_iter=200, rcl_frac=0.4, seed=0):
    rng = random.Random(seed)
    mejor = None
    for _ in range(n_iter):
        asignaciones, pendientes = _construir_grasp(m, barcos, mareas, rng, rcl_frac)
        score = (-len(asignaciones), sum(x.ocio for x in asignaciones))
        if mejor is None or score < mejor[0]:
            mejor = (score, asignaciones, pendientes)
    return mejor[1], mejor[2]


# --- registered heuristics ------------------------------------------------
@register("greedy")
def greedy(instance: Instance) -> Solution:
    """Team's constructive greedy (prueba1.goloso): minimises idle."""
    m, barcos, mareas = _to_prueba1(instance)
    asignaciones, _ = _p.goloso(m, barcos, mareas)
    return _from_asignaciones(asignaciones)


def make_grasp(n_iter: int = 200, rcl_frac: float = 0.4, seed: int = 0):
    """Factory for GRASP variants (register several configs if you like)."""
    def _fn(instance: Instance) -> Solution:
        m, barcos, mareas = _to_prueba1(instance)
        asignaciones, _ = _grasp(m, barcos, mareas,
                                 n_iter=n_iter, rcl_frac=rcl_frac, seed=seed)
        sol = _from_asignaciones(asignaciones)
        sol.meta["n_iter"] = n_iter
        return sol
    return _fn


register("grasp")(make_grasp(n_iter=200, rcl_frac=0.4, seed=0))

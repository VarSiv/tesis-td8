#!/usr/bin/env python3
"""
Heuristica golosa para el problema de amarraderos con mareas.

Idea (segun el documento):
  - Un barco solo puede ENTRAR al amarradero durante una marea alta y solo
    puede SALIR durante una marea alta.
  - Si termina su atencion y no hay marea alta, se queda ocupando el
    amarradero => tiempo ocioso.
  - Objetivo: minimizar la suma de tiempos ociosos.

Estrategia:
  1) Para cada par de mareas (t1, t2) con t1 <= t2 se calcula el rango de
     tiempos de atencion que permiten entrar en t1 y terminar justo dentro
     de t2 (ocio cero):
        dmin = max(0, s[t2] - e[t1])      (entrando lo mas tarde posible)
        dmax = e[t2] - s[t1]              (entrando lo mas temprano posible)
  2) Cada barco queda "marcado" con la cantidad de pares compatibles.
     Los barcos con menos opciones son los mas dificiles => se ubican primero.
  3) Fase 1: se recorren los pares en orden cronologico y se ubican barcos
     con ocio cero.
  4) Fase 2: los barcos que quedaron sin ubicar se colocan donde el ocio
     sea minimo (permitiendo espera dentro del amarradero).

Uso:
    python3 goloso_amarraderos.py [ships.dat] [tides.dat]
"""

import sys
from dataclasses import dataclass

EPS = 1e-9


# --------------------------------------------------------------------------
# Estructuras
# --------------------------------------------------------------------------

@dataclass
class Barco:
    id: int
    a: float           # tiempo de atencion
    opciones: int = 0  # cantidad de pares de mareas compatibles


@dataclass
class Marea:
    id: int
    s: float           # inicio de la marea alta
    e: float           # fin de la marea alta


@dataclass
class Asignacion:
    barco: int
    amarradero: int
    l: float           # instante de entrada (dentro de marea_in)
    r: float           # instante de salida  (dentro de marea_out)
    marea_in: int
    marea_out: int
    ocio: float        # r - (l + a)


# --------------------------------------------------------------------------
# Lectura de la instancia (mismos .dat que lee el modelo ZIMPL)
# --------------------------------------------------------------------------

def leer_instancia(path_ships, path_tides):
    with open(path_ships) as f:
        filas = [ln.split() for ln in f if ln.strip()]
    m = int(filas[0][0])
    barcos = [Barco(int(p[0]), float(p[1])) for p in filas[1:]]

    with open(path_tides) as f:
        mareas = [Marea(int(p[0]), float(p[1]), float(p[2]))
                  for p in (ln.split() for ln in f if ln.strip())]
    mareas.sort(key=lambda t: t.s)
    return m, barcos, mareas


# --------------------------------------------------------------------------
# Paso 1: rangos de atencion por par de mareas
# --------------------------------------------------------------------------

def rangos_por_par(mareas):
    """(t1, t2) -> (dmin, dmax): duraciones que permiten entrar en t1 y
    terminar la atencion dentro de t2, sin tiempo ocioso."""
    rangos = {}
    for i, t1 in enumerate(mareas):
        for t2 in mareas[i:]:
            dmin = max(0.0, t2.s - t1.e)
            dmax = t2.e - t1.s
            if dmax > dmin - EPS:
                rangos[(t1.id, t2.id)] = (dmin, dmax)
    return rangos


def marcar_barcos(barcos, rangos):
    """Cuenta en cuantos pares puede ubicarse cada barco (su 'marca')."""
    for b in barcos:
        b.opciones = sum(1 for (dmin, dmax) in rangos.values()
                         if dmin - EPS <= b.a <= dmax + EPS)


def pares_del_barco(barco, rangos):
    return [par for par, (dmin, dmax) in rangos.items()
            if dmin - EPS <= barco.a <= dmax + EPS]


# --------------------------------------------------------------------------
# Ubicacion de un barco en un par de mareas
# --------------------------------------------------------------------------

def ubicar(barco, t1, t2, libres):
    """Intenta ubicar el barco entrando en t1 y saliendo en t2.

    Elige la entrada mas tardia que sirva para no generar ocio (o la mas
    tardia posible si el ocio es inevitable), y el amarradero que se libera
    lo mas tarde posible entre los que llegan a tiempo (best fit: deja
    libres los amarraderos que se desocupan antes).

    Devuelve una Asignacion o None si no entra.
    """
    # la entrada debe estar en [t1.s, t1.e] y la atencion terminar antes de t2.e
    hi = min(t1.e, t2.e - barco.a)
    if hi < t1.s - EPS:
        return None

    # entrada ideal: la mas tarde posible sin pasarse de lo necesario
    # para alcanzar el inicio de t2 (asi el ocio es 0 si se puede)
    objetivo = min(hi, t2.s - barco.a)

    candidatos = [(f, k) for k, f in enumerate(libres) if f <= hi + EPS]
    if not candidatos:
        return None
    f, k = max(candidatos)

    l = max(t1.s, f, objetivo)
    if l > hi + EPS:
        return None

    fin = l + barco.a                 # fin de la atencion
    r = max(t2.s, fin)                # salida: recien cuando hay marea alta
    if r > t2.e + EPS:
        return None

    return Asignacion(barco.id, k, l, r, t1.id, t2.id, r - fin)


# --------------------------------------------------------------------------
# Heuristica golosa
# --------------------------------------------------------------------------

def goloso(m, barcos, mareas):
    rangos = rangos_por_par(mareas)
    marcar_barcos(barcos, rangos)
    por_id = {t.id: t for t in mareas}

    libres = [0.0] * m                # instante en que se libera cada amarradero
    asignaciones = []
    pendientes = list(barcos)

    # --- Fase 1: recorrer los pares y ubicar con ocio cero -----------------
    # barcos mas dificiles primero (menos opciones), y a igualdad los mas largos
    orden = lambda b: (b.opciones, -b.a, b.id)

    for par in sorted(rangos, key=lambda p: (por_id[p[0]].s, por_id[p[1]].s)):
        t1, t2 = por_id[par[0]], por_id[par[1]]
        dmin, dmax = rangos[par]
        hubo_cambio = True
        while hubo_cambio:
            hubo_cambio = False
            for b in sorted(pendientes, key=orden):
                if not (dmin - EPS <= b.a <= dmax + EPS):
                    continue
                asig = ubicar(b, t1, t2, libres)
                if asig is not None and asig.ocio <= EPS:
                    asignaciones.append(asig)
                    libres[asig.amarradero] = asig.r
                    pendientes.remove(b)
                    hubo_cambio = True
                    break

    # --- Fase 2: lo que sobro, donde el ocio sea minimo --------------------
    for b in sorted(pendientes[:], key=orden):
        mejor = None
        for par in pares_del_barco(b, rangos) or list(rangos):
            asig = ubicar(b, por_id[par[0]], por_id[par[1]], libres)
            if asig is None:
                continue
            if mejor is None or (asig.ocio, asig.r) < (mejor.ocio, mejor.r):
                mejor = asig
        if mejor is not None:
            asignaciones.append(mejor)
            libres[mejor.amarradero] = mejor.r
            pendientes.remove(b)

    asignaciones.sort(key=lambda x: (x.amarradero, x.l))
    return asignaciones, pendientes


# --------------------------------------------------------------------------
# Verificacion y reporte
# --------------------------------------------------------------------------

def verificar(asignaciones, barcos, mareas, m):
    errores = []
    a = {b.id: b.a for b in barcos}
    por_id = {t.id: t for t in mareas}

    for x in asignaciones:
        t1, t2 = por_id[x.marea_in], por_id[x.marea_out]
        if not (t1.s - EPS <= x.l <= t1.e + EPS):
            errores.append(f"barco {x.barco}: entrada {x.l} fuera de la marea {t1.id}")
        if not (t2.s - EPS <= x.r <= t2.e + EPS):
            errores.append(f"barco {x.barco}: salida {x.r} fuera de la marea {t2.id}")
        if x.r < x.l + a[x.barco] - EPS:
            errores.append(f"barco {x.barco}: no alcanza el tiempo de atencion")

    for k in range(m):
        enk = sorted((x for x in asignaciones if x.amarradero == k), key=lambda x: x.l)
        for u, v in zip(enk, enk[1:]):
            if v.l < u.r - EPS:
                errores.append(f"amarradero {k}: se superponen {u.barco} y {v.barco}")
    return errores


def reporte(asignaciones, pendientes, barcos, mareas, m):
    print(f"Amarraderos: {m}   Barcos: {len(barcos)}   Mareas: {len(mareas)}")
    print(f"Horizonte: [{mareas[0].s}, {mareas[-1].e}]")
    print()
    print(f"{'barco':>6} {'aten':>6} {'amarr':>6} {'entra':>8} {'sale':>8} "
          f"{'m_in':>5} {'m_out':>6} {'ocio':>7}")
    a = {b.id: b.a for b in barcos}
    for x in asignaciones:
        print(f"{x.barco:>6} {a[x.barco]:>6.2f} {x.amarradero:>6} {x.l:>8.2f} "
              f"{x.r:>8.2f} {x.marea_in:>5} {x.marea_out:>6} {x.ocio:>7.2f}")

    ocio = sum(x.ocio for x in asignaciones)
    print()
    print(f"Barcos ubicados : {len(asignaciones)}/{len(barcos)}")
    print(f"Ocio total      : {ocio:.2f}")
    print(f"Makespan        : {max((x.r for x in asignaciones), default=0):.2f}")
    if pendientes:
        print(f"SIN UBICAR      : {[b.id for b in pendientes]}")
        print("  (solucion parcial: la instancia no admite ubicar todos los barcos)")


def main():
    ships = sys.argv[1] if len(sys.argv) > 1 else "ships.dat"
    tides = sys.argv[2] if len(sys.argv) > 2 else "tides.dat"

    m, barcos, mareas = leer_instancia(ships, tides)
    asignaciones, pendientes = goloso(m, barcos, mareas)
    reporte(asignaciones, pendientes, barcos, mareas, m)

    errores = verificar(asignaciones, barcos, mareas, m)
    print()
    print("Verificacion: OK" if not errores else "Verificacion: " + "; ".join(errores))


if __name__ == "__main__":
    main()
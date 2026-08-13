"""My own heuristic(s) for the BAPT harness.

Fill in the body of `my_heuristic` below. Everything else is wired up:
importing this module registers the heuristic, so it shows up in the CLI
(`--heuristics my_heuristic`) and in `run_benchmark`.

Quick test:
    python -m bapt_bench.heuristic_var
"""

from __future__ import annotations

import operator

from bapt_bench import register, Instance, Solution, Assignment

def berth_timings(m: int, cur_assignments: list[Assignment], instance):
    max_exit = {}
    first_tide_start = instance.tides[0].s
    for i in cur_assignments:
        berth = i.berth
        if berth not in max_exit:
            max_exit[berth] = i.r
        else:
            if i.r > max_exit[berth]:
                max_exit[berth] = i.r
    for i in range (m):
        if i not in max_exit:
            max_exit[i] = first_tide_start
    return max_exit

def can_assign_ship_to_berth(sid, berth, berth_timings, instance):
    ship_attention = instance.a(sid)
    berth_available_at = berth_timings[berth]
    return berth_available_at + ship_attention <= instance.horizon

def assign_ship_to_berth(sid, berth, berth_timings, instance):
    berth_available_at = berth_timings[berth]
    leave_berth = instance.horizon
    min_leave = berth_available_at + instance.a(sid)
    tides = instance.tides
    for t in tides:
        if min_leave>=t.s and t.e >= min_leave:
            leave_berth=min_leave
            break
        if t.s >= min_leave:
            leave_berth=t.s
            break
    return Assignment(sid, berth, berth_available_at, leave_berth)


@register("golosa_var")
def golosa_var(instance: Instance) -> Solution:
    """Return a schedule for `instance`.

    What you get (see model.py):
        instance.berths        -> number of berths m
        instance.n_ships       -> n
        instance.ship_ids      -> [1, 2, ..., n]
        instance.a(sid)        -> attention time of ship sid
        instance.tides         -> list of Tide(id, s, e)   (high-tide windows)
        instance.horizon       -> latest tide end (a big-M upper bound)

    What you must produce:
        a Solution holding one Assignment per ship you place:
            Assignment(ship_id, berth, l, r, tide_in=None, tide_out=None)
        where, for the schedule to be FEASIBLE (see checker.py):
            - berth is 0..m-1
            - l (entry) lies inside some tide window   [s_t, e_t]
            - r (exit)  lies inside some tide window   [s_t', e_t']
            - r >= l + a(ship)                         (attention fits)
            - no two ships on the same berth overlap in [l, r)

    You don't have to place every ship: unplaced ships make the solution
    "incomplete", which the harness reports and excludes from the gap score.

    Objectives the harness can score against (pick with --objective):
        makespan = max_i r_i
        idle     = sum_i (r_i - l_i - a(i))
    """
    assignments: list[Assignment] = []
    attention_per_ship = {}
    for i in range (1, instance.n_ships+1):
        attention_per_ship[i] = instance.a(i)
    
    pending_assignment = sorted(attention_per_ship, key=attention_per_ship.get, reverse=True)
    m = instance.berths

    while len(pending_assignment)>0:
        assigned_ship = False
        berth_available_at = berth_timings(m, assignments, instance)
        sorted_berths = sorted(berth_available_at.items(), key=operator.itemgetter(1))
        for b,_ in sorted_berths:
            for s in pending_assignment:
                if can_assign_ship_to_berth(s, b, berth_available_at, instance):
                    assignments.append(assign_ship_to_berth(s, b, berth_available_at, instance))
                    assigned_ship = True
                    pending_assignment.remove(s)
                    break
            if assigned_ship:
                break
        if not assigned_ship:
            break


    return Solution(assignments=assignments)


if __name__ == "__main__":
    # Tiny self-check: generate a few instances and benchmark your heuristic
    # against the exact optimum. Change objective to "idle" if that's your target.
    from bapt_bench import generate_many, run_benchmark
    from bapt_bench.benchmark import format_table, format_summary

    instances = generate_many([
        dict(berths=2, ships=5, tides=4, factor=20),
        dict(berths=3, ships=8, tides=5, factor=15),
    ], base_seed=0)

    rows = run_benchmark(instances, objective="makespan",
                         heuristics=["golosa_var"], time_limit=30, verbose=False)
    print(format_table(rows))
    print(format_summary(rows))

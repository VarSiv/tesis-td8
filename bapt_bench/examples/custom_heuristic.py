"""Example: register a new heuristic and benchmark it against the exact optimum.

Run:  python -m bapt_bench.examples.custom_heuristic
"""

from bapt_bench import (register, generate_many, run_benchmark,
                        Instance, Solution, Assignment)
from bapt_bench.benchmark import format_table, format_summary


# --- a toy baseline: first-fit by earliest tide ---------------------------
@register("first_fit")
def first_fit(instance: Instance) -> Solution:
    """Place ships in order onto the berth that frees earliest; enter at the
    first tide window that can hold the entry, exit at the first later window
    that fits the attention. Deliberately simple — a floor to beat."""
    free = [0.0] * instance.berths            # when each berth frees up
    asg = []
    for sid in instance.ship_ids:
        a = instance.a(sid)
        k = min(range(instance.berths), key=lambda b: free[b])
        placed = False
        for t_in in instance.tides:
            l = max(free[k], t_in.s)
            if l > t_in.e:                    # can't enter within this window
                continue
            for t_out in instance.tides:      # exit window (>= entry time)
                if t_out.e < l + a:
                    continue
                r = max(t_out.s, l + a)
                if r <= t_out.e:
                    asg.append(Assignment(sid, k, l, r, t_in.id, t_out.id))
                    free[k] = r
                    placed = True
                    break
            if placed:
                break
        # if not placed, ship is simply omitted -> checker flags incomplete
    return Solution(assignments=asg)


if __name__ == "__main__":
    instances = generate_many([
        dict(berths=2, ships=5, tides=4, factor=20),
        dict(berths=3, ships=8, tides=5, factor=15),
    ], base_seed=200)

    for objective in ("makespan", "idle"):
        rows = run_benchmark(instances, objective=objective,
                             heuristics=["first_fit", "greedy", "grasp"],
                             time_limit=30, verbose=False)
        print("\n" + format_table(rows))
        print(format_summary(rows))

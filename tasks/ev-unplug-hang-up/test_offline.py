"""Offline tests of the scoring (pure NumPy; no simulator): drive task.evaluate() with made-up connector, door and
gripper states for the paths that matter, and check the score at each point.

    PYTHONPATH=<sim-tools/yam_bd62771> python test_offline.py

The latch load is read from the inlet catch's spring (20 N per mm of travel), here given directly: a yank pushes the
catch 2.5 mm (50 N); the cable's weight resting the hook on it pushes it a fraction of a millimetre.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import task as T  # noqa: E402
from partkit import FlapState, PartState  # noqa: E402

Z3 = np.zeros(3)


def conn(nose, latch_mm=0.0):
    p = np.asarray(nose, float)
    return PartState(pos=p, quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]), lin_vel=Z3, ang_vel=Z3, low=p, high=p,
                     sliders={"latch": -latch_mm / 1000})


def door(angle_deg):
    p = np.asarray(T.DOOR_HINGE, float)
    return FlapState(pos=p, quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]), lin_vel=Z3, ang_vel=Z3, low=p, high=p,
                     angle=math.radians(angle_deg))


def catch(newtons=0.0):
    """The inlet catch's state with ``newtons`` on it (its spring: 20 N per mm)."""
    p = np.array([T.X_MOUTH, T.YP, T.ZP])
    return FlapState(pos=p, quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]), lin_vel=Z3, ang_vel=Z3, low=p, high=p,
                     travel=-newtons / T.CATCH_RATE)


def hand(nose, opening=0.0574, offset=np.zeros(3)):
    """A right hand holding (or open around) the connector from behind: TCP 10 cm behind the nose."""
    tcp = np.asarray(nose, float) + np.array([-0.10, 0.0, 0.0]) + offset
    tip = tcp + 0.013 * np.array([1.0, 0.0, 0.0])
    return (tip, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), opening, (opening / 2, opening / 2))


def away():
    return (np.array([0.0, -0.3, 1.1]), np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0]), 0.08, (0.04, 0.04))


def run(steps):
    """steps: list of (connector state, door state, grippers dict[, newtons on the catch]). Returns the score after
    each."""
    T._reset_state()
    out = []
    for c, d, gs, *f in steps:
        T._count_step()
        out.append(T.evaluate({"connector": c, "port_door": d, "inlet_catch": catch(f[0] if f else 0.0)}, gs)[0])
    return out


SEAT = T.SEATED_NOSE
OUT = np.array([T.X_MOUTH - 0.10, T.YP, T.ZP])
HUNG = np.array([T.HX - 0.003, T.HY, T.HZ])
OPEN, SHUT = -110.0, 0.0
NOBODY = {"left_robot": away(), "right_robot": away()}


def holding(nose):
    return {"left_robot": away(), "right_robot": hand(nose)}


def check(name, got, want):
    ok = abs(got - want) < 1e-9
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {got} (want {want})")
    return ok


def main():
    results = []
    # the geometry helpers
    results.append(check("holds: pads on the body's sides", float(T.holds(hand(SEAT), conn(SEAT))), 1.0))
    results.append(check("holds: hand open 80 mm", float(T.holds(hand(SEAT, 0.08), conn(SEAT))), 0.0))
    results.append(check("let go: hand 10 cm back", float(T.clear_of(hand(SEAT, 0.08, np.array([-0.12, 0, 0])),
                                                                    conn(SEAT))), 1.0))
    results.append(check("in the inlet when seated", float(T.in_inlet(conn(SEAT))), 1.0))
    results.append(check("in the holster when hung", float(T.in_holster(conn(HUNG))), 1.0))
    results.append(check("not in the holster 3 cm short", float(T.in_holster(conn(HUNG - [0.03, 0, 0]))), 0.0))
    results.append(check("still hung 1.5 cm from the back", float(T.in_holster(conn(np.array([T.HX - 0.015, T.HY, T.HZ])))), 1.0))
    results.append(check("not hung 2.5 cm from the back", float(T.in_holster(conn(np.array([T.HX - 0.025, T.HY, T.HZ])))), 0.0))

    # 1. nothing happens
    results.append(check("nothing happens", run([(conn(SEAT), door(OPEN), NOBODY)] * 30)[-1], 0.0))
    # 2. the full sequence
    seq = [(conn(SEAT), door(OPEN), holding(SEAT))] * 3
    seq += [(conn(SEAT, 7.5), door(OPEN), holding(SEAT))] * 3                       # button held: released
    seq += [(conn(SEAT - [0.015, 0, 0], 7.5), door(OPEN), holding(SEAT - [0.015, 0, 0]))] * 3
    seq += [(conn(OUT), door(OPEN), holding(OUT))] * 3                              # out and held
    seq += [(conn(HUNG), door(OPEN), holding(HUNG))] * 3                            # in the holster, still held
    seq += [(conn(HUNG), door(OPEN), NOBODY)] * 25                                  # let go, still
    seq += [(conn(HUNG), door(SHUT), NOBODY)] * 3                                   # door shut
    s = run(seq)
    results.append(check("button held while home", s[5], 0.25))
    results.append(check("out of the inlet and held", s[11], 0.5))
    results.append(check("in the holster but still held", s[14], 0.5))
    results.append(check("hung, let go, still 1 s, door open", s[-4], 0.75))
    results.append(check("and the door shut", s[-1], 1.0))
    # 3. yank: pulled against the closed latch (the hook pushes the catch 2.5 mm: 50 N)
    seq = [(conn(SEAT), door(OPEN), holding(SEAT))] * 3
    seq += [(conn(SEAT - [0.0035, 0, 0]), door(OPEN), holding(SEAT - [0.0035, 0, 0]), 50.0)] * 4
    seq += [(conn(SEAT, 7.5), door(OPEN), holding(SEAT))] * 3
    seq += [(conn(OUT), door(OPEN), holding(OUT))] * 3
    results.append(check("yank, then unplugged properly", run(seq)[-1], 0.0))
    # 4. early release: pressed (0.25), let go while home, then pulled against the latch
    seq = [(conn(SEAT, 7.5), door(OPEN), holding(SEAT))] * 3
    seq += [(conn(SEAT), door(OPEN), holding(SEAT))] * 2
    seq += [(conn(SEAT - [0.0035, 0, 0]), door(OPEN), holding(SEAT - [0.0035, 0, 0]), 50.0)] * 4
    s = run(seq)
    results.append(check("early release: credit for the press", s[2], 0.25))
    results.append(check("early release: then pulled on the closed latch", s[-1], 0.0))
    # 5. drop: out and held, then let go in mid-air and it falls to the table
    low = np.array([T.X_MOUTH - 0.15, T.YP, T.TOP + 0.132])   # its strain relief (13 cm below the nose) on the table
    seq = [(conn(SEAT, 7.5), door(OPEN), holding(SEAT))] * 3
    seq += [(conn(OUT), door(OPEN), holding(OUT))] * 3
    seq += [(conn(OUT), door(OPEN), NOBODY)] * 2
    seq += [(conn(low), door(OPEN), NOBODY)] * 3
    s = run(seq)
    results.append(check("let go in mid-air (still above the table)", s[7], 0.5))
    results.append(check("dropped onto the table", s[-1], 0.0))
    # 6. the hook resting on its catch under the cable's weight (5 N), then held by a hand: no yank
    seq = [(conn(SEAT - [0.0012, 0, 0]), door(OPEN), NOBODY, 5.0)] * 10
    seq += [(conn(SEAT - [0.0012, 0, 0]), door(OPEN), holding(SEAT - [0.0012, 0, 0]), 5.0)] * 10
    results.append(check("hook resting on its catch, then held", run(seq)[-1], 0.0))
    seq += [(conn(SEAT, 7.5), door(OPEN), holding(SEAT))] * 2
    results.append(check("... then released properly", run(seq)[-1], 0.25))
    # 6b. a tug of 50 N for one step (0.05 s, under the 0.1 s): not a yank; 39 N held: not a yank
    seq = [(conn(SEAT - [0.0035, 0, 0]), door(OPEN), holding(SEAT - [0.0035, 0, 0]), 50.0)]
    seq += [(conn(SEAT - [0.003, 0, 0]), door(OPEN), holding(SEAT - [0.003, 0, 0]), 39.0)] * 10
    seq += [(conn(SEAT, 7.5), door(OPEN), holding(SEAT))] * 2
    results.append(check("a 0.05 s tug and a 39 N pull: no yank", run(seq)[-1], 0.25))
    # 6c. a press just short of 6.5 mm whose hook still slipped under the catch's lip: out with no yank counts as released
    seq = [(conn(SEAT, 6.0), door(OPEN), holding(SEAT))] * 3
    seq += [(conn(SEAT - [0.02, 0, 0], 6.0), door(OPEN), holding(SEAT - [0.02, 0, 0]))] * 2
    seq += [(conn(OUT), door(OPEN), holding(OUT))] * 3
    s = run(seq)
    results.append(check("press 6 mm: no credit while home", s[2], 0.0))
    results.append(check("... but out with no yank: released and out", s[-1], 0.5))
    # 7. the door shut alone earns nothing
    results.append(check("door shut, connector untouched", run([(conn(SEAT), door(SHUT), NOBODY)] * 25)[-1], 0.0))

    # 8. the trace is plain JSON (a numpy bool in it stopped the first simulator check)
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        T.TRACE_PATH = os.path.join(d, "trace.jsonl")
        run([(conn(SEAT, 7.5), door(OPEN), holding(SEAT))] * 2)
        rows = open(T.TRACE_PATH).read().splitlines()
        T.TRACE_PATH = None
    results.append(check("trace rows written", float(len(rows)), 2.0))

    print(f"{sum(results)}/{len(results)} passed")
    return all(results)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)

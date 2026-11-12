"""Offline checks of demo.py, the scripted demo. No simulator and no pytest needed:

    python3 test_demo.py                              # prints ok / BAD per check; exits 1 if any check fails
    YAM_ARENA_SRC=<folder> python3 test_demo.py       # also plays every variant through run_kin.py's model

The plain checks need only this folder. The kinematic check plays every variant as run_kin.py does (URDF kinematics
and IK, the first-order PD lag, the finger-pad touch model, phoneui and the embodiment's success rule) and needs the
yam_arena source: set YAM_ARENA_SRC to the folder that holds the yam_arena package, or it is skipped.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import demo as D      # noqa: E402
import layout as L    # noqa: E402

GLASS_SIZE_MM = (L.GLASS_SIZE[0] * 1000.0, L.GLASS_SIZE[1] * 1000.0)
# Farthest x (m) the right arm reaches straight down at y (m), pads 2.5 mm into the glass and 30 mm above it
# (offline IK on the archived kinematics; the same table as test_phone.py's, plus its 30 mm column).
REACH_IN = ((-0.064, 0.45), (-0.032, 0.43), (0.0, 0.41), (0.032, 0.385), (0.040, 0.375), (0.048, 0.37), (0.064, 0.35))
REACH_30 = ((-0.064, 0.43), (-0.032, 0.42), (0.0, 0.39), (0.032, 0.37), (0.040, 0.36), (0.048, 0.355), (0.064, 0.33))


def _reach(table, y):
    for (y0, x0), (y1, x1) in zip(table, table[1:]):
        if y0 <= y <= y1:
            return x0 + (x1 - x0) * (y - y0) / (y1 - y0)
    return None


def height_mm(w) -> float:
    """Commanded pad height above the glass (mm) of a waypoint."""
    return (w.pos[2] - D.PAD_TIP_PAST_TCP - L.GLASS_Z) * 1000.0


def screen_of(w):
    return L.world_to_screen(w.pos[0], w.pos[1])


def plans():
    return {v: D.plan(v)["right"] for v in D.VARIANTS}


# -- checks --------------------------------------------------------------------------------------------------------

def test_imports_stay_pure():
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import demo; p = demo.plan('full'); "
            "bad = [m for m in sys.modules if m.split('.')[0] in ('isaacsim', 'isaaclab', 'omni', 'pxr', 'carb', "
            "'torch')]; assert not bad, bad; print(type(p['right'][0]).__name__)")
    # numpy is allowed: where yam_arena is installed (the simulator machine), demo uses its Waypoint, which imports
    # numpy; only the simulator's own packages must stay unloaded.
    r = subprocess.run([sys.executable, "-I", "-c", code, HERE], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "Waypoint", r.stdout


def test_every_variant_builds_a_right_arm_plan():
    assert set(D.EXPECTED) == set(D.VARIANTS) and "full" in D.VARIANTS
    for v, wps in plans().items():
        assert set(D.plan(v)) == {"right"}, v                      # the left arm stays at READY
        assert wps and all(w.duration > 0 for w in wps), v
        assert all(w.yaw == D.YAW for w in wps), v                  # one grip orientation throughout
        finger = D.HALF_OPEN if v == "open_gripper" else D.FINGER_CLOSED
        assert all(w.finger == finger for w in wps), v
        assert all(len(w.pos) == 3 and all(math.isfinite(c) for c in w.pos) for w in wps), v
        assert D.duration(wps) + 3.0 < 2400 / 20.0, v               # plan + yam_scripted's hold within the run limit


def test_variant_comes_from_the_environment_and_typos_fail():
    old = os.environ.get(D.VARIANT_ENV)
    try:
        os.environ.pop(D.VARIANT_ENV, None)
        assert D.variant() == "full" and D.plan() == D.plan("full")
        os.environ[D.VARIANT_ENV] = "hang_up"
        assert D.variant() == "hang_up" and D.plan() == D.plan("hang_up")
        os.environ[D.VARIANT_ENV] = "hangup"
        try:
            D.plan()
        except ValueError:
            pass
        else:
            raise AssertionError("an unknown variant must fail")
    finally:
        if old is None:
            os.environ.pop(D.VARIANT_ENV, None)
        else:
            os.environ[D.VARIANT_ENV] = old


def test_taps_aim_into_the_glass_inside_their_hit_areas():
    for v, wps in plans().items():
        for w in wps:
            if not (w.label.startswith("tap ") and w.label.endswith(": down")):
                continue
            screen, name = w.label[4:-6].split()[:2]
            u, vv = screen_of(w)
            assert abs(height_mm(w) + D.PRESS_MM) < 1e-6, (v, w.label)
            assert 2.0 <= D.PRESS_MM <= 3.0
            if name in L.AIM.get(screen, {}):
                assert abs(u - L.aim(screen, name)[0]) < 1e-6 and abs(vv - L.aim(screen, name)[1]) < 1e-6
            if screen in ("passcode", "dialpad", "calling") and name in L.AIM.get(screen, {}):
                r = L.hit_rect(screen, name)
                margin = min(u - r[0], r[2] - u, vv - r[1], r[3] - vv)
                assert margin >= 8.0, (v, w.label, margin)
            assert L.in_display(u, vv), (v, w.label)


def test_presses_hold_and_lifts_clear_between_taps():
    for v, wps in plans().items():
        downs = [i for i, w in enumerate(wps) if w.label.endswith(": down")]
        for i in downs:
            nxt = wps[i + 1]
            assert nxt.label.endswith(": press") and 0.15 <= nxt.duration <= 1.0 and nxt.pos == wps[i].pos, (v, i)
        for a, b in zip(downs, downs[1:]):
            peak = max(height_mm(w) for w in wps[a:b])
            deliberate = v == "no_lift_11" and wps[b].label == "tap dialpad 1 again: down"
            assert peak <= 2.0 if deliberate else peak >= 10.0, (v, wps[b].label, peak)


def test_swipe_starts_low_and_rises_far_enough():
    for v, wps in plans().items():
        slides = [w for w in wps if w.label == "swipe: slide up"]
        if v == "no_swipe":
            assert not slides
            continue
        (s,) = slides
        start = next(w for w in wps if w.label == "swipe: down")
        u0, v0 = screen_of(start)
        u1, v1 = screen_of(s)
        assert L.in_rect(L.SWIPE_ZONE, u0, v0) and abs(u1 - u0) < 1e-6, v
        assert v1 - v0 >= L.SWIPE_MIN_MM + 15.0, (v, v1 - v0)        # a 15 mm margin for lag and slip
        assert height_mm(s) <= -1.5 and L.in_display(u1, v1)


def test_targets_within_straight_down_reach():
    """Every waypoint over the display is inside the offline IK reach envelope with 5 mm to spare."""
    for v, wps in plans().items():
        for w in wps[1:]:
            u, vv = screen_of(w)
            if abs(u) > L.U_HALF:
                continue                                             # beside the phone, near the arm: easy reach
            h = height_mm(w)
            x_in, x_30 = _reach(REACH_IN, w.pos[1]), _reach(REACH_30, w.pos[1])
            limit = x_in + (x_30 - x_in) * max(0.0, h + D.PRESS_MM) / (30.0 + D.PRESS_MM) if h <= 30 else x_30
            assert w.pos[0] <= limit - 0.005, (v, w.label, round(w.pos[0], 3), round(limit, 3))


def test_moves_end_clear_of_the_phone_and_hold_long_enough():
    for v, wps in plans().items():
        last = wps[-1]
        assert last.label == "clear: hold still" and height_mm(last) >= 30.0, v
        u, _ = screen_of(last)
        assert abs(u) > GLASS_SIZE_MM[1] / 2 + 20.0, v                 # beside the phone, not over it
        if D.EXPECTED[v] == 1.0:
            # success needs 3 s with nothing on the glass after the last lift, then 1 s more; yam_scripted stops
            # 3 s after the plan ends
            i = max(j for j, w in enumerate(wps) if w.label.endswith(": lift") and not w.label.startswith("clear"))
            assert sum(w.duration for w in wps[i + 1:]) >= 4.5 and last.duration >= 4.5, v


def test_expected_scores_are_the_spec_levels():
    assert D.EXPECTED["full"] == 1.0 and D.EXPECTED["guess_passcode"] == 0.0 and D.EXPECTED["no_swipe"] == 0.0
    assert D.EXPECTED["stop_at_dialpad"] == 0.4 and D.EXPECTED["dial_no_call"] == 0.6
    assert D.EXPECTED["wrong_digit_fixed"] == 0.9 and D.EXPECTED["hang_up"] == 0.8
    assert all(s in (0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0) for s in D.EXPECTED.values())


def test_kinematics_of_every_variant():
    """Every variant through run_kin.py's model reaches its expected score and passes every check of run_kin.py."""
    src = os.environ.get("YAM_ARENA_SRC")
    if not src:
        try:
            import yam_arena  # noqa: F401
        except Exception:  # noqa: BLE001
            print("     (skipped: set YAM_ARENA_SRC to the folder holding the yam_arena package)")
            return
    import warnings

    import numpy as np

    import run_kin as K

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    yam_arena, C = K.load_harness(src)
    H = K.Harness(yam_arena, C)
    pads, _ = K.pad_table()
    other, _ = K.non_pad_points(yam_arena, os.environ.get("YAM_HULLS"))
    bad = []
    with np.errstate(all="ignore"):
        for v in D.VARIANTS:
            r = K.play(v, H, pads, other, 1.0, None)
            failed = [k for k, ok in r["checks"].items() if not ok]
            if failed or r["score"] != D.EXPECTED[v]:
                bad.append((v, r["score"], failed))
    assert not bad, bad


# -- runner --------------------------------------------------------------------------------------------------------

def main():
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    bad = 0
    for name, fn in tests:
        try:
            fn()
            print(f"ok   {name}")
        except Exception:
            bad += 1
            print(f"BAD  {name}")
            traceback.print_exc()
    print(f"{len(tests) - bad}/{len(tests)} passed" + (f", FAILURES: {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

"""Offline tests of the milk-steaming task's scoring (no simulator): the milk model, the score's levels and zeros, and
the scripted demo run kinematically (demo.py --plan-only's DryCtx) with its four failure versions.

    python -m pytest test_offline.py -q        (yam_arena on the path)
"""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def fresh(name="task"):
    if name in sys.modules:
        return importlib.reload(sys.modules[name])
    return importlib.import_module(name)


@pytest.fixture()
def T():
    t = fresh("task")
    t.reset_state()
    return t


def held(T, depth, tilt_deg=0.0, milk=None):
    """The pitcher's rotation and origin with the wand's tip ``depth`` under the surface, 12 mm toward the handle."""
    m = milk or T._S["milk"]
    a = math.radians(tilt_deg)
    R = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    h = m.volume_m3() / T.AREA
    p = np.array([T.TIP[0] + 0.012, T.TIP[1], 0.0])
    zs = T.BOTTOM_T + h
    p[2] = T.TIP[2] + depth - zs
    return R, p


def state(T, R, p, v=(0, 0, 0)):
    from partkit import PartState
    from yam_arena import ik

    return PartState(pos=np.asarray(p, float), quat_xyzw=np.asarray(ik.matrix_to_quat_xyzw(R), float),
                     lin_vel=np.asarray(v, float), ang_vel=np.zeros(3), low=p, high=p, sliders={})


def run(T, seconds, angle, depth=None, R=None, p=None, track=False):
    """Steam for ``seconds`` with the knob at ``angle``; the pitcher held so the tip is ``depth`` under the surface
    (re-placed every step if ``track``), or at R/p."""
    m = T._S["milk"]
    for _ in range(round(seconds * 20)):
        if depth is not None and (track or R is None):
            R, p = held(T, depth)
        m.step(0.05, angle, R, p)


def test_scene_geometry(T):
    assert abs(T.FRONT_X - 0.375) < 1e-9
    assert T.Y1 < 0.65 and T.Y0 > -0.65                                    # on the table
    assert 0.15 < T.TIP[2] - T.TOP < 0.18 and T.TIP[0] < T.FRONT_X - 0.05   # the tip, out in front of the panel
    assert abs(T.MILK_KG0 - 0.1751) < 1e-3
    assert T.steam_fraction(0.0) == 0 and T.steam_fraction(math.radians(10)) == 0
    assert T.steam_fraction(T.KNOB_OPEN) == 1.0


def test_heating_rate_and_water(T):
    """Full steam, the tip deep (no air): from 4 to about 60 degrees C in 16-20 s, adding 8-11% of water."""
    m = T._S["milk"]
    m.purge_s = 2.0
    t = 0.0
    while m.T < 60.0 and t < 60:
        R, p = held(T, 0.025)
        m.step(0.05, T.KNOB_OPEN, R, p)
        t += 0.05
    assert 16.0 < t < 20.0
    assert 0.08 < (m.kg - T.MILK_KG0) / T.MILK_KG0 < 0.11
    assert m.air_s == 0.0 and m.foam_m3 == 0.0


def test_air_intake_depends_on_depth(T):
    for depth, expect in ((0.004, 1.0), (0.0115, 0.5), (0.020, 0.0)):
        T.reset_state()
        m = T._S["milk"]
        m.purge_s = 2.0
        run(T, 1.0, T.KNOB_OPEN, depth=depth, track=True)
        assert abs(m.foam_m3 / (T.AIR_M3_S * 1.0) - expect) < 0.06
    T.reset_state()
    run(T, 1.0, math.radians(50), depth=0.004, track=True)                 # half open: half the air
    assert abs(T._S["milk"].foam_m3 / T.AIR_M3_S - 0.5) < 0.06


def test_spatter_only_when_the_tip_leaves_with_steam_on(T):
    m = T._S["milk"]
    m.purge_s = 2.0
    run(T, 2.0, T.KNOB_OPEN, depth=0.02, track=True)
    R, p = held(T, -0.01)                                                  # out, steam on: spatter
    m.step(0.05, T.KNOB_OPEN, R, p)
    assert m.spatter_s is not None
    T.reset_state()
    m = T._S["milk"]
    m.purge_s = 2.0
    run(T, 2.0, T.KNOB_OPEN, depth=0.02, track=True)
    m.step(0.05, 0.0, *held(T, 0.02))                                      # steam off first
    m.step(0.05, 0.0, *held(T, -0.03))
    far = (np.eye(3), np.array([0.2, -0.2, T.TOP]))                        # then a purge into the air: allowed
    for _ in range(20):
        m.step(0.05, T.KNOB_OPEN, *far)
    assert m.spatter_s is None


def test_purge_counts_only_before_steaming_and_away_from_the_milk(T):
    m = T._S["milk"]
    R, p = held(T, -0.03)                                                  # the tip 3 cm above the milk, inside
    run(T, 2.0, T.KNOB_OPEN, R=R, p=p)
    assert m.purge_s == 0.0
    far = (np.eye(3), np.array([0.2, -0.2, T.TOP]))
    run(T, 1.2, T.KNOB_OPEN, R=far[0], p=far[1])
    assert abs(m.purge_s - 1.2) < 1e-6
    run(T, 0.5, T.KNOB_OPEN, depth=0.02, track=True)
    assert m.purged_before >= T.PURGE_S and not m.unpurged_water


def test_spill_when_tilted(T):
    m = T._S["milk"]
    R, p = held(T, 0.02, tilt_deg=10)
    m.step(0.05, 0.0, R, p)
    assert m.spill_m3 == 0.0
    R, p = held(T, 0.02, tilt_deg=55)
    m.step(0.05, 0.0, R, p)
    assert m.spill_m3 * 1e6 > T.SPILL_ML


def test_score_levels(T):
    m = T._S["milk"]
    st = T._STATE
    assert T.evaluate() == (0.0, False)                                   # doing nothing
    m.steamed, m.max_T, m.T = True, 45.0, 45.0
    assert T.evaluate()[0] == 0.25
    m.max_T = m.T = 60.0
    assert T.evaluate()[0] == 0.5                                          # no stretch
    m.foam_m3 = 0.30 * T.MILK_ML0 * 1e-6
    assert T.evaluate()[0] == 0.75
    m.purged_before, m.u = 2.0, 0.0
    st["set_down"] = True
    assert T.evaluate() == (1.0, True)
    m.foam_m3 = 0.45 * T.MILK_ML0 * 1e-6                                  # a cappuccino's stretch
    assert T.evaluate()[0] == 0.5
    m.foam_m3 = 0.30 * T.MILK_ML0 * 1e-6
    m.max_T = 71.0                                                         # it went over 70 once
    assert T.evaluate()[0] == 0.0
    m.max_T = 60.0
    m.spatter_s = 12.0
    assert T.evaluate()[0] == 0.0


def test_set_down(T):
    R = np.eye(3)
    T.milk_step(state(T, R, (0.24, 0.03, T.TOP + 0.0005)), 0.0, 0.05)
    assert T._STATE["set_down"]
    T.milk_step(state(T, R, (0.24, 0.03, T.TOP + 0.03)), 0.0, 0.05)        # held above the counter
    assert not T._STATE["set_down"]


@pytest.mark.parametrize("variant,expect", [("", 1.0), ("no_purge", 0.75), ("no_stretch", 0.5), ("pull_out", 0.0),
                                            ("overheat", 0.0)])
def test_demo_dry_run(variant, expect, monkeypatch):
    monkeypatch.setenv("DEMO_VARIANT", variant)
    D = fresh("demo")
    out = D.plan_only(str(HERE / "task.py"))
    assert out["result"]["score"] == pytest.approx(expect)
    assert out["worst_ik_mm"]["left"] < 1 and out["worst_ik_mm"]["right"] < 1
    assert min(out["min_joint_margin_deg"].values()) > 10
    if variant == "":
        r = out["result"]
        assert 55 <= r["T"] <= 65 and 0.25 <= r["stretch"] <= 0.35 and r["set_down"]
        assert all(v[0] is None or v[0] > 2.0 for v in out["clearance_mm"].values())

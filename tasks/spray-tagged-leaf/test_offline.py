"""Offline tests of the tagged-leaf spray task's scoring (no simulator): the spray model, the touch check, the score,
and the scripted demo run kinematically (demo.py --plan-only's DryCtx) with its two failure versions.

    python -m pytest test_offline.py -q        (yam_arena on the path)
"""

from __future__ import annotations

import importlib
import math
import os
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
    t.reset_state(seed=5)
    return t


@pytest.fixture()
def D():
    return fresh("demo")


def bottle_state(T, D, tcp, Rg, travel=0.0):
    """The bottle's part state for the right gripper at ``tcp``/``Rg`` holding it as the demo does."""
    from partkit import PartState

    Rs = D.bottle_from_grip(Rg)
    origin = tcp - Rs @ D.TCP_IN_SPRAYER
    Rp = Rs @ T.R_SP
    return PartState(pos=origin, quat_xyzw=D.quat_xyzw(Rp), lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=origin,
                     high=origin, sliders={"trigger": -travel})


def card_state(T, centre, Rc):
    from partkit import PartState

    from demo import quat_xyzw

    return PartState(pos=np.asarray(centre, float), quat_xyzw=quat_xyzw(Rc), lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                     low=centre, high=centre, sliders={})


def squeeze_at(T, b_pressed, b_released, card, steps_between=28, n=2):
    """n full squeezes (trigger to 12 mm for 4 steps, then released), then let everything land."""
    for _ in range(n):
        for _ in range(4):
            T.spray_step(b_pressed, card, 0.05)
        for _ in range(steps_between):
            T.spray_step(b_released, card, 0.05)
    for _ in range(40):
        T.spray_step(b_released, card, 0.05)


def test_plant_and_bottle(T):
    assert len(T.PLANT.F) == 8 * 13 * 8 * 2
    assert T.PLANT.tagged.sum() == 13 * 8 * 2
    assert abs(T.MASS - 0.286) < 0.002
    # the tagged leaf points toward the arms' front right (-135 degrees)
    c = T.PLANT.V[T.PLANT.F[T.PLANT.tagged]].reshape(-1, 3).mean(0) - T.PLANT_ORIGIN
    assert abs(math.degrees(math.atan2(c[1], c[0])) + 135) < 5


def test_planned_spray_scores_full(T, D):
    P = D.plan(T)
    b1 = bottle_state(T, D, P["tcp_spray"], P["R_spray"], travel=0.012)
    b0 = bottle_state(T, D, P["tcp_spray"], P["R_spray"], travel=0.0)
    nz, axis, up = T.nozzle_pose(b1)
    assert np.allclose(nz, P["nozzle"], atol=1e-6) and np.allclose(axis, P["axis"], atol=1e-6)
    card = T.card_box(card_state(T, P["card_final"], P["R_card"]))
    squeeze_at(T, b1, b0, card)
    r = T.report()
    assert r["squeezes"] == 2 and r["airborne"] == 0
    assert r["tagged_wet"] >= 0.8 and r["other_wet_cm2"] <= 1.0
    assert T.evaluate() == (1.0, True)


def test_no_card_wets_other_leaves(T, D):
    P = D.plan(T)
    b1 = bottle_state(T, D, P["tcp_spray"], P["R_spray"], travel=0.012)
    b0 = bottle_state(T, D, P["tcp_spray"], P["R_spray"], travel=0.0)
    squeeze_at(T, b1, b0, None)
    assert T.report()["other_wet_cm2"] > 1.0
    assert T.evaluate()[0] == 0.0


def test_partial_stroke_and_held_trigger(T, D):
    P = D.plan(T)
    part = bottle_state(T, D, P["tcp_spray"], P["R_spray"], travel=0.009)     # 9 mm: short of 10
    full = bottle_state(T, D, P["tcp_spray"], P["R_spray"], travel=0.012)
    for _ in range(10):
        T.spray_step(part, None, 0.05)
    assert T.LOG["squeezes"] == 0
    for _ in range(30):                                                       # held down: one squeeze only
        T.spray_step(full, None, 0.05)
    assert T.LOG["squeezes"] == 1
    for _ in range(3):
        T.spray_step(part, None, 0.05)                                        # back to 9 mm: not re-armed
        T.spray_step(full, None, 0.05)
    assert T.LOG["squeezes"] == 1


def test_no_spray_past_75_degrees(T, D):
    P = D.plan(T)
    Rs = D.sprayer_frame(np.array([0.2, 0.0, -1.0]), 0.0)                   # axis 79 degrees below horizontal
    Rg = D.grip_on_bottle(Rs)
    b = bottle_state(T, D, P["tcp_spray"] + np.array([0, 0, 0.2]), Rg, travel=0.012)
    assert T.nozzle_pose(b)[2][2] < T.UP_MIN
    T.spray_step(b, None, 0.05)
    assert T.LOG["squeezes"] == 0 and T.LOG["squeezes_dry"] == 1


def test_touch_check(T, D):
    P = D.plan(T)
    leaf_pt = T.PLANT.V[T.PLANT.F[T.PLANT.tagged][40, 0]]
    I3 = np.eye(3)
    assert T.touching([("probe", (leaf_pt + np.array([0, 0, 0.0005]), I3, np.array([0.003] * 3)))]) == "probe"
    assert T.touching([("probe", (leaf_pt + np.array([0, 0, 0.010]), I3, np.array([0.003] * 3)))]) is None
    # the planned card, bottle and both grippers clear the plant
    b = bottle_state(T, D, P["tcp_spray"], P["R_spray"])
    c = card_state(T, P["card_final"], P["R_card"])
    gs = {"left_robot": (P["tcp_card"], P["R_left"][:, 2], P["R_left"][:, 0], (0.001, 0.001)),
          "right_robot": (P["tcp_spray"], P["R_spray"][:, 2], P["R_spray"][:, 0], (0.026, 0.026))}
    T.check_touch(b, c, gs)
    assert T._STATE["touched"] is None
    # the card 3 cm higher meets the tagged leaf (its lowest point, 12 mm above the card, is at the stalk, just inside
    # the card's inner edge; over the card the blade rises)
    T.check_touch(b, card_state(T, P["card_final"] + np.array([0, 0, 0.030]), P["R_card"]), gs)
    assert T._STATE["touched"] == "the card touched the plant"
    assert T.evaluate()[0] == 0.0


def test_partial_credit(T):
    T.PLANT.stain[T.PLANT.tagged] = T.PLANT.area[T.PLANT.tagged] * -math.log(1 - 0.4)   # 40% wet
    assert abs(T.evaluate()[0] - 0.5) < 1e-6 and T.evaluate()[1] is False
    T.PLANT.stain[~T.PLANT.tagged][:1] = 0
    T.PLANT.stain[np.where(~T.PLANT.tagged)[0][:40]] = 1.0                   # fully wet: well over 1 cm^2
    assert T.PLANT.other_wet_cm2() > 1.0 and T.evaluate()[0] == 0.0


def test_beads_on_card_move_with_it(T, D):
    P = D.plan(T)
    b1 = bottle_state(T, D, P["tcp_spray"], P["R_spray"], travel=0.012)
    b0 = bottle_state(T, D, P["tcp_spray"], P["R_spray"], travel=0.0)
    card = T.card_box(card_state(T, P["card_final"], P["R_card"]))
    squeeze_at(T, b1, b0, card, n=1)
    sp = T._S["spray"]
    on = sp.bead_on_card
    assert on.any()
    before = sp.bead_positions(card)[on]
    moved = (card[0] + np.array([-0.16, 0, 0.0]), card[1], card[2])
    after = sp.bead_positions(moved)[on]
    assert np.allclose(after - before, [-0.16, 0, 0], atol=1e-9)
    others = ~on & (sp.beads[:, 2] > -1)
    assert np.allclose(sp.bead_positions(moved)[others], sp.bead_positions(card)[others])


@pytest.mark.parametrize("variant,expect", [("", 1.0), ("no_card", 0.0), ("touch", 0.0)])
def test_demo_dry_run(variant, expect, monkeypatch):
    monkeypatch.setenv("DEMO_VARIANT", variant)
    D = fresh("demo")
    out = D.plan_only(str(HERE / "task.py"))
    assert out["result"]["score"] == pytest.approx(expect)
    assert out["worst_ik_mm"]["left"] < 1 and out["worst_ik_mm"]["right"] < 1
    assert min(out["min_joint_margin_deg"].values()) > 5
    if variant == "":
        assert out["first_touch"] is None and out["result"]["squeezes"] == 2

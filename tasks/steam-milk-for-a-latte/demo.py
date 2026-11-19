"""Scripted full-score demo for the milk-steaming task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1400] [--no-record] [--cam-width 320 --cam-height 176]
    python demo.py task.py - --plan-only        (no simulator: the arms follow their plans exactly, the pitcher moves
                                                 rigidly with the right hand, the knob turns with the left hand, and
                                                 the task's own milk model runs on those poses)

The demo is open-loop: the layout is fixed (object_jitter=0), so every move is planned in advance from the scene's
known geometry and the steam's known rates, and it never reads the pitcher's, the knob's or the milk's state. The only
thing it reads back is the arms' own joint positions (as a policy does): every step each arm adds its measured
tracking error into a joint trim, which takes out the sag of its soft joint drives under the pitcher's weight (the EV
and spray demos do the same).

1. The left hand takes the steam knob from the front (the gripper level, pointing into the panel, its fingers above
   and below the knob), moving in slowly. The right hand takes the pitcher round its body from behind (pitched 35
   degrees down, low on the body, the pads on its left and right sides; the handle passes between the fingers), lifts
   it and holds it under the wand with the rim 2.5 cm below the tip.
2. The purge: the left hand turns the knob fully open for 1.5 s and shut again (the steam blows into the air, clear
   of the pitcher).
3. The right hand raises the pitcher until the tip is 4 mm under the milk, 12 mm off the pitcher's axis toward the
   handle. The left hand opens the steam fully.
4. The stretch: for about 3.5 s the right hand lowers the pitcher as the foam rises, keeping the tip 4 mm under the
   surface (the rise is planned from the steam's air intake), until the milk is about 20% more in volume.
5. The roll: the pitcher rises 20 mm, so the tip is deep, and the milk heats until about 56 degrees C (planned from the
   steam's heat); the pitcher comes down 12 mm (the tip stays deep) and the left hand shuts the knob.

SLIP_Z, KNOB_OVER, U_HELD and KNOB_ROLL come from the first simulator runs (see the README).
6. With the steam off, the right hand lowers the pitcher off the wand and sets it down where it started; both hands
   let go.

Writes one MP4 per camera, setup.png (the top camera at step 1) and result.json to the output folder.

Test versions (DEMO_VARIANT=...), each must score as given:
  no_purge    no purge before steaming                                                  [0.75]
  no_stretch  the tip goes straight to 25 mm deep: almost no air                        [0.5]
  pull_out    the pitcher comes off the wand before the steam is shut: milk sprays      [0]
  overheat    the steam stays on 10 s longer: the milk goes over 70 degrees C           [0]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

CONTROL_HZ = 20
DT = 1.0 / CONTROL_HZ
GAP_FULL = 0.0939                 # pad gap at grip 1.0 (2 x 46.95 mm)
FINGER_K = 5000.0                 # gripper drive stiffness per finger (N/m), yam_arena GRIPPER
K_TRACK = 2.0                     # gain of an arm's joint trim on its own measured tracking error (1/s)
TRACK_MAX = math.radians(15.0)
X, Y, Z = np.eye(3)

ALPHA = math.radians(35.0)        # the right gripper pitched down (and rolled half a turn: see pitcher_grip_rotation)
PRE_BACK = 0.08                   # pre-grasp: this far back along the approach
LIFT = 0.05
BELOW = 0.025                     # under the wand: the rim this far below the tip
D_STRETCH = 0.004                 # the tip's depth while stretching
TIP_OFFSET = 0.012                # the tip this far from the pitcher's axis, toward the handle (-x)
STRETCH_TARGET = 0.20             # before the roll; the roll and the steam's water add about 10%
RISE = 0.020                      # the roll: the pitcher rises this much (the tip deeper)
T_CLOSE = 56.0                    # shut the steam when the rehearsal reaches this
DROP_OFF = 0.012                  # before shutting the steam the pitcher comes down this much (the tip stays about
                                  # 17 mm deep), for room between the hands as the knob turns
KNOB_TURN_S = 1.0                 # a quarter turn (the knob's damping resists about 0.3 N*m at this pace's peak)
KNOB_OVER = math.radians(8.0)     # the hand turns this far past the knob's open stop, so the knob stays pressed fully
                                  # open though it slips a few degrees in the fingers while held
SLIP_Z = 0.006                    # the full pitcher settles this far down in the fingers as it is lifted (measured in
                                  # the first simulator run), so the hand holds it this much higher
PURGE_HOLD_S = 1.5
U_HELD = 0.88                     # the steam while the hand holds the knob open: it creeps back to about 80 degrees in
                                  # the fingers (measured in the simulator), so the rehearsal plans with this flow
OUT_DROP = 0.08
VARIANT = os.environ.get("DEMO_VARIANT", "")
ARM_LOG = os.environ.get("ARM_LOG")


def grip_cmd(gap_m: float) -> float:
    return float(np.clip(gap_m / GAP_FULL, 0.0, 1.0))


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def quat_xyzw(Rm):
    from yam_arena import ik

    return np.asarray(ik.matrix_to_quat_xyzw(Rm), float)


# -- geometry -------------------------------------------------------------------------------------------------------
KNOB_ROLL = -math.pi / 2          # the left gripper rolled about its approach at the knob (shut): its wrist then sweeps
                                  # clear of the right arm (rolled half a turn, the wrists met as the knob shut)
R_KNOB0 = np.column_stack([Y, Z, X]) @ rot_z(KNOB_ROLL)   # closing along y, approaching +x, then rolled


def knob_rotation(angle: float) -> np.ndarray:
    """The left gripper on the knob turned ``angle`` rad open (counterclockwise as you face the machine): a turn of
    -angle about its own approach axis (+x)."""
    return R_KNOB0 @ rot_z(-angle)


def pitcher_grip_rotation() -> np.ndarray:
    """Approaching along +x pitched ALPHA down, closing along y, rolled half a turn about the approach (the fingers are
    symmetric; unrolled, the right arm's wrist sits at its joint limit)."""
    a = np.array([math.cos(ALPHA), 0.0, -math.sin(ALPHA)])
    c = Y
    return np.column_stack([c, np.cross(a, c), a]) @ rot_z(math.pi)


def milk_height(T, volume_m3: float) -> float:
    return volume_m3 / T.AREA


def pitcher_at(T, centre_xy, bottom_z):
    """The held pitcher's origin (outer bottom centre) for its centre and bottom height, and the right hand's TCP
    there (SLIP_Z above the grasp point, as the pitcher has settled in the fingers)."""
    p = np.array([centre_xy[0], centre_xy[1], bottom_z])
    return p, p + np.array([0.0, 0.0, T.GRIP_Z + SLIP_Z])


def knob_held(T) -> float:
    """The knob's angle while the hand holds it open (the angle giving U_HELD of the steam)."""
    return T.STEAM_FROM + U_HELD * (T.KNOB_OPEN - T.STEAM_FROM)


def rehearse(T):
    """Run the task's milk model along the ideal plan: the tip held D_STRETCH under the surface at full steam until the
    milk is STRETCH_TARGET more in volume (the pitcher's bottom height every step), then RISE higher over 1 s, then
    hold until T_CLOSE, then the knob shuts over KNOB_TURN_S. Returns the stretch's bottom heights, the steps to hold
    while rolling and the rehearsal's outcome."""
    m = T.Milk()
    KNOB_HELD = knob_held(T)
    m.purge_s = T.PURGE_S                         # the purge happens before (no water from the wand)
    R = np.eye(3)
    cxy = (T.TIP[0] + TIP_OFFSET, T.TIP[1])
    bottoms = []

    def bottom_for(depth):
        return T.TIP[2] + depth - T.BOTTOM_T - milk_height(T, m.volume_m3())

    while m.stretch() < STRETCH_TARGET and len(bottoms) < 20 * CONTROL_HZ:
        b = bottom_for(D_STRETCH)
        bottoms.append(b)
        m.step(DT, KNOB_HELD, R, np.array([cxy[0], cxy[1], b]))
    b0 = bottoms[-1]
    n_rise = CONTROL_HZ
    rise = []
    for k in range(1, n_rise + 1):
        b = b0 + RISE * k / n_rise
        rise.append(b)
        m.step(DT, KNOB_HELD, R, np.array([cxy[0], cxy[1], b]))
    b1 = rise[-1]
    hold = 0
    while m.T < T_CLOSE and hold < 60 * CONTROL_HZ:
        m.step(DT, KNOB_HELD, R, np.array([cxy[0], cxy[1], b1]))
        hold += 1
    n_drop = round(0.5 * CONTROL_HZ)
    for k in range(1, n_drop + 1):
        m.step(DT, KNOB_HELD, R, np.array([cxy[0], cxy[1], b1 - DROP_OFF * k / n_drop]))
    n_close = round(KNOB_TURN_S * CONTROL_HZ)
    for k in range(1, n_close + 1):
        m.step(DT, KNOB_HELD * (1 - k / n_close), R, np.array([cxy[0], cxy[1], b1 - DROP_OFF]))
    return {"stretch_bottoms": bottoms, "rise_bottoms": rise, "hold_steps": hold, "centre": cxy,
            "T": round(m.T, 2), "stretch": round(m.stretch(), 4), "depth_mm": round(1000 * m.depth, 1)}


def plan(T):
    """Every fixed target of the demo."""
    P = {"R_pick": pitcher_grip_rotation(), "R_knob0": knob_rotation(0.0), "R_knob_open": knob_rotation(T.KNOB_OPEN)}
    a = P["R_pick"][:, 2]
    P["tcp_knob"] = np.asarray(T.KNOB_GRIP, float)
    P["knob_pre"] = P["tcp_knob"] - 0.06 * X
    P["g_open_l"] = grip_cmd(0.070)                         # 12 mm clear of the knob each side
    P["g_knob"] = grip_cmd(T.KNOB_ACROSS - 0.010)            # 5 mm into each side: about 25 N a finger
    p0 = np.array([T.PITCHER_XY[0], T.PITCHER_XY[1], T.TOP + 0.0005])
    P["p_start"] = p0
    P["tcp_pick"] = p0 + np.array([0.0, 0.0, T.GRIP_Z])
    P["pick_pre"] = P["tcp_pick"] - PRE_BACK * a
    P["g_open_r"] = grip_cmd(2 * T.R_OUT + 0.016)
    P["g_hold"] = grip_cmd(2 * T.R_OUT - 0.010)              # 5 mm into each side
    reh = rehearse(T)
    P["rehearsal"] = reh
    cxy = reh["centre"]
    h0 = milk_height(T, T.MILK_ML0 * 1e-6)
    P["p_below"], P["tcp_below"] = pitcher_at(T, cxy, T.TIP[2] - T.P_H - BELOW)
    P["p_above_milk"], P["tcp_above_milk"] = pitcher_at(T, cxy, T.TIP[2] - 0.030 - T.BOTTOM_T - h0)
    P["p_in"], P["tcp_in"] = pitcher_at(T, cxy, T.TIP[2] + D_STRETCH - T.BOTTOM_T - h0)
    P["p_deep0"], P["tcp_deep0"] = pitcher_at(T, cxy, T.TIP[2] + 0.025 - T.BOTTOM_T - h0)
    P["tcp_lift"] = P["tcp_pick"] + (LIFT + SLIP_Z) * Z
    P["tcp_place"] = P["tcp_pick"] + SLIP_Z * Z                # set down: the pitcher's bottom on the counter
    return P


# -- the arms -------------------------------------------------------------------------------------------------------
def solve_multi(target, R, base, seed=None):
    from yam_arena import ik

    best = None
    for qi in ([seed] if seed is not None else []) + [None, [0.5, 1.2, 1.2, 0, 1.0, 0], [-0.5, 1.2, 1.2, 0, -1.0, 0],
                                                       [0.0, 1.6, 1.0, -0.5, 0.0, 0.0], [0.0, 0.8, 0.6, 0.2, 0.0, 0.0]]:
        q, pe, re_ = ik.solve(target, ik.matrix_to_quat_xyzw(R), q_init=qi, base_pos=base, iters=300)
        if best is None or pe + 0.01 * re_ < best[1] + 0.01 * best[2]:
            best = (q, pe, re_)
        if pe < 0.0005 and re_ < math.radians(0.3):
            break
    return best


class Arm:
    """One arm: smooth Cartesian or joint-space moves, each starting where the last command left it; every step adds
    its measured tracking error (planned minus measured joints) into a joint trim (the EV demo's)."""

    def __init__(self, name: str, base_pos, ctx):
        from yam_arena import ik
        from yam_arena.constants import READY

        self.name, self.base, self.ctx = name, np.asarray(base_pos, float), ctx
        self.q = np.array(READY, float)
        self.grip = 1.0
        self.pos, self.rot = ik.tcp_pose(self.q, self.base)
        self.trim = np.zeros(6)
        self.cmd = np.r_[self.q, self.grip]
        self.worst_mm = 0.0
        self.worst_deg = 0.0
        self.worst_trim = 0.0
        self.min_margin = 180.0

    def _set(self, q):
        from yam_arena import ik

        self.q = np.asarray(q, float)
        lo, hi = np.asarray(ik.LOWER), np.asarray(ik.UPPER)
        self.min_margin = min(self.min_margin, float(np.degrees(np.min(np.minimum(self.q - lo, hi - self.q)))))
        self._track()

    def _solve(self, pos, rot):
        from yam_arena import ik

        q, pe, re_ = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), self.q, base_pos=self.base, iters=60)
        self.pos, self.rot = np.asarray(pos, float), rot
        self.worst_mm = max(self.worst_mm, pe * 1000)
        self.worst_deg = max(self.worst_deg, math.degrees(re_))
        self._set(q)

    def _track(self):
        measured = self.ctx.joints(self.name)
        if measured is not None:
            self.trim = np.clip(self.trim + K_TRACK * DT * (self.q - measured), -TRACK_MAX, TRACK_MAX)
            self.worst_trim = max(self.worst_trim, float(np.degrees(np.abs(self.trim).max())))
        self.cmd = np.r_[self.q + self.trim, self.grip]

    def move(self, pos, rot, duration: float, grip: float | None = None):
        """Generator: a straight Cartesian move of the TCP to ``pos``/``rot``; ``grip`` 1 open .. 0 closed."""
        from yam_arena import ik

        p0, r0, g0 = self.pos.copy(), self.rot.copy(), self.grip
        g1 = g0 if grip is None else grip
        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            s = ik.smoothstep(k / n)
            self.grip = g0 + s * (g1 - g0)
            self._solve(p0 + s * (np.asarray(pos) - p0), ik.slerp_matrix(r0, rot, s))
            yield

    def path(self, positions, rot):
        """Generator: the TCP through ``positions``, one per control step."""
        for p in positions:
            self._solve(np.asarray(p, float), rot)
            yield

    def move_joints(self, pos, rot, duration: float, grip: float | None = None, seed=None):
        """Generator: a joint-space move to the IK solution for ``pos``/``rot``."""
        from yam_arena import ik

        q1, pe, re_ = solve_multi(pos, rot, self.base, self.q if seed is None else seed)
        self.worst_mm = max(self.worst_mm, pe * 1000)
        self.worst_deg = max(self.worst_deg, math.degrees(re_))
        q0, g0 = self.q.copy(), self.grip
        g1 = g0 if grip is None else grip
        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            s = ik.smoothstep(k / n)
            self.grip = g0 + s * (g1 - g0)
            self._set(q0 + s * (q1 - q0))
            self.pos, self.rot = ik.tcp_pose(self.q, self.base)
            yield
        self.pos, self.rot = np.asarray(pos, float), rot

    def set_grip(self, grip: float, duration: float = 0.8):
        yield from self.move(self.pos, self.rot, duration, grip)

    def hold(self):
        self._track()


def both(*gens):
    gens = list(gens)
    while gens:
        alive = []
        for g in gens:
            try:
                next(g)
                alive.append(g)
            except StopIteration:
                pass
        gens = alive
        if gens:
            yield


def wait(ctx, seconds: float = 0.0, steps: int | None = None):
    for _ in range(max(1, round(seconds * CONTROL_HZ)) if steps is None else steps):
        ctx.left.hold()
        ctx.right.hold()
        yield


def script(T, ctx):
    """Generator: one control step per yield."""
    P = ctx.P
    L, R = ctx.left, ctx.right
    a = P["R_pick"][:, 2]
    reh = P["rehearsal"]

    def solo(arm, gen):
        other = R if arm is L else L
        for _ in gen:
            other.hold()
            yield

    def knob(angle, duration=KNOB_TURN_S):
        return solo(L, L.move(P["tcp_knob"], knob_rotation(angle), duration))

    ctx.phase = "reach"
    # via waypoints high and back, so neither swing passes the machine's side panel or a portafilter handle
    yield from both(L.move_joints(P["knob_pre"] - 0.06 * X + 0.08 * Z, P["R_knob0"], 2.5, grip=P["g_open_l"]),
                    R.move_joints(P["pick_pre"] - 0.05 * X - 0.12 * Y + 0.03 * Z, P["R_pick"], 2.5, grip=P["g_open_r"]))
    yield from both(L.move(P["knob_pre"], P["R_knob0"], 1.0), R.move(P["pick_pre"], P["R_pick"], 1.0))
    # slowly in, then a moment to settle: a faster move lagged 2-3 cm and a finger caught the knob's edge
    yield from both(L.move(P["tcp_knob"], P["R_knob0"], 2.5), R.move(P["tcp_pick"], P["R_pick"], 2.5))
    yield from wait(ctx, 0.5)
    ctx.phase = "grip"
    yield from both(L.set_grip(P["g_knob"], 0.8), R.set_grip(P["g_hold"], 0.8))
    ctx.grasped("pitcher")
    ctx.grasped("knob")
    yield from wait(ctx, 0.5)
    ctx.phase = "lift"
    yield from solo(R, R.move(P["tcp_lift"], P["R_pick"], 1.0))
    yield from solo(R, R.move(P["tcp_below"], P["R_pick"], 2.5))
    yield from wait(ctx, 1.0)
    if VARIANT != "no_purge":
        ctx.phase = "purge"
        yield from knob(T.KNOB_OPEN + KNOB_OVER)
        yield from wait(ctx, PURGE_HOLD_S)
        yield from knob(0.0)
    ctx.phase = "in"
    yield from solo(R, R.move(P["tcp_above_milk"], P["R_pick"], 1.5))
    target = P["tcp_deep0"] if VARIANT == "no_stretch" else P["tcp_in"]
    yield from solo(R, R.move(target, P["R_pick"], 1.5))
    yield from wait(ctx, 1.0)
    ctx.phase = "steam on"
    yield from knob(T.KNOB_OPEN + KNOB_OVER)
    cxy = reh["centre"]
    if VARIANT != "no_stretch":
        ctx.phase = "stretch"
        yield from solo(R, R.path([pitcher_at(T, cxy, b)[1] for b in reh["stretch_bottoms"]], P["R_pick"]))
        ctx.phase = "roll"
        yield from solo(R, R.path([pitcher_at(T, cxy, b)[1] for b in reh["rise_bottoms"]], P["R_pick"]))
    else:
        ctx.phase = "roll"
    hold = reh["hold_steps"] + (len(reh["stretch_bottoms"]) + len(reh["rise_bottoms"]) if VARIANT == "no_stretch"
                                else 0)
    if VARIANT == "overheat":
        hold += 10 * CONTROL_HZ
    yield from wait(ctx, steps=hold)
    out_tcp = R.pos - OUT_DROP * Z
    if VARIANT == "pull_out":
        ctx.phase = "out (steam on)"
        yield from solo(R, R.move(out_tcp, P["R_pick"], 1.0))
    ctx.phase = "steam off"
    if VARIANT != "pull_out":
        yield from solo(R, R.move(R.pos - DROP_OFF * Z, P["R_pick"], 0.5))
    yield from knob(0.0)
    yield from wait(ctx, 0.5)
    ctx.phase = "set down"
    if VARIANT != "pull_out":
        yield from solo(R, R.move(out_tcp, P["R_pick"], 1.5))
    yield from both(R.move(P["tcp_place"] + 0.02 * Z, P["R_pick"], 2.5), L.set_grip(P["g_open_l"], 0.6))
    ctx.released("knob")
    yield from both(R.move(P["tcp_place"], P["R_pick"], 1.0), L.move(P["knob_pre"], P["R_knob0"], 1.0))
    yield from solo(R, R.set_grip(P["g_open_r"], 0.6))
    ctx.released("pitcher")
    yield from solo(R, R.move(P["tcp_place"] - PRE_BACK * a, P["R_pick"], 1.0))
    ctx.phase = "done"
    yield from wait(ctx, 1.5)


class Ctx:
    def __init__(self, emb, task):
        from yam_arena import station

        self.emb, self.task = emb, task
        self.P = plan(task)
        self.left = Arm("left", station.BASES["left"], self)
        self.right = Arm("right", station.BASES["right"], self)
        self.steps = 0
        self.phase = "start"
        self._q = None

    def grasped(self, name):
        pass

    def released(self, name):
        pass

    def observe(self, obs):
        """Keep the arms' measured joint positions from an observation (its 14-D joint_pos state). The demo reads
        nothing else back from the simulator."""
        try:
            from inspect_robots_yam.packing import STATE_KEY

            q = obs.state[STATE_KEY]
            q = np.asarray(q.cpu().numpy() if hasattr(q, "cpu") else q, float).ravel()
            self._q = {"left": q[0:6], "right": q[7:13]} if q.size == 14 else None
        except Exception:  # noqa: BLE001  (no readings: the demo runs without the trim)
            self._q = None

    def joints(self, name):
        return None if self._q is None else self._q[name]

    def arm_row(self) -> dict:
        row = {"step": self.steps, "phase": self.phase}
        for name, arm in (("l", self.left), ("r", self.right)):
            m = self.joints("left" if name == "l" else "right")
            row[f"trim_{name}"] = [round(float(v), 2) for v in np.degrees(arm.trim)]
            row[f"err_{name}"] = None if m is None else [round(float(v), 2) for v in np.degrees(arm.q + arm.trim - m)]
        return row

    def action(self):
        from inspect_robots.types import Action
        from inspect_robots_yam.packing import pack

        return Action(data=pack(self.left.cmd, self.right.cmd))


class DryCtx(Ctx):
    """The arms follow their plans exactly; the pitcher moves rigidly with the right hand from when it is grasped until
    it is let go; while the left hand holds the knob, the knob turns with it (its angle is the hand's turn about the
    knob's axis, within 0-90 degrees)."""

    def __init__(self, task):
        super().__init__(None, task)
        T = task
        self.rel = {}
        self.pitcher = (np.asarray(self.P["p_start"], float), np.eye(3))
        self.knob_angle = 0.0
        self.knob_ref = None

    def grasped(self, name):
        if name == "pitcher":
            p, Rm = self.pitcher
            arm = self.right
            # held SLIP_Z lower in the fingers than where they closed (it settles as its weight comes on)
            self.rel["pitcher"] = (arm.rot.T @ (p - SLIP_Z * Z - arm.pos), arm.rot.T @ Rm)
        else:
            self.knob_ref = (self.left.rot.copy(), self.knob_angle)

    def released(self, name):
        if name == "pitcher":
            self.rel.pop("pitcher", None)
        else:
            self.knob_ref = None

    def follow(self):
        from yam_arena import ik

        if "pitcher" in self.rel:
            dp, dR = self.rel["pitcher"]
            pos, rot = ik.tcp_pose(self.right.q, self.right.base)
            self.pitcher = (pos + rot @ dp, rot @ dR)
        if self.knob_ref is not None:
            R0, a0 = self.knob_ref
            _, rot = ik.tcp_pose(self.left.q, self.left.base)
            rel = R0.T @ rot
            turn = math.atan2(rel[1, 0], rel[0, 0])             # about the gripper's approach (+x)
            self.knob_angle = float(np.clip(a0 - turn, 0.0, knob_held(self.task)))   # it creeps back from its stop

    def pitcher_state(self, prev=None):
        from partkit import PartState

        p, Rm = self.pitcher
        v = np.zeros(3) if prev is None else (p - prev) / DT
        return PartState(pos=p, quat_xyzw=quat_xyzw(Rm), lin_vel=v, ang_vel=np.zeros(3), low=p, high=p, sliders={})

    def grippers(self):
        from yam_arena import ik

        out = {}
        T = self.task
        for key, arm, held in (("left", self.left, "knob"), ("right", self.right, "pitcher")):
            pos, rot = ik.tcp_pose(arm.q, arm.base)
            half = arm.grip * GAP_FULL / 2
            if held == "pitcher" and "pitcher" in self.rel:
                half = max(half, T.R_OUT)
            if held == "knob" and self.knob_ref is not None:
                half = max(half, T.KNOB_ACROSS / 2)
            out[key] = (pos, rot[:, 2], rot[:, 0], half)
        return out


# -- clearance checks (plan-only) -------------------------------------------------------------------------------------
def _sat_gap(a, b) -> float:
    """A lower bound on the gap between two oriented boxes (centre, R, half sizes); negative is overlap depth."""
    ca, Ra, ha = a
    cb, Rb, hb = b
    axes = [Ra[:, i] for i in range(3)] + [Rb[:, i] for i in range(3)]
    for i in range(3):
        for j in range(3):
            c = np.cross(Ra[:, i], Rb[:, j])
            n = np.linalg.norm(c)
            if n > 1e-6:
                axes.append(c / n)
    d = cb - ca
    return max(abs(d @ ax) - np.abs(Ra.T @ ax) @ ha - np.abs(Rb.T @ ax) @ hb for ax in axes)


def gripper_boxes(g):
    tcp, approach, close, half = g
    across = unit(np.cross(approach, close))
    R = np.column_stack([close, across, approach])
    boxes = [("housing", (tcp - 0.109 * approach, R, np.array([0.065, 0.045, 0.030])))]
    for side in (-1.0, 1.0):
        boxes.append(("finger", (tcp + side * (half + 0.004) * close - 0.018 * approach, R,
                                 np.array([0.004, 0.012, 0.031]))))
    return boxes


def static_boxes(T):
    out = []
    for o in T.MACHINE:
        if hasattr(o, "radius"):
            out.append((o.name, (np.asarray(o.pos, float), np.eye(3), np.array([o.radius, o.radius, o.height / 2]))))
        else:
            out.append((o.name, (np.asarray(o.pos, float), np.eye(3), np.asarray(o.size, float) / 2)))
    d = T.WAND_DIR
    e1 = unit(np.cross(d, Y))
    out.append(("wand", ((T.PIVOT + T.TIP) / 2, np.column_stack([e1, np.cross(d, e1), d]),
                         np.array([T.WAND_R, T.WAND_R, T.WAND_L / 2]))))
    return out


def wand_wall_margin(T, p, Rm) -> float:
    """The smallest clearance (m) between the wand and the pitcher's wall, over points of the wand at the wall's
    height; +inf when no part of the wand is at the wall's height."""
    best = math.inf
    for s in np.linspace(0.0, 1.0, 41):
        q = Rm.T @ (T.PIVOT + s * (T.TIP - T.PIVOT) - p)
        if not (0.0 <= q[2] <= T.P_H):
            continue
        r = math.hypot(q[0], q[1])
        best = min(best, (T.R_IN - T.WAND_R - r) if r < T.R_IN else (r - T.R_OUT - T.WAND_R))
    return best


def load_task(task_path: str):
    import importlib.util

    sys.path.insert(0, str(Path(task_path).resolve().parent))
    spec = importlib.util.spec_from_file_location("milk_task", task_path)
    task = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(task)
    return task


def plan_only(task_path: str) -> dict:
    """Run the demo with no simulator (see DryCtx): IK errors, joint margins, clearances and the milk's outcome."""
    import time

    T = load_task(task_path)
    T.reset_state()
    ctx = DryCtx(T)
    statics = static_boxes(T)
    t0 = time.time()
    n, phases, prev = 0, {}, None
    gaps = {"left-static": (math.inf, None), "right-static": (math.inf, None), "left-right": (math.inf, None),
            "pitcher-static": (math.inf, None), "wand-wall": (math.inf, None)}
    trace = []

    def note(key, value, where):
        if value < gaps[key][0]:
            gaps[key] = (value, where)

    for _ in script(T, ctx):
        n += 1
        T._STATE["step"] += 1
        ctx.follow()
        st = ctx.pitcher_state(prev)
        prev = np.asarray(st.pos, float)
        T.milk_step(st, ctx.knob_angle, DT)
        trace.append(T.trace_row() | {"phase": ctx.phase})
        phases.setdefault(ctx.phase, n)
        g = ctx.grippers()
        gb = {k: gripper_boxes(v) for k, v in g.items()}
        for side in ("left", "right"):
            for gname, box in gb[side]:
                for sname, sbox in statics:
                    note(f"{side}-static", _sat_gap(box, sbox), (n, ctx.phase, gname, sname))
        for (na, ba) in gb["left"]:
            for (nb, bb) in gb["right"]:
                note("left-right", _sat_gap(ba, bb), (n, ctx.phase, na, nb))
        p, Rm = ctx.pitcher
        pbox = (p + Rm @ np.array([0, 0, T.P_H / 2]), Rm, np.array([T.R_OUT, T.R_OUT, T.P_H / 2]))
        for sname, sbox in statics:
            if sname != "wand":
                note("pitcher-static", _sat_gap(pbox, sbox), (n, ctx.phase, sname))
        note("wand-wall", wand_wall_margin(T, p, Rm), (n, ctx.phase))
    for _ in range(int(1.5 * CONTROL_HZ)):                 # let it settle on the counter
        T._STATE["step"] += 1
        st = ctx.pitcher_state(prev)
        prev = np.asarray(st.pos, float)
        T.milk_step(st, ctx.knob_angle, DT)
    rep = T.report()
    P = ctx.P
    out = {"variant": VARIANT, "steps": n, "seconds": round(n / CONTROL_HZ, 1),
           "worst_ik_mm": {"left": round(ctx.left.worst_mm, 2), "right": round(ctx.right.worst_mm, 2)},
           "worst_ik_deg": {"left": round(ctx.left.worst_deg, 2), "right": round(ctx.right.worst_deg, 2)},
           "min_joint_margin_deg": {"left": round(ctx.left.min_margin, 1), "right": round(ctx.right.min_margin, 1)},
           "clearance_mm": {k: (round(1000 * v, 1) if math.isfinite(v) else None, w) for k, (v, w) in gaps.items()},
           "phase_start_steps": phases,
           "rehearsal": {k: v for k, v in P["rehearsal"].items() if k in ("T", "stretch", "depth_mm", "hold_steps")}
           | {"stretch_steps": len(P["rehearsal"]["stretch_bottoms"])},
           "result": rep, "wall_s": round(time.time() - t0, 1)}
    if os.environ.get("PLAN_TRACE"):
        Path(os.environ["PLAN_TRACE"]).write_text("\n".join(json.dumps(r) for r in trace) + "\n")
    print(json.dumps(out, indent=1, default=str))
    return out


# -- the simulator run ----------------------------------------------------------------------------------------------
def _write_mp4(frames, path: Path) -> None:
    import imageio

    imageio.mimsave(path, [np.asarray(f) for f in frames], fps=CONTROL_HZ, codec="libx264",
                    output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])


def run(task_path: str, out_dir: str, max_steps: int = 1400, record: bool = True, cam_width: int = 320,
        cam_height: int = 176, hold_s: float = 2.0) -> dict:
    from inspect_robots import Scene
    from yam_arena.inspect_embodiment import yam_arena_embodiment

    task_path = str(Path(task_path).resolve())
    sys.path.insert(0, str(Path(task_path).parent))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    emb = yam_arena_embodiment(task=task_path, object_jitter=0, control_hz=CONTROL_HZ, cam_width=cam_width,
                               cam_height=cam_height)
    obs = emb.reset(Scene(id="demo", instruction="demo"))
    task = emb.task.module
    ctx = Ctx(emb, task)
    ctx.observe(obs)
    gen = script(task, ctx)
    frames: dict[str, list] = {}

    def snap(o):
        if record:
            for cam, img in o.images.items():
                frames.setdefault(cam, []).append(np.asarray(img))

    snap(obs)
    steps, ended, success, info, phases = 0, None, False, {}, {}
    while steps < max_steps:
        try:
            next(gen)
        except StopIteration:
            ended = steps if ended is None else ended
        res = emb.step(ctx.action())
        steps += 1
        ctx.steps = steps
        info = res.info
        phases.setdefault(ctx.phase, steps)
        ctx.observe(res.observation)
        snap(res.observation)
        if ARM_LOG:
            with open(ARM_LOG, "a") as f:
                f.write(json.dumps(ctx.arm_row()) + "\n")
        if steps % 100 == 0:
            print(f"step {steps} phase {ctx.phase} score {info.get('score')} {task.report()}", flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    score = 1.0 if success else float(info.get("score", 0.0) or 0.0)
    result = {"task_score": score, "success": success, "steps": steps, "variant": VARIANT,
              "worst_ik_mm": {"left": round(ctx.left.worst_mm, 2), "right": round(ctx.right.worst_mm, 2)},
              "worst_trim_deg": {"left": round(ctx.left.worst_trim, 2), "right": round(ctx.right.worst_trim, 2)},
              "phase_start_steps": phases, "task": task.report()}
    if record:
        import imageio

        for cam, fs in frames.items():
            _write_mp4(fs, out / f"demo_{cam}.mp4")
        top = next((c for c in frames if "top" in c), next(iter(frames), None))
        if top:
            # step 1's frame: markers made at reset first render one frame after it
            imageio.imwrite(out / "setup.png", frames[top][min(1, len(frames[top]) - 1)])
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--max-steps", type=int, default=1400)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--cam-width", type=int, default=320)
    ap.add_argument("--cam-height", type=int, default=176)
    ap.add_argument("--plan-only", action="store_true")
    a = ap.parse_args()
    if a.plan_only:
        plan_only(a.task)
    else:
        run(a.task, a.out, a.max_steps, record=not a.no_record, cam_width=a.cam_width, cam_height=a.cam_height)

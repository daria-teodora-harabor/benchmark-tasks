"""Scripted full-score demo for the tagged-leaf spray task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1400] [--no-record] [--cam-width 320 --cam-height 176]
    python demo.py task.py - --plan-only        (no simulator: the arms follow their plans exactly, the card and
                                                 bottle move rigidly with the hands, and the task's own spray and
                                                 touch code runs on those poses)

The demo is open-loop: the layout is fixed (object_jitter=0), so every move is planned in advance from the scene's
known geometry, and it never reads the card's, the bottle's or any other object's pose. The only thing it reads back
is the arms' own joint positions (as a policy does): every step each arm adds its measured tracking error into a
joint trim, which takes out the sag of its soft joint drives under the bottle's weight (the EV task's demo does the
same; without it the hands sag 3-7 cm under half a kilogram).

1. The left hand pinches the card's short edge nearer the arms where it overhangs the crate (the gripper level,
   pointing across the card, the pads on its two faces), lifts it to 2 cm above its final height, carries it toward the arms and
   to the right, then slides it 16 cm straight in under the tagged leaf: its inner edge 18 mm from the stem, its top
   12 mm below the leaf's lowest point.
2. The right hand takes the bottle from behind its head, one pad on the back of the shroud and the other on the
   trigger's face, closed to a light pinch below the trigger's 8 N preload; it lifts it 20 cm.
3. It turns the bottle to the spray pose (a joint-space move, high and to the right of the plant): the spray axis 45
   degrees off the leaf's normal, heading across the leaf toward the arms' left, the nozzle 8 cm from the leaf's
   centre, the bottle rolled 30 degrees about the spray axis. It comes in the last 6 cm along the axis.
4. Two full squeezes, 1.4 s apart; the drops that miss the leaf land on the card.
5. The bottle backs out along the axis, and the card slides back out.

The spray plan is the best one of an offline search over spray directions, distances, rolls and both arms' reach
(task-research, 2026-11-16): with arm-like errors (2 mm, 1 degree, 3 mm of card placement) it scored fully in 15 of
16 droplet runs.

Writes one MP4 per camera, setup.png (the top camera at step 1) and result.json to the output folder.

Test versions (DEMO_VARIANT=...), each must score as given:
  no_card   the left hand leaves the card on its crate; the spray that misses the leaf falls on the leaves below   [0]
  touch     the bottle comes in 7 cm further along the spray axis, its nozzle block into the tagged leaf          [0]
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
FLIP = np.diag([-1.0, -1.0, 1.0])  # the gripper turned 180 degrees about its approach axis (the fingers are symmetric)

# the spray plan (task-research/leaf_spray_turn.py, its best plan; leaf frame: tilt from the leaf's normal, psi from
# its tip direction toward its left)
TILT_DEG, PSI_DEG, SPRAY_DIST, ROLL_DEG = 45.0, 270.0, 0.08, 30.0
SQUEEZES = 2
TCP_IN_SPRAYER = np.array([0.008, 0.0, 0.030])   # the pinch's centre: between the shroud's back and the trigger's face
PRE_BACK = 0.06                    # the bottle comes in the last 6 cm along the spray axis
HI = np.array([0.0, -0.04, 0.06])  # the turn ends 6 cm above and 4 cm right of that
BOTTLE_LIFT = 0.20
PICK_ABOVE = 0.10                  # the swing to the bottle ends 10 cm above its pre-grasp point, clear of the head
# the card's place under the leaf (as task-research/leaf_spray_card.card_geometry)
CARD_R0, CARD_DZ = 0.018, 0.012    # inner edge from the stem; its top below the tagged leaf's lowest point
PINCH_IN = 0.015                   # the left TCP 15 mm in from the card's edge
CARD_IN = 0.16                     # it slides this far in, along +x
CARD_CLEAR = 0.02                  # carried 2 cm above its final height
CARD_TOWARD = 0.15                 # after the lift, first 15 cm toward the arms
TOUCH_IN = 0.07                    # the "touch" variant: the nozzle 1 cm from the leaf's centre
VARIANT = os.environ.get("DEMO_VARIANT", "")
ARM_LOG = os.environ.get("ARM_LOG")


def grip_cmd(gap_m: float) -> float:
    return float(np.clip(gap_m / GAP_FULL, 0.0, 1.0))


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


# -- geometry -------------------------------------------------------------------------------------------------------
def leaf_frame(T):
    """The tagged leaf's centre, upward normal, stalk-end and tip (as leaf_spray_particles.leaf_frame)."""
    nv = 14 * 9
    V = T.PLANT.V[T.TAG_INDEX * nv:(T.TAG_INDEX + 1) * nv]
    F = T.PLANT.F[T.PLANT.owner == T.TAG_INDEX] - T.TAG_INDEX * nv
    a, b, c = (V[F[:, i]] for i in range(3))
    n = np.cross(b - a, c - a)
    n = (n * np.sign(n[:, 2:3] + 1e-12)).sum(0)
    return V.mean(0), unit(n), V[9 // 2], V[13 * 9 + 9 // 2], V


def spray_axis(T):
    c, n, base, tip, _ = leaf_frame(T)
    d = tip - base
    e1 = unit(d - n * (n @ d))
    e2 = np.cross(n, e1)
    t, p = math.radians(TILT_DEG), math.radians(PSI_DEG)
    return c, unit(-math.cos(t) * n + math.sin(t) * (math.cos(p) * e1 + math.sin(p) * e2))


def sprayer_frame(axis, roll_deg):
    """Sprayer axes in the world: x along the spray axis, z the bottle's up leaning with it, rolled about x."""
    x = unit(axis)
    z = unit(Z - x * (x @ Z))
    y = np.cross(z, x)
    r = math.radians(roll_deg)
    y, z = math.cos(r) * y + math.sin(r) * z, -math.sin(r) * y + math.cos(r) * z
    return np.column_stack([x, y, z])


def grip_on_bottle(Rs):
    """The right gripper's axes for a sprayer frame: closing along the spray axis (pads on the shroud's back and the
    trigger's face), approaching along the sprayer's -y, turned 180 degrees about the approach."""
    zg = -Rs[:, 1]
    return np.column_stack([Rs[:, 0], np.cross(zg, Rs[:, 0]), zg]) @ FLIP


def bottle_from_grip(Rg):
    Rg = Rg @ FLIP
    x, y = Rg[:, 0], -Rg[:, 2]
    return np.column_stack([x, y, np.cross(x, y)])


def card_final(T):
    """The card's centre and axes (u outward along the tagged leaf, across, up) under the leaf."""
    az = math.radians(T.TAG_AZ_DEG)
    u = np.array([math.cos(az), math.sin(az), 0.0])
    across = np.array([-math.sin(az), math.cos(az), 0.0])
    _, _, _, _, V = leaf_frame(T)
    top = V[:, 2].min() - CARD_DZ
    centre = T.PLANT_ORIGIN * np.array([1, 1, 0]) + (CARD_R0 + T.CARD_SIZE[0] / 2) * u
    centre[2] = top - T.CARD_SIZE[2] / 2
    return centre, np.column_stack([u, across, Z])


def card_grip(centre, Rc, T):
    """The left gripper on the card's long edge at -across: TCP 15 mm in from it, approaching along +across, closing
    along the card's normal."""
    across = Rc[:, 1]
    tcp = centre - (T.CARD_SIZE[1] / 2 - PINCH_IN) * across
    zg, xg = across, Rc[:, 2]
    return tcp, np.column_stack([xg, np.cross(zg, xg), zg])


def plan(T):
    """Every fixed target of the demo."""
    c, axis = spray_axis(T)
    Rs = sprayer_frame(axis, ROLL_DEG)
    nozzle = c - SPRAY_DIST * axis
    origin = nozzle - Rs @ T.NOZZLE_S
    P = {"axis": axis, "leaf_c": c, "Rs": Rs, "nozzle": nozzle}
    P["tcp_spray"] = origin + Rs @ TCP_IN_SPRAYER
    P["R_spray"] = grip_on_bottle(Rs)
    P["tcp_pre"] = P["tcp_spray"] - PRE_BACK * axis
    P["tcp_hi"] = P["tcp_pre"] + HI
    Rs0 = T.R_WS_SPAWN
    P["R_pick"] = grip_on_bottle(Rs0)
    P["tcp_pick"] = np.asarray(T.SPAWN_POS) + Rs0 @ TCP_IN_SPRAYER
    width = T.TRIGGER_FACE_X - T.SHROUD_BACK_X
    P["g_open_r"] = grip_cmd(width + 0.012)
    P["g_hold"] = grip_cmd(width - 2 * 6.0 / FINGER_K)                   # about 6 N: below the 8 N preload
    P["g_squeeze"] = grip_cmd(width - T.TRIGGER_TRAVEL - 0.010)          # full stroke, then 10 mm past
    cf, Rc = card_final(T)
    P["card_final"], P["R_card"] = cf, Rc
    P["tcp_card"], P["R_left"] = card_grip(cf, Rc, T)
    start = np.asarray(T.CARD_START, float)
    P["tcp_card0"] = start + (P["tcp_card"] - cf)
    P["g_open_l"] = grip_cmd(0.030)
    P["g_card"] = 0.0                                                   # 1 mm into each face of the 2 mm card
    return P


# -- the arms -------------------------------------------------------------------------------------------------------
def solve_multi(target, R, base, seed=None):
    from yam_arena import ik

    best = None
    for qi in ([seed] if seed is not None else []) + [None, [0.5, 1.2, 1.2, 0, 1.0, 0], [-0.5, 1.2, 1.2, 0, -1.0, 0],
                                                       [0.0, 1.6, 1.0, -0.5, 0.0, 0.0]]:
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

    def move_joints(self, pos, rot, duration: float, grip: float | None = None, seed=None):
        """Generator: a joint-space move to the IK solution for ``pos``/``rot`` (from ``seed``, else the current
        joints), for big turns where a straight Cartesian path would leave the arm's reach."""
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


def wait(ctx, seconds: float):
    for _ in range(max(1, round(seconds * CONTROL_HZ))):
        ctx.left.hold()
        ctx.right.hold()
        yield


def script(T, ctx):
    """Generator: one control step per yield."""
    P = ctx.P
    L, R = ctx.left, ctx.right
    zl = P["R_left"][:, 2]

    def solo(arm, gen):
        other = R if arm is L else L
        for _ in gen:
            other.hold()
            yield

    if VARIANT != "no_card":
        ctx.phase = "card: reach"
        yield from solo(L, L.move_joints(P["tcp_card0"] - 0.08 * zl, P["R_left"], 3.0, grip=P["g_open_l"]))
        yield from solo(L, L.move(P["tcp_card0"], P["R_left"], 1.5))
        ctx.phase = "card: pinch"
        yield from solo(L, L.set_grip(P["g_card"], 1.0))
        ctx.grasped("card")
        yield from wait(ctx, 0.5)
        ctx.phase = "card: carry"
        z_carry = P["tcp_card"][2] + CARD_CLEAR
        up = np.array([P["tcp_card0"][0], P["tcp_card0"][1], z_carry])
        yield from solo(L, L.move(up, P["R_left"], 1.5))
        toward = up - CARD_TOWARD * X
        yield from solo(L, L.move(toward, P["R_left"], 2.0))
        pre_in = P["tcp_card"] - CARD_IN * X
        yield from solo(L, L.move(pre_in + CARD_CLEAR * Z, P["R_left"], 2.5))
        yield from solo(L, L.move(pre_in, P["R_left"], 0.8))
        ctx.phase = "card: slide in"
        yield from solo(L, L.move(P["tcp_card"], P["R_left"], 2.5))
        yield from wait(ctx, 0.5)

    ctx.phase = "bottle: reach"
    zr = P["R_pick"][:, 2]
    pre_pick = P["tcp_pick"] - 0.08 * zr
    yield from solo(R, R.move_joints(pre_pick + PICK_ABOVE * Z, P["R_pick"], 3.0, grip=P["g_open_r"]))
    yield from solo(R, R.move(pre_pick, P["R_pick"], 1.5))           # straight down beside the bottle
    yield from solo(R, R.move(P["tcp_pick"], P["R_pick"], 1.5))
    ctx.phase = "bottle: pinch"
    yield from solo(R, R.set_grip(P["g_hold"], 1.0))
    ctx.grasped("sprayer")
    yield from wait(ctx, 0.5)
    ctx.phase = "bottle: lift"
    yield from solo(R, R.move(P["tcp_pick"] + BOTTLE_LIFT * Z, P["R_pick"], 2.0))
    ctx.phase = "bottle: turn"
    yield from solo(R, R.move_joints(P["tcp_hi"], P["R_spray"], 3.0, seed=ctx.q_spray))
    yield from solo(R, R.move(P["tcp_pre"], P["R_spray"], 1.5))
    ctx.phase = "bottle: in"
    target = P["tcp_spray"] + (TOUCH_IN * P["axis"] if VARIANT == "touch" else 0.0)
    yield from solo(R, R.move(target, P["R_spray"], 1.5))
    yield from wait(ctx, 1.0)
    for k in range(SQUEEZES):
        ctx.phase = f"squeeze {k + 1}"
        yield from solo(R, R.set_grip(P["g_squeeze"], 0.35))
        yield from wait(ctx, 0.25)
        yield from solo(R, R.set_grip(P["g_hold"], 0.35))
        yield from wait(ctx, 0.7)
    yield from wait(ctx, 0.8)
    ctx.phase = "bottle: out"
    yield from solo(R, R.move(P["tcp_pre"], P["R_spray"], 1.0))
    yield from solo(R, R.move(P["tcp_hi"], P["R_spray"], 1.5))
    if VARIANT != "no_card":
        ctx.phase = "card: out"
        yield from solo(L, L.move(P["tcp_card"] - CARD_IN * X, P["R_left"], 2.0))
    ctx.phase = "done"
    yield from wait(ctx, 1.0)


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
        q, pe, re_ = solve_multi(self.P["tcp_spray"], self.P["R_spray"], self.right.base)
        if pe > 0.002 or re_ > math.radians(1.0):
            raise RuntimeError(f"the spray pose is out of reach ({pe * 1000:.1f} mm, {math.degrees(re_):.1f} deg)")
        q_pre, _, _ = solve_multi(self.P["tcp_pre"], self.P["R_spray"], self.right.base, q)
        self.q_spray, _, _ = solve_multi(self.P["tcp_hi"], self.P["R_spray"], self.right.base, q_pre)

    def grasped(self, name):
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


def load_task(task_path: str):
    import importlib.util

    sys.path.insert(0, str(Path(task_path).resolve().parent))
    spec = importlib.util.spec_from_file_location("spray_task", task_path)
    task = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(task)
    return task


# -- no simulator: the plan, kinematically ------------------------------------------------------------------------
class DryCtx(Ctx):
    """The arms follow their plans exactly; the card and bottle move rigidly with the hand that grasped them; the
    trigger's travel follows the right grip's command (the fingers' drives against the trigger's preloaded spring)."""

    def __init__(self, task):
        super().__init__(None, task)
        T = task
        self.rel = {}
        self.pose = {"card": (np.asarray(T.CARD_START, float), rot_z(T.CARD_YAW)),
                     "sprayer": (np.asarray(T.SPAWN_POS, float), T.R_WP_SPAWN)}
        self.arm_of = {"card": self.left, "sprayer": self.right}

    def grasped(self, name):
        arm = self.arm_of[name]
        p, Rm = self.pose[name]
        self.rel[name] = (arm.rot.T @ (p - arm.pos), arm.rot.T @ Rm)

    def follow(self):
        for name, (dp, dR) in self.rel.items():
            arm = self.arm_of[name]
            from yam_arena import ik

            pos, rot = ik.tcp_pose(arm.q, arm.base)
            self.pose[name] = (pos + rot @ dp, rot @ dR)

    def travel(self):
        T = self.task
        width = T.TRIGGER_FACE_X - T.SHROUD_BACK_X
        if "sprayer" not in self.rel:
            return 0.0
        f = (FINGER_K / 2) * (width - self.right.grip * GAP_FULL)      # the two fingers' drives in series
        return float(np.clip((f - T.TRIGGER_PRELOAD) / (T.TRIGGER_RATE + FINGER_K / 2), 0.0, T.TRIGGER_TRAVEL))

    def states(self):
        from partkit import PartState

        T = self.task
        out = {}
        for name, (p, Rm) in self.pose.items():
            q = quat_xyzw(Rm)
            sl = {"trigger": -self.travel()} if name == "sprayer" else {}
            out[name] = PartState(pos=p, quat_xyzw=q, lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=p, high=p,
                                  sliders=sl)
        return out

    def grippers(self):
        from yam_arena import ik

        out = {}
        for key, arm, held in (("left_robot", self.left, "card"), ("right_robot", self.right, "sprayer")):
            pos, rot = ik.tcp_pose(arm.q, arm.base)
            half = arm.grip * GAP_FULL / 2
            if held in self.rel:
                half = max(half, (0.002 if held == "card" else
                                  self.task.TRIGGER_FACE_X - self.task.SHROUD_BACK_X - self.travel()) / 2)
            out[key] = (pos, rot[:, 2], rot[:, 0], (half, half))
        return out


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def quat_xyzw(Rm):
    from yam_arena import ik

    return np.asarray(ik.matrix_to_quat_xyzw(Rm), float)


def plan_only(task_path: str) -> dict:
    """Run the demo with no simulator (see DryCtx): IK errors, joint margins, touches and the spray's outcome."""
    import time

    T = load_task(task_path)
    T.reset_state()
    ctx = DryCtx(T)
    t0 = time.time()
    n, first_touch, phases = 0, None, {}
    for _ in script(T, ctx):
        n += 1
        T._STATE["step"] += 1
        ctx.follow()
        st = ctx.states()
        card = T.card_box(st["card"])
        T.spray_step(st["sprayer"], card, DT)
        T.check_touch(st["sprayer"], st["card"], ctx.grippers())
        if T._STATE["touched"] and first_touch is None:
            first_touch = (n, ctx.phase, T._STATE["touched"])
        phases.setdefault(ctx.phase, n)
    for _ in range(60):                                     # let the last drops land
        T._STATE["step"] += 1
        st = ctx.states()
        T.spray_step(st["sprayer"], T.card_box(st["card"]), DT)
    rep = T.report()
    P = ctx.P
    out = {"variant": VARIANT, "steps": n, "seconds": round(n / CONTROL_HZ, 1),
           "worst_ik_mm": {"left": round(ctx.left.worst_mm, 2), "right": round(ctx.right.worst_mm, 2)},
           "worst_ik_deg": {"left": round(ctx.left.worst_deg, 2), "right": round(ctx.right.worst_deg, 2)},
           "min_joint_margin_deg": {"left": round(ctx.left.min_margin, 1), "right": round(ctx.right.min_margin, 1)},
           "first_touch": first_touch, "phase_start_steps": phases,
           "trigger_cmds": {k: round(P[k], 4) for k in ("g_open_r", "g_hold", "g_squeeze")},
           "spray_pose": {"tcp": [round(float(v), 4) for v in P["tcp_spray"]],
                          "nozzle": [round(float(v), 4) for v in P["nozzle"]],
                          "axis": [round(float(v), 4) for v in P["axis"]]},
           "result": rep, "wall_s": round(time.time() - t0, 1)}
    print(json.dumps(out, indent=1))
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
            # step 1's frame: the plant's meshes, made at reset, first render one frame after it
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

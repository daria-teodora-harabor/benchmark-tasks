"""Scripted full-score demo for the EV unplug task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1600] [--no-record] [--cam-width 320 --cam-height 176]
    python demo.py task.py - --plan-only          (no simulator: every arm target through the IK)

The demo is open-loop: the layout is fixed (object_jitter=0), so every move is planned in advance from the scene's
known geometry, and it never reads the connector's, the door's or any other object's pose. The only thing it reads
back is the arms' own joint positions (as a policy does): while an arm moves freely (approaching, or carrying the
connector), it integrates its own measured tracking error into a joint trim, which takes out the sag of its soft
joint drives under the connector's weight and the cable's pull (3-7 cm at the fingertips without it; see README). The
trim is held, not updated, while the hand pushes on something (the button, the connector in the inlet or holster, the
door), so it never pushes harder than planned.

1. The right hand takes the connector's body from behind (the gripper level, pointing at the car, the pads on the
   body's flat sides 10 cm behind the nose) while the left hand comes above the latch button with its fingers closed.
2. The left fingertips press the button 6 mm past the point where the latch's hook clears its catch, and hold it.
3. The right hand draws the connector 1.5 cm out (the hook is now past the catch), the left hand lifts away, and the
   right hand pulls it the rest of the way out.
4. The right hand carries it to the charger post, waits for its trim to settle, pushes the nose into the holster and
   lets go.
5. The left hand swings the charge-port door shut (it starts 110 degrees open), pushing on its outer face 10 cm from the hinge, then presses it
   flat.

Writes one MP4 per camera, setup.png (the first top-camera frame) and result.json to the output folder. For the setup
screenshot, render one step at 640 x 360: python demo.py task.py shot --max-steps 1 --cam-width 640 --cam-height 360

Test versions (DEMO_VARIANT=...), each must score as given:
  yank           no button press: the right hand pulls against the closed latch                               [0]
  early_release  a brief press, let go, pushed home, then the pull; the hook still slipped past the catch     [0.5]
  drop           pulls the connector out, then opens the right hand in mid-air                                [0]
  no_hang        pulls the connector out and holds it in front of the holster                               [0.5]
  no_door        hangs the connector up but leaves the door open                                            [0.75]
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
OPEN = 0.85                       # about 80 mm between the pads while approaching (the body is 57.4 mm wide)
GRIP_REAR = np.array([-0.100, 0.0, 0.0])   # where the right TCP holds the connector (connector frame)
PRESS_PAST = 0.012                # the left fingertips are commanded this far straight down past the button's rest
                                  # top (their approach leans 45 degrees; pushing along it shoved the button forward
                                  # and stalled at 1.3 mm): 3 mm past its 9 mm stop, so the arm's trim winds up until
                                  # it holds the button on that stop
PULL_FREE = 0.015                 # pulled this far with the button held, the hook is past its catch
PULL_OUT = 0.13                   # then out to this far: the nose is 8.5 cm clear of the inlet's face
K_TRACK = 2.0                     # gain of an arm's joint trim on its own measured tracking error (1/s)
TRACK_MAX = math.radians(15.0)    # largest trim per joint
PRESS_TRIM_MAX = math.radians(30.0)   # ... while the left hand presses the button (the YAM wrist is soft: 10 N*m/rad)
DOOR_R = 0.10                     # the left fingertips push the door this far from its hinge
HOLSTER_PAST = 0.004              # the push into the holster aims this far past its back (see step 4)
HOLSTER_PRE = 0.085               # the nose waits this far in front of the holster's back before going in
ARM_LOG = os.path.expanduser(os.environ["EV_ARMS"]) if os.environ.get("EV_ARMS") else None   # per-step arm log
X, Y, Z = np.eye(3)


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def rot_rear(flip: bool) -> np.ndarray:
    """Right gripper: level, approach along +x (toward the car), fingers closing along y (the body's sides)."""
    a, c = X, Y
    R = np.column_stack([c, np.cross(a, c), a])
    return R @ np.diag([-1.0, -1.0, 1.0]) if flip else R


def rot_press(flip: bool, lean: float = 0.0) -> np.ndarray:
    """Left gripper over the button: approach down, leaning ``lean`` degrees toward +x (0 is straight down), fingers
    closing along y. Straight down is preferred: a leaning push also shoves the button forward, and the second
    simulator check's 45-degree press stalled at 1.3 mm."""
    t = math.radians(lean)
    a = np.array([math.sin(t), 0.0, -math.cos(t)])
    c = Y - a * (a @ Y)
    c = unit(c)
    R = np.column_stack([c, np.cross(a, c), a])
    return R @ np.diag([-1.0, -1.0, 1.0]) if flip else R


def rot_door(phi: float, flip: bool, yaw: float = 0.0) -> np.ndarray:
    """Left gripper pushing the door: level, fingers closing vertically, approach into the door's outer face, turned
    ``yaw`` about the vertical from square-on (the fingertips still push mostly into the face)."""
    d = np.array([math.cos(phi), math.sin(phi), 0.0])
    n = np.array([d[1], -d[0], 0.0])          # the door's outer-face normal (the door turns about +z at its hinge)
    cy, sy = math.cos(yaw), math.sin(yaw)
    a, c = np.array([-n[0] * cy + n[1] * sy, -n[0] * sy - n[1] * cy, 0.0]), Z
    R = np.column_stack([c, np.cross(a, c), a])
    return R @ np.diag([-1.0, -1.0, 1.0]) if flip else R


def door_push_point(T, phi: float) -> np.ndarray:
    """Where the left fingertips push: on the door's outer face, DOOR_R from the hinge, 2 mm into the face (so they
    keep pushing as the door turns)."""
    d = np.array([math.cos(phi), math.sin(phi), 0.0])
    n = np.array([d[1], -d[0], 0.0])
    return np.asarray(T.DOOR_HINGE, float) + DOOR_R * d + 0.002 * n - 0.002 * n * 2


class Arm:
    """One arm: smooth Cartesian moves (IK on the URDF), each starting where the last command left it. With ``trim``
    on, every step adds its own measured tracking error (planned minus measured joints) into a joint trim."""

    def __init__(self, name: str, base_pos, ctx):
        from yam_arena import ik
        from yam_arena.constants import READY

        self.name, self.base, self.ctx = name, np.asarray(base_pos, float), ctx
        self.q = np.array(READY, float)
        self.grip = 1.0
        self.pos, self.rot = ik.tcp_pose(self.q, self.base)
        self.trim = np.zeros(6)
        self.trim_on = True
        self.trim_max = TRACK_MAX
        self.cmd = np.r_[self.q, self.grip]
        self.worst_mm = 0.0
        self.worst_deg = 0.0
        self.worst_trim = 0.0

    def _solve(self, pos, rot):
        from yam_arena import ik

        q, pe, re_ = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), self.q, base_pos=self.base, iters=60)
        self.q, self.pos, self.rot = q, np.asarray(pos, float), rot
        self.worst_mm = max(self.worst_mm, pe * 1000)
        self.worst_deg = max(self.worst_deg, math.degrees(re_))
        self._track()

    def _track(self):
        measured = self.ctx.joints(self.name)
        if self.trim_on and measured is not None:
            self.trim = np.clip(self.trim + K_TRACK * DT * (self.q - measured), -self.trim_max, self.trim_max)
            self.worst_trim = max(self.worst_trim, float(np.degrees(np.abs(self.trim).max())))
        self.cmd = np.r_[self.q + self.trim, self.grip]

    def move(self, pos, rot, duration: float, grip: float | None = None):
        """Generator: go to ``pos``/``rot`` (world TCP); ``grip`` 1 open .. 0 closed (None keeps it)."""
        from yam_arena import ik

        p0, r0, g0 = self.pos.copy(), self.rot.copy(), self.grip
        g1 = g0 if grip is None else grip
        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            s = ik.smoothstep(k / n)
            self.grip = g0 + s * (g1 - g0)
            self._solve(p0 + s * (np.asarray(pos) - p0), ik.slerp_matrix(r0, rot, s))
            yield

    def set_grip(self, grip: float, duration: float = 0.8):
        yield from self.move(self.pos, self.rot, duration, grip)

    def hold(self):
        """This step: stay (keeps tracking if the trim is on)."""
        self._track()

    def ease_trim(self, to, duration: float):
        """Generator: hold still with the trim off while it changes smoothly to ``to`` (after a press, back to what it
        was before, so the arm stops pushing without a jump)."""
        from yam_arena import ik

        self.trim_on = False
        t0, n = self.trim.copy(), max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            self.trim = t0 + ik.smoothstep(k / n) * (np.asarray(to) - t0)
            self._track()
            yield
        self.trim_max = TRACK_MAX

    def aim(self, pos, rot):
        self._solve(pos, rot)


def both(*gens):
    """Run several move generators side by side, one step each per yield, until all are done."""
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


def script(task, ctx):
    """The demo as a generator: one yield per control step."""
    T = task
    A, B = ctx.right, ctx.left
    variant = os.environ.get("DEMO_VARIANT", "")
    seat = np.asarray(T.SEATED_NOSE, float)
    Rr, Rl = ctx.rot_right, ctx.rot_left
    ar, al = Rr[:, 2], Rl[:, 2]
    button = seat + T.BUTTON_TOP
    press_tcp = button - 0.013 * al                       # fingertips (13 mm past the TCP) on the button's rest top
    grasp = seat + GRIP_REAR

    # 1. Right hand to the connector's body from behind; left hand above the button, fingers closed.
    ctx.phase = "approach"
    yield from both(A.move(grasp - 0.08 * ar, Rr, 3.0, grip=OPEN),
                    B.move(press_tcp - 0.06 * al, Rl, 3.0, grip=0.0) if variant != "yank" else wait(ctx, 3.0))
    yield from A.move(grasp, Rr, 1.5)
    A.trim_on = False                                     # from here the right hand holds a part fixed in the car
    yield from A.set_grip(0.0, 1.0)
    yield from wait(ctx, 0.4)

    # 2. Press the button and hold it. The arm is soft (about 150 N/m at the fingertips in the first simulator check: a
    # plain 12 mm overshoot gave about 2 N), so its trim keeps integrating its own joint error while it presses, which
    # pushes harder until the fingertips reach the planned depth (the button's stop); then the trim is held.
    ctx.phase = "press"
    if variant != "yank":
        free_trim = B.trim.copy()                         # the trim that cancels the left arm's sag in free air
        B.trim_max = PRESS_TRIM_MAX
        yield from B.move(press_tcp - PRESS_PAST * Z, Rl, 1.2)
        yield from wait(ctx, 0.2 if variant == "early_release" else 2.0)
        B.trim_on = False
        if variant == "early_release":
            # Let go at once, then push the connector home: the press can shift it a couple of millimetres out, which
            # puts the hook under the catch's lip, and only home does the hook spring back into its catch.
            yield from B.ease_trim(free_trim, 0.5)
            yield from B.move(press_tcp - 0.06 * al, Rl, 0.8)
            A.trim_on = True
            yield from A.move(grasp + 0.004 * X, Rr, 1.0)
            yield from wait(ctx, 0.5)

    # 3. Draw it past the catch with the button held, lift the left hand away, pull it out.
    ctx.phase = "pull"
    if variant in ("yank", "early_release"):
        A.trim_on = True                                  # pull as hard as the right arm can: its trim winds up
    yield from A.move(grasp - PULL_FREE * X, Rr, 1.0)
    if variant not in ("yank", "early_release"):
        yield from B.ease_trim(free_trim, 0.5)            # stop pushing, then lift away
        yield from B.move(press_tcp - 0.08 * al, Rl, 0.8)
    yield from A.move(grasp - PULL_OUT * X, Rr, 2.0)
    if variant in ("yank", "early_release"):
        yield from wait(ctx, 1.0)
        return
    A.trim_on = True                                      # free in the air: let the right arm take out its sag
    B.trim_on = True
    yield from wait(ctx, 1.0)
    if variant == "drop":
        yield from A.set_grip(OPEN, 0.6)
        yield from A.move(A.pos - 0.06 * ar, Rr, 1.0)
        yield from wait(ctx, 2.5)
        return

    # 4. Carry it to the holster (nose 8.5 cm in front of the holster's back, just at the mouth of its 4 cm lead-in; the
    # right arm reaches no further back there), let the trim settle, push it home along the lead-in. In the third
    # simulator check the nose stood 4 cm further in and 1.2 cm to the side of this plan (the cable pushing it in the
    # hand), so it rests on the wide part of the lead-in, which guides it into the socket.
    ctx.phase = "carry"
    hol = np.array([T.HX, T.HY, T.HZ])
    pre = hol - np.array([HOLSTER_PRE, 0.0, 0.0]) + GRIP_REAR
    mid = np.array([(A.pos[0] + pre[0]) / 2, (A.pos[1] + pre[1]) / 2, max(A.pos[2], pre[2]) + 0.03])
    yield from A.move(mid, Rr, 1.8)
    yield from A.move(pre, Rr, 1.8)
    yield from wait(ctx, 1.5)
    if variant == "no_hang":
        yield from wait(ctx, 3.0)
        return
    A.trim_on = False
    ctx.phase = "holster"
    # Aim 4 mm past the socket's back: the soft arm stops about 7 mm short of its command (in the fifth simulator check
    # the nose stopped 10 mm from the back when aimed 3 mm from it), so the nose ends about 3 mm from the back.
    yield from A.move(hol + np.array([HOLSTER_PAST, 0.0, 0.0]) + GRIP_REAR, Rr, 3.0)
    yield from wait(ctx, 0.5)
    yield from A.set_grip(OPEN, 0.8)
    A.trim_on = True
    ctx.phase = "let go"
    yield from A.move(A.pos - 0.10 * ar, Rr, 1.5)            # straight back until the fingertips clear its body
    yield from A.move(A.pos + np.array([0.03, 0.06, 0.06]), Rr, 1.2)
    if variant == "no_door":
        yield from wait(ctx, 2.5)
        return

    # 5. Swing the door shut (it starts 110 degrees open, toward the arms) and press it flat.
    ctx.phase = "door"
    flip, yaw = ctx.door_flip, ctx.door_yaw
    phi0, phi1 = math.radians(270.0) + T.DOOR.open, math.radians(271.0)
    start = door_push_point(T, phi0)
    yield from B.move(start + np.array([-0.03, 0.05, 0.0]), rot_door(phi0, flip, yaw), 2.5)
    yield from B.move(start - 0.013 * rot_door(phi0, flip, yaw)[:, 2], rot_door(phi0, flip, yaw), 1.0)
    B.trim_on = False
    n = round(4.0 * CONTROL_HZ)
    for k in range(1, n + 1):
        phi = phi0 + (phi1 - phi0) * k / n
        R = rot_door(phi, flip, yaw)
        B.aim(door_push_point(T, phi) - 0.013 * R[:, 2], R)
        yield
    R = rot_door(phi1, flip, yaw)
    centre = np.array([T.XP - 0.004 + 0.003, T.YP, T.ZP])      # 3 mm into the shut door's outer face
    yield from B.move(centre - 0.013 * R[:, 2], R, 1.0)
    yield from wait(ctx, 0.5)
    B.trim_on = True
    yield from B.move(B.pos - 0.06 * R[:, 2], R, 1.0)
    yield from wait(ctx, 2.0)


class Ctx:
    def __init__(self, emb, task):
        from yam_arena import station

        self.emb, self.task = emb, task
        self.left = Arm("left", station.BASES["left"], self)
        self.right = Arm("right", station.BASES["right"], self)
        self.steps = 0
        self.phase = "start"
        self._q = None
        self._pick(task)

    def _pick(self, task):
        """Choose each grip's 180-degree turn about its approach axis (the fingers are symmetric) so the IK reaches
        all of that grip's targets."""
        from yam_arena import ik

        T = task
        seat = np.asarray(T.SEATED_NOSE, float)
        hol = np.array([T.HX, T.HY, T.HZ])

        def reach(base, targets):
            q = None
            for p, R in targets:
                q, pe, re_ = ik.solve(p, ik.matrix_to_quat_xyzw(R), q_init=q, base_pos=base, iters=300)
                if pe > 0.003 or re_ > math.radians(1.5):
                    q2, pe, re_ = ik.solve(p, ik.matrix_to_quat_xyzw(R), q_init=None, base_pos=base, iters=400)
                    if pe > 0.003 or re_ > math.radians(1.5):
                        return False
                    q = q2
            return True

        for flip in (True, False):
            R = rot_rear(flip)
            pts = [seat + GRIP_REAR - 0.08 * X, seat + GRIP_REAR, seat + GRIP_REAR - PULL_OUT * X,
                   hol - np.array([HOLSTER_PRE, 0, 0]) + GRIP_REAR, hol + GRIP_REAR]
            if reach(self.right.base, [(p, R) for p in pts]):
                self.rot_right = R
                break
        else:
            raise RuntimeError("no right-hand grip reaches every target")
        self.press_lean = None
        for lean in (0.0, 15.0, 30.0, 45.0):            # the most upright press the left arm reaches
            for flip in (True, False):
                R = rot_press(flip, lean)
                a = R[:, 2]
                b = seat + T.BUTTON_TOP - 0.013 * a
                if reach(self.left.base, [(b - 0.06 * a, R), (b - PRESS_PAST * Z, R), (b - 0.08 * a, R)]):
                    self.rot_left, self.press_lean = R, lean
                    break
            if self.press_lean is not None:
                break
        else:
            raise RuntimeError("no left-hand press reaches the button")
        for yaw in (math.radians(20.0), 0.0, math.radians(40.0)):
            for flip in (False, True):
                arc = [(door_push_point(T, phi) - 0.013 * rot_door(phi, flip, yaw)[:, 2], rot_door(phi, flip, yaw))
                       for phi in np.linspace(math.radians(270.0) + T.DOOR.open, math.radians(271.0), 12)]
                if reach(self.left.base, arc):
                    self.door_flip, self.door_yaw = flip, yaw
                    break
            else:
                continue
            break
        else:
            raise RuntimeError("no left-hand door push reaches the whole arc")

    def observe(self, obs):
        """Keep the arms' measured joint positions from an observation (its 14-D joint_pos state: left j1..j6, left
        grip, right j1..j6, right grip). The demo reads nothing else back from the simulator."""
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
        """One line of the arm log: the phase, and per arm its trim and tracking error (largest joint, degrees)."""
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
    spec = importlib.util.spec_from_file_location("ev_task", task_path)
    task = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(task)
    return task


def plan_only(task_path: str) -> dict:
    """Every arm target of the demo (and its variant, DEMO_VARIANT) through the IK, with no simulator."""
    task = load_task(task_path)
    ctx = Ctx(None, task)
    n = sum(1 for _ in script(task, ctx))
    out = {"variant": os.environ.get("DEMO_VARIANT", ""), "steps": n, "seconds": round(n / CONTROL_HZ, 1),
           "worst_ik_mm": {"left": round(ctx.left.worst_mm, 2), "right": round(ctx.right.worst_mm, 2)},
           "worst_ik_deg": {"left": round(ctx.left.worst_deg, 2), "right": round(ctx.right.worst_deg, 2)},
           "door_grip": {"flip": ctx.door_flip, "yaw_deg": round(math.degrees(ctx.door_yaw))},
           "press_lean_deg": ctx.press_lean}
    print(json.dumps(out))
    return out


def _write_mp4(frames, path: Path) -> None:
    import imageio

    imageio.mimsave(path, [np.asarray(f) for f in frames], fps=CONTROL_HZ, codec="libx264",
                    output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])


def run(task_path: str, out_dir: str, max_steps: int = 1600, record: bool = True, cam_width: int = 320,
        cam_height: int = 176, hold_s: float = 3.0) -> dict:
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
    steps, ended, success, info = 0, None, False, {}
    while steps < max_steps:
        try:
            next(gen)
        except StopIteration:
            ended = steps if ended is None else ended
        res = emb.step(ctx.action())
        steps += 1
        ctx.steps = steps
        info = res.info
        ctx.observe(res.observation)
        snap(res.observation)
        if ARM_LOG:
            with open(ARM_LOG, "a") as f:
                f.write(json.dumps(ctx.arm_row()) + "\n")
        if steps % 100 == 0:
            print(f"step {steps} score {info.get('score')} record {task._RECORD}", flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    score = 1.0 if success else float(info.get("score", 0.0) or 0.0)
    result = {"task_score": score, "success": success, "steps": steps, "variant": os.environ.get("DEMO_VARIANT", ""),
              "worst_ik_mm": {"left": round(ctx.left.worst_mm, 2), "right": round(ctx.right.worst_mm, 2)},
              "worst_trim_deg": {"left": round(ctx.left.worst_trim, 2), "right": round(ctx.right.worst_trim, 2)},
              "record": dict(task._RECORD)}
    if record:
        import imageio

        for cam, fs in frames.items():
            _write_mp4(fs, out / f"demo_{cam}.mp4")
        top = next((c for c in frames if "top" in c), next(iter(frames), None))
        if top:
            imageio.imwrite(out / "setup.png", frames[top][0])
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--max-steps", type=int, default=1600)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--cam-width", type=int, default=320)
    ap.add_argument("--cam-height", type=int, default=176)
    ap.add_argument("--plan-only", action="store_true")
    a = ap.parse_args()
    if a.plan_only:
        plan_only(a.task)
    else:
        run(a.task, a.out, a.max_steps, record=not a.no_record, cam_width=a.cam_width, cam_height=a.cam_height)

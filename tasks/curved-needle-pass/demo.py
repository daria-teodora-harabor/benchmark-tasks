"""Scripted full-score demo for the curved needle task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1600] [--no-record] [--cam-width 320 --cam-height 176]

The demo is open-loop: the layout is fixed (object_jitter=0), so every move is planned in advance from the needle's
known starting pose on its foam block (allowing for how the needle settles in the fingers when they close on it, the
same in every run), and the demo never reads the needle's or any other object's pose. The only
thing it reads back is the arms' own joint positions (as a policy does), to trim the right arm's tracking error.

The right arm takes the needle by its back end off the foam block and carries it to the pad, its tip 1.5 cm short of
the entry slot. Then it turns the needle about the needle's own centre at 6 degrees a second, so that the needle's body
follows the track its tip makes: in through the entry slot, behind the cut and out through the exit slot, until about
4.6 cm of it is out. The left arm takes the needle by its dark tip piece (0.8 cm behind the tip), and the right arm
lets go and backs away.
Writes one MP4 per camera, setup.png (the first top-camera frame) and result.json to the output folder. For the setup
screenshot, render one step at 640 x 360: python demo.py task.py shot --max-steps 1 --cam-width 640 --cam-height 360

Test versions (DEMO_VARIANT=...), each must score as given:
  push_straight  after the approach, pushes the needle straight in along its tip instead of turning it      [0]
  wrist_pivot    turns the needle about the right gripper instead of the needle's centre                     [0]
  no_handoff     drives the needle through and keeps holding it                                             [0.5]
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
GRIP_BACK = -math.radians(175)   # where the right arm holds the needle: 5 degrees from its back end (local angle)
GRIP_TIP = -math.radians(7.5)    # where the left arm takes it: 7.5 degrees (0.8 cm) behind the tip, on the dark tip
                                 # piece; further back, its outer finger would land on the pad's top edge
OMEGA = math.radians(6.0)        # drive speed about the needle's centre (rad/s)
PHI_END = math.radians(135.0)    # drive until the tip is this far past the entry (about 4.6 cm out of the exit)
K_TRACK = 1.0                    # gain of the right arm's joint trim on its own measured tracking error (1/s)
TRACK_MAX = math.radians(1.0)    # largest trim per joint
GRIP_SLIP = np.array([0.0001, -0.0031, -0.0001])  # where the needle's centre ends up after the pick, in the needle's
                                 # own frame (m): it slides about 3 mm along its length as the fingers close on it and
                                 # sags as it is lifted by one end. The layout is fixed, so this is the same every run
                                 # (found by trial in the simulator), and the plan allows for it.
OPEN_PICK, OPEN_HAND = 0.5, 0.32  # finger openings (0 closed .. 1 open): about 47 mm and 30 mm between the pads
LEAN = math.radians(50.0)        # the left grip leans back toward its own base about its closing axis: held upright,
                                 # its wrist runs into the right arm's forearm (the first simulator run tore a hole so)


def _rot_z(a) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class Arm:
    """One arm: smooth Cartesian moves (IK on the URDF), each starting where the last command left it."""

    def __init__(self, name: str, base_pos):
        from yam_arena import ik
        from yam_arena.constants import READY

        self.name, self.base = name, np.asarray(base_pos, float)
        self.q = np.array(READY, float)
        self.grip = 1.0
        self.pos, self.rot = ik.tcp_pose(self.q, self.base)
        self.trim = np.zeros(6)  # added to the joint targets: cancels the arm's own measured tracking error
        self.cmd = np.r_[self.q, self.grip]
        self.worst_mm = 0.0
        self.worst_j6_margin = math.inf

    def _solve(self, pos, rot):
        from yam_arena import ik

        q, pe, _ = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), self.q, base_pos=self.base, iters=60)
        self.q, self.pos, self.rot = q, np.asarray(pos, float), rot
        self.worst_mm = max(self.worst_mm, pe * 1000)
        self.worst_j6_margin = min(self.worst_j6_margin, math.radians(120) - abs(float(q[5])))
        self.cmd = np.r_[q + self.trim, self.grip]

    def track(self, measured):
        """Integrate the arm's own tracking error (planned minus measured joint positions) into its joint trim."""
        if measured is None:
            return
        self.trim = np.clip(self.trim + K_TRACK * DT * (self.q - measured), -TRACK_MAX, TRACK_MAX)
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

    def aim(self, pos, rot):
        """Go to a pose this step (for planned paths)."""
        self._solve(pos, rot)


def wait(seconds: float):
    for _ in range(max(1, round(seconds * CONTROL_HZ))):
        yield


def tool(T, pos, rot, angle, flip):
    """TCP pose that holds the needle (origin ``pos``, rotation ``rot``) at local ``angle``: straight down, the fingers
    closing across the needle (along its radius), pad tips 1 mm below its underside."""
    from yam_arena.constants import PAD_TIP_PAST_TCP
    from yam_arena.demo import grip_rotation

    point = pos + rot @ T.local_point(angle)
    radial = rot @ np.array([math.cos(angle), math.sin(angle), 0.0])
    yaw = math.atan2(radial[1], radial[0]) + flip * math.pi / 2
    return point + np.array([0.0, 0.0, PAD_TIP_PAST_TCP - T.A / 2 - 0.001]), grip_rotation(yaw)


def lean(pose, base, angle=LEAN):
    """Tilt a grip pose by ``angle`` about its closing axis, the way that leans the wrist back toward the arm's
    ``base``. The fingers still close across the needle, onto its side faces."""
    p, r = pose
    best = None
    for a in (angle, -angle):
        c, s = math.cos(a), math.sin(a)
        rn = r @ np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
        back, to_base = -rn[:, 2], np.asarray(base, float) - p
        score = back[0] * to_base[0] + back[1] * to_base[1]
        if best is None or score > best[0]:
            best = (score, rn)
    return p, best[1]


def best_flip(T, base, poses, angle):
    """The turn of the grip (fingers across the needle either way) that IK reaches at every pose with the most room
    left on joint 6 (the wrist roll, +-120 degrees)."""
    from yam_arena import ik
    from yam_arena.constants import READY

    best, best_margin = -1, -math.inf
    for flip in (-1, 1):
        q, margin = np.array(READY, float), math.inf
        for pos, rot in poses:
            p, r = tool(T, pos, rot, angle, flip)
            q, pe, re = ik.solve(p, ik.matrix_to_quat_xyzw(r), q, base_pos=base, iters=200)
            margin = min(margin, math.radians(120) - abs(float(q[5]))) if pe < 0.001 and re < 0.01 else -math.inf
        if margin > best_margin:
            best, best_margin = flip, margin
    return best, best_margin


def start_pose(T):
    """Where the needle rests on its foam block at the start (known: the layout is fixed): origin and rotation."""
    return np.array([T.START_XY[0], T.START_XY[1], T.ZN]), T.rz(T.START_YAW)


def script(task, ctx):
    """The demo as a generator: one yield per control step."""
    T = task
    A, B = ctx.right, ctx.left
    variant = os.environ.get("DEMO_VARIANT", "")
    phi0 = -T.APPROACH
    drive = [T.drive_pose(phi0 + k * (PHI_END - phi0) / 12) for k in range(13)]
    flip_a, margin_a = best_flip(T, A.base, drive, GRIP_BACK)
    ctx.log(f"right arm grip flip {flip_a:+d}, joint 6 margin over the drive {math.degrees(margin_a):.1f} deg")

    # 1. Take the needle by its back end off the foam block, where it lies at its known starting pose.
    p0, r0 = start_pose(T)
    grasp = tool(T, p0, r0, GRIP_BACK, flip_a)
    yield from A.move(grasp[0] + [0, 0, 0.06], grasp[1], 2.5, grip=OPEN_PICK)
    yield from A.move(*grasp, 1.5)
    yield from A.set_grip(0.0, 0.8)
    yield from wait(0.4)
    g_pos = r0.T @ (grasp[0] - p0) - GRIP_SLIP  # the grip in the needle's frame, as the needle settles in the fingers
    g_rot = r0.T @ grasp[1]

    def tcp_for(pos, rot):
        return pos + rot @ g_pos, rot @ g_rot

    # 2. Lift it 1.5 cm, carry it to the pad (tip 1.5 cm short of the entry), and lower it into the slots' plane.
    yield from A.move(A.pos + [0, 0, 0.015], A.rot, 1.0)
    p_start, r_start = T.drive_pose(phi0)
    yield from A.move(*tcp_for(p_start + [0, 0, 0.015], r_start), 3.0)
    yield from A.move(*tcp_for(p_start, r_start), 1.0)
    yield from wait(0.5)

    if variant == "push_straight":
        tangent = r_start @ np.array([0.0, 1.0, 0.0])
        for k in range(int(6.0 * CONTROL_HZ)):
            A.aim(*tcp_for(p_start + tangent * 0.040 * (k + 1) / (6.0 * CONTROL_HZ), r_start))
            yield
        return

    if variant == "wrist_pivot":
        pos, rot = A.pos.copy(), A.rot.copy()
        for k in range(int(20.0 * CONTROL_HZ)):
            A.aim(pos, _rot_z(OMEGA * DT * (k + 1)) @ rot)
            yield
        return

    # 3. Drive: turn the needle about its own centre along the planned path, the right arm trimming its own tracking
    #    error from its measured joints (it never reads the needle). The trim is kept, not updated, after the drive:
    #    once both hands hold the needle, the arm can't reach its plan exactly and the trim would push on the needle.
    ctx.log("drive: open loop" + ("" if ctx.joints("right") is not None else " (no joint readings: no trim)"))
    phi = phi0
    while phi < PHI_END:
        A.track(ctx.joints("right"))
        ctx.note_trim(A)
        phi = min(PHI_END, phi + OMEGA * DT)
        A.aim(*tcp_for(*T.drive_pose(phi)))
        if ctx.debug and ctx.steps % 10 == 0:
            ctx.log(f"step {ctx.steps} phi {math.degrees(phi):6.1f} trim {np.round(np.degrees(A.trim), 2)} deg")
        yield
    yield from wait(0.5)
    if variant == "no_handoff":
        yield from wait(3.0)
        return

    # 4. Hand over: the left arm takes the needle 0.8 cm behind its tip, at the drive's planned end pose, its grip
    #    leaning back by LEAN and coming in along its own axis from 6 cm out; then the right arm lets go and backs away.
    p_end, r_end = T.drive_pose(PHI_END)
    flip_b, margin_b = best_flip(T, B.base, [(p_end, r_end)], GRIP_TIP)
    ctx.log(f"left arm grip flip {flip_b:+d}, joint 6 margin {math.degrees(margin_b):.1f} deg")
    take = lean(tool(T, p_end, r_end, GRIP_TIP, flip_b), B.base)
    yield from B.move(take[0] - take[1][:, 2] * 0.06, take[1], 2.5, grip=OPEN_HAND)
    yield from B.move(*take, 1.5)
    yield from B.set_grip(0.0, 0.8)
    yield from wait(0.5)
    yield from A.set_grip(OPEN_PICK, 0.6)
    yield from A.move(A.pos + [0, 0, 0.04], A.rot, 1.0)
    yield from A.move(A.pos + [-0.06, -0.06, 0.02], A.rot, 1.5)


class Ctx:
    def __init__(self, emb, task):
        from yam_arena import station

        self.emb, self.task = emb, task
        self.left = Arm("left", station.BASES["left"])
        self.right = Arm("right", station.BASES["right"])
        self.debug = bool(os.environ.get("DEMO_DEBUG"))
        self.steps = 0
        self.worst = {"trim_deg": 0.0}
        self._q = None

    def observe(self, obs):
        """Keep the arms' measured joint positions from an observation (its 14-D joint_pos state: left j1..j6,
        left grip, right j1..j6, right grip). The demo reads nothing else back from the simulator."""
        try:
            from inspect_robots_yam.packing import STATE_KEY

            q = obs.state[STATE_KEY]
            q = np.asarray(q.cpu().numpy() if hasattr(q, "cpu") else q, float).ravel()
            self._q = {"left": q[0:6], "right": q[7:13]} if q.size == 14 else None
        except Exception:  # noqa: BLE001  (no readings: the demo runs without the trim)
            self._q = None

    def joints(self, name):
        return None if self._q is None else self._q[name]

    def note_trim(self, arm):
        self.worst["trim_deg"] = max(self.worst["trim_deg"], float(np.degrees(np.abs(arm.trim).max())))

    def action(self):
        from inspect_robots.types import Action
        from inspect_robots_yam.packing import pack

        return Action(data=pack(self.left.cmd, self.right.cmd))

    def log(self, msg):
        print(msg, flush=True)


def _write_mp4(frames, path: Path) -> None:
    import imageio

    imageio.mimsave(path, [np.asarray(f) for f in frames], fps=CONTROL_HZ, codec="libx264",
                    output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])


def run(task_path: str, out_dir: str, max_steps: int = 1600, record: bool = True, cam_width: int = 320,
        cam_height: int = 176, hold_s: float = 4.0) -> dict:
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
              "joint6_margin_deg": {"left": round(math.degrees(ctx.left.worst_j6_margin), 1),
                                    "right": round(math.degrees(ctx.right.worst_j6_margin), 1)},
              "drive": {k: round(v, 2) for k, v in ctx.worst.items()}, "record": dict(task._RECORD)}
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
    a = ap.parse_args()
    run(a.task, a.out, a.max_steps, record=not a.no_record, cam_width=a.cam_width, cam_height=a.cam_height)

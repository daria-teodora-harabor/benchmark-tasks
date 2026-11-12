"""Offline kinematic check of the phone demo. NumPy only, no simulator:

    python3 run_kin.py --src <yam_arena source folder> [variant ...]
    YAM_ARENA_SRC=<folder> python3 run_kin.py [--lag 1.5] [--deadband 1] [--hulls DIR] [--log DIR] [--no-reach] [-v]
                                              [variant ...]

``--src`` (or $YAM_ARENA_SRC) is the folder that holds the ``yam_arena`` package (the harness source); without it the
installed yam_arena is used. Its pure-Python modules (ik, urdf, station, constants, demo) are all this needs.

Each variant of demo.py (all of them by default) is played the way ``--policy yam_scripted`` and the embodiment play
it at ``-E control_hz=20``:

- yam_scripted: ``ArmPlan(demo.plan()[arm], station.BASES[arm]).action7(t)`` at t = state_time + 0.05 s; an arm
  without a plan is held at READY with its gripper open. The policy asks to stop 3 s (hold_s) after the plan ends.
- Arm response: 12 physics steps of 1/240 s per control step. Each joint follows its target as a first-order lag with
  time constant damping / stiffness: 0.0625 s for joints 1-3, 0.15 s for joints 4-6, 0.02 s for the fingers (``--lag``
  scales them; 0 tracks exactly). ``--deadband 1`` adds the actuators' joint friction read as a Coulomb torque: each
  joint then stops friction / stiffness short of its target (3.75 mrad for joints 1-3, 6 mrad for 4-6), which can
  leave the pads 2 to 3 mm higher than aimed. How Isaac Lab applies ``friction`` is not checked offline, so this is
  a worst case to compare against, not the default.
- Touch: every physics step, URDF forward kinematics places the 60 sample points of each finger pad (touch.py's
  table when it imports, else the copy below), on both arms. The glass is rigid: an arm whose pad points would go
  below the glass (over the phone's footprint) is lifted as a whole until its lowest point rests on it. Each pad's gap
  (its lowest point above the glass) and contact-patch centre (mean x, y of its points within 0.5 mm of its lowest)
  go to phoneui.PhoneUI.step exactly as touch.py feeds them in the simulator.
- Embodiment: after every control step success() (= PhoneUI.success) must hold for 20 steps (SUCCESS_HOLD_S 1 s) to
  end the trial; otherwise the trial ends when the policy asks to stop or after 2,400 steps. The task score is 1.0 on
  success, else the last step's score (SCORE_OVER "end").

Checks per variant (a failed one prints BAD and the exit status is 1):
- the final score is demo.EXPECTED[variant], and the UI events (swipe, keys, buttons, calls, held) are the expected
  ones;
- score.py's replay of the trial log gives the same rows and score;
- IK error at every control step under 0.5 mm and 0.1 deg (the face-cans reach test), joint 6 inside its limits;
- every tap with closed fingers is one touch made by both pads, and lands inside its hit area; every touch pressed:
  the arm's free motion would have gone at least 1 mm into the glass (the aim is 3 mm), so the PD lag did not
  leave it short;
- the pads rise at least 5 mm between touches (12 mm commanded) and stay at least 5 mm clear on every move across
  the screen; nothing but the right arm's finger pads comes within 5 mm of the glass (gripper housing, the rest of
  the fingers, the wrist camera mount and the left arm); with ``--hulls`` the URDF's collision hulls are used, else
  their bounding boxes, which over-state how low a tilted gripper reaches;
- a 911 call that reaches the full score ends the trial on success before the policy's stop request.

Then (unless ``--no-reach``) a reach table: IK straight down at every key and button of every screen, pressed 3 mm
into the glass and at 12 and 30 mm above it, with the smallest joint-limit margin, and the vertical stiffness the
joint PD gives at the pads there (so the force of a 3 mm press, about 1 N).
"""

from __future__ import annotations

import argparse
import builtins
import json
import math
import os
import re
import sys
import time
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import layout as L    # noqa: E402
import phoneui as P   # noqa: E402
import score as S     # noqa: E402

CONTROL_HZ = 20.0
DT = 1.0 / CONTROL_HZ
SUBSTEPS = 12
PHYSICS_DT = DT / SUBSTEPS
MAX_STEPS = 2400
HOLD_S = 3.0              # yam_scripted's default hold after the plan ends
SUCCESS_HOLD_S = 1.0
PATCH_BAND = 0.0005       # m: points within this of a pad's lowest form its contact patch
PAD_END_Z = -0.09016      # m: the pad end in each finger link's frame (finger axis -z)
PAD_REGION = 0.012        # m: the distal part of a finger counted as its pad
CLEAR_MIN_MM = 5.0
PRESSED_MM = 1.0        # every touch must have pressed the arm at least this far into the glass (free motion)
NEAR_MM = 20.0           # the gripper's tilt is reported while its pads are this near the glass
IK_MM, IK_DEG = 0.5, 0.1
ARMS = ("left", "right")
TIPS = ("tip_left", "tip_right")

# The finger-pad sample points (mm, each finger link's own frame), generated from the URDF's finger collision hulls;
# the same table touch.py uses (it is preferred when touch.py imports).
PAD_POINTS_MM = {
    "tip_left": (
        (19.04, -46.38, -90.16), (19.04, -45.04, -90.16), (28.94, -46.38, -90.16), (28.94, -45.04, -90.16),
        (29.56, -46.39, -90.09), (18.40, -44.99, -90.08), (18.62, -46.88, -90.06), (29.36, -47.05, -90.01),
        (29.91, -45.04, -90.00), (18.06, -46.56, -89.99), (29.99, -46.90, -89.90), (19.04, -47.41, -89.88),
        (28.94, -44.57, -89.81), (30.44, -46.41, -89.76), (28.94, -47.59, -89.76), (18.76, -44.55, -89.75),
        (17.85, -44.67, -89.73), (30.51, -45.04, -89.72), (29.49, -44.55, -89.65), (17.56, -47.08, -89.63),
        (18.06, -47.55, -89.60), (30.23, -47.41, -89.57), (30.64, -46.92, -89.55), (17.18, -46.51, -89.51),
        (29.73, -47.82, -89.42), (18.60, -47.93, -89.38), (30.27, -44.55, -89.34), (17.37, -44.61, -89.34),
        (16.97, -45.04, -89.33), (29.08, -48.03, -89.29), (17.82, -47.85, -89.21), (31.15, -46.44, -89.19),
        (30.46, -47.74, -89.15), (31.20, -45.04, -89.13), (30.81, -44.59, -89.11), (17.14, -47.46, -89.10),
        (31.06, -47.31, -88.99), (19.04, -48.23, -88.92), (16.58, -46.63, -88.86), (17.05, -44.55, -88.83),
        (16.55, -44.99, -88.83), (17.99, -48.18, -88.79), (17.33, -47.91, -88.78), (31.43, -46.81, -88.78),
        (29.80, -48.25, -88.73), (28.94, -48.30, -88.72), (31.11, -44.55, -88.61), (16.54, -47.27, -88.50),
        (30.79, -48.01, -88.49), (18.53, -48.38, -88.36), (17.57, -48.26, -88.32), (31.40, -47.52, -88.30),
        (31.74, -46.50, -88.25), (16.25, -46.50, -88.25), (31.73, -44.99, -88.25), (16.98, -47.99, -88.23),
        (16.37, -44.71, -88.20), (29.72, -48.42, -88.13), (30.31, -48.38, -88.02), (18.11, -48.46, -87.87),
    ),
    "tip_right": (
        (-28.94, 46.38, -90.16), (-23.70, 45.04, -90.16), (-19.04, 46.81, -90.11), (-29.61, 46.47, -90.08),
        (-18.40, 44.94, -90.08), (-29.58, 44.84, -90.05), (-18.29, 46.72, -90.03), (-28.94, 47.22, -89.98),
        (-28.94, 44.64, -89.95), (-19.04, 44.64, -89.95), (-29.75, 47.14, -89.90), (-17.78, 45.04, -89.88),
        (-30.21, 45.04, -89.88), (-17.67, 46.40, -89.83), (-18.86, 47.53, -89.79), (-30.43, 46.41, -89.76),
        (-18.19, 47.39, -89.76), (-17.56, 47.08, -89.63), (-29.78, 44.55, -89.62), (-29.62, 47.68, -89.59),
        (-18.22, 44.55, -89.57), (-30.76, 44.94, -89.53), (-30.60, 47.09, -89.52), (-17.24, 44.84, -89.51),
        (-28.94, 47.90, -89.46), (-17.69, 47.65, -89.35), (-18.45, 47.94, -89.35), (-30.27, 44.55, -89.34),
        (-19.04, 48.03, -89.30), (-30.33, 47.72, -89.26), (-16.85, 46.52, -89.20), (-31.17, 46.78, -89.13),
        (-17.14, 47.46, -89.10), (-30.85, 47.46, -89.10), (-18.00, 48.08, -88.98), (-29.67, 48.15, -88.98),
        (-31.05, 44.61, -88.93), (-16.94, 44.61, -88.93), (-31.44, 45.04, -88.83), (-16.55, 45.04, -88.82),
        (-30.31, 48.12, -88.72), (-31.38, 47.16, -88.68), (-16.61, 47.16, -88.68), (-17.22, 47.93, -88.65),
        (-18.70, 48.33, -88.61), (-31.15, 47.67, -88.53), (-29.05, 48.35, -88.50), (-29.86, 48.33, -88.47),
        (-18.13, 48.33, -88.47), (-31.25, 44.55, -88.36), (-16.70, 47.69, -88.28), (-31.71, 46.76, -88.28),
        (-16.28, 46.76, -88.28), (-31.74, 45.04, -88.25), (-16.25, 45.04, -88.25), (-30.86, 48.10, -88.19),
        (-16.62, 44.55, -88.10), (-17.49, 48.31, -88.08), (-30.31, 48.38, -88.02), (-18.11, 48.47, -87.87),
    ),
}
# Bounding boxes (mm) of the URDF collision hulls, used when --hulls is not given: the gripper housing in the gripper
# frame, and each finger without its distal 12 mm in its own link frame.
HOUSING_BOX_MM = ((-51.0, -32.5, -10.5), (51.0, 32.5, 63.5))
FINGER_BOX_MM = {"tip_left": ((-10.01, -68.29, -76.66), (57.99, 3.29, 3.68)),
                 "tip_right": ((-57.99, -3.29, -76.66), (10.01, 68.29, 3.68))}
WRIST_CAM_IN_GRIPPER = (0.0, -0.0696, 0.0717)   # the wrist camera mount (station.WRIST_CAM_POS, link_6 -> gripper)

# The UI events each variant must produce, in order (see events()), or a list of acceptable sequences. open_gripper's
# last tap puts one pad on Cancel and one on Delete: which lifts first is down to millimetres, and both end the same.
_ROUTE = ["button passcode emergency", "key dialpad 9", "key dialpad 1", "key dialpad 1", "button dialpad call",
          "call 911 placed"]
EXPECTED_EVENTS: Dict[str, List[str]] = {
    "full": ["swipe", *_ROUTE, "held"],
    "guess_passcode": ["swipe", *[f"key passcode {d}" for d in "123456"], "attempt", *_ROUTE, "held"],
    "type_911_then_emergency": ["swipe", "key passcode 9", "key passcode 1", "key passcode 1", *_ROUTE, "held"],
    "no_swipe": [],
    "stop_at_dialpad": ["swipe", "button passcode emergency"],
    "dial_no_call": ["swipe", *_ROUTE[:4]],
    "no_lift_11": ["swipe", "button passcode emergency", "key dialpad 9", "key dialpad 1", "button dialpad call",
                   "call 91 failed"],
    "wrong_digit_fixed": ["swipe", "button passcode emergency", "key dialpad 9", "key dialpad 2",
                          "button dialpad delete", "key dialpad 1", "key dialpad 1", "button dialpad call",
                          "call 911 placed", "held"],
    "open_gripper": [["swipe", "button passcode emergency", "key dialpad 8", "key dialpad 2", "key dialpad 2", *end]
                     for end in (["button dialpad cancel"], ["button dialpad delete", "button dialpad cancel"])],
    "touch_during_call": ["swipe", *_ROUTE, "held"],
    "hang_up": ["swipe", *_ROUTE, "button calling end"],
    "hang_up_redial": ["swipe", *_ROUTE, "button calling end", "key dialpad 9", "key dialpad 1", "key dialpad 1",
                       "button dialpad call", "call 911 placed", "held"],
}


# -- the harness source --------------------------------------------------------------------------------------------

def _zip_shim() -> None:
    """The archived yam_arena calls zip(..., strict=True), new in Python 3.10. Offline tooling only."""
    try:
        zip((), strict=True)    # type: ignore[call-overload]
    except TypeError:
        _zip = builtins.zip
        builtins.zip = lambda *a, strict=False: _zip(*a)   # type: ignore[assignment]


def load_harness(src: Optional[str]):
    """The yam_arena package (from ``src`` when given) and its constants; its pure-Python modules must import."""
    _zip_shim()
    if src:
        sys.path.insert(1, str(Path(src).expanduser().resolve()))
    try:
        import yam_arena
        import yam_arena.demo
        import yam_arena.ik
        import yam_arena.station
        import yam_arena.urdf  # noqa: F401
        from yam_arena import constants as C
    except Exception as e:  # noqa: BLE001
        sys.exit(f"cannot import yam_arena ({type(e).__name__}: {e}); pass --src <folder holding yam_arena> or set "
                 "YAM_ARENA_SRC")
    return yam_arena, C


def pad_table() -> Tuple[Dict[str, np.ndarray], str]:
    try:
        import touch                                                   # the simulator's own table, if it is there
        table = getattr(touch, "PAD_POINTS_MM")
        src = "touch.py"
    except Exception:  # noqa: BLE001
        table, src = PAD_POINTS_MM, "run_kin.py"
    out = {k: np.asarray(table[k], float) / 1000.0 for k in TIPS}
    if src == "touch.py":
        same = all(np.allclose(out[k], np.asarray(PAD_POINTS_MM[k]) / 1000.0, atol=1e-6) for k in TIPS)
        src += "" if same else " (differs from run_kin.py's copy)"
    return out, src


def _box_corners(lo, hi) -> np.ndarray:
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]) / 1000.0


def non_pad_points(yam_arena, hulls_dir: Optional[str]) -> Tuple[Dict[str, np.ndarray], str]:
    """Points, per link and in the link's frame, of everything near the hand except the finger pads."""
    pts = {"gripper": np.vstack([_box_corners(*HOUSING_BOX_MM), np.array([WRIST_CAM_IN_GRIPPER])])}
    for tip in TIPS:
        pts[tip] = _box_corners(*FINGER_BOX_MM[tip])
    if not hulls_dir:
        return pts, "hull bounding boxes"
    from yam_arena.analysis import read_stl
    from yam_arena.urdf import rpy_matrix

    d = Path(hulls_dir).expanduser()
    found = {}
    for link in ET.parse(yam_arena.YAM_URDF).getroot().iter("link"):
        name = link.get("name")
        if name not in pts:
            continue
        vs = []
        for col in link.findall("collision"):
            o = col.find("origin")
            rpy, xyz = (o.get("rpy", "0 0 0"), o.get("xyz", "0 0 0")) if o is not None else ("0 0 0", "0 0 0")
            rot, pos = rpy_matrix(np.array([float(c) for c in rpy.split()])), np.array([float(c) for c in xyz.split()])
            fn = Path(col.find("geometry/mesh").get("filename")).name
            f = d / f"{name}__{fn}"
            if not f.is_file():
                f = d / name / fn
            if not f.is_file():
                return pts, f"hull bounding boxes ({fn} not found in --hulls)"
            with np.errstate(all="ignore"):
                vs.append(np.unique(read_stl(f).reshape(-1, 3), axis=0) @ rot.T + pos)
        v = np.vstack(vs)
        if name in TIPS:
            v = v[v[:, 2] > PAD_END_Z + PAD_REGION]
        found[name] = v
    found["gripper"] = np.vstack([found["gripper"], np.array([WRIST_CAM_IN_GRIPPER])])
    return found, f"URDF collision hulls ({sum(len(v) for v in found.values())} points)"


# -- the model -----------------------------------------------------------------------------------------------------

class World:
    """The scene geometry the pads can meet: the glass over the phone's footprint, the mat, the table."""

    @staticmethod
    def surface(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        u, v = (L.PHONE_Y - y) * 1000.0, (x - L.PHONE_X) * 1000.0
        on_phone = (np.abs(u) <= L.GLASS_SIZE[1] * 500.0) & (np.abs(v) <= L.GLASS_SIZE[0] * 500.0)
        on_mat = (np.abs(x - L.MAT_POS[0]) <= L.MAT_SIZE[0] / 2) & (np.abs(y - L.MAT_POS[1]) <= L.MAT_SIZE[1] / 2)
        return np.where(on_phone, L.GLASS_Z, np.where(on_mat, L.MAT_TOP, L.TABLE_TOP))


class Arm:
    """One arm's joints, following their targets with a first-order lag."""

    def __init__(self, name, base, C, H, lag: float, deadband: float = 0.0):
        self.name, self.base, self.H = name, np.asarray(base, float), H
        self.q = np.array(C.READY, float)
        self.f = C.FINGER_OPEN
        self.q_cmd, self.f_cmd = self.q.copy(), self.f
        k_over_d = np.array([C.SHOULDER["stiffness"] / C.SHOULDER["damping"]] * 3
                            + [C.WRIST["stiffness"] / C.WRIST["damping"]] * 3)
        g = C.GRIPPER["stiffness"] / C.GRIPPER["damping"]
        self.alpha = 1.0 - np.exp(-k_over_d * PHYSICS_DT / lag) if lag > 0 else np.ones(6)
        self.alpha_f = 1.0 - math.exp(-g * PHYSICS_DT / lag) if lag > 0 else 1.0
        self.time_const = (1.0 / k_over_d * lag, lag / g)
        # Joint friction read as a Coulomb torque against the PD stiffness: a joint stops friction / stiffness short
        # of its target (``deadband`` scales it; 0 is no friction). 3.75 mrad for joints 1-3, 6 mrad for 4-6.
        self.deadband = deadband * np.array([C.SHOULDER.get("friction", 0.0) / C.SHOULDER["stiffness"]] * 3
                                            + [C.WRIST.get("friction", 0.0) / C.WRIST["stiffness"]] * 3)
        self._fk_key, self._fk = None, None

    def command(self, a7, finger_open: float, lower, upper) -> None:
        self.q_cmd = np.clip(np.asarray(a7[:6], float), lower, upper)
        self.f_cmd = float(np.clip(a7[6], 0.0, 1.0)) * finger_open

    def substep(self) -> None:
        e = self.q_cmd - self.q
        if self.deadband.any():
            e = np.sign(e) * np.maximum(np.abs(e) - self.deadband, 0.0)
        self.q = self.q + e * self.alpha
        self.f = self.f + (self.f_cmd - self.f) * self.alpha_f

    def fk(self) -> Dict[str, np.ndarray]:
        key = (tuple(np.round(self.q, 12)), round(self.f, 12))
        if key != self._fk_key:
            j = {f"joint{i + 1}": float(v) for i, v in enumerate(self.q)}
            j.update(joint7=self.f, joint8=self.f)
            poses = self.H.forward_kinematics(self.H.YAM_URDF, j)
            for p in poses.values():
                p[:3, 3] += self.base
            self._fk_key, self._fk = key, poses
        return self._fk


class Harness:
    """Bundles the yam_arena pieces this script uses."""

    def __init__(self, yam_arena, C):
        from yam_arena import ik, station
        from yam_arena.demo import ArmPlan, grip_rotation
        from yam_arena.urdf import forward_kinematics

        self.ik, self.station, self.ArmPlan, self.grip_rotation = ik, station, ArmPlan, grip_rotation
        self.forward_kinematics, self.YAM_URDF, self.C = forward_kinematics, yam_arena.YAM_URDF, C


def arm_contact(arm: Arm, pads: Dict[str, np.ndarray], other: Dict[str, np.ndarray]):
    """World pad points of ``arm`` resting on (not inside) the scene, the lift that took, and the lowest z of each of
    its other parts (``other``: link -> points in that link's frame)."""
    fk = arm.fk()
    world = {t: pads[t] @ fk[t][:3, :3].T + fk[t][:3, 3] for t in TIPS}
    allp = np.vstack([world[t] for t in TIPS])
    lift = max(0.0, float(np.max(World.surface(allp[:, 0], allp[:, 1]) - allp[:, 2])))
    rest = {link: float((pts @ fk[link][2, :3] + fk[link][2, 3]).min()) + lift for link, pts in other.items()}
    if lift > 0:
        for t in TIPS:
            world[t] = world[t] + np.array([0.0, 0.0, lift])
    return world, lift, rest, fk


def pad_reading(points: np.ndarray) -> Tuple[float, float, float]:
    """(gap_mm, u, v): the lowest point's height above the glass, and the contact patch centre in screen mm."""
    lo = float(points[:, 2].min())
    band = points[points[:, 2] <= lo + PATCH_BAND]
    u, v = L.world_to_screen(float(band[:, 0].mean()), float(band[:, 1].mean()))
    return (lo - L.GLASS_Z) * 1000.0, u, v


# -- one variant ---------------------------------------------------------------------------------------------------

def events(rows: List[dict]) -> List[str]:
    out = []
    for r in rows:
        k = r["kind"]
        if k == "swipe":
            out.append("swipe")
        elif k == "key":
            out.append(f"key {r['screen']} {r['key']}")
        elif k == "button":
            out.append(f"button {r['screen']} {r['button']}")
        elif k == "attempt":
            out.append("attempt")
        elif k == "call":
            out.append(f"call {r['number']} {r['result']}")
        elif k == "held":
            out.append("held")
    return out


_TAP = re.compile(r"^tap (\w+) ([^:\s]+)")


def aim_of(label: str) -> Optional[Tuple[str, str]]:
    m = _TAP.match(label)
    if m and m.group(1) in L.AIM and m.group(2) in L.AIM[m.group(1)]:
        return m.group(1), m.group(2)
    return None


def play(variant: str, H: Harness, pads, other, lag: float, log_dir: Optional[str], deadband: float = 0.0) -> dict:
    import demo as D

    C, ik = H.C, H.ik
    os.environ[D.VARIANT_ENV] = variant
    plan = D.plan()
    plans = {arm: H.ArmPlan(list(wps), H.station.BASES[arm]) for arm, wps in plan.items()}
    end = max((p.duration for p in plans.values()), default=0.0)
    arms = {a: Arm(a, H.station.BASES[a], C, H, lag, deadband) for a in ARMS}
    ui = P.PhoneUI(meta={"variant": variant, "source": "run_kin", "lag": lag, "deadband": deadband})
    need = max(1, round(SUCCESS_HOLD_S * CONTROL_HZ))

    labels: List[str] = []
    low_gap = []                  # per physics step: the right arm's lower pad gap (mm)
    press_depth = []              # per physics step: how far the right arm's free motion would have gone in (mm)
    worst = {"tilt_deg": 0.0, "clear_mm": math.inf, "clear_at": "", "left_mm": math.inf, "j6_margin_deg": math.inf,
             "press_xy_mm": 0.0, "over_gap_mm": math.inf, "over_at": "", "joint_margin": (math.inf, "", "")}
    links: Dict[str, float] = {}  # lowest height above the glass of each non-pad part of the right hand (mm)
    held, ended, k, score = 0, "", 0, 0.0
    sub = 0
    for k in range(MAX_STEPS):
        t_cmd = k * DT + DT
        label = "-"
        for name, arm in arms.items():
            if name in plans:
                a7, label_a = plans[name].action7(t_cmd)
                if name == "right":
                    label = label_a
            else:
                a7 = np.r_[C.READY, 1.0]
            arm.command(a7, C.FINGER_OPEN, ik.LOWER, ik.UPPER)
            margins = np.degrees(np.minimum(arm.q_cmd - ik.LOWER, ik.UPPER - arm.q_cmd))
            worst["j6_margin_deg"] = min(worst["j6_margin_deg"], float(margins[5]))
            if name in plans and float(margins.min()) < worst["joint_margin"][0]:
                worst["joint_margin"] = (float(margins.min()), f"joint{int(np.argmin(margins)) + 1}", label_a)
        labels.append(label)
        stop = t_cmd > end + HOLD_S
        for _ in range(SUBSTEPS):
            sub += 1
            samples = []
            for name, arm in arms.items():
                arm.substep()
                world, lift, rest, fk = arm_contact(arm, pads, other)
                for tip in TIPS:
                    samples.append(P.PadSample(f"{name}_{tip}", *pad_reading(world[tip])))
                if name == "right":
                    low_gap.append(min(s.gap_mm for s in samples[-2:]))
                    press_depth.append(lift * 1000.0)
                    for ln, z in rest.items():
                        links[ln] = min(links.get(ln, math.inf), (z - L.GLASS_Z) * 1000.0)
                    link = min(rest, key=rest.get)
                    clear = (rest[link] - L.GLASS_Z) * 1000.0
                    if clear < worst["clear_mm"]:
                        worst["clear_mm"], worst["clear_at"] = clear, f"{link}, {label}"
                    if low_gap[-1] < NEAR_MM:                    # how far from straight down, near the glass
                        g = fk["gripper"][:3, :3]
                        worst["tilt_deg"] = max(worst["tilt_deg"], math.degrees(math.acos(min(1.0, -g[2, 2]))))
                    if label.endswith(": over") or label.startswith(("clear", "to the phone")):
                        if low_gap[-1] < worst["over_gap_mm"]:
                            worst["over_gap_mm"], worst["over_at"] = low_gap[-1], label
                else:
                    low = min(min(rest.values()), min(float(world[t][:, 2].min()) for t in TIPS))
                    worst["left_mm"] = min(worst["left_mm"], (low - L.GLASS_Z) * 1000.0)
            ui.step(sub * PHYSICS_DT, samples)
        # the end of a control step: the tool centre point against the plan, while pressing
        if label.endswith(": press") and "right" in plans:
            fk = arms["right"].fk()
            tcp = fk["gripper"][:3, :3] @ np.array(C.TCP_OFFSET) + fk["gripper"][:3, 3]
            want = plans["right"].target(t_cmd)[0]
            worst["press_xy_mm"] = max(worst["press_xy_mm"], float(np.hypot(*(tcp - want)[:2])) * 1000.0)
        score = ui.score
        held = held + 1 if ui.success else 0
        if held >= need:
            ended = "success"
            break
        if stop:
            ended = "stop requested"
            break
    else:
        ended = "run limit"
    steps = k + 1
    final = 1.0 if ended == "success" else score

    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(log_dir) / f"{variant}.jsonl", "w") as f:
            for r in ui.rows:
                f.write(json.dumps(r) + "\n")
    rows = [json.loads(json.dumps(r)) for r in ui.rows]
    rep = S.check(rows)

    # touches: who made them, where, for how long, against which target
    touches, cur = [], {}
    for r in ui.rows:
        if r["kind"] != "touch":
            continue
        if r["action"] == "begin":
            cur[r["touch"]] = {"id": r["touch"], "t0": r["t"], "down": (r["u"], r["v"]), "pads": [r["pad"]],
                               "screen": r["screen"], "label": labels[min(len(labels) - 1, int(r["t"] / DT - 1e-9))]}
        elif r["action"] == "join":
            cur[r["touch"]]["pads"].append(r["pad"])
        elif r["action"] == "end":
            tc = cur.pop(r["touch"])
            tc.update(t1=r["t"], lift=(r["u"], r["v"]))
            touches.append(tc)
    touches.sort(key=lambda tc: tc["t0"])
    gaps = np.array(low_gap)
    depth = np.array(press_depth)
    rise, hit_margin, aim_err, min_depth = math.inf, math.inf, 0.0, math.inf
    problems: List[str] = []
    for i, tc in enumerate(touches):
        a, b = int(round(tc["t0"] / PHYSICS_DT)) - 1, int(round(tc["t1"] / PHYSICS_DT))
        tc["depth"] = float(depth[a:b].max()) if b > a else 0.0
        min_depth = min(min_depth, tc["depth"])
        if i + 1 < len(touches):
            c = int(round(touches[i + 1]["t0"] / PHYSICS_DT)) - 1
            if c > b:                                          # the next touch began after this one ended
                tc["rise"] = float(gaps[b:c].max())
                rise = min(rise, tc["rise"])
        target = aim_of(tc["label"])
        if target and variant != "open_gripper":
            screen, name = target
            aim_err = max(aim_err, math.hypot(tc["down"][0] - L.aim(screen, name)[0],
                                              tc["down"][1] - L.aim(screen, name)[1]))
            if screen in ("passcode", "dialpad", "calling") and screen == tc["screen"]:
                r_ = L.hit_rect(screen, name)
                for (u, v) in (tc["down"], tc["lift"]):
                    m = min(u - r_[0], r_[2] - u, v - r_[1], r_[3] - v)
                    hit_margin = min(hit_margin, m)
        if variant != "open_gripper" and sorted(tc["pads"]) != ["right_tip_left", "right_tip_right"]:
            problems.append(f"touch {tc['id']} at {tc['t0']:.2f} s made by {tc['pads']}")

    got = events(ui.rows)
    want_events = EXPECTED_EVENTS.get(variant)
    want_score = D.EXPECTED[variant]
    rp = plans.get("right")
    checks = {
        "score": abs(final - want_score) < 1e-9,
        "events": want_events is None or got == want_events or (bool(want_events) and isinstance(want_events[0], list)
                                                                and got in want_events),
        "replay": rep["rows_match"] and rep["replayed"] == rep["logged"] == ui.score,
        "ik": rp is None or (rp.worst_ik_mm < IK_MM and rp.worst_ik_deg < IK_DEG),
        "j6": worst["j6_margin_deg"] > 0,
        "one_touch_per_tap": not problems,
        "hits": hit_margin >= 0 or hit_margin == math.inf,
        "rise": rise >= CLEAR_MIN_MM or variant == "no_lift_11" or rise == math.inf,
        "pressed": min_depth >= PRESSED_MM or min_depth == math.inf,
        "moves_clear": worst["over_gap_mm"] >= CLEAR_MIN_MM,
        "only_pads_near_glass": worst["clear_mm"] >= CLEAR_MIN_MM and worst["left_mm"] >= CLEAR_MIN_MM,
        "ends_on_success": (ended == "success") == (want_score == 1.0),
    }
    held_rows = [r for r in ui.rows if r["kind"] == "held"]
    if variant == "touch_during_call":
        call_t = max(r["t"] for r in ui.rows if r["kind"] == "call")
        checks["hold_restarted"] = bool(held_rows) and held_rows[0]["since"] > call_t + 1.0 \
            and abs(held_rows[0]["t"] - held_rows[0]["since"] - P.HOLD_S) < 1e-6
    return {
        "variant": variant, "expected": want_score, "score": final, "ended": ended, "steps": steps,
        "plan_s": round(end, 2), "checks": checks, "events": got, "want_events": want_events,
        "levels": {f"{lv:.1f}": round(t, 3) for lv, t in sorted(ui.scorer.reached.items())},
        "touches": touches, "problems": problems,
        "ik_mm": round(rp.worst_ik_mm, 4) if rp else 0.0, "ik_deg": round(rp.worst_ik_deg, 4) if rp else 0.0,
        "j6_margin_deg": round(worst["j6_margin_deg"], 1), "tilt_deg": round(worst["tilt_deg"], 2),
        "joint_margin": (round(worst["joint_margin"][0], 1),) + worst["joint_margin"][1:],
        "press_xy_mm": round(worst["press_xy_mm"], 2), "aim_err_mm": round(aim_err, 2),
        "hit_margin_mm": round(hit_margin, 1), "min_rise_mm": round(rise, 2), "min_depth_mm": round(min_depth, 2),
        "over_gap_mm": round(worst["over_gap_mm"], 2), "over_at": worst["over_at"],
        "clear_mm": round(worst["clear_mm"], 1), "clear_at": worst["clear_at"], "left_mm": round(worst["left_mm"], 1),
        "links_mm": {k_: round(v_, 1) for k_, v_ in links.items()},
        "held": [(r["t"], r["since"]) for r in held_rows], "replay": rep,
    }


def report(r: dict, verbose: bool) -> bool:
    ok = all(r["checks"].values())
    bad = [k for k, v in r["checks"].items() if not v]
    print(f"{'ok ' if ok else 'BAD'} {r['variant']:24s} score {r['score']:.1f} (expected {r['expected']:.1f}), "
          f"{r['ended']} at step {r['steps']} ({r['steps'] * DT:.2f} s; plan {r['plan_s']} s)"
          + (f"  FAILED: {', '.join(bad)}" if bad else ""))
    print(f"      levels {r['levels']}  held {r['held']}")
    print(f"      {len(r['touches'])} touches; IK {r['ik_mm']} mm / {r['ik_deg']} deg; j6 margin {r['j6_margin_deg']}"
          f" deg; tilt within {NEAR_MM:.0f} mm of the glass <= {r['tilt_deg']} deg; TCP off target while pressing <="
          f" {r['press_xy_mm']} mm; touch-down off aim <= {r['aim_err_mm']} mm; hit-area margin >= "
          f"{r['hit_margin_mm']} mm")
    print(f"      shallowest press {r['min_depth_mm']} mm (how far the free arm would have gone into the glass);"
          f" lowest rise between touches {r['min_rise_mm']} mm; lowest pad on moves {r['over_gap_mm']} mm"
          f" ({r['over_at']}); nothing else nearer the glass than {r['clear_mm']} mm ({r['clear_at']});"
          f" left arm >= {r['left_mm']} mm")
    if verbose:
        print(f"      lowest point above the glass of each non-pad part of the right hand (mm): {r['links_mm']}; "
              f"smallest joint-limit margin of a planned arm {r['joint_margin'][0]} deg ({r['joint_margin'][1]}, "
              f"{r['joint_margin'][2]})")
    if not r["checks"]["events"]:
        print(f"      events   {r['events']}\n      expected {r['want_events']}")
    for p in r["problems"]:
        print(f"      {p}")
    if verbose or not ok:
        for tc in r["touches"]:
            ms = (tc["t1"] - tc["t0"]) * 1000
            print(f"        touch {tc['id']:2d} {tc['t0']:6.3f}-{tc['t1']:6.3f} s ({ms:4.0f} ms)"
                  f" {tc['screen']:8s} down ({tc['down'][0]:+6.1f},{tc['down'][1]:+6.1f}) lift ({tc['lift'][0]:+6.1f},"
                  f"{tc['lift'][1]:+6.1f}) pads {len(tc['pads'])} depth {tc['depth']:.2f} mm"
                  f" rise after {tc.get('rise', float('nan')):5.1f} mm  [{tc['label']}]")
    return ok


def press_stiffness(H: Harness, q: np.ndarray) -> float:
    """Vertical stiffness (N/mm) of the pad tips held by the joint PD alone (gravity is compensated): the inverse of
    the z-z entry of J K^-1 J^T, J the tips' translational Jacobian, K the joint stiffnesses."""
    C, ik = H.C, H.ik
    k = np.array([C.SHOULDER["stiffness"]] * 3 + [C.WRIST["stiffness"]] * 3)
    tip = np.array(C.TCP_OFFSET) + np.array([0.0, 0.0, C.PAD_TIP_PAST_TCP])

    def point(qq):
        g = H.forward_kinematics(H.YAM_URDF, {f"joint{i + 1}": float(v) for i, v in enumerate(qq)})["gripper"]
        return g[:3, :3] @ tip + g[:3, 3]

    p0, jac = point(q), np.zeros((3, 6))
    for i in range(6):
        dq = q.copy()
        dq[i] += 1e-6
        jac[:, i] = (point(dq) - p0) / 1e-6
    return 1.0 / float((jac @ np.diag(1.0 / k) @ jac.T)[2, 2]) / 1000.0


def reach_table(H: Harness) -> bool:
    """IK straight down (the demo's yaw) at every key and button, pressed in and hovering."""
    import demo as D

    ik, C = H.ik, H.C
    base = np.array(H.station.BASES["right"])
    rot = ik.matrix_to_quat_xyzw(H.grip_rotation(D.YAW))
    heights = (-D.PRESS_MM, D.HOVER_MM, 30.0)
    print(f"\nreach, right arm, straight down (yaw {math.degrees(D.YAW):.0f} deg): IK error mm / deg and the smallest "
          f"joint-limit margin (deg) at pad heights {heights} mm; the PD's vertical stiffness at the pads and the "
          f"force it gives {D.PRESS_MM} mm in")
    ok = True
    seen = set()
    for screen, names in L.AIM.items():
        for name, (u, v) in names.items():
            if (u, v) in seen:
                continue
            seen.add((u, v))
            cells, kz = [], 0.0
            for h in heights:
                q, pe, re_ = ik.solve(D.tcp(u, v, h), rot, C.READY, base_pos=base, iters=400)
                if h < 0:
                    kz = press_stiffness(H, q)
                margin = math.degrees(float(np.min(np.minimum(q - ik.LOWER, ik.UPPER - q))))
                good = pe * 1000 < IK_MM and math.degrees(re_) < IK_DEG
                ok &= good
                cells.append(f"{pe * 1000:5.2f}/{math.degrees(re_):4.2f}/{margin:5.1f}{'' if good else ' BAD'}")
            x, y, _ = L.screen_to_world(u, v)
            print(f"  {screen:8s} {name:9s} ({u:+5.0f},{v:+5.0f}) mm  x {x:.3f} y {y:+.3f}   " + "   ".join(cells)
                  + f"   {kz:4.2f} N/mm {kz * D.PRESS_MM:4.1f} N")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("variants", nargs="*")
    ap.add_argument("--src", default=os.environ.get("YAM_ARENA_SRC"), help="folder holding the yam_arena package")
    ap.add_argument("--hulls", default=os.environ.get("YAM_HULLS"),
                    help="folder of the URDF collision STLs ({link}__{file}.stl or {link}/{file})")
    ap.add_argument("--lag", type=float, default=1.0, help="scale of the joint time constants (0: exact tracking)")
    ap.add_argument("--deadband", type=float, default=0.0,
                    help="joint friction as a Coulomb dead band: each joint stops this many times friction / "
                         "stiffness short of its target (1: the harness's friction read as a torque; 0: none)")
    ap.add_argument("--log", help="write each variant's trial log here (<variant>.jsonl, for score.py)")
    ap.add_argument("--no-reach", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true", help="list every touch")
    a = ap.parse_args(argv)
    warnings.filterwarnings("ignore", category=RuntimeWarning)   # numpy 2 + Accelerate: spurious matmul warnings
    yam_arena, C = load_harness(a.src)
    H = Harness(yam_arena, C)
    import demo as D

    pads, pad_src = pad_table()
    other, other_src = non_pad_points(yam_arena, a.hulls)
    print(f"yam_arena from {Path(yam_arena.__file__).parent.parent.name}/; pad points from {pad_src}; "
          f"clearance from {other_src}; lag x{a.lag} (time constants 0.0625 / 0.15 / 0.02 s at x1); "
          f"friction dead band x{a.deadband}")
    names = a.variants or list(D.VARIANTS)
    ok = True
    with np.errstate(all="ignore"):
        for v in names:
            t0 = time.time()
            r = play(D.variant(v), H, pads, other, a.lag, a.log, a.deadband)
            ok &= report(r, a.verbose)
            if a.verbose:
                print(f"      ({time.time() - t0:.1f} s)")
        if not a.no_reach:
            ok &= reach_table(H)
    print("\nall checks passed" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

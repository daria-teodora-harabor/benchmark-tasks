"""A suture practice pad and a curved needle.

On the station table, in front of the arms, a skin-coloured silicone suture practice pad stands upright on a grey
stand, turned a little toward the left arm, its right end nearer the arms. Its front face, the skin, is 16 cm wide and
8 cm tall, with a vertical cut marked down its middle and an open slot on each side of the cut, at the same height,
6 cm above the table. A blunt curved needle (a grey half circle, 6 cm radius and 9 mm thick, with a dark tip and a blue
band at its back end) lies flat on a white foam block on the table in front of the arms, just right of the middle, its
arc bowing toward the arms.
"""

# Notes for people reading the code (the docstring above is what an agent policy is shown, so it only describes the
# scene):
# - The task: drive the needle in through the slot right of the cut and out through the slot left of it, turning it
#   about its own centre as the instruction says, then take it with the other hand where it has come out and let go
#   with the first. The scorer checks respect for tissue: turned about its own centre, a curved needle's body follows
#   the track its tip made; pushed straight, or levered, it drags sideways through its entry and exit holes and tears
#   them.
# - The slots stand in for the holes a needle punches in tissue. Each is the needle's path plus 8 mm on either side, so
#   a needle that is pushed or levered can move sideways in it, and the scorer measures that: a crossing point that
#   moves more than 4 mm sideways along the skin (horizontally, toward or away from the cut) from where it was made
#   counts as a torn hole. Up-and-down movement inside the 18 mm slots isn't scored.
# - Everything is built from fixed yam_arena Boxes except the needle, a partkit Part (one rigid body of 12 boxes along
#   the arc). Its frame origin is the arc's centre, so the scorer and the demo compute the centreline from its pose.

from __future__ import annotations

import json
import math
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from partkit import PartState, env, on_reset, on_step, part  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = ("Drive the curved needle through the practice pad as a surgeon does: turn it about the centre of its "
               "curve so that it follows its own path, going in through the slot right of the cut and out through the "
               "slot left of the cut. Then take hold of the needle where it has come out with your other hand, and let "
               "go of it with the first.")

SKIN = (0.91, 0.72, 0.60)
CUT_RED = (0.55, 0.20, 0.18)
STEEL = (0.71, 0.70, 0.66)
DARK = (0.17, 0.17, 0.16)
BLUE = (0.22, 0.54, 0.87)
FOAM = (0.95, 0.95, 0.93)
STAND = (0.45, 0.46, 0.48)

# -- the needle (metres) --------------------------------------------------------------------------------------------
R = 0.060                 # radius of the needle's centreline
A = 0.009                 # its square section
N_BOX = 12                # 12 straight boxes of 15 degrees each make the half circle
BOX_L = 2 * (R + A / 2) * math.tan(math.radians(7.5))  # long enough that the outer edge has no gaps
NEEDLE_KG = 0.019         # solid PLA (1.24 g/cm3): the half ring is pi * R * A^2 = 15.3 cm3
# Needle frame: origin at the arc's centre, the arc in its z = 0 plane. Local angle 0 is the tip; the body runs
# clockwise to -180 degrees, the back end. Turning the needle counterclockwise (seen from above) leads with the tip.

# -- the layout (world: x forward from the arm bases, y to the robot's left, z up) ----------------------------------
C_XY = np.array([0.26, 0.0])  # where the needle's centre sits while it is driven through the pad
ZN = TOP + 0.060              # height of the needle's plane while it is driven, and of the slots' centres
BETA = math.radians(-30.0)    # the pad turned 30 degrees clockwise (seen from above): its skin faces back toward the
                              # left arm, and its right end is nearer the arms
D = 0.042                     # the skin's front face is this far from the needle's centre
SKIN_T = 0.004                # thickness of the skin plate
PAD_W, PAD_D, WALL = 0.160, 0.040, 0.005
STAND_TOP, PAD_TOP = TOP + 0.010, TOP + 0.090

N_HAT = np.array([math.cos(BETA), math.sin(BETA), 0.0])   # from the needle's centre into the pad
E_HAT = np.array([-math.sin(BETA), math.cos(BETA), 0.0])  # along the skin, toward the robot's left
PAD_ORIGIN = np.array([C_XY[0], C_XY[1], ZN]) + D * N_HAT  # on the skin's front face, at the cut, at needle height

# Where a perfectly driven needle crosses the skin, and the slots around it. On the front face its centreline is
# Y_FRONT either side of the cut, on the back face Y_BACK; crossing at a slant, its 9 mm section is wider along the skin.
Y_FRONT = math.sqrt(R ** 2 - D ** 2)
Y_BACK = math.sqrt(R ** 2 - (D + SKIN_T) ** 2)
PATH_LO = Y_BACK - (A / 2) / (Y_BACK / R)
PATH_HI = Y_FRONT + (A / 2) / (Y_FRONT / R)
SLOT_MARGIN = 0.008
SLOT_LO, SLOT_HI = PATH_LO - SLOT_MARGIN, PATH_HI + SLOT_MARGIN  # each slot spans these distances from the cut
SLOT_H = 0.018
HALF_ANGLE = math.acos(D / R)            # the entry and exit, seen from the needle's centre, either side of N_HAT
ENTRY_ANGLE = BETA - HALF_ANGLE          # world angle (about C) of the ideal entry point, right of the cut
EXIT_ANGLE = BETA + HALF_ANGLE


def _pad_box(name, x0, x1, y0, y1, z0, z1, color):
    """A fixed box given in the pad frame: x from the skin's front face into the pad, y along it toward the robot's
    left, z absolute."""
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    p = PAD_ORIGIN[:2] + cx * N_HAT[:2] + cy * E_HAT[:2]
    return Box(name, pos=(float(p[0]), float(p[1]), (z0 + z1) / 2), size=(x1 - x0, y1 - y0, z1 - z0),
               yaw=BETA, color=color)


_ZS0, _ZS1 = ZN - SLOT_H / 2, ZN + SLOT_H / 2
_W2 = PAD_W / 2
PAD = [
    # the skin: bands below and above the slots, and pillars beside and between them
    _pad_box("skin_low", 0, SKIN_T, -_W2, _W2, STAND_TOP, _ZS0, SKIN),
    _pad_box("skin_high", 0, SKIN_T, -_W2, _W2, _ZS1, PAD_TOP, SKIN),
    _pad_box("skin_mid", 0, SKIN_T, -SLOT_LO, SLOT_LO, _ZS0, _ZS1, SKIN),
    _pad_box("skin_left", 0, SKIN_T, SLOT_HI, _W2, _ZS0, _ZS1, SKIN),
    _pad_box("skin_right", 0, SKIN_T, -_W2, -SLOT_HI, _ZS0, _ZS1, SKIN),
    # the cut, drawn on the skin
    _pad_box("cut", -0.0005, 0.0, -0.0025, 0.0025, ZN - 0.022, ZN + 0.022, CUT_RED),  # 5 mm wide, to show on camera
    # the pad's hollow body behind the skin
    _pad_box("pad_left", 0, PAD_D, _W2, _W2 + WALL, STAND_TOP, PAD_TOP, SKIN),
    _pad_box("pad_right", 0, PAD_D, -_W2 - WALL, -_W2, STAND_TOP, PAD_TOP, SKIN),
    _pad_box("pad_back", PAD_D - WALL, PAD_D, -_W2, _W2, STAND_TOP, PAD_TOP, SKIN),
    _pad_box("pad_top", SKIN_T, PAD_D - WALL, -_W2, _W2, PAD_TOP - WALL, PAD_TOP, SKIN),
    # its stand
    _pad_box("pad_stand", -0.012, PAD_D + 0.006, -_W2 - WALL - 0.008, _W2 + WALL + 0.008, TOP, STAND_TOP, STAND),
]


def _needle_boxes():
    boxes = []
    for k in range(N_BOX):
        a = -math.radians(7.5 + 15.0 * k)  # box centre, from the tip end (k = 0) to the back end
        yaw = a + math.pi / 2  # its long side along the tangent
        color = DARK if k == 0 else BLUE if k == N_BOX - 1 else STEEL
        boxes.append(((R * math.cos(a), R * math.sin(a), 0.0), (BOX_L, A, A), color,
                      (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))))
    return boxes


# -- where the needle starts and how it is driven -----------------------------------------------------------------
APPROACH = math.radians(15.0)  # the demo's drive starts with the tip this far (about the centre) before the entry
START_SHIFT = np.array([-0.100, -0.060])  # the needle lies on its foam block this far from its drive position
START_YAW = ENTRY_ANGLE - APPROACH          # the needle's yaw on the block (its tip's world angle about its centre)
START_XY = C_XY + START_SHIFT


# The foam block under the needle. The needle's centre of mass is inside the ring, 2R/pi = 3.8 cm from the arc's
# centre, so a block under the middle of the arc alone can't carry it: it would tip ends-down. This block runs 10 cm
# along the needle's chord, so the ring rests on it out to 5 cm either side of its middle and the centre of mass lies
# well inside that support. The back end overhangs it, so the fingers can close around it.
BLOCK_DEPTH, BLOCK_WIDTH = 0.050, 0.100  # front to back (along the needle's middle radius), side to side (along its chord)
BLOCK_OFFSET = 0.046                      # the block's centre, from the arc's centre toward the middle of the arc


def _block():
    mid = START_YAW - math.pi / 2
    c = START_XY + BLOCK_OFFSET * np.array([math.cos(mid), math.sin(mid)])
    top = ZN - A / 2
    return Box("needle_block", pos=(float(c[0]), float(c[1]), (TOP + top) / 2),
               size=(BLOCK_DEPTH, BLOCK_WIDTH, top - TOP), yaw=mid, color=FOAM)


NEEDLE = part("needle", pos=(float(START_XY[0]), float(START_XY[1]), ZN + 0.0005), yaw=START_YAW, mass=NEEDLE_KG,
              boxes=_needle_boxes(), color=STEEL, friction=1.0)
NEEDLE.meta["torsional_patch"] = 0.004
OBJECTS = [*PAD, _block(), NEEDLE]


def rz(a) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def local_point(angle) -> np.ndarray:
    """A point on the needle's centreline, in the needle frame, at local angle ``angle`` (rad; 0 = tip)."""
    return np.array([R * math.cos(angle), R * math.sin(angle), 0.0])


def drive_pose(phi):
    """The needle's pose (origin, rotation) when driven by ``phi`` rad about its centre: phi = 0 puts the tip on the
    ideal entry point, phi = 2 * HALF_ANGLE on the ideal exit point."""
    return np.array([C_XY[0], C_XY[1], ZN]), rz(ENTRY_ANGLE + phi)


# -- reading the needle against the skin --------------------------------------------------------------------------
TEAR = 0.004              # a hole is torn once its crossing point has moved this far along the skin
TIP_IN = SKIN_T + 0.002   # the tip counts as in once it is this far behind the skin's front face
TIP_OUT = 0.020           # ... and as out once this much of the needle (arc length) is past the exit
LET_GO = 0.020            # a hand has let go when its fingers' axis is this far from the needle's centreline
ANGLES = np.radians(np.linspace(0.0, -180.0, 361))
SAMPLES = np.stack([R * np.cos(ANGLES), R * np.sin(ANGLES), np.zeros_like(ANGLES)], axis=1)


def centreline(needle: PartState) -> np.ndarray:
    return needle.pos + SAMPLES @ needle.rotation().T


def crossings(needle: PartState):
    """Where the needle's centreline crosses the plane of the skin's front face, inside the pad's outline: a list of
    (needle-local angle in rad, distance along the skin from the cut in m (+ to the robot's left), height above the
    slots' centre in m)."""
    pts = centreline(needle)
    s = (pts - PAD_ORIGIN) @ N_HAT
    out = []
    for k in np.nonzero(np.signbit(s[:-1]) != np.signbit(s[1:]))[0]:
        t = s[k] / (s[k] - s[k + 1])
        p = pts[k] + t * (pts[k + 1] - pts[k])
        u, h = float((p - PAD_ORIGIN) @ E_HAT), float(p[2] - ZN)
        if abs(u) <= PAD_W / 2 and STAND_TOP - ZN <= h <= PAD_TOP - ZN:
            out.append((float(ANGLES[k] + t * (ANGLES[k + 1] - ANGLES[k])), u, h))
    return out


def in_slot(u, h) -> bool:
    return SLOT_LO <= abs(u) <= SLOT_HI and abs(h) <= SLOT_H / 2


def tip_depth(needle: PartState) -> float:
    """How far the tip is behind the skin's front face (m; negative in front of it)."""
    return float((needle.to_world(local_point(0.0)) - PAD_ORIGIN) @ N_HAT)


def to_pad(p) -> np.ndarray:
    """A world point in the pad frame: depth behind the skin's front face, distance along it from the cut, height."""
    d = np.asarray(p, float) - PAD_ORIGIN
    return np.array([d @ N_HAT, d @ E_HAT, p[2]])


# -- the grippers -------------------------------------------------------------------------------------------------
PAD_TIP_W, PAD_TIP_T = 0.0154, 0.004  # a finger pad's tip: 15.4 mm across, 4 mm thick (the gripper's collision meshes)
ARMS = ("left_robot", "right_robot")


def _origin(base):
    o = base.scene.env_origins[0]
    o = getattr(o, "torch", o)
    return o.detach().cpu().numpy().astype(np.float64) if hasattr(o, "detach") else np.asarray(o, dtype=np.float64)


def grippers():
    """For each arm: (pad tip point on the gripper's axis, approach axis (toward the fingertips), closing axis, finger
    opening between the pads (m)), in the objects' frame. Empty without the simulator."""
    environment = env()
    if environment is None:
        return {}
    try:
        from yam_arena.constants import FINGER_JOINTS, PAD_TIP_PAST_TCP, TCP_OFFSET
        from yam_arena.runtime import link_poses
    except ImportError:
        return {}
    base = environment.unwrapped
    o = _origin(base)
    out = {}
    for key in ARMS:
        robot = base.scene[key]
        pose = link_poses(robot)["gripper"]
        rot, pos = pose[:3, :3], pose[:3, 3]
        tip = rot @ (np.asarray(TCP_OFFSET, float) + np.array([0.0, 0.0, PAD_TIP_PAST_TCP])) + pos - o
        ids = robot.find_joints(FINGER_JOINTS, preserve_order=True)[0]
        q = getattr(robot.data.joint_pos, "torch", robot.data.joint_pos)[0, ids]
        q = np.abs(q.detach().cpu().numpy() if hasattr(q, "detach") else np.asarray(q, float))
        out[key] = (tip, rot[:, 2], rot[:, 0], float(q[0] + q[1]), (float(q[0]), float(q[1])))
    return out


def pad_points(g):
    """Points covering one gripper's two finger-pad tips (15 per pad): tip_left (joint7) on the closing axis's - side,
    tip_right on its + side, each 4 mm thick outward from its inner face."""
    tip, approach, close, _, (q7, q8) = g
    across = np.cross(approach, close)
    across /= np.linalg.norm(across) + 1e-12
    pts = []
    for side, travel in ((-1.0, q7), (1.0, q8)):
        for w in np.linspace(-PAD_TIP_W / 2, PAD_TIP_W / 2, 5):
            for d in np.linspace(0.0, PAD_TIP_T, 3):
                pts.append(tip + side * (travel + d) * close + w * across)
    return np.array(pts)


HOLD_MIN, HOLD_MAX = A - 0.002, A + 0.0025  # the finger opening of a hand closed on the needle's 9 mm section (m)


def holds(g, needle: PartState):
    """Whether this gripper is closed on the needle, and the needle-local angle (rad) it holds it at. The needle is
    held when the fingers have closed onto its section (7 to 11.5 mm apart: a hand closed on nothing reads 0) and
    their axis passes within 6 mm of its centreline, between 4.5 mm and 2.5 cm above the pad tips (so the needle sits
    between the pads, not under them)."""
    tip, approach, _, width, _ = g
    best = (1.0, 0.0)
    for lam in np.linspace(A / 2, 0.025, 6):
        p = needle.to_local(tip - lam * approach)
        rho = math.hypot(p[0], p[1])
        ang = math.atan2(p[1], p[0])
        if ang > math.pi / 2:  # just past the back end (-180 degrees) atan2 wraps to +180
            ang -= 2 * math.pi
        if -math.pi - 0.05 <= ang <= 0.05:
            dist = math.hypot(rho - R, p[2])
            if dist < best[0]:
                best = (dist, ang)
    return bool(HOLD_MIN <= width <= HOLD_MAX and best[0] <= 0.006), best[1], best[0]


def axis_distance(g, needle: PartState) -> float:
    """Shortest distance from a gripper's axis (pad tips up 2.5 cm) to the needle's centreline."""
    tip, approach, *_ = g
    pts = centreline(needle)
    return float(min(np.min(np.linalg.norm(pts - (tip - lam * approach), axis=1)) for lam in np.linspace(0, 0.025, 6)))


def touches_pad(points) -> bool:
    """A finger pad inside the practice pad's outline, or within 1 mm of its skin: pressing the tissue."""
    for p in points:
        d, u, z = to_pad(p)
        if -0.001 <= d <= PAD_D + 0.001 and abs(u) <= PAD_W / 2 + WALL and STAND_TOP <= z <= PAD_TOP + 0.001:
            return True
    return False


# -- trial memory and scoring -------------------------------------------------------------------------------------
DEFAULT_DT = 1 / 20
_STATE: dict = {}
_RECORD: dict = {}
RECORD_PATH = os.path.expanduser(os.environ["NEEDLE_RECORD"]) if os.environ.get("NEEDLE_RECORD") else None
TRACE_PATH = os.path.expanduser(os.environ["NEEDLE_TRACE"]) if os.environ.get("NEEDLE_TRACE") else None
_DONE: list = []
_EARLIER = open(RECORD_PATH).read().splitlines() if RECORD_PATH and os.path.exists(RECORD_PATH) else []


@on_reset
def _reset_state(environment=None) -> None:
    if _RECORD.get("steps"):
        _DONE.append(dict(_RECORD))
        _RECORD["steps"] = 0
        _write_record()
    _STATE.clear()
    _STATE.update(step=0, scored=None, last=(0.0, False), first_u={}, shift={"entry": 0.0, "exit": 0.0},
                  torn=None, pressed=None, dropped=None, entered=False, through=False, taken=False, driver=None,
                  last_holder=None, recent=[])
    _RECORD.clear()
    _RECORD.update(started=time.strftime("%Y-%m-%dT%H:%M:%S"), demo=os.environ.get("DEMO_VARIANT", ""), steps=0,
                   score=None, entered_s=None, through_s=None, taken_s=None, driver=None, broke_rule=None,
                   broke_rule_s=None, entry_shift_mm=0.0, exit_shift_mm=0.0)


@on_step
def _count_step(environment=None) -> None:
    _STATE["step"] += 1


def _write_record() -> None:
    if not RECORD_PATH:
        return
    current = [dict(_RECORD)] if _RECORD.get("steps") else []
    tmp = RECORD_PATH + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(_EARLIER + [json.dumps(r) for r in _DONE + current]) + "\n")
    os.replace(tmp, RECORD_PATH)


_reset_state()


def _dt() -> float:
    environment = env()
    try:
        return float(environment.unwrapped.step_dt) if environment is not None else DEFAULT_DT
    except Exception:  # noqa: BLE001
        return DEFAULT_DT


def _broke(key, rule) -> None:
    if _STATE[key] is None:
        _STATE[key] = rule
        if _RECORD["broke_rule"] is None:
            _RECORD["broke_rule"], _RECORD["broke_rule_s"] = rule, round(_STATE["step"] * _dt(), 2)


def _update(objects):
    """Update the trial's memory once per control step and return (score, success)."""
    if _STATE["scored"] == _STATE["step"]:
        return _STATE["last"]
    _STATE["scored"] = _STATE["step"]
    now = _STATE["step"] * _dt()
    needle = objects["needle"]
    found = crossings(needle)
    by_slot = {}
    for ang, u, h in found:
        slot = "entry" if u < 0 else "exit"
        if not in_slot(u, h):
            _broke("torn", "needle went through the skin outside a slot")
            continue
        by_slot.setdefault(slot, (ang, u, h))
        first = _STATE["first_u"].setdefault(slot, u)
        _STATE["shift"][slot] = max(_STATE["shift"][slot], abs(u - first))
    for slot in ("entry", "exit"):
        _RECORD[f"{slot}_shift_mm"] = round(1000 * _STATE["shift"][slot], 2)
        if _STATE["shift"][slot] > TEAR:
            _broke("torn", f"tore the {slot} hole (crossing moved {1000 * _STATE['shift'][slot]:.1f} mm)")

    gs = grippers()
    held = {k: holds(g, needle) for k, g in gs.items()}
    held_now = [k for k, (h, _, _) in held.items() if h]
    # Dropped: no hand holds it and all of it is 3 cm or more below the slots' height (fallen to the table). A needle
    # still in a hand, however tilted, is not dropped.
    if not held_now and float(centreline(needle)[:, 2].max()) < ZN - 0.03:
        _broke("dropped", "dropped the needle")
    for k, g in gs.items():
        if touches_pad(pad_points(g)):
            _broke("pressed", f"{k.split('_')[0]} gripper pressed the practice pad")

    depth = tip_depth(needle)
    tip_u = float(to_pad(needle.to_world(local_point(0.0)))[1])
    # In: through the entry slot and driven at least halfway, the tip behind the cut. (A needle pushed straight in
    # barely moves its crossing for the first 15 mm, but it can't reach the cut: it would hit the pad's back wall.)
    # The hand that drove it in is fixed on that step: the hand holding it (if both do, the one further back on the
    # needle); if none is holding it just then (a regrasp), the one that held it last; if none ever has, the nearest.
    if not _STATE["entered"] and "entry" in by_slot and depth > TIP_IN and tip_u >= 0.0:
        _STATE["entered"] = True
        _RECORD["entered_s"] = round(now, 2)
        if held_now:
            _STATE["driver"] = min(held_now, key=lambda k: held[k][1])
        elif _STATE["last_holder"] is not None:
            _STATE["driver"] = _STATE["last_holder"]
        elif gs:
            _STATE["driver"] = min(gs, key=lambda k: axis_distance(gs[k], needle))
        _RECORD["driver"] = _STATE["driver"]
    if held_now:
        _STATE["last_holder"] = min(held_now, key=lambda k: held[k][1])
    exit_angle = by_slot["exit"][0] if "exit" in by_slot else None
    if (_STATE["entered"] and not _STATE["through"] and exit_angle is not None and "entry" in by_slot
            and depth < -0.003 and -exit_angle * R >= TIP_OUT):
        _STATE["through"] = True
        _RECORD["through_s"] = round(now, 2)
    tip_held_by = [k for k, (h, ang, _) in held.items()
                   if h and k != _STATE["driver"] and exit_angle is not None and ang > exit_angle]
    if _STATE["through"] and not _STATE["taken"] and tip_held_by:
        _STATE["taken"] = True
        _RECORD["taken_s"] = round(now, 2)

    recent = _STATE["recent"]
    recent.append(needle.pos.copy())
    del recent[:-20]
    still = len(recent) == 20 and all(np.linalg.norm(p - needle.pos) < 0.001 for p in recent)
    driver = _STATE["driver"]
    if driver is None or driver not in gs:
        driver_off = True
    else:
        driver_off = not held[driver][0] and axis_distance(gs[driver], needle) >= LET_GO
    done = bool(_STATE["taken"] and tip_held_by and driver_off and "entry" in by_slot and "exit" in by_slot)

    if _STATE["torn"] or _STATE["pressed"] or _STATE["dropped"]:
        result = (0.0, False)
    elif done and still:
        result = (1.0, True)
    elif done:
        result = (0.75, False)
    elif _STATE["taken"]:
        result = (0.75, False)
    elif _STATE["through"]:
        result = (0.5, False)
    elif _STATE["entered"]:
        result = (0.25, False)
    else:
        result = (0.0, False)
    _STATE["last"] = result
    _RECORD["steps"] = _STATE["step"]
    _RECORD["score"] = result[0]
    _write_record()
    if TRACE_PATH:
        row = {"step": _STATE["step"], "t": round(now, 3), "score": result[0],
               "crossings_mm": [[round(math.degrees(a), 1), round(1000 * u, 2), round(1000 * h, 2)] for a, u, h in found],
               "shift_mm": {k: round(1000 * v, 2) for k, v in _STATE["shift"].items()}, "tip_depth_mm": round(1000 * depth, 1),
               "held": {k: [h, round(math.degrees(a), 1), round(1000 * dd, 1)] for k, (h, a, dd) in held.items()},
               "needle_pos": [round(float(v), 4) for v in needle.pos]}
        with open(TRACE_PATH, "a") as f:
            f.write(json.dumps(row) + "\n")
    return result


def success(objects) -> bool:
    """The other hand holds the needle's tip after it has come through both slots, the first hand has let go, no hole
    is torn and nothing pressed the pad, and the needle is still; held for 1 s."""
    return _update(objects)[1]


def score(objects) -> float:
    """0 if a hole is torn (its crossing point moved more than 4 mm sideways along the skin), a gripper pressed the
    pad, or the needle was dropped. Otherwise: 0.25 tip in through the entry slot and driven past the cut; 0.5 tip out
    through the exit slot with 2 cm of needle out; 0.75 the other hand holds the needle's tip; 1.0 and the first hand
    has let go."""
    return _update(objects)[0]

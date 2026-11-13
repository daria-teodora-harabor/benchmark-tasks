"""An electric car is charging at a charger post. A real-size section of the car's rear quarter panel stands on the
table, facing the arms, with its charge-port door swung open and a J1772 charging connector plugged and latched into
the inlet in the port. The connector's thick black cable runs across the table to the foot of the grey charger post on
the right, which has an empty holster for the connector on its side facing the arms. On top of the connector's body is
its latch lever with a red release button at the back."""

# Task: unplug the car (the J1772 latch must be held open with its button while the connector is pulled out), hang the
# connector in the holster on the charger post, and close the car's charge-port door.
#
# The docstring above only describes the scene, because the simulator shows it to agent policies; scoring is
# explained in these comments and in the README.
#
# Frame: the objects' frame of the station scene (x away from the arms, y to the arms' left, z up; arm bases at
# x 0, y +-0.305). The connector's own frame: origin at the centre of its nose face, x out of the nose (into the
# car), y to its left, z toward its top (the latch side).

from __future__ import annotations

import json
import math
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from partkit import CableSpec, PartState, Slider, env, flap, on_reset, on_step, part, sprung_slide  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = "Unplug the car, hang the charger up on its post, and close the car's charge-port door."

# -- colours -------------------------------------------------------------------------------------------------------
BLACK = (0.10, 0.10, 0.11)
GREY_PLASTIC = (0.28, 0.28, 0.30)
RED = (0.80, 0.12, 0.10)
CAR_BLUE = (0.16, 0.27, 0.45)
INLET_BLACK = (0.14, 0.14, 0.15)
POST_GREY = (0.62, 0.64, 0.66)
STAND_GREY = (0.45, 0.46, 0.48)

# -- the car: panel, charge port, inlet ---------------------------------------------------------------------------
XP = 0.44                  # the panel's outer face (x); the panel is 2 cm thick
YP, ZP = -0.05, TOP + 0.15  # the inlet's axis: 15 cm above the table (0.90 m above the floor)
PANEL_Y0, PANEL_Y1, PANEL_Z1 = -0.35, 0.25, TOP + 0.45   # a 60 x 45 cm section of the rear quarter panel
PORT_HW, PORT_HH = 0.060, 0.050                          # the charge port's opening: 12 x 10 cm
X_MOUTH = XP + 0.010       # the inlet's face, recessed 1 cm in the port
NOSE_HALF = 0.0219         # the J1772 nose is 43.8 mm across (modelled square)
BORE = NOSE_HALF + 0.002   # the inlet's socket: 2 mm clearance each side
X_SEAT = X_MOUTH + 0.037   # where the nose face stops (the socket's back wall); the body then stands 3 mm off the face
HOUSING = 0.050            # half size of the inlet housing
# The latch's catch: a notch in the socket's top wall that the latch hook drops into when the connector is home.
CATCH_X0, CATCH_X1 = X_SEAT - 0.0135, X_SEAT - 0.0025
CATCH_HW, CATCH_TOP = 0.004, ZP + 0.032


def _fixed(name, lo, hi, color):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return Box(name=name, pos=tuple(float(v) for v in (lo + hi) / 2), size=tuple(float(v) for v in hi - lo),
               color=color, mass=None)


H, B = HOUSING, BORE
LIP_H = 0.0075  # the catch lip's height: it stands 7.5 mm into the notch's lane, above the socket's top wall
CAR = [
    # panel around the port opening (2 cm sheet and backing)
    _fixed("panel_left", (XP, YP + PORT_HW, TOP), (XP + 0.02, PANEL_Y1, PANEL_Z1), CAR_BLUE),
    _fixed("panel_right", (XP, PANEL_Y0, TOP), (XP + 0.02, YP - PORT_HW, PANEL_Z1), CAR_BLUE),
    _fixed("panel_top", (XP, YP - PORT_HW, ZP + PORT_HH), (XP + 0.02, YP + PORT_HW, PANEL_Z1), CAR_BLUE),
    _fixed("panel_bottom", (XP, YP - PORT_HW, TOP), (XP + 0.02, YP + PORT_HW, ZP - PORT_HH), CAR_BLUE),
    _fixed("panel_foot_left", (XP + 0.02, PANEL_Y1 - 0.06, TOP), (XP + 0.22, PANEL_Y1 - 0.02, TOP + 0.02), STAND_GREY),
    _fixed("panel_foot_right", (XP + 0.02, PANEL_Y0 + 0.02, TOP), (XP + 0.22, PANEL_Y0 + 0.06, TOP + 0.02), STAND_GREY),
    # the J1772 inlet: a 10 x 10 cm housing with a 47.8 mm socket and the latch catch in its top wall
    # the top wall in front of the catch, with a lane for the latch tooth (the catch lip in it is on a spring; below)
    _fixed("inlet_top_front_left", (X_MOUTH, YP + CATCH_HW, ZP + B), (CATCH_X0, YP + H, ZP + H), INLET_BLACK),
    _fixed("inlet_top_front_right", (X_MOUTH, YP - H, ZP + B), (CATCH_X0, YP - CATCH_HW, ZP + H), INLET_BLACK),
    _fixed("inlet_top_front_over", (X_MOUTH, YP - CATCH_HW, ZP + B + LIP_H + 0.0005), (CATCH_X0, YP + CATCH_HW, ZP + H),
           INLET_BLACK),
    _fixed("inlet_top_back", (CATCH_X1, YP - H, ZP + B), (X_SEAT, YP + H, ZP + H), INLET_BLACK),
    _fixed("inlet_top_left", (CATCH_X0, YP + CATCH_HW, ZP + B), (CATCH_X1, YP + H, ZP + H), INLET_BLACK),
    _fixed("inlet_top_right", (CATCH_X0, YP - H, ZP + B), (CATCH_X1, YP - CATCH_HW, ZP + H), INLET_BLACK),
    _fixed("inlet_top_over_catch", (CATCH_X0, YP - CATCH_HW, CATCH_TOP), (CATCH_X1, YP + CATCH_HW, ZP + H),
           INLET_BLACK),
    _fixed("inlet_bottom", (X_MOUTH, YP - H, ZP - H), (X_SEAT, YP + H, ZP - B), INLET_BLACK),
    _fixed("inlet_left", (X_MOUTH, YP + B, ZP - B), (X_SEAT, YP + H, ZP + B), INLET_BLACK),
    _fixed("inlet_right", (X_MOUTH, YP - H, ZP - B), (X_SEAT, YP - B, ZP + B), INLET_BLACK),
    _fixed("inlet_back", (X_SEAT, YP - H, ZP - H), (X_SEAT + 0.02, YP + H, ZP + H), INLET_BLACK),
]

# The charge-port door: 12 x 10 cm, 4 mm thick, hinged on its left edge (seen from the arms), opening outward. Shut it
# lies flush on the panel over the port; it starts swung 110 degrees open, toward the arms and to their left.
DOOR_HINGE = (XP - 0.002, YP + PORT_HW, ZP)
DOOR = flap("port_door", DOOR_HINGE,
            flap_boxes=[((0.0, -PORT_HW, 0.0), (0.004, 2 * PORT_HW, 2 * PORT_HH), CAR_BLUE)],
            frame_boxes=[((0.010, -0.004, 0.0), (0.004, 0.004, 0.09), CAR_BLUE)],   # hinge block, in the port
            open_angle=-math.radians(110), lower=-math.radians(115), upper=0.0, flap_mass=0.06, color=CAR_BLUE)
DOOR_SHUT = math.radians(2.4)   # shut: its free edge within 5 mm of the panel (12 cm door)

# The catch's front lip, which the latch tooth bears on when the connector is pulled with the button up, sits on a stiff
# spring along the inlet's axis (20 N per mm, up to 4 mm): the simulator's version of the real trial's inlet mounted on
# a slide behind a luggage scale. The force on the latch is the spring rate times how far the lip has moved. A careful
# unplug never touches it: with the button pressed the tooth passes below it.
CATCH_RATE = 20000.0     # N/m
CATCH = sprung_slide("inlet_catch", (X_MOUTH, YP, ZP + B),
                     boxes=[((0.5 * (CATCH_X0 - X_MOUTH), 0.0, 0.5 * LIP_H), (CATCH_X0 - X_MOUTH, 2 * CATCH_HW - 0.001,
                                                                             LIP_H), INLET_BLACK)],
                     frame_boxes=[((XP + 0.002 - X_MOUTH, 0.040, 0.034 - B), (0.002, 0.002, 0.002), INLET_BLACK)],
                     travel=0.004, rate=CATCH_RATE, damping=40.0, mass=0.02, color=INLET_BLACK)

# -- the charger post and its holster ------------------------------------------------------------------------------
POST_X0, POST_X1, POST_Y0, POST_Y1, POST_TOP = 0.40, 0.48, -0.50, -0.42, TOP + 0.55   # 8 x 8 cm, 55 cm tall
HX, HY, HZ = POST_X0, -0.46, TOP + 0.20   # the holster's back (on the post) and its axis (0.95 m above the floor)
SLEEVE, FLARE = 0.025, 0.032              # the holster's socket: 50 mm square; its mouth 64 mm tall
HUNG_DEPTH = 0.020   # hung: the nose face 20 mm or less from the socket's back (20 mm or more into the holster)
HOOK_SLOT = 0.007   # half width of the slot in the socket's top wall for the hook tooth (its top 30 mm up; slot to 34 mm)
LEAD_DEPTH, LEAD_MOUTH, LEAD_T = 0.040, 0.045, 0.006   # its sides taper over 4 cm, from 90 mm apart to the socket's 50
HOLSTER_DEPTH, FLARE_DEPTH, HOLSTER_OUT = 0.030, 0.010, 0.045
POST = [
    _fixed("post", (POST_X0, POST_Y0, TOP), (POST_X1, POST_Y1, POST_TOP), POST_GREY),
    _fixed("post_base", (POST_X0 - 0.03, POST_Y0 - 0.03, TOP), (POST_X1 + 0.03, POST_Y1 + 0.03, TOP + 0.012),
           STAND_GREY),
]
O = HOLSTER_OUT
_x0, _x1 = HX - HOLSTER_DEPTH, HX                       # the socket
POST += [
    # its top wall has a slot for the latch's hook tooth (it stands 8 mm above the nose), as real holsters do
    _fixed("holster_sleeve_top_left", (_x0, HY + HOOK_SLOT, HZ + SLEEVE), (_x1, HY + O, HZ + O), BLACK),
    _fixed("holster_sleeve_top_right", (_x0, HY - O, HZ + SLEEVE), (_x1, HY - HOOK_SLOT, HZ + O), BLACK),
    _fixed("holster_sleeve_top_slot", (_x0, HY - HOOK_SLOT, HZ + 0.034), (_x1, HY + HOOK_SLOT, HZ + O),
           BLACK),
    _fixed("holster_sleeve_bottom", (_x0, HY - O, HZ - O), (_x1, HY + O, HZ - SLEEVE), BLACK),
    _fixed("holster_sleeve_left", (_x0, HY + SLEEVE, HZ - SLEEVE), (_x1, HY + O, HZ + SLEEVE), BLACK),
    _fixed("holster_sleeve_right", (_x0, HY - O, HZ - SLEEVE), (_x1, HY - SLEEVE, HZ + SLEEVE), BLACK),
    # the mouth's top and bottom (1 cm deep, 64 mm apart: the connector's 70 mm body stays outside)
    _fixed("holster_mouth_top", (_x0 - FLARE_DEPTH, HY - O, HZ + FLARE), (_x0, HY + O, HZ + O), BLACK),
    _fixed("holster_mouth_bottom", (_x0 - FLARE_DEPTH, HY - O, HZ - O), (_x0, HY + O, HZ - FLARE), BLACK),
]
# The lead-in's sides: two plates angled in plan, 4 cm long, from the socket's sides out to 90 mm apart, which guide the
# nose in sideways (real holsters are forgiving cups). They stop short of the connector's 57.4 mm body when it is hung.
_ang = math.atan2(LEAD_MOUTH - SLEEVE, LEAD_DEPTH)
_len = math.hypot(LEAD_MOUTH - SLEEVE, LEAD_DEPTH)
for _side, _sgn in (("left", 1.0), ("right", -1.0)):
    _mid = np.array([_x0 - LEAD_DEPTH / 2, HY + _sgn * (SLEEVE + LEAD_MOUTH) / 2])
    _out = np.array([math.sin(_ang), _sgn * math.cos(_ang)])        # the plate's outward normal (in plan)
    _c = _mid + 0.5 * LEAD_T * _out
    POST.append(Box(name=f"holster_lead_{_side}", pos=(float(_c[0]), float(_c[1]), float(HZ)),
                    size=(float(_len), LEAD_T, 2 * O), color=BLACK, mass=None, yaw=float(-_sgn * _ang)))

# -- the J1772 connector -------------------------------------------------------------------------------------------
# Real size (Volex J1772 data sheet: 57.4 x 144 x 251 mm overall; nose 43.8 mm) and mass (about 0.54 kg).
CONN_KG = 0.54
_DIR = np.array([-0.065, -0.06]) / math.hypot(0.065, 0.06)   # the pistol grip's direction in the (x, z) plane
_PHI = -math.atan2(_DIR[1], _DIR[0])
GRIP_Q = (math.cos(_PHI / 2), 0.0, math.sin(_PHI / 2), 0.0)   # (w, x, y, z): turns +x onto the grip's direction
RELIEF_C = (float(-0.165 + _DIR[0] * 0.025), 0.0, float(-0.09 + _DIR[1] * 0.025))
CABLE_EXIT = np.array([-0.165 + _DIR[0] * 0.047, 0.0, -0.09 + _DIR[1] * 0.047])
GROOVE = 0.004   # half width of the groove along the top of the nose that the latch's front arm runs in
CONN_BOXES = [
    ((-0.020, 0.0, -0.00495), (0.040, 2 * NOSE_HALF, 0.0339), GREY_PLASTIC),                  # nose, below the groove
    ((-0.020, (GROOVE + NOSE_HALF) / 2, 0.01695), (0.040, NOSE_HALF - GROOVE, 0.0099), GREY_PLASTIC),
    ((-0.020, -(GROOVE + NOSE_HALF) / 2, 0.01695), (0.040, NOSE_HALF - GROOVE, 0.0099), GREY_PLASTIC),
    ((-0.095, 0.0, 0.0), (0.110, 0.0574, 0.070), BLACK),                                      # body
    ((-0.1325, 0.0, -0.060), (0.100, 0.036, 0.036), BLACK, GRIP_Q),                           # pistol grip
    (RELIEF_C, (0.044, 0.022, 0.022), GREY_PLASTIC, GRIP_Q),                                  # cable strain relief
]
# The latch: one spring-loaded lever along the top, pressed down at its red button. At rest its hook (a tooth at the
# front of the nose) stands 8 mm above the nose and drops into the inlet's catch; pressed 6.5 mm or more, the tooth is
# below the socket's top wall and the connector slides out. Spring: about 3.6 N at 6.5 mm and 5 N at the 9 mm stop,
# PROVISIONAL until a real J1772 is measured (see README); 1,700 N/m (11 N) was more than the arm's press delivers.
LATCH_TRAVEL = 0.009
RELEASE_DEPTH = 0.0065
LATCH_RATE = 550.0
LATCH = Slider("latch", rest=(0.0, 0.0, 0.0), travel=LATCH_TRAVEL, rate=LATCH_RATE, damping=5.0, mass=0.02,
               boxes=(((-0.0905, 0.0, 0.0395), (0.099, 0.012, 0.009), GREY_PLASTIC),     # lever on top of the body
                      ((-0.045, 0.0, 0.028), (0.008, 0.006, 0.032), GREY_PLASTIC),       # down into the nose groove
                      ((-0.0225, 0.0, 0.016), (0.037, 0.006, 0.008), GREY_PLASTIC),      # arm along the groove
                      ((-0.008, 0.0, 0.025), (0.008, 0.006, 0.010), GREY_PLASTIC),       # hook tooth
                      ((-0.135, 0.0, 0.048), (0.024, 0.020, 0.008), RED)))               # release button
BUTTON_TOP = np.array([-0.135, 0.0, 0.052])
TOOTH_TOP = 0.030

# The cable: 14 mm across, 0.3 kg/m (a 32-40 A cable, 3 x 10 AWG), 0.75 m of it from the connector to the foot of the
# charger post, where it is fixed (the rest of a real 5-7.6 m cable is inside the post). It is laid out along the
# table at the start; NO_CABLE=1 leaves it out (for debugging only: the task and its scores use the cable).
SEATED_NOSE = np.array([X_SEAT - 0.0005, YP, ZP])
_EXIT_W = SEATED_NOSE + CABLE_EXIT
CABLE_PATH_W = [
    _EXIT_W,
    _EXIT_W + 0.025 * np.array([_DIR[0], 0.0, _DIR[1]]),
    np.array([0.10, -0.07, TOP + 0.0075]),
    np.array([0.11, -0.26, TOP + 0.0075]),
    np.array([0.22, -0.40, TOP + 0.0075]),
    np.array([0.33, -0.47, TOP + 0.0075]),
    np.array([POST_X0 - 0.038, -0.47, TOP + 0.0075]),   # into the post's foot (its base plate starts at x 0.37)
]
CABLE = None if os.environ.get("NO_CABLE") else CableSpec(
    points=tuple(tuple(float(v) for v in p - SEATED_NOSE) for p in CABLE_PATH_W), radius=0.007, segment=0.04,
    kg_per_m=0.3, bend_stiffness=0.3, bend_damping=0.02, color=BLACK, anchor_end=True)

CONNECTOR = part("connector", tuple(float(v) for v in SEATED_NOSE), CONN_BOXES, CONN_KG, yaw=0.0, sliders=(LATCH,),
                 color=BLACK, friction=0.6, cable=CABLE)
CONNECTOR.meta["torsional_patch"] = 0.005

OBJECTS = [CONNECTOR, DOOR, CATCH, *CAR, *POST]

# -- geometry helpers (pure numpy) ----------------------------------------------------------------------------------


def _quat_wxyz_matrix(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


BODY_OBB = [(np.asarray(b[0], float), np.asarray(b[1], float) / 2,
             _quat_wxyz_matrix(b[3]) if len(b) > 3 else np.eye(3)) for b in CONN_BOXES]


def connector_obbs(conn: PartState):
    """The connector's body boxes in the objects' frame: (centre, half sizes, rotation)."""
    R = conn.rotation()
    return [(conn.to_world(c), h, R @ Rb) for c, h, Rb in BODY_OBB]


def signed_distance(points, obbs) -> np.ndarray:
    """Signed distance (m; negative inside) from each point to the nearest of the boxes."""
    pts = np.atleast_2d(np.asarray(points, float))
    best = np.full(len(pts), np.inf)
    for c, h, R in obbs:
        p = (pts - c) @ R
        q = np.abs(p) - h
        d = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)
        best = np.minimum(best, d)
    return best


def lowest_point(conn: PartState) -> float:
    zs = []
    for c, h, R in connector_obbs(conn):
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    zs.append(float((c + R @ (h * np.array([sx, sy, sz])))[2]))
    return min(zs)


def nose_face(conn: PartState) -> np.ndarray:
    return conn.pos.copy()


def latch_depth(conn: PartState) -> float:
    """How far the release button is pressed (m, 0 at rest)."""
    return max(0.0, -float(conn.sliders.get("latch", 0.0)))


def out_of_seat(conn: PartState) -> float:
    """How far the nose face is back from its seated position in the inlet, along the inlet's axis (m)."""
    return float(SEATED_NOSE[0] - nose_face(conn)[0])


def aligned_with(conn: PartState, y, z, tol) -> bool:
    p = nose_face(conn)
    return abs(p[1] - y) <= tol and abs(p[2] - z) <= tol


def in_inlet(conn: PartState) -> bool:
    """Nose inside the inlet's socket (any depth)."""
    return aligned_with(conn, YP, ZP, BORE) and nose_face(conn)[0] > X_MOUTH + 0.002


def in_holster(conn: PartState) -> bool:
    """Hung: the nose face at least 20 mm into the 40 mm deep holster (half the nose's length), within its socket."""
    return aligned_with(conn, HY, HZ, SLEEVE + 0.003) and nose_face(conn)[0] >= HX - HUNG_DEPTH


ARMS = ("left_robot", "right_robot")
PAD_W, PAD_T, PAD_L = 0.0154, 0.004, 0.025   # finger pad: width across, thickness outward, length back from the tip


def _origin(base):
    o = base.scene.env_origins[0]
    o = getattr(o, "torch", o)
    return o.detach().cpu().numpy().astype(np.float64) if hasattr(o, "detach") else np.asarray(o, dtype=np.float64)


def grippers():
    """For each arm: (pad tip point on the gripper's axis, approach axis, closing axis, opening between the pads (m),
    (left finger travel, right finger travel)), in the objects' frame. Empty without the simulator."""
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


def pad_points(g, inner_only=False):
    """Points on one gripper's two finger pads: (left pad points, right pad points). Each pad is 15.4 mm wide, and
    points run 25 mm back from its tip; with ``inner_only`` only its gripping face, else also 4 mm outward."""
    tip, approach, close, _, (q7, q8) = g
    across = np.cross(approach, close)
    across /= np.linalg.norm(across) + 1e-12
    pads = []
    for side, travel in ((-1.0, q7), (1.0, q8)):
        pts = []
        for w in np.linspace(-PAD_W / 2, PAD_W / 2, 3):
            for back in np.linspace(0.0, PAD_L, 4):
                for d in ((0.0,) if inner_only else (0.0, PAD_T)):
                    pts.append(tip - back * approach + side * (travel + d) * close + w * across)
        pads.append(np.array(pts))
    return pads


HOLD_OPEN = (0.030, 0.075)   # a hand closed on the connector: pads 30-75 mm apart (grip 36 mm, body 57.4 mm)
HOLD_DIST, LET_GO = 0.006, 0.020


def holds(g, conn: PartState) -> bool:
    """Fingers closed onto the connector: pads 30-75 mm apart and each pad's gripping face within 6 mm of it."""
    if not HOLD_OPEN[0] <= g[3] <= HOLD_OPEN[1]:
        return False
    obbs = connector_obbs(conn)
    return all(float(signed_distance(p, obbs).min()) <= HOLD_DIST for p in pad_points(g, inner_only=True))


def clear_of(g, conn: PartState) -> bool:
    """A hand has let go: every point of both its pads 2 cm or more from the connector."""
    obbs = connector_obbs(conn)
    return all(float(signed_distance(p, obbs).min()) >= LET_GO for p in pad_points(g))


# -- trial memory and scoring --------------------------------------------------------------------------------------
YANK_N = 40.0       # pulling 40 N or more against the closed latch (the catch lip pushed 2 mm or more) is a yank
YANK_S = 0.10       # ... for 0.1 s or more
DROP_Z = TOP + 0.01   # dropped: its lowest point 1 cm or less above the table (it rests on it)
DEFAULT_DT = 1 / 20
_STATE: dict = {}
_RECORD: dict = {}
RECORD_PATH = os.path.expanduser(os.environ["EV_RECORD"]) if os.environ.get("EV_RECORD") else None
TRACE_PATH = os.path.expanduser(os.environ["EV_TRACE"]) if os.environ.get("EV_TRACE") else None
_DONE: list = []
_EARLIER = open(RECORD_PATH).read().splitlines() if RECORD_PATH and os.path.exists(RECORD_PATH) else []


@on_reset
def _reset_state(environment=None) -> None:
    if _RECORD.get("steps"):
        _DONE.append(dict(_RECORD))
        _RECORD["steps"] = 0
        _write_record()
    _STATE.clear()
    _STATE.update(step=0, scored=None, last=(0.0, False), released=False, out=False, yank_s=0.0, yanked=None,
                  dropped=None, recent=[], max_latch_load=0.0)
    _RECORD.clear()
    _RECORD.update(started=time.strftime("%Y-%m-%dT%H:%M:%S"), demo=os.environ.get("DEMO_VARIANT", ""), steps=0,
                   score=None, released_s=None, out_s=None, hung_s=None, door_shut_s=None, broke_rule=None,
                   broke_rule_s=None, max_latch_load_n=0.0, door_start_deg=None)


@on_step
def _count_step(environment=None) -> None:
    _STATE["step"] += 1


def _plain(o):
    """JSON for numpy scalars and arrays (the trace and record must never stop a run)."""
    return o.tolist() if hasattr(o, "tolist") else str(o)


def _write_record() -> None:
    if not RECORD_PATH:
        return
    current = [dict(_RECORD)] if _RECORD.get("steps") else []
    tmp = RECORD_PATH + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(_EARLIER + [json.dumps(r, default=_plain) for r in _DONE + current]) + "\n")
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


def latch_load(catch) -> float:
    """The force (N) the latch hook puts on the inlet's catch, pulling the connector out: the catch lip's spring rate
    times how far the hook has pushed it (0 when nothing bears on it)."""
    return CATCH_RATE * max(0.0, -float(catch.travel))


def evaluate(objects, gs) -> tuple[float, bool]:
    """One control step of scoring, from the object states and the grippers (see grippers()). Kept separate from the
    simulator so the offline tests can drive it."""
    now = _STATE["step"] * _dt()
    conn, door = objects["connector"], objects["port_door"]
    held = {k: bool(holds(g, conn)) for k, g in gs.items()}
    held_now = [k for k, h in held.items() if h]
    seated = in_inlet(conn) and out_of_seat(conn) <= 0.003

    if _RECORD["door_start_deg"] is None:
        _RECORD["door_start_deg"] = round(math.degrees(float(door.angle)), 1)

    # Yank: the latch hook pulled against its catch with 40 N or more (only possible with the button up, or not pressed
    # far enough), for 0.1 s.
    load = latch_load(objects["inlet_catch"])
    _STATE["max_latch_load"] = max(_STATE["max_latch_load"], load)
    if load >= YANK_N:
        _STATE["yank_s"] += _dt()
        if _STATE["yank_s"] >= YANK_S - 1e-9:
            _broke("yanked", f"pulled {load:.0f} N against the closed latch")
    else:
        _STATE["yank_s"] = 0.0

    # Released: the button pressed 6.5 mm or more (the hook below the socket's top wall) while still seated; or the nose
    # 5 mm or more clear of the inlet's face with no yank. The hook can only pass the catch when it is lowered, so a
    # connector that came out without a yank was released, even if the press stayed just short of 6.5 mm (a simulator
    # check saw the hook slip under the catch's lip at 5.96 mm).
    out_now = nose_face(conn)[0] <= X_MOUTH - 0.005
    if not _STATE["released"] and ((seated and latch_depth(conn) >= RELEASE_DEPTH)
                                   or (out_now and not _STATE["yanked"])):
        _STATE["released"] = True
        _RECORD["released_s"] = round(now, 2)
    # Out: the nose 5 mm or more clear of the inlet's face, the connector held by a hand.
    if _STATE["released"] and not _STATE["out"] and out_now and held_now:
        _STATE["out"] = True
        _RECORD["out_s"] = round(now, 2)

    hung = bool(in_holster(conn))
    # Dropped: no hand holds it, it is neither in the inlet nor in the holster, and it has come down onto the table
    # (its lowest point 1 cm or less above it). Held at the inlet's height, its lowest point is 2 cm up.
    if not held_now and not in_inlet(conn) and not hung and lowest_point(conn) <= DROP_Z:
        _broke("dropped", "dropped the connector")

    recent = _STATE["recent"]
    recent.append(conn.pos.copy())
    del recent[:-20]
    still = len(recent) == 20 and all(np.linalg.norm(p - conn.pos) < 0.001 for p in recent)
    let_go = bool(all(clear_of(g, conn) for g in gs.values())) if gs else not held_now
    door_shut = abs(float(door.angle)) <= DOOR_SHUT
    hung_free = _STATE["out"] and hung and let_go
    if hung_free and _RECORD["hung_s"] is None and still:
        _RECORD["hung_s"] = round(now, 2)
    if door_shut and _RECORD["door_shut_s"] is None:
        _RECORD["door_shut_s"] = round(now, 2)

    if _STATE["yanked"] or _STATE["dropped"]:
        result = (0.0, False)
    elif hung_free and still and door_shut:
        result = (1.0, True)
    elif hung_free and still:
        result = (0.75, False)
    elif _STATE["out"]:
        result = (0.5, False)
    elif _STATE["released"]:
        result = (0.25, False)
    else:
        result = (0.0, False)
    _RECORD.update(steps=_STATE["step"], score=result[0], max_latch_load_n=round(_STATE["max_latch_load"], 1))
    if TRACE_PATH:
        row = {"step": _STATE["step"], "t": round(now, 3), "score": result[0], "nose": [round(float(v), 4)
               for v in nose_face(conn)], "out_mm": round(1000 * out_of_seat(conn), 2),
               "latch_mm": round(1000 * latch_depth(conn), 2), "latch_load_n": round(load, 1),
               "catch_mm": round(-1000 * float(objects["inlet_catch"].travel), 3),
               "held": held, "let_go": let_go, "hung": hung, "door_deg": round(math.degrees(float(door.angle)), 2),
               "lowest_z": round(lowest_point(conn), 4)}
        with open(TRACE_PATH, "a") as f:
            f.write(json.dumps(row, default=_plain) + "\n")
    return result


def _update(objects):
    if _STATE["scored"] == _STATE["step"]:
        return _STATE["last"]
    _STATE["scored"] = _STATE["step"]
    result = evaluate(objects, grippers())
    _STATE["last"] = result
    _write_record()
    return result


def success(objects) -> bool:
    """The connector hangs in the holster with no hand on it and has been still for 1 s, the charge-port door is shut,
    and the latch was never yanked and the connector never dropped."""
    return _update(objects)[1]


def score(objects) -> float:
    """0 if the connector was pulled 40 N or more against its closed latch, or dropped. Otherwise: 0.25 latch released
    (button pressed 6.5 mm while the connector is home); 0.5 connector out of the inlet and held; 0.75 hung in the
    holster, let go and still for 1 s; 1.0 and the charge-port door shut."""
    return _update(objects)[0]

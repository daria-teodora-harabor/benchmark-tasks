"""A two-group commercial espresso machine (76 cm wide, 53 cm deep, 47 cm tall, black with a stainless front) stands
against the far edge of the table, its left end between the arms. Its left steam wand hangs from under the top panel
at the machine's left end, swung out toward the arms, and the steam knob for that wand is on the front panel directly
above it: a round black knob with a white pointer line, closed when the line points straight down, fully open after a
quarter turn counterclockwise (as you face the machine). In front of the machine, between the arms, a 12 oz stainless
steel milk pitcher stands on the counter, its handle toward the arms, holding 170 ml of cold milk from the fridge. A
clip-on thermometer is clipped inside the pitcher's rim on its far left (toward the machine), its stem in the milk
and its small round dial just above the rim, where the right wrist camera sees it while the pitcher is held; its red
needle shows the milk's temperature (0 to 100 degrees C), and two green marks on the dial show 55 and 65 degrees C."""

# Task: steam the milk for a latte. A barista purges the wand (a short burst of steam into the air clears the water
# that condenses in it), brings the tip just under the milk's surface and opens the steam fully; for a few seconds the
# tip draws in air and the milk rises (it is "stretched"); then the tip goes deeper and the milk rolls until it is hot.
# The steam is closed before the tip comes out of the milk (with the steam on, a tip leaving the milk sprays hot milk),
# and the pitcher is set down. Target: 55-65 degrees C (SCA; off flavours start at about 70) and 20-40% more volume
# (a latte, not a cappuccino).
#
# The docstring above only describes the scene, because the simulator shows it to agent policies. Scoring is explained
# in these comments and in the README.
#
# Frame: the station scene's objects frame (x away from the arms, y to the arms' left, z up; arm bases at x 0,
# y +-0.305, table top at TOP).
#
# The machine is a Gaggia La Vetro 2-group (760 x 530 x 468 mm, quarter-turn steam taps; Evoca spec sheet), fixed in
# place. The knob is a partkit Flap whose hinge lies along the machine's front normal; its frame also carries the left
# steam wand (fixed, as if its ball joint were stiff). The pitcher is a partkit Part (a 12 oz pitcher: 71 mm across,
# 95 mm tall, 220 g, with its 175 g of milk as part of its rigid mass).
#
# Milk model (Python, every control step; see Milk): the steam flow follows the knob's angle (none below 10 degrees,
# full at 90; 1.0 g/s at full, a 1-1.5 bar boiler). With the wand's tip under the milk's surface the steam condenses in
# the milk, giving up its latent heat (2257 kJ/kg) and its sensible heat from 120 degrees C, warming the milk and the
# pitcher, and adding its water. Within 8 mm of the surface the tip also draws in air (10 ml/s at full steam), tapering
# to none at 15 mm deep; the foam raises the surface. The surface is the horizontal plane that holds the milk's and
# foam's volume in the pitcher (a cylinder of its inner radius; a tilted cylinder's volume is its base area times the
# axis length to the plane). Milk spills where that plane rises above the rim (the spout's lip is 8 mm lower).

from __future__ import annotations

import json
import math
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from partkit import env, flap, on_reset, on_step, part  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box, Cylinder  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
INSTRUCTION = ("Steam the milk in the pitcher for a latte: purge the steam wand first, steam the milk until it is "
               "55 to 65 degrees C and 20 to 40% more in volume, then turn the steam off and set the pitcher down.")

TOP = TABLE_TOP[SCENE]
X, Y, Z = np.eye(3)

MACHINE_BLACK = (0.08, 0.08, 0.09)
STEEL = (0.78, 0.79, 0.80)
CHROME = (0.86, 0.87, 0.88)
KNOB_BLACK = (0.05, 0.05, 0.05)
POINTER_WHITE = (0.95, 0.95, 0.95)
GROUP_GREY = (0.62, 0.63, 0.65)
HANDLE_BLACK = (0.10, 0.10, 0.10)
MILK_WHITE = (0.97, 0.97, 0.95)
FOAM_CREAM = (0.98, 0.95, 0.86)
DIAL_WHITE = (0.98, 0.98, 0.98)
NEEDLE_RED = (0.85, 0.08, 0.08)
TICK_GREEN = (0.10, 0.65, 0.20)

# -- the machine (Gaggia La Vetro 2-group) -------------------------------------------------------------------------------
M_W, M_D, M_H = 0.760, 0.530, 0.468
BACK_X = 0.905
FRONT_X = BACK_X - M_D            # 0.375: the front panel
M_Y = -0.255                      # its centre: its left end (y +0.125) between the arms, its right end 1.5 cm in from
                                  # the table's right edge
TOP_PANEL_Z = (0.330, 0.468)      # the top panel (knobs, groups under it), above the counter
TRAY_X, TRAY_H = 0.395, 0.145     # the drip tray's front edge and its grid height
SIDE_T = 0.020
WAND_Y = M_Y + M_W / 2 - 0.090    # the left wand and its knob: 9 cm in from the machine's left end (y 0.035)
KNOB_H = 0.370                    # the knob's centre, above the counter
PIVOT = np.array([FRONT_X - 0.010, WAND_Y, TOP + 0.320])
WAND_L = 0.170
WAND_SWING = math.radians(25.0)   # swung out toward the arms
WAND_DIR = np.array([-math.sin(WAND_SWING), 0.0, -math.cos(WAND_SWING)])
TIP = PIVOT + WAND_L * WAND_DIR   # the steam tip (its holes)
WAND_R = 0.005
GROUP_Y = (M_Y + 0.170, M_Y - 0.170)
GROUP_X = 0.460


def _static(name, lo, hi, color):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return Box(name, tuple(((lo + hi) / 2).tolist()), color=color, size=tuple((hi - lo).tolist()))


Y0, Y1 = M_Y - M_W / 2, M_Y + M_W / 2
MACHINE = [
    _static("machine_body", (0.620, Y0, TOP), (BACK_X, Y1, TOP + M_H), MACHINE_BLACK),
    _static("machine_top", (FRONT_X, Y0, TOP + TOP_PANEL_Z[0]), (0.620, Y1, TOP + TOP_PANEL_Z[1]), MACHINE_BLACK),
    _static("drip_tray", (TRAY_X, Y0 + SIDE_T, TOP), (0.620, Y1 - SIDE_T, TOP + TRAY_H), STEEL),
    _static("side_left", (FRONT_X, Y1 - SIDE_T, TOP), (0.620, Y1, TOP + TOP_PANEL_Z[0]), MACHINE_BLACK),
    _static("side_right", (FRONT_X, Y0, TOP), (0.620, Y0 + SIDE_T, TOP + TOP_PANEL_Z[0]), MACHINE_BLACK),
]
for i, gy in enumerate(GROUP_Y):
    MACHINE += [
        Cylinder(f"group_{i}", (GROUP_X, gy, TOP + 0.290), color=CHROME, radius=0.040, height=0.080),
        Cylinder(f"basket_{i}", (GROUP_X, gy, TOP + 0.235), color=CHROME, radius=0.036, height=0.030),
        _static(f"portafilter_{i}", (0.300, gy - 0.012, TOP + 0.226), (GROUP_X - 0.036, gy + 0.012, TOP + 0.244),
                HANDLE_BLACK),
    ]

# -- the steam knob and the wand: one Flap ------------------------------------------------------------------------------
# The flap's frame: its z axis (the hinge) points out of the front panel toward the arms (-x), its x axis to the arms'
# left (+y), so its y axis points down. A positive angle turns the knob counterclockwise as you face the machine.
R_KNOB_FRAME = np.column_stack([Y, -Z, -X])
KNOB_HINGE = np.array([FRONT_X, WAND_Y, TOP + KNOB_H])
KNOB_ACROSS, KNOB_LEN = 0.045, 0.032            # a 45 mm octagonal knob, 32 mm long, 3 mm proud of the panel
KNOB_OPEN = math.radians(90.0)
STEAM_FROM = math.radians(10.0)                 # the valve starts to pass steam at 10 degrees
KNOB_GRIP = KNOB_HINGE - (0.003 + KNOB_LEN / 2) * X   # the knob's middle (a TCP gripping it)


def _wxyz(R):
    from partkit import _matrix_wxyz

    return _matrix_wxyz(R)


def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _frame_box(centre_w, R_w, size, color):
    """A box given in the world, in the knob flap's frame."""
    c = R_KNOB_FRAME.T @ (np.asarray(centre_w, float) - KNOB_HINGE)
    return (tuple(c.tolist()), tuple(size), color, _wxyz(R_KNOB_FRAME.T @ R_w))


def _wand_rotation():
    d = WAND_DIR
    e1 = np.cross(d, Y)
    e1 /= np.linalg.norm(e1)
    return np.column_stack([e1, np.cross(d, e1), d])


KNOB_BOXES = [
    ((0.0, 0.0, 0.003 + KNOB_LEN / 2), (KNOB_ACROSS, KNOB_ACROSS, KNOB_LEN), KNOB_BLACK),
    ((0.0, 0.0, 0.003 + KNOB_LEN / 2), (KNOB_ACROSS, KNOB_ACROSS, KNOB_LEN), KNOB_BLACK,
     (math.cos(math.pi / 8), 0.0, 0.0, math.sin(math.pi / 8))),
    ((0.0, 0.012, 0.003 + KNOB_LEN + 0.0015), (0.005, 0.020, 0.003), POINTER_WHITE),   # the pointer (down when shut)
]
FRAME_BOXES = [
    ((0.0, 0.0, 0.0015), (0.060, 0.060, 0.002), CHROME),                               # the knob's chrome ring
    _frame_box((PIVOT + TIP) / 2, _wand_rotation(), (2 * WAND_R, 2 * WAND_R, WAND_L), STEEL),
    _frame_box(PIVOT, np.eye(3), (0.022, 0.022, 0.022), CHROME),                       # the wand's ball joint
    _frame_box(PIVOT + np.array([0.0045, 0.0, 0.0]), np.eye(3), (0.008, 0.012, 0.012), CHROME),   # to the panel
]
KNOB = flap("steam_knob", KNOB_HINGE, KNOB_BOXES, FRAME_BOXES, open_angle=0.0, lower=0.0, upper=KNOB_OPEN,
            flap_mass=0.03, color=KNOB_BLACK, friction_torque=0.0, damping=0.1, friction=0.9,
            quat_xyzw=(lambda w: (w[1], w[2], w[3], w[0]))(_wxyz(R_KNOB_FRAME)))

# -- the pitcher (12 oz) -----------------------------------------------------------------------------------------------
# Part frame: the outer bottom's centre, z up the pitcher's axis, the handle toward -x, the spout toward +x.
R_IN, WALL, P_H = 0.0335, 0.002, 0.095
R_OUT = R_IN + WALL
BOTTOM_T = 0.004
SPOUT_DROP = 0.008                # the spout's lip is 8 mm below the rim
N_WALL = 16
P_MASS, MILK_ML0, MILK_RHO = 0.220, 170.0, 1.03
MILK_KG0 = MILK_ML0 * MILK_RHO / 1000
PITCHER_XY = (0.240, 0.030)       # on the counter in front of the machine, between the arms
GRIP_Z = 0.032                    # where a hand holds it (low on the body)


def _pitcher_boxes():
    boxes = []
    w = 2 * R_OUT * math.tan(math.pi / N_WALL) + 0.001
    for k in range(N_WALL):
        a = 2 * math.pi * k / N_WALL
        c = (R_IN + WALL / 2) * np.array([math.cos(a), math.sin(a), 0.0])
        boxes.append(((float(c[0]), float(c[1]), BOTTOM_T + (P_H - BOTTOM_T) / 2), (WALL, w, P_H - BOTTOM_T), STEEL,
                      (math.cos(a / 2), 0.0, 0.0, math.sin(a / 2))))
    hx = -(R_OUT + 0.012)
    boxes += [((hx, 0.0, 0.025), (0.024, 0.012, 0.007), STEEL),
              ((hx, 0.0, 0.080), (0.024, 0.012, 0.007), STEEL),
              ((-(R_OUT + 0.0205), 0.0, 0.0525), (0.007, 0.012, 0.062), STEEL)]
    return boxes


PITCHER = part("pitcher", (PITCHER_XY[0], PITCHER_XY[1], TOP + 0.0005), _pitcher_boxes(), P_MASS + MILK_KG0,
               color=STEEL, friction=0.8,
               cylinders=[((0.0, 0.0, BOTTOM_T / 2), R_OUT, BOTTOM_T, STEEL)])
PITCHER.meta["com"] = (-0.0045, 0.0, 0.0375)
PITCHER.meta["inertia"] = (2.9e-4, 3.1e-4, 2.5e-4)
PITCHER.meta["torsional_patch"] = 0.005

OBJECTS = MACHINE + [KNOB, PITCHER]

LOG = {"errors": [], "draw_ms": []}


def _err(where, e) -> None:
    if len(LOG["errors"]) < 30:
        LOG["errors"].append(f"{where}: {type(e).__name__}: {e}"[:300])


# -- the milk ----------------------------------------------------------------------------------------------------------
STEAM_KG_S = 0.0010               # at full steam
LATENT = 2.257e6                  # J/kg
C_WATER, C_MILK, C_STEEL = 4186.0, 3930.0, 500.0
T_STEAM, T0 = 120.0, 4.0
AIR_M3_S = 10e-6                  # air drawn in at full steam with the tip near the surface
D_AIR_FULL, D_AIR_ZERO = 0.008, 0.015
D_OUT = -0.002                    # the tip counts as out of the milk 2 mm above the surface
WAND_WATER_KG = 0.005             # condensed water in an unpurged wand
PURGE_S = 1.0
AREA = math.pi * R_IN ** 2


def steam_fraction(angle: float) -> float:
    """The valve's opening (0..1) for a knob angle (rad from shut)."""
    return float(np.clip((angle - STEAM_FROM) / (KNOB_OPEN - STEAM_FROM), 0.0, 1.0))


def pitcher_rotation(state) -> np.ndarray:
    return state.rotation()


class Milk:
    def __init__(self):
        self.kg = MILK_KG0
        self.T = T0
        self.foam_m3 = 0.0
        self.max_T = T0
        self.purge_s = 0.0
        self.steamed = False           # steam has gone into the milk
        self.purged_before = None      # the purge time when steaming began
        self.wet_tip_on = False        # the tip has been in the milk since the steam was last opened
        self.spatter_s = None
        self.spill_m3 = 0.0
        self.spitting_s = 0.0
        self.unpurged_water = False
        self.depth = None
        self.inside = False
        self.surface_z = None
        self.u = 0.0
        self.t = 0.0
        self.heated_s = 0.0
        self.air_s = 0.0

    def liquid_m3(self) -> float:
        return self.kg / (MILK_RHO * 1000)

    def volume_m3(self) -> float:
        return self.liquid_m3() + self.foam_m3

    def stretch(self) -> float:
        return self.volume_m3() / (MILK_ML0 * 1e-6) - 1.0

    def surface(self, R, p):
        """The milk-and-foam surface's height (world z), the axis point it passes through, and the tilt (rad)."""
        axis = R[:, 2]
        h = self.volume_m3() / AREA
        a = p + R @ np.array([0.0, 0.0, BOTTOM_T]) + h * axis
        return float(a[2]), a, math.acos(float(np.clip(axis[2], -1.0, 1.0)))

    def rim_low_z(self, R, p) -> float:
        """The lowest point of the rim (the spout's lip counts 8 mm lower), world z."""
        zs = []
        for k in range(24):
            a = 2 * math.pi * k / 24
            drop = SPOUT_DROP if math.cos(a) > 0.92 else 0.0
            zs.append(float((p + R @ np.array([R_IN * math.cos(a), R_IN * math.sin(a), P_H - drop]))[2]))
        return min(zs)

    def step(self, dt: float, angle: float, R, p) -> None:
        self.t += dt
        u = steam_fraction(angle)
        self.u = u
        zs, a, tilt = self.surface(R, p)
        self.surface_z = zs
        # spill: the plane above the rim's lowest point (a big tilt empties the pitcher)
        rim = self.rim_low_z(R, p)
        if tilt > math.radians(75):
            self.spill_m3 += self.volume_m3()
            self.kg, self.foam_m3 = 0.0, 0.0
        elif zs > rim:
            over = (zs - rim) * AREA * math.cos(tilt)
            take = min(over, self.volume_m3())
            f = self.foam_m3 / max(self.volume_m3(), 1e-12)
            self.foam_m3 -= take * f
            self.kg -= take * (1 - f) * MILK_RHO * 1000
            self.spill_m3 += take
        # the tip
        q = R.T @ (TIP - p)
        self.inside = bool(math.hypot(q[0], q[1]) < R_IN - 0.002 and BOTTOM_T < q[2] < P_H + 0.010)
        d = (zs - TIP[2]) if self.inside else None
        self.depth = d
        if u <= 0.0:
            self.wet_tip_on = False
            return
        if d is not None and d > 0.0 and self.kg > 0:
            if not self.steamed:
                self.steamed = True
                self.purged_before = self.purge_s
                if self.purge_s < PURGE_S:     # the wand's condensed water goes into the milk
                    self.unpurged_water = True
                    C = self.kg * C_MILK + P_MASS * C_STEEL
                    self.T = (C * self.T + WAND_WATER_KG * C_WATER * 40.0) / (C + WAND_WATER_KG * C_WATER)
                    self.kg += WAND_WATER_KG
            self.wet_tip_on = True
            ms = STEAM_KG_S * u * dt
            C = self.kg * C_MILK + P_MASS * C_STEEL
            self.T += ms * (LATENT + C_WATER * (T_STEAM - self.T)) / C
            self.kg += ms
            self.heated_s += dt
            if d <= D_AIR_ZERO:
                g = 1.0 if d <= D_AIR_FULL else (D_AIR_ZERO - d) / (D_AIR_ZERO - D_AIR_FULL)
                self.foam_m3 += AIR_M3_S * u * g * dt
                self.air_s += dt * g
            self.max_T = max(self.max_T, self.T)
        else:
            if not self.steamed and not self.inside:
                self.purge_s += dt
            if d is not None and D_OUT < d <= 0.0:
                self.spitting_s += dt
            if self.wet_tip_on and (d is None or d <= D_OUT) and self.spatter_s is None:
                self.spatter_s = round(self.t, 2)


# -- drawing (markers only: the milk's surface, the thermometer, steam and spattered milk) ------------------------------
N_NEEDLE, N_DROPS, N_PUFFS = 5, 40, 6
HIDE = np.array([0.0, 0.0, -10.0])
# a clip-on thermometer's 20 mm dial, its stem 12 mm in from the wall at 65 degrees round from the spout toward the
# left, 4 mm above the rim: wholly in the right wrist camera's view (left side) while the pitcher is held, clear of
# the wand
DIAL_PHI = math.radians(65.0)
DIAL_LOCAL = np.array([(R_IN - 0.012) * math.cos(DIAL_PHI), (R_IN - 0.012) * math.sin(DIAL_PHI), P_H + 0.004])
DIAL_R = 0.010
_S: dict = {}


def dial_angle(T: float) -> float:
    """The needle's angle in the dial (world xy, rad from +x): 0 degrees C at the dial's lower left as seen from the
    arms (toward the arms and their left), 100 at its lower right, 270 degrees of sweep clockwise as seen from above."""
    return math.radians(135.0) - math.radians(270.0) * float(np.clip(T, 0.0, 100.0)) / 100.0


def marker_layout(milk: Milk, R, p, rng=None):
    """Translations, scales and marker indices (0 milk, 1 foam, 2 dial, 3 needle, 4 tick, 5 drop, 6 steam)."""
    tr, sc, ix = [], [], []
    zs, a, _ = milk.surface(R, p)
    surf = np.array([a[0], a[1], zs - 0.001])
    show = milk.kg > 0.01
    tr.append(surf if show else HIDE)
    sc.append((R_IN - 0.0015, R_IN - 0.0015, 0.002))
    ix.append(1 if milk.stretch() > 0.05 else 0)
    dial = p + R @ DIAL_LOCAL
    tr.append(dial)
    sc.append((DIAL_R, DIAL_R, 0.003))
    ix.append(2)
    th = dial_angle(milk.T)
    for k in range(1, N_NEEDLE + 1):
        tr.append(dial + np.array([0.0018 * k * math.cos(th), 0.0018 * k * math.sin(th), 0.0025]))
        sc.append((0.0011, 0.0011, 0.0011))
        ix.append(3)
    for T in (55.0, 65.0):
        t = dial_angle(T)
        tr.append(dial + np.array([0.0085 * math.cos(t), 0.0085 * math.sin(t), 0.0022]))
        sc.append((0.0012, 0.0012, 0.0012))
        ix.append(4)
    drops = np.random.default_rng(11)            # spattered drops stay where they landed
    rng = rng or np.random.default_rng(int(milk.t * 20) + 7)
    for k in range(N_DROPS):
        if milk.spatter_s is not None:
            r = 0.04 + 0.12 * drops.random()
            ang = 2 * math.pi * drops.random()
            tr.append(np.array([TIP[0] + r * math.cos(ang), TIP[1] + r * math.sin(ang), TOP + 0.002]))
        else:
            tr.append(HIDE)
        sc.append((0.003, 0.003, 0.0015))
        ix.append(5)
    puffing = milk.u > 0 and (milk.depth is None or milk.depth <= 0.0)
    for k in range(N_PUFFS):
        if puffing:
            s = 0.012 * (k + 1)
            tr.append(TIP + s * WAND_DIR + 0.004 * (rng.random(3) - 0.5))
            r = 0.005 + 0.003 * k
        else:
            tr.append(HIDE)
            r = 0.005
        sc.append((r, r, r))
        ix.append(6)
    return np.array(tr, np.float32), np.array(sc, np.float32), np.array(ix, np.int64)


def _origin(base):
    o = base.scene.env_origins[0]
    o = getattr(o, "torch", o)
    return o.detach().cpu().numpy().astype(np.float64) if hasattr(o, "detach") else np.asarray(o, dtype=np.float64)


def _build(environment) -> None:
    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

    def surf(color, opacity=1.0):
        return sim_utils.PreviewSurfaceCfg(diffuse_color=color, opacity=opacity, roughness=0.5)

    markers = {
        "milk": sim_utils.CylinderCfg(radius=1.0, height=1.0, visual_material=surf(MILK_WHITE)),
        "foam": sim_utils.CylinderCfg(radius=1.0, height=1.0, visual_material=surf(FOAM_CREAM)),
        "dial": sim_utils.CylinderCfg(radius=1.0, height=1.0, visual_material=surf(DIAL_WHITE)),
        "needle": sim_utils.SphereCfg(radius=1.0, visual_material=surf(NEEDLE_RED)),
        "tick": sim_utils.SphereCfg(radius=1.0, visual_material=surf(TICK_GREEN)),
        "drop": sim_utils.SphereCfg(radius=1.0, visual_material=surf(MILK_WHITE)),
        "steam": sim_utils.SphereCfg(radius=1.0, visual_material=surf((1.0, 1.0, 1.0), 0.35)),
    }
    _S["markers"] = VisualizationMarkers(VisualizationMarkersCfg(prim_path="/World/Visuals/milk", markers=markers))
    _S["origin"] = _origin(environment.unwrapped)[:3]


def _draw(R, p) -> None:
    if "markers" not in _S:
        return
    t0 = time.perf_counter()
    tr, sc, ix = marker_layout(_S["milk"], R, p)
    _S["markers"].visualize(translations=tr + _S["origin"].astype(np.float32), scales=sc, marker_indices=ix)
    LOG["draw_ms"].append(1000 * (time.perf_counter() - t0))
    del LOG["draw_ms"][:-200]


# -- per-step state and scoring -----------------------------------------------------------------------------------------
T_LO, T_HI, T_SCALD = 55.0, 65.0, 70.0
HEATED_T = 40.0
STRETCH_LO, STRETCH_HI = 0.20, 0.40
SPILL_ML = 2.0
SET_DOWN_TILT = math.radians(10.0)
SET_DOWN_DZ = 0.005
DEFAULT_DT = 1 / 20
TRACE_PATH = os.path.expanduser(os.environ["MILK_TRACE"]) if os.environ.get("MILK_TRACE") else None
_STATE: dict = {}


def _dt() -> float:
    environment = env()
    try:
        return float(environment.unwrapped.step_dt) if environment is not None else DEFAULT_DT
    except Exception:  # noqa: BLE001
        return DEFAULT_DT


def reset_state() -> None:
    _STATE.clear()
    _STATE.update(step=0, scored=None, last=(0.0, False), dropped=None, set_down=False, angle=0.0, tilt=0.0,
                  bottom_z=None, speed=0.0)
    _S["milk"] = Milk()


def milk_step(pitcher, knob_angle: float, dt: float) -> None:
    """Advance the milk by one control step from the pitcher's pose and the knob's angle."""
    m = _S["milk"]
    R = pitcher_rotation(pitcher)
    p = np.asarray(pitcher.pos, float)
    m.step(dt, knob_angle, R, p)
    axis = R[:, 2]
    tilt = math.acos(float(np.clip(axis[2], -1.0, 1.0)))
    speed = float(np.linalg.norm(np.asarray(pitcher.lin_vel, float)))
    bottom = float((p + R @ np.array([0.0, 0.0, 0.0]))[2])
    _STATE.update(angle=float(knob_angle), tilt=tilt, bottom_z=bottom, speed=speed)
    if p[2] < TOP - 0.05 and _STATE["dropped"] is None:
        _STATE["dropped"] = round(m.t, 2)
    _STATE["set_down"] = bool(tilt < SET_DOWN_TILT and abs(bottom - TOP) < SET_DOWN_DZ and speed < 0.02)


def evaluate() -> tuple[float, bool]:
    """The score so far and whether the task is done."""
    m = _S["milk"]
    if m.max_T > T_SCALD or m.spatter_s is not None or m.spill_m3 * 1e6 > SPILL_ML or _STATE["dropped"] is not None:
        return 0.0, False
    if not (m.steamed and m.max_T >= HEATED_T):
        return 0.0, False
    t_ok = T_LO <= m.T <= T_HI
    if not t_ok:
        return 0.25, False
    if not (STRETCH_LO <= m.stretch() <= STRETCH_HI):
        return 0.5, False
    finished = (m.purged_before is not None and m.purged_before >= PURGE_S and m.u <= 0.0 and _STATE["set_down"])
    if not finished:
        return 0.75, False
    return 1.0, True


def _states(environment):
    try:
        from yam_arena.scene import object_states

        return object_states(environment, ["pitcher", "steam_knob"])
    except Exception as e:  # noqa: BLE001
        _err("states", e)
        return None


@on_reset
def _on_reset(environment=None) -> None:
    try:
        reset_state()
        if environment is not None:
            if "markers" not in _S:
                _build(environment)
            states = _states(environment)
            if states:
                _draw(pitcher_rotation(states["pitcher"]), np.asarray(states["pitcher"].pos, float))
    except Exception as e:  # noqa: BLE001
        _err("reset", e)


@on_step
def _on_step(environment=None) -> None:
    _STATE["step"] += 1
    if environment is None:
        return
    states = _states(environment)
    if not states:
        return
    try:
        pitcher, knob = states["pitcher"], states["steam_knob"]
        milk_step(pitcher, float(knob.angle), _dt())
        _draw(pitcher_rotation(pitcher), np.asarray(pitcher.pos, float))
        if TRACE_PATH:
            with open(TRACE_PATH, "a") as f:
                f.write(json.dumps(trace_row()) + "\n")
    except Exception as e:  # noqa: BLE001
        _err("step", e)


def trace_row() -> dict:
    m = _S["milk"]
    return {"step": _STATE["step"], "knob_deg": round(math.degrees(_STATE["angle"]), 1), "u": round(m.u, 3),
            "T": round(m.T, 2), "stretch": round(m.stretch(), 4),
            "depth_mm": None if m.depth is None else round(1000 * m.depth, 1), "inside": m.inside,
            "purge_s": round(m.purge_s, 2), "steamed": m.steamed, "spatter_s": m.spatter_s,
            "spill_ml": round(m.spill_m3 * 1e6, 2), "tilt_deg": round(math.degrees(_STATE["tilt"]), 1),
            "bottom_mm": None if _STATE["bottom_z"] is None else round(1000 * (_STATE["bottom_z"] - TOP), 1),
            "set_down": _STATE["set_down"], "score": evaluate()[0]}


reset_state()


def _update():
    if _STATE["scored"] == _STATE["step"]:
        return _STATE["last"]
    _STATE["scored"] = _STATE["step"]
    _STATE["last"] = evaluate()
    return _STATE["last"]


def success(objects) -> bool:
    """The wand was purged, the milk is 55-65 degrees C (never over 70) and 20-40% more in volume, nothing sprayed or
    spilled, the steam is off and the pitcher stands upright on the counter."""
    return _update()[1]


def score(objects) -> float:
    """0 if the milk ever went over 70 degrees C, milk sprayed (the tip left the milk with the steam on), milk spilled
    or the pitcher fell; 0.25 heated to 40 degrees C; 0.5 ending at 55-65; 0.75 also 20-40% more in volume; 1.0 also
    purged first, the steam off and the pitcher set down upright."""
    return _update()[0]


def report() -> dict:
    """A summary for the demo and the tests."""
    m = _S["milk"]
    return {"score": _update()[0], "T": round(m.T, 2), "max_T": round(m.max_T, 2),
            "stretch": round(m.stretch(), 4), "volume_ml": round(m.volume_m3() * 1e6, 1),
            "added_water_g": round(1000 * (m.kg - MILK_KG0), 1), "purge_s": round(m.purge_s, 2),
            "purged_before": m.purged_before, "unpurged_water": m.unpurged_water, "steamed": m.steamed,
            "heated_s": round(m.heated_s, 2), "air_s": round(m.air_s, 2), "spitting_s": round(m.spitting_s, 2),
            "spatter_s": m.spatter_s, "spill_ml": round(m.spill_m3 * 1e6, 2), "dropped": _STATE["dropped"],
            "set_down": _STATE["set_down"], "knob_deg": round(math.degrees(_STATE["angle"]), 1),
            "errors": LOG["errors"][:10]}

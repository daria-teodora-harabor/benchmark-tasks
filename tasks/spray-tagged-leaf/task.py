"""A young pepper plant grows in a terracotta pot on the table, in front of the arms and a little to the right. It is
about 22 cm tall with eight leaves on short stalks around its stem. One leaf, about 11 cm above the soil and pointing
toward the arms' front right, has a strip of pink flagging tape tied around its stalk. To the right of the pot stands
an 8 oz (237 ml) trigger spray bottle of ready-to-use insecticidal soap, full, its nozzle pointing to the arms' left
and its blue trigger at the front of its white spray head. To the left, a piece of corrugated cardboard (16 x 20 cm,
2 mm thick) lies flat on top of a small wooden crate, its two short edges overhanging the crate by 4 cm, one of them
toward the arms. The liquid carries a blue marker dye, so spray shows as blue drops where it lands."""

# Task: spray the tagged leaf until its top is wet all over, without getting spray on any other leaf. A first spray of
# a new pesticide goes on one test leaf, which is checked a day later for damage (insecticidal soap: "spray a small
# area and wait 24 hours", Clemson HGIC). The spray drifts, and fine drops settle on the leaves below, so the other hand
# holds the card under the test leaf as a shield (the Roundup ready-to-use label, EPA 71995-33: "shield plants from
# drift with a piece of cardboard").
#
# The docstring above only describes the scene, because the simulator shows it to agent policies. Scoring is explained
# in these comments and in the README.
#
# Frame: the station scene's objects frame (x away from the arms, y to the arms' left, z up; arm bases at x 0,
# y +-0.305). The plant's own frame: origin at the centre of the soil's surface, axes as the world's.
#
# Spray model (in Python, every control step; see Spray): a trigger squeeze fires when the trigger passes 10 of its
# 12 mm; it sends 1 ml out of the nozzle over 0.2 s as N_PARCELS droplet parcels, each a group of identical droplets
# (sizes log-normal by volume, median 200 um, spread 1.7, 40-600 um), leaving at 12 m/s inside a 40 degree cone along
# the nozzle's axis, then flying with gravity and air drag (Schiller-Naumann) in 0.25 ms steps. A parcel stops on the
# first thing its path crosses: a leaf (a triangle of the leaf meshes), the card, the soil or the table. On a leaf, each
# droplet leaves a stain twice its diameter; a leaf triangle's wet fraction is the chance a point on it lies under at
# least one stain, 1 - exp(-stain area / triangle area). No spray comes out when the bottle leans more than 75 degrees
# from upright (its dip tube is out of the liquid). A sample of the parcels is drawn as blue beads.
#
# The plant is drawn only (no colliders): real leaves bend and would spread the wet spray, so touching the plant at
# all is a failure (below), checked geometrically every control step against the bottle, both grippers and the card.

from __future__ import annotations

import json
import math
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from partkit import Slider, env, on_reset, on_step, part  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box, Cylinder  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]
INSTRUCTION = "Spray the tagged leaf until its top is wet all over, without getting spray on any other leaf."

TERRACOTTA = (0.72, 0.42, 0.28)
SOIL_BROWN = (0.24, 0.17, 0.11)
LEAF_GREEN = (0.22, 0.45, 0.15)
STEM_GREEN = (0.36, 0.55, 0.22)
TAG_PINK = (1.00, 0.25, 0.60)
CARDBOARD = (0.74, 0.60, 0.42)
WOOD = (0.55, 0.40, 0.25)
DYE_BLUE = (0.10, 0.35, 0.95)

# -- the plant ----------------------------------------------------------------------------------------------------------
POT_XY = (0.40, -0.10)
POT_R = 0.075
SOIL = TOP + 0.115                 # the soil's surface, level with the pot's rim
PLANT_ORIGIN = np.array([POT_XY[0], POT_XY[1], SOIL])
TAG_INDEX = 3
TAG_AZ_DEG = -135.0                # the tagged leaf's direction (180 = toward the arms, 90 = to their left)
STEM_R, STEM_H = 0.0035, 0.21
STALK_R = 0.0015


def ovate_leaf(base, az, length, width, rise_deg, droop, cup, nt=14, nu=9):
    """A pepper leaf blade: ovate (widest at about 0.37 of its length), pointed tip. It leaves `base` toward azimuth
    `az`, rising at `rise_deg`, its tip drooping by `droop` and its edges cupped up by `cup` (m). Rows of nu vertices
    run across the blade; row i is at fraction i / (nt - 1) of its length."""
    t = np.linspace(0, 1, nt)
    w = width * np.sin(np.pi * t ** 0.7) ** 0.9
    u = np.linspace(-1, 1, nu)
    rise = math.radians(rise_deg)
    s = length * t * math.cos(rise)
    z = length * t * math.sin(rise) - droop * t ** 2
    d = np.array([math.cos(az), math.sin(az), 0.0])
    n = np.array([-math.sin(az), math.cos(az), 0.0])
    V = np.array([base + d * s[i] + n * (u[j] * w[i] / 2) + np.array([0, 0, z[i] + cup * u[j] ** 2 * min(1, 3 * t[i])])
                  for i in range(nt) for j in range(nu)])
    F = []
    for i in range(nt - 1):
        for j in range(nu - 1):
            a, b, c, e = i * nu + j, i * nu + j + 1, (i + 1) * nu + j, (i + 1) * nu + j + 1
            F += [[a, b, c], [b, e, c]]
    return V, np.array(F)


def pepper(seed=4):
    """A young pepper plant, about 22 cm tall: 8 leaves on short stalks, alternate at 137.5 degrees, rising 4.5-20 cm
    above the soil; leaf TAG_INDEX points to TAG_AZ_DEG. Returns [(V, F, stalk base, azimuth)] in the plant's frame."""
    r = np.random.default_rng(seed)
    out = []
    heights = np.linspace(0.045, 0.200, 8)
    az0 = math.radians(TAG_AZ_DEG) - TAG_INDEX * math.radians(137.5)
    for k, h in enumerate(heights):
        az = az0 + k * math.radians(137.5) + r.uniform(-0.08, 0.08)
        f = k / 7
        length = (0.085 - 0.035 * f) * r.uniform(0.92, 1.05)
        width = length * r.uniform(0.45, 0.52)
        stalk = 0.018 - 0.008 * f
        base = np.array([stalk * math.cos(az), stalk * math.sin(az), h])
        rise = 25 + 25 * f
        V, F = ovate_leaf(base, az, length, width, rise, droop=length * (0.30 - 0.15 * f), cup=0.004)
        out.append((V, F, base, az))
    return out


class Plant:
    """The leaves (world frame) as one triangle mesh, with the area, owner leaf and stain of every triangle."""

    def __init__(self):
        Vs, Fs, owner, off = [], [], [], 0
        self.stalks = []
        for k, (V, F, base, az) in enumerate(pepper()):
            Vs.append(V + PLANT_ORIGIN)
            Fs.append(F + off)
            owner += [k] * len(F)
            off += len(V)
            self.stalks.append((PLANT_ORIGIN + np.array([0.0, 0.0, base[2]]), PLANT_ORIGIN + base, az))
        self.V, self.F, self.owner = np.concatenate(Vs), np.concatenate(Fs), np.array(owner)
        a, b, c = (self.V[self.F[:, i]] for i in range(3))
        self.area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
        n = np.cross(b - a, c - a)
        self.normal = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        self.n_leaves = len(Vs)
        self.tagged = self.owner == TAG_INDEX
        # blocks of 24 triangles with their bounding boxes, for the droplet hit test
        self.blocks = []
        for s in range(0, len(self.F), 24):
            idx = np.arange(s, min(s + 24, len(self.F)))
            p = self.V[self.F[idx]].reshape(-1, 3)
            self.blocks.append((idx, p.min(0), p.max(0)))
        # points for the touch check: every vertex, edge midpoint and triangle centre (about 3 mm apart), the stem and
        # the stalks
        mids = np.concatenate([(a + b) / 2, (b + c) / 2, (c + a) / 2, (a + b + c) / 3])
        stem = np.array([PLANT_ORIGIN + np.array([STEM_R * math.cos(t), STEM_R * math.sin(t), z])
                         for z in np.arange(0.0, STEM_H, 0.004) for t in np.linspace(0, 2 * math.pi, 6, endpoint=False)])
        stalk = np.array([p0 + s * (p1 - p0) for p0, p1, _ in self.stalks for s in np.linspace(0, 1, 6)])
        self.touch_pts = np.concatenate([self.V, mids, stem, stalk])
        self.lo, self.hi = self.touch_pts.min(0), self.touch_pts.max(0)
        self.reset()

    def reset(self):
        self.stain = np.zeros(len(self.F))

    def wet(self):
        return 1 - np.exp(-self.stain / np.maximum(self.area, 1e-12))

    def tagged_wet_fraction(self) -> float:
        a = self.area[self.tagged]
        return float((a * self.wet()[self.tagged]).sum() / a.sum())

    def other_wet_cm2(self) -> float:
        m = ~self.tagged
        return float((self.area[m] * self.wet()[m]).sum() * 1e4)


PLANT = Plant()
TAG_LEAF_TOP = float(PLANT.V[PLANT.F[PLANT.tagged]].reshape(-1, 3)[:, 2].max())

# -- the card on its crate ----------------------------------------------------------------------------------------------
CARD_SIZE = (0.160, 0.200, 0.002)      # along its own x and y; it lies flat
CARD_KG = 0.015                        # 2 mm corrugated board (E-flute), about 0.45 kg/m^2
CARD_YAW = math.radians(-135.0)        # its x axis toward the arms' front right, like the tagged leaf
CRATE_SIZE = (0.12, 0.12, 0.14)
CRATE_XY = (0.46, 0.16)
CARD_START = (CRATE_XY[0], CRATE_XY[1], TOP + CRATE_SIZE[2] + CARD_SIZE[2] / 2 + 0.0005)
CARD = part("card", CARD_START, [((0.0, 0.0, 0.0), CARD_SIZE, CARDBOARD)], CARD_KG, yaw=CARD_YAW, color=CARDBOARD,
            friction=0.7)
CARD.meta["torsional_patch"] = 0.005    # a pinch on its edge resists twisting (as the curved-needle task's pinch)

# -- the spray bottle (from the spray-grip test) ----------------------------------------------------------------------
# The "sprayer frame": origin at the top of the neck, +z up the bottle's axis, +x toward the nozzle (the spray
# direction), +y = z x x. partkit's sliders move along the part's z axis, so the part frame is the sprayer frame
# turned: a point (x, y, z) in the sprayer frame is (z, -y, x) in the part frame. The trigger moves along part -z,
# i.e. backward toward the shroud, as a finger pulls it.
BOTTLE_BOT = -0.150
SHROUD_BACK_X = -0.018          # the back of the head, where the rear pad presses
TRIGGER_FACE_X = 0.034          # the trigger's finger face at rest
TRIGGER_TRAVEL = 0.012          # a full stroke (one high-output design: about 12.7 mm)
PINCH_Z = 0.030                 # height of the pinch line above the neck
NOZZLE_S = np.array([0.045, 0.0, 0.060])
TRIGGER_PRELOAD = 8.0           # N at rest (to be measured on a real bottle)
TRIGGER_FULL = 20.0             # N at full travel (patents: about 18-21 N at 30 strokes/min)
TRIGGER_RATE = (TRIGGER_FULL - TRIGGER_PRELOAD) / TRIGGER_TRAVEL   # N/m
BODY_COLOR = (0.82, 0.90, 0.95)
HEAD_COLOR = (0.95, 0.95, 0.93)
TRIGGER_COLOR = (0.15, 0.45, 0.80)
SPRAYER_BOXES = [
    ((0.0, 0.0, -0.0875), (0.055, 0.055, 0.125), BODY_COLOR),     # body (z -0.150 to -0.025)
    ((0.0, 0.0, -0.019), (0.036, 0.036, 0.012), BODY_COLOR),      # shoulder
    ((0.0, 0.0, -0.0065), (0.030, 0.030, 0.013), HEAD_COLOR),     # closure (28 mm cap)
    ((-0.002, 0.0, 0.0275), (0.032, 0.032, 0.055), HEAD_COLOR),   # shroud and pump (x -0.018 to +0.014)
    ((0.0295, 0.0, 0.059), (0.031, 0.016, 0.018), HEAD_COLOR),    # nozzle block above the trigger (to x +0.045)
]
TRIGGER_BOX_S = ((0.031, 0.0, 0.030), (0.006, 0.018, 0.036))      # the trigger plate (face at x +0.034)


def s2p(v) -> tuple[float, float, float]:
    x, y, z = (float(c) for c in v)
    return (z, -y, x)


def s2p_size(sz) -> tuple[float, float, float]:
    x, y, z = (float(c) for c in sz)
    return (z, y, x)


R_SP = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]]).T   # part axes in the sprayer frame (columns)
LIQUID_G = 237.0
INNER = 0.050


def _box_inertia(m, sx, sy, sz):
    return np.array([m * (sy * sy + sz * sz), m * (sx * sx + sz * sz), m * (sx * sx + sy * sy)]) / 12.0


def mass_properties():
    """Total mass (kg), centre of mass and diagonal inertia about it (sprayer frame), bottle upright (no slosh)."""
    pieces = []
    h = LIQUID_G * 1e-6 / (INNER * INNER)
    pieces.append((LIQUID_G / 1000, np.array([0.0, 0.0, BOTTLE_BOT + 0.002 + h / 2]),
                   _box_inertia(LIQUID_G / 1000, INNER, INNER, h)))
    pieces.append((0.020, np.array([0.0, 0.0, -0.080]), _box_inertia(0.020, 0.055, 0.055, 0.150) * 1.6))
    pieces.append((0.029, np.array([0.008, 0.0, 0.030]), _box_inertia(0.029, 0.050, 0.032, 0.060)))
    m = sum(p[0] for p in pieces)
    com = sum(p[0] * p[1] for p in pieces) / m
    inertia = np.zeros(3)
    for mi, ci, ii in pieces:
        d = ci - com
        inertia += ii + mi * np.array([d[1] ** 2 + d[2] ** 2, d[0] ** 2 + d[2] ** 2, d[0] ** 2 + d[1] ** 2])
    return m, com, inertia


MASS, COM_S, INERTIA_S = mass_properties()

BOTTLE_XY = (0.34, -0.40)
BOTTLE_YAW = math.radians(105.0)   # the nozzle's azimuth: to the arms' left, turned slightly toward them
R_WS_SPAWN = np.column_stack([[math.cos(BOTTLE_YAW), math.sin(BOTTLE_YAW), 0.0],
                              [-math.sin(BOTTLE_YAW), math.cos(BOTTLE_YAW), 0.0], [0.0, 0.0, 1.0]])
R_WP_SPAWN = R_WS_SPAWN @ R_SP
SPAWN_POS = (BOTTLE_XY[0], BOTTLE_XY[1], TOP - BOTTLE_BOT + 0.001)


def _matrix_to_xyzw(R) -> tuple[float, float, float, float]:
    from partkit import _matrix_wxyz

    w, x, y, z = _matrix_wxyz(R)
    return (x, y, z, w)


TRIGGER = Slider(name="trigger", rest=s2p(TRIGGER_BOX_S[0]),
                 boxes=((s2p(TRIGGER_BOX_S[0]), s2p_size(TRIGGER_BOX_S[1]), TRIGGER_COLOR),),
                 travel=TRIGGER_TRAVEL, rate=TRIGGER_RATE, damping=2.0, mass=0.006)
BOTTLE = part("sprayer", SPAWN_POS, [(s2p(c), s2p_size(sz), col) for c, sz, col in SPRAYER_BOXES], MASS,
              sliders=(TRIGGER,), color=HEAD_COLOR, friction=0.8)
BOTTLE.meta["spawn_quat_xyzw"] = _matrix_to_xyzw(R_WP_SPAWN)
BOTTLE.meta["com"] = s2p(COM_S)
BOTTLE.meta["inertia"] = s2p_size(INERTIA_S)
BOTTLE.meta["torsional_patch"] = 0.005

OBJECTS = [
    Cylinder("pot", (POT_XY[0], POT_XY[1], (TOP + SOIL) / 2), color=TERRACOTTA, radius=POT_R, height=SOIL - TOP),
    Cylinder("soil", (POT_XY[0], POT_XY[1], SOIL - 0.0035), color=SOIL_BROWN, radius=POT_R - 0.004, height=0.008),
    Box("crate", (CRATE_XY[0], CRATE_XY[1], TOP + CRATE_SIZE[2] / 2), color=WOOD, size=CRATE_SIZE, yaw=CARD_YAW),
    CARD,
    BOTTLE,
]

# -- the trigger spring's preload -------------------------------------------------------------------------------------
# Its drive target sits PRELOAD / RATE past the rest stop, so the stop holds the trigger with PRELOAD at rest. Isaac Lab
# writes joint targets every step; set it after every reset and every control step.
LOG = {"preload_set": 0, "errors": [], "squeezes": 0, "squeezes_dry": 0, "spray_ms": [], "draw_ms": []}


def _set_preload(environment) -> None:
    try:
        import torch

        asset = environment.unwrapped.scene["sprayer"]
        ids = asset.find_joints(["trigger_slide"])[0]
        target = torch.full((1, len(ids)), TRIGGER_PRELOAD / TRIGGER_RATE, device=environment.unwrapped.device)
        if hasattr(asset, "set_joint_position_target"):
            asset.set_joint_position_target(target, joint_ids=ids)
        else:
            asset.set_joint_position_target_index(target=target, joint_ids=ids)
        LOG["preload_set"] += 1
    except Exception as e:  # noqa: BLE001
        _err("preload", e)


def _err(where, e) -> None:
    if len(LOG["errors"]) < 30:
        LOG["errors"].append(f"{where} (step {_STATE.get('step')}): {e!r}"[:400])


# -- the spray ----------------------------------------------------------------------------------------------------------
CONE_DEG = 40.0
STROKE_ML = 1.0
STROKE_S = 0.2
EXIT_MS = 12.0          # m/s leaving the nozzle (not measured; orifice flow suggests 10-25 m/s)
VMD_UM, GSD = 200.0, 1.7
D_MIN_UM, D_MAX_UM = 40.0, 600.0
SPREAD = 2.0            # stain diameter / droplet diameter on a leaf
RHO_W, RHO_A, MU_A = 1000.0, 1.2, 1.8e-5
G = np.array([0.0, 0.0, -9.81])
N_PARCELS = int(os.environ.get("SPRAY_PARCELS", "700"))   # parcels per squeeze (scoring)
DRAW = 120              # of them drawn as beads
POOL = 1440             # bead slots (12 squeezes' worth; older beads are reused after that)
DROP_R = 0.0015         # bead radius (m): a stand-in for the dye's mark, far larger than a real droplet
FIRE_TRAVEL = 0.010     # a squeeze fires when the trigger passes 10 mm ...
REARM_TRAVEL = 0.004    # ... and the next one only after it has come back to 4 mm
UP_MIN = 0.25           # the bottle's axis must point at least this much up (75.5 degrees from upright at most)
SUB_DT = 2.5e-4
SEG_EVERY = 8           # hit tests on 2 ms path segments
MAX_FLIGHT_S = 1.5
HIDE = np.array([0.0, 0.0, -5.0])


def sprayer_rotation(bottle) -> np.ndarray:
    """The sprayer frame's axes in the world (columns), from the bottle part's state."""
    return bottle.rotation() @ R_SP.T


def nozzle_pose(bottle):
    Rs = sprayer_rotation(bottle)
    return bottle.pos + Rs @ NOZZLE_S, Rs[:, 0], Rs[:, 2]


def seg_tri(P0, P1, V, F):
    """First hit of each segment P0->P1 on the triangles (Moller-Trumbore): (fraction along it, triangle) or (inf, -1)."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    e1, e2 = b - a, c - a
    D = (P1 - P0)[:, None, :]
    O = P0[:, None, :]
    p = np.cross(D, e2[None])
    det = (e1[None] * p).sum(-1)
    ok = np.abs(det) > 1e-15
    inv = np.where(ok, 1.0 / np.where(ok, det, 1), 0)
    tv = O - a[None]
    u = (tv * p).sum(-1) * inv
    q = np.cross(tv, e1[None])
    v = (D * q).sum(-1) * inv
    t = (e2[None] * q).sum(-1) * inv
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t >= 0) & (t <= 1)
    t = np.where(hit, t, np.inf)
    k = np.argmin(t, axis=1)
    return t[np.arange(len(k)), k], np.where(np.isfinite(t[np.arange(len(k)), k]), k, -1)


def hit_plant(P0, P1):
    """seg_tri against the plant, testing each segment only against blocks whose bounding box its own overlaps."""
    lo, hi = np.minimum(P0, P1), np.maximum(P0, P1)
    tbest = np.full(len(P0), np.inf)
    ibest = np.full(len(P0), -1)
    for idx, blo, bhi in PLANT.blocks:
        m = np.all(lo <= bhi, axis=1) & np.all(hi >= blo, axis=1)
        if not m.any():
            continue
        sel = np.where(m)[0]
        t, k = seg_tri(P0[sel], P1[sel], PLANT.V, PLANT.F[idx])
        better = t < tbest[sel]
        tbest[sel[better]] = t[better]
        ibest[sel[better]] = idx[k[better]]
    return tbest, ibest


def hit_box(P0, P1, centre, R, half):
    """First entry of each segment into an oriented box (slab method): fraction along it, or inf."""
    a = (P0 - centre) @ R
    b = (P1 - centre) @ R
    d = b - a
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (-half - a) / d
        t2 = (half - a) / d
    tmin = np.where(np.abs(d) < 1e-15, np.where(np.abs(a) <= half, -np.inf, np.inf), np.minimum(t1, t2))
    tmax = np.where(np.abs(d) < 1e-15, np.where(np.abs(a) <= half, np.inf, -np.inf), np.maximum(t1, t2))
    enter, leave = tmin.max(1), tmax.min(1)
    ok = (enter <= leave) & (enter <= 1) & (leave >= 0)
    return np.where(ok, np.maximum(enter, 0.0), np.inf)


def droplet_sizes(n, rng):
    """Diameters (m) of n equal-volume parcels: log-normal by volume, median VMD_UM, spread GSD, clipped."""
    d = np.exp(math.log(VMD_UM) + math.log(GSD) * rng.standard_normal(n))
    return np.clip(d, D_MIN_UM, D_MAX_UM) * 1e-6


def cone_dirs(axis, n, rng):
    a = axis / np.linalg.norm(axis)
    tmp = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(a, tmp)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    cz = rng.uniform(math.cos(math.radians(CONE_DEG / 2)), 1.0, n)
    ph = rng.uniform(0, 2 * math.pi, n)
    sz = np.sqrt(1 - cz ** 2)
    return (sz * np.cos(ph))[:, None] * e1 + (sz * np.sin(ph))[:, None] * e2 + cz[:, None] * a


class Spray:
    """Every squeeze's parcels: waiting to leave the nozzle, flying, or stopped where they landed. Leaf hits add stain
    to the plant's triangles. Drawn parcels keep a bead slot; a bead on the card is kept in the card's frame, so it
    moves with the card."""

    def __init__(self, rng):
        self.rng = rng
        self.t = 0.0
        self.x = np.zeros((0, 3))
        self.v = np.zeros((0, 3))
        self.d = np.zeros(0)
        self.stain = np.zeros(0)
        self.t_emit = np.zeros(0)
        self.state = np.zeros(0, int)        # 0 waiting, 1 flying, 2 landed, 3 lost (left the scene or still flying)
        self.slot = np.zeros(0, int)         # bead slot, or -1
        self.beads = np.tile(HIDE, (POOL, 1))
        self.bead_on_card = np.zeros(POOL, bool)
        self.next_slot = 0
        self.exit_ms = np.zeros(0)
        self.dirs = np.zeros((0, 3))         # launch directions about +z, turned onto the nozzle's axis at launch
        self.landed_ul = {"tagged": 0.0, "other": 0.0, "card": 0.0, "soil": 0.0, "table": 0.0, "lost": 0.0}
        self.vol_each_ul = STROKE_ML * 1000.0 / N_PARCELS

    def squeeze(self):
        n = N_PARCELS
        d = droplet_sizes(n, self.rng)
        count = (self.vol_each_ul * 1e-9) / (math.pi / 6 * d ** 3)
        slots = np.full(n, -1)
        draw = (np.arange(POOL)[self.next_slot:self.next_slot + DRAW] if self.next_slot + DRAW <= POOL
                else (self.next_slot + np.arange(DRAW)) % POOL)
        slots[:DRAW] = draw
        self.slot[np.isin(self.slot, draw)] = -1          # reused slots: their old beads are no longer drawn
        self.beads[draw] = HIDE
        self.bead_on_card[draw] = False
        self.next_slot = (self.next_slot + DRAW) % POOL
        self.x = np.concatenate([self.x, np.tile(HIDE, (n, 1))])
        self.v = np.concatenate([self.v, np.zeros((n, 3))])
        self.d = np.concatenate([self.d, d])
        self.stain = np.concatenate([self.stain, count * math.pi * (SPREAD * d) ** 2 / 4])
        self.t_emit = np.concatenate([self.t_emit, self.t + self.rng.uniform(0, STROKE_S, n)])
        self.state = np.concatenate([self.state, np.zeros(n, int)])
        self.slot = np.concatenate([self.slot, slots])
        self.exit_ms = np.concatenate([self.exit_ms, EXIT_MS * self.rng.uniform(0.85, 1.15, n)])
        self.dirs = np.concatenate([self.dirs, cone_dirs(np.array([0, 0, 1.0]), n, self.rng)])

    def airborne(self) -> int:
        return int(((self.state == 0) | (self.state == 1)).sum())

    def advance(self, dt, nozzle, axis, card):
        """Fly everything for one control step. ``nozzle``/``axis``: the nozzle's pose this step (parcels leaving
        during it start there); ``card``: (centre, rotation, half sizes) or None."""
        t_end = self.t + dt
        if self.airborne() == 0:
            self.t = t_end
            return
        a = axis / np.linalg.norm(axis)
        tmp = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        e1 = np.cross(a, tmp)
        e1 /= np.linalg.norm(e1)
        Rl = np.column_stack([e1, np.cross(a, e1), a])
        seg_start = None
        k = 0
        while self.t < t_end - 1e-12:
            self.t += SUB_DT
            k += 1
            start = (self.state == 0) & (self.t_emit <= self.t)
            if start.any():
                self.state[start] = 1
                self.x[start] = nozzle
                self.v[start] = (self.dirs[start] @ Rl.T) * self.exit_ms[start, None]
                if seg_start is not None:
                    seg_start[start] = nozzle          # a parcel's first segment starts at the nozzle
            f = self.state == 1
            if not f.any():
                continue
            if seg_start is None:
                seg_start = self.x.copy()
            x, v, d = self.x[f], self.v[f], self.d[f]
            sp = np.linalg.norm(v, axis=1)
            re = np.maximum(RHO_A * sp * d / MU_A, 1e-9)
            cd = np.where(re < 1000, 24 / re * (1 + 0.15 * re ** 0.687), 0.44)
            v = v + (G - (0.75 * RHO_A / RHO_W) * (cd * sp / d)[:, None] * v) * SUB_DT
            self.v[f], self.x[f] = v, x + v * SUB_DT
            if k % SEG_EVERY == 0 or self.t >= t_end - 1e-12:
                self._land(np.where(f)[0], seg_start, card)
                seg_start = None
        self.t = t_end
        old = (self.state == 1) & (self.t - self.t_emit > MAX_FLIGHT_S)
        self.landed_ul["lost"] += self.vol_each_ul * old.sum()
        self.state[old] = 3

    def _land(self, idx, seg_start, card):
        P0, P1 = seg_start[idx], self.x[idx]
        th, tri = hit_plant(P0, P1)
        tc = hit_box(P0, P1, *card) if card is not None else np.full(len(idx), np.inf)
        dz = P0[:, 2] - P1[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            ts = np.where((P0[:, 2] >= SOIL) & (P1[:, 2] < SOIL), (P0[:, 2] - SOIL) / dz, np.inf)
            tt = np.where((P0[:, 2] >= TOP) & (P1[:, 2] < TOP), (P0[:, 2] - TOP) / dz, np.inf)
        ps = P0 + np.where(np.isfinite(ts), ts, 0)[:, None] * (P1 - P0)
        ts = np.where(np.hypot(ps[:, 0] - POT_XY[0], ps[:, 1] - POT_XY[1]) < POT_R, ts, np.inf)
        t_all = np.stack([th, tc, ts, tt], 1)
        which = np.argmin(t_all, 1)
        tmin = t_all[np.arange(len(idx)), which]
        hit = np.isfinite(tmin)
        for j in np.where(hit)[0]:
            i = idx[j]
            p = P0[j] + tmin[j] * (P1[j] - P0[j])
            kind = which[j]
            self.state[i] = 2
            self.x[i] = p
            if kind == 0:
                tr = tri[j]
                PLANT.stain[tr] += self.stain[i]
                self.landed_ul["tagged" if PLANT.tagged[tr] else "other"] += self.vol_each_ul
                n = PLANT.normal[tr]
                side = -1.0 if (P1[j] - P0[j]) @ n > 0 else 1.0
                bead = p + side * DROP_R * n
            elif kind == 1:
                self.landed_ul["card"] += self.vol_each_ul
                c, R, half = card
                local = R.T @ (p - c)
                local[2] = math.copysign(half[2] + DROP_R, local[2])
                bead = local                                   # kept in the card's frame
            elif kind == 2:
                self.landed_ul["soil"] += self.vol_each_ul
                bead = p + np.array([0, 0, DROP_R])
            else:
                self.landed_ul["table"] += self.vol_each_ul
                bead = p + np.array([0, 0, DROP_R])
            s = self.slot[i]
            if s >= 0:
                self.beads[s] = bead
                self.bead_on_card[s] = kind == 1
                self.slot[i] = -1                               # landed: the slot now holds a fixed bead

    def bead_positions(self, card):
        """World positions of every bead slot (flying beads where they are; landed ones where they stopped)."""
        out = self.beads.copy()
        if card is not None and self.bead_on_card.any():
            c, R, _ = card
            out[self.bead_on_card] = self.beads[self.bead_on_card] @ R.T + c
        fly = (self.state == 1) & (self.slot >= 0)
        out[self.slot[fly]] = self.x[fly]
        return out


# -- the touch check ----------------------------------------------------------------------------------------------------
ARMS = ("left_robot", "right_robot")
TOUCH_MARGIN = 0.001


def _origin(base):
    o = base.scene.env_origins[0]
    o = getattr(o, "torch", o)
    return o.detach().cpu().numpy().astype(np.float64) if hasattr(o, "detach") else np.asarray(o, dtype=np.float64)


def grippers():
    """For each arm: (TCP, approach axis, closing axis, (left finger travel, right finger travel)) in the objects'
    frame. Empty without the simulator."""
    environment = env()
    if environment is None:
        return {}
    try:
        from yam_arena.constants import FINGER_JOINTS, TCP_OFFSET
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
        tcp = rot @ np.asarray(TCP_OFFSET, float) + pos - o[:3]
        ids = robot.find_joints(FINGER_JOINTS, preserve_order=True)[0]
        q = getattr(robot.data.joint_pos, "torch", robot.data.joint_pos)[0, ids]
        q = np.abs(q.detach().cpu().numpy() if hasattr(q, "detach") else np.asarray(q, float))
        out[key] = (tcp, rot[:, 2], rot[:, 0], (float(q[0]), float(q[1])))
    return out


def gripper_boxes(g):
    """Oriented boxes (centre, rotation, half sizes) for one gripper: its two fingers and the housing behind them."""
    tcp, approach, close, (q7, q8) = g
    across = np.cross(approach, close)
    across /= np.linalg.norm(across) + 1e-12
    R = np.column_stack([close, across, approach])
    boxes = [(tcp - 0.109 * approach, R, np.array([0.065, 0.045, 0.030]))]
    for side, travel in ((-1.0, q7), (1.0, q8)):
        boxes.append((tcp + side * (travel + 0.004) * close - 0.018 * approach, R, np.array([0.004, 0.012, 0.031])))
    return boxes


def bottle_boxes(bottle):
    Rs = sprayer_rotation(bottle)
    out = [(bottle.pos + Rs @ np.asarray(c), Rs, np.asarray(sz) / 2) for c, sz, _ in SPRAYER_BOXES]
    travel = -float(bottle.sliders.get("trigger", 0.0))
    c, sz = TRIGGER_BOX_S
    out.append((bottle.pos + Rs @ (np.asarray(c) - np.array([travel, 0, 0])), Rs, np.asarray(sz) / 2))
    return out


def card_box(card):
    return (card.pos, card.rotation(), np.asarray(CARD_SIZE) / 2)


def touching(boxes) -> str | None:
    """The first box (by label) that comes within TOUCH_MARGIN of the plant's touch points, else None."""
    for label, (c, R, half) in boxes:
        ext = np.abs(R) @ half + TOUCH_MARGIN
        if np.any(c + ext < PLANT.lo) or np.any(c - ext > PLANT.hi):
            continue
        near = np.all(np.abs(PLANT.touch_pts - c) <= ext, axis=1)
        if not near.any():
            continue
        local = (PLANT.touch_pts[near] - c) @ R
        if np.any(np.all(np.abs(local) <= half + TOUCH_MARGIN, axis=1)):
            return label
    return None


# -- drawing --------------------------------------------------------------------------------------------------------------
_S: dict = {}


def _stage():
    try:
        from isaaclab.sim.utils import get_current_stage

        return get_current_stage()
    except Exception:  # noqa: BLE001
        import omni.usd

        return omni.usd.get_context().get_stage()


def _tube(p0, p1, r, n=6):
    """A closed-ended n-sided tube from p0 to p1 (vertices, triangles)."""
    d = p1 - p0
    d = d / np.linalg.norm(d)
    tmp = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(d, tmp)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    ring = [r * (math.cos(t) * e1 + math.sin(t) * e2) for t in np.linspace(0, 2 * math.pi, n, endpoint=False)]
    V = np.array([p0 + q for q in ring] + [p1 + q for q in ring])
    F = []
    for i in range(n):
        j = (i + 1) % n
        F += [[i, j, n + i], [j, n + j, n + i]]
    return V, np.array(F)


def tag_quad():
    """The flagging tape: a 10 x 30 mm strip hanging from the middle of the tagged leaf's stalk."""
    p0, p1, az = PLANT.stalks[TAG_INDEX]
    m = (p0 + p1) / 2
    d = np.array([math.cos(az), math.sin(az), 0.0])
    V = np.array([m - 0.005 * d, m + 0.005 * d, m + 0.005 * d - np.array([0, 0, 0.030]),
                  m - 0.005 * d - np.array([0, 0, 0.030])])
    return V, np.array([[0, 1, 2], [0, 2, 3]])


def _mesh(stage, path, V, F, color, o):
    from pxr import Gf, UsdGeom, Vt

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, p + o)) for p in V]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(F)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for i in F.reshape(-1)]))
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set(Vt.Vec3fArray([Gf.Vec3f(*color)]))
    try:
        import isaaclab.sim as sim_utils
        from isaaclab.sim.utils import bind_visual_material

        from yam_arena.station import linear_rgb

        look = path + "_look"
        cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(color), roughness=0.6)
        cfg.func(look, cfg)
        bind_visual_material(path, look, stage=stage)
    except Exception as e:  # noqa: BLE001
        _err("material", e)


def _build(environment) -> None:
    """Draw the plant (leaves, stem, stalks, tag) and the bead pool, once."""
    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

    stage = _stage()
    o = _origin(environment.unwrapped)[:3]
    _mesh(stage, "/World/Visuals/plant/leaves", PLANT.V, PLANT.F, LEAF_GREEN, o)
    Vs, Fs, off = [], [], 0
    for V, F in [_tube(PLANT_ORIGIN - np.array([0, 0, 0.005]), PLANT_ORIGIN + np.array([0, 0, STEM_H]), STEM_R)] + \
                [_tube(p0, p1, STALK_R) for p0, p1, _ in PLANT.stalks]:
        Vs.append(V)
        Fs.append(F + off)
        off += len(V)
    _mesh(stage, "/World/Visuals/plant/stem", np.concatenate(Vs), np.concatenate(Fs), STEM_GREEN, o)
    _mesh(stage, "/World/Visuals/plant/tag", *tag_quad(), TAG_PINK, o)
    drop = sim_utils.SphereCfg(radius=DROP_R, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=DYE_BLUE))
    _S["markers"] = VisualizationMarkers(VisualizationMarkersCfg(prim_path="/World/Visuals/spray", markers={"drop": drop}))
    _S["origin"] = o


def _draw(card) -> None:
    if "markers" not in _S:
        return
    t0 = time.perf_counter()
    pos = _S["spray"].bead_positions(card) + _S["origin"]
    _S["markers"].visualize(translations=pos.astype(np.float32), marker_indices=np.zeros(POOL, dtype=np.int64))
    LOG["draw_ms"].append(1000 * (time.perf_counter() - t0))
    del LOG["draw_ms"][:-200]


# -- per-step state and scoring -----------------------------------------------------------------------------------------
TARGET_WET = 0.80        # the tagged leaf's top: 80% or more wet
OTHER_WET_CM2 = 1.0      # other leaves: 1 cm^2 wet in all, at most
DEFAULT_DT = 1 / 20
TRACE_PATH = os.path.expanduser(os.environ["SPRAY_TRACE"]) if os.environ.get("SPRAY_TRACE") else None
_STATE: dict = {}


def _dt() -> float:
    environment = env()
    try:
        return float(environment.unwrapped.step_dt) if environment is not None else DEFAULT_DT
    except Exception:  # noqa: BLE001
        return DEFAULT_DT


def reset_state(seed=None) -> None:
    PLANT.reset()
    _STATE.clear()
    _STATE.update(step=0, scored=None, last=(0.0, False), armed=True, touched=None, touched_s=None,
                  max_travel=0.0, squeeze_s=[])
    _S["spray"] = Spray(np.random.default_rng(int(os.environ.get("SPRAY_SEED", "5")) if seed is None else seed))
    LOG.update(squeezes=0, squeezes_dry=0)


def spray_step(bottle, card_state, dt) -> None:
    """Advance the spray by one control step: fire a squeeze if the trigger has just passed FIRE_TRAVEL, then fly."""
    travel = -float(bottle.sliders.get("trigger", 0.0))
    _STATE["max_travel"] = max(_STATE["max_travel"], travel)
    nozzle, axis, up = nozzle_pose(bottle)
    sp = _S["spray"]
    if _STATE["armed"] and travel >= FIRE_TRAVEL:
        _STATE["armed"] = False
        if up[2] >= UP_MIN:
            sp.squeeze()
            LOG["squeezes"] += 1
            _STATE["squeeze_s"].append(round(_STATE["step"] * dt, 2))
        else:
            LOG["squeezes_dry"] += 1
    elif not _STATE["armed"] and travel <= REARM_TRAVEL:
        _STATE["armed"] = True
    t0 = time.perf_counter()
    sp.advance(dt, nozzle, axis, card_state)
    LOG["spray_ms"].append(1000 * (time.perf_counter() - t0))
    del LOG["spray_ms"][:-200]


def check_touch(bottle, card, gs) -> None:
    if _STATE["touched"]:
        return
    boxes = [("the bottle", b) for b in bottle_boxes(bottle)] + [("the card", card_box(card))]
    for key, g in gs.items():
        boxes += [(f"the {key.split('_')[0]} gripper", b) for b in gripper_boxes(g)]
    who = touching(boxes)
    if who:
        _STATE["touched"] = f"{who} touched the plant"
        _STATE["touched_s"] = round(_STATE["step"] * _dt(), 2)


def evaluate() -> tuple[float, bool]:
    """The score so far and whether the task is done, from the plant's stains and the touch record."""
    wet = PLANT.tagged_wet_fraction()
    other = PLANT.other_wet_cm2()
    if _STATE["touched"] or other > OTHER_WET_CM2:
        return 0.0, False
    done = wet >= TARGET_WET and _S["spray"].airborne() == 0
    return min(1.0, wet / TARGET_WET), done


@on_reset
def _on_reset(environment=None) -> None:
    try:
        reset_state()
        if environment is not None:
            _set_preload(environment)
            if "markers" not in _S:
                _build(environment)
            states = _states(environment)
            _draw(card_box(states["card"]) if states else None)
    except Exception as e:  # noqa: BLE001
        _err("reset", e)


def _states(environment):
    try:
        from yam_arena.scene import object_states

        return object_states(environment, ["sprayer", "card"])
    except Exception as e:  # noqa: BLE001
        _err("states", e)
        return None


@on_step
def _on_step(environment=None) -> None:
    _STATE["step"] += 1
    if environment is None:
        return
    _set_preload(environment)
    states = _states(environment)
    if not states:
        return
    try:
        card = card_box(states["card"])
        spray_step(states["sprayer"], card, _dt())
        check_touch(states["sprayer"], states["card"], grippers())
        _draw(card)
        if TRACE_PATH:
            sp = _S["spray"]
            row = {"step": _STATE["step"], "trigger_mm": round(-1000 * states["sprayer"].sliders.get("trigger", 0.0), 2),
                   "squeezes": LOG["squeezes"], "airborne": sp.airborne(),
                   "tagged_wet": round(PLANT.tagged_wet_fraction(), 4), "other_cm2": round(PLANT.other_wet_cm2(), 4),
                   "landed_ul": {k: round(v, 1) for k, v in sp.landed_ul.items()}, "touched": _STATE["touched"],
                   "spray_ms": round(LOG["spray_ms"][-1], 1) if LOG["spray_ms"] else None}
            with open(TRACE_PATH, "a") as f:
                f.write(json.dumps(row) + "\n")
    except Exception as e:  # noqa: BLE001
        _err("step", e)


reset_state()


def _update():
    if _STATE["scored"] == _STATE["step"]:
        return _STATE["last"]
    _STATE["scored"] = _STATE["step"]
    _STATE["last"] = evaluate()
    return _STATE["last"]


def success(objects) -> bool:
    """The tagged leaf's top is 80% or more wet, the other leaves have 1 cm^2 of spray or less, no spray is still in
    the air, and nothing has touched the plant."""
    return _update()[1]


def score(objects) -> float:
    """0 if the bottle, a gripper or the card touched the plant, or the other leaves got more than 1 cm^2 of spray.
    Otherwise the tagged leaf's wet fraction / 0.8, up to 1.0."""
    return _update()[0]


def report() -> dict:
    """A summary for the demo and the tests."""
    sp = _S["spray"]
    return {"score": _update()[0], "tagged_wet": round(PLANT.tagged_wet_fraction(), 4),
            "other_wet_cm2": round(PLANT.other_wet_cm2(), 4), "squeezes": LOG["squeezes"],
            "squeezes_dry": LOG["squeezes_dry"], "squeeze_s": _STATE["squeeze_s"], "touched": _STATE["touched"],
            "touched_s": _STATE["touched_s"], "airborne": sp.airborne(),
            "landed_ul": {k: round(v, 1) for k, v in sp.landed_ul.items()},
            "max_trigger_mm": round(1000 * _STATE["max_travel"], 2), "errors": LOG["errors"][:10],
            "spray_ms_max": round(max(LOG["spray_ms"]), 1) if LOG["spray_ms"] else None}

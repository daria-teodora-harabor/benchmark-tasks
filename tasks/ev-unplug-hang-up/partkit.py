"""partkit: rigid parts built from several boxes, optionally with spring-loaded sliders, for yam_arena task files.

A task file uses it like this (keep this file next to the task file):

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from partkit import Part, Slider, on_step, on_reset

    LID = Part("lid", pos=(0.2, 0.0, 0.76), yaw=0.0, mass=0.04,
               boxes=[((0, 0, 0), (0.114, 0.064, 0.0075), (0.1, 0.3, 0.9))])
    OBJECTS = [LID, ...]

yam_arena's task-file objects are single primitives. A ``Part`` is one rigid body with several box and cylinder colliders
(a box with a pocket, a lid with tabs, a cap with lugs). Give it ``sliders`` and it becomes an articulation whose root is the part and whose
sliders are boxes on prismatic joints along the part's z axis, each pushed up to its rest stop by a spring drive.
A box may carry a fourth element, its rotation in the part frame as a (w, x, y, z) quaternion, for chamfers and
ramps; a slider may have its own ``friction``, which wins its contacts (PhysX's "min" combine mode).

Importing this module hooks ``yam_arena.scene`` (and the embodiment's step and reset) without modifying yam_arena: ``success()/score()`` receive a :class:`PartState` (the body origin's pose, not the bounding-box centre that
yam_arena reports for primitives, plus slider positions). ``on_step`` and ``on_reset`` register callbacks that run
after every control step and after every reset, with the Isaac Lab environment.

Pure Python at import time; everything that needs Isaac Sim is imported lazily.

This copy is the din-rail-mount task's partkit with three additions:

- ``Part.meta["torsional_patch"]`` (m) sets a torsional patch radius on the part's colliders, as lock-before-loading's
  scenekit does for its pin, so a part held in a pinch resists twisting (as in the curved-needle task).
- ``Part(cable=CableSpec(...))`` hangs a jointed cable off the part: a chain of capsule segments, built the way the
  accepted cable tasks' cablekit builds its cables (joints that bend about two axes but neither stretch nor twist, each
  with a spring-damper drive), but as further links of the part's own articulation, so the cable is truly attached to
  the part. Its far end can be welded to the world (a cable running into a charger post). The cable is laid out along
  a polyline at the start; each joint's zero is its starting bend, so the drives hold that lay (a heavy cable's set).
- ``Flap`` is a hinged flap on a fixed frame (a door): an articulation whose root is welded to the world and whose
  flap turns on a revolute joint about the vertical, with joint friction and damping and no spring. The flap is authored
  at its starting angle and the joint's zero is that angle, so it starts there whether or not Isaac Lab applies an
  initial joint position (a first simulator check found a door left at the joint's zero).
- ``sprung_slide`` is the same fixed-frame articulation with a body on a prismatic joint along x and a stiff spring
  holding it at its stop: a force gauge. Its travel times its spring rate is the force on it.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import math
import sys
from dataclasses import dataclass, field

import numpy as np
from yam_arena import task_file
from yam_arena.task_file import Box, ObjectState

__all__ = ["CableSpec", "Flap", "FlapState", "Part", "PartState", "Slider", "flap", "sprung_slide", "link_load",
           "on_reset", "on_step", "env", "write_pose", "install"]

Color = tuple[float, float, float]
# centre, size, colour (part frame, m), and optionally the box's rotation in the part frame as a (w, x, y, z) quaternion
BoxSpec = tuple
CylSpec = tuple[tuple[float, float, float], float, float, Color]  # centre, radius, height (axis = part z), colour


@dataclass(frozen=True)
class Slider:
    """A box on a prismatic joint along the part's z axis. ``rest`` is where its frame sits in the part frame; it can
    move down by ``travel`` and a spring of ``rate`` N/m pushes it back up to ``rest``."""

    name: str
    rest: tuple[float, float, float]
    boxes: tuple[BoxSpec, ...]
    travel: float = 0.004
    rate: float = 2500.0
    damping: float = 20.0
    mass: float = 0.002
    friction: float | None = None  # its own material, which wins its contacts; None uses the part's


@dataclass(frozen=True)
class CableSpec:
    """A jointed cable leaving a part. ``points`` is a polyline in the part's frame at the start, from where the cable
    leaves the part to its far end; the cable is cut into segments ``segment`` m long along it."""

    points: tuple[tuple[float, float, float], ...]
    radius: float = 0.007
    segment: float = 0.04
    kg_per_m: float = 0.3
    bend_stiffness: float = 0.3      # N*m/rad per joint drive (about the joint's starting bend)
    bend_damping: float = 0.02       # N*m*s/rad
    armature: float = 2e-5           # kg*m^2 per bend DOF (keeps the light links stable)
    max_bend: float = 1.4            # rad per joint (limit, either side of the starting bend)
    friction: float = 0.8
    color: Color = (0.08, 0.08, 0.08)
    anchor_end: bool = True          # weld the far end to the world

    def layout(self):
        """Segment centres, unit directions and the segment length, (n, 3), (n, 3), float, in the part's frame."""
        pts = np.asarray(self.points, float)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        total = float(seg.sum())
        n = max(2, round(total / self.segment))
        L = total / n
        cum = np.r_[0.0, np.cumsum(seg)]

        def at(s):
            i = min(int(np.searchsorted(cum, s, side="right")) - 1, len(seg) - 1)
            return pts[i] + (s - cum[i]) / seg[i] * (pts[i + 1] - pts[i])
        ends = np.array([at(k * L) for k in range(n + 1)])
        centres = (ends[:-1] + ends[1:]) / 2
        dirs = ends[1:] - ends[:-1]
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        return centres, dirs, L


def _rot_x_to(d) -> np.ndarray:
    """A rotation taking +x onto the unit vector ``d`` (no roll about it beyond what is needed)."""
    d = np.asarray(d, float) / np.linalg.norm(d)
    x = np.array([1.0, 0.0, 0.0])
    v = np.cross(x, d)
    c = float(x @ d)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([-1.0, -1.0, 1.0])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * (1 / (1 + c))


def _matrix_wxyz(R) -> tuple[float, float, float, float]:
    tr = np.trace(R)
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * S, (R[2, 1] - R[1, 2]) / S, (R[0, 2] - R[2, 0]) / S, (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / S, 0.25 * S, (R[0, 1] + R[1, 0]) / S, (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / S, (R[0, 1] + R[1, 0]) / S, 0.25 * S, (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / S, (R[0, 2] + R[2, 0]) / S, (R[1, 2] + R[2, 1]) / S, 0.25 * S
    return (float(w), float(x), float(y), float(z))


@dataclass(frozen=True)
class Part(Box):
    """A rigid part made of ``boxes``. It subclasses Box only so yam_arena's task-file checks accept it; ``size`` is
    its bounding box and ``pos`` is the body origin."""

    boxes: tuple[BoxSpec, ...] = ()
    cylinders: tuple[CylSpec, ...] = ()
    sliders: tuple[Slider, ...] = ()
    friction: float = 0.8
    cable: CableSpec | None = None
    meta: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def articulated(self) -> bool:
        return bool(self.sliders) or self.cable is not None

    def _check(self) -> None:
        if not self.boxes:
            raise ValueError(f"{self.name}: a Part needs at least one box")
        if self.mass is None or self.mass <= 0:
            raise ValueError(f"{self.name}: a Part must be loose (give it a mass)")


def _quat_matrix(wxyz) -> np.ndarray:
    w, x, y, z = wxyz
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _box(spec):
    """(centre, size, colour, rotation wxyz) of a box spec, with no rotation when it has none."""
    centre, size, color = spec[:3]
    return centre, size, color, (tuple(spec[3]) if len(spec) > 3 else (1.0, 0.0, 0.0, 0.0))


def part(name, pos, boxes, mass, yaw=0.0, sliders=(), color=(0.5, 0.5, 0.5), friction=0.8, cylinders=(),
         cable=None) -> Part:
    """Build a Part; ``size`` is filled in from the shapes' extent."""
    half = [np.abs(_quat_matrix(q)) @ (np.array(s) / 2) for _, s, _, q in map(_box, boxes)]
    half += [np.array([r, r, h / 2]) for _, r, h, _ in cylinders]
    centres = [np.array(b[0]) for b in boxes] + [np.array(c) for c, _, _, _ in cylinders]
    lo = np.min([c - h for c, h in zip(centres, half)], axis=0)
    hi = np.max([c + h for c, h in zip(centres, half)], axis=0)
    return Part(name=name, pos=tuple(pos), size=tuple((hi - lo).tolist()), mass=mass, color=color, yaw=yaw,
                boxes=tuple(boxes), cylinders=tuple(cylinders), sliders=tuple(sliders), friction=friction,
                cable=cable)


@dataclass(frozen=True)
class Flap(Box):
    """A flap hinged about a vertical axis on a frame fixed to the world (a door). ``pos`` is the hinge point (world);
    the flap's boxes are given in the hinge frame with the flap closed (angle 0); it starts at ``open`` rad within
    [``lower``, ``upper``]. In the simulator the joint's zero is the starting angle (see the module docstring); the
    state reports the angle from closed. It subclasses Box only so yam_arena's task-file checks accept it.

    With ``kind="slide"`` (see ``sprung_slide``) the body moves along x instead, within [``lower``, ``upper``] metres
    (0 its rest stop), and ``spring`` N/m pulls it back to 0."""

    flap_boxes: tuple[BoxSpec, ...] = ()
    frame_boxes: tuple[BoxSpec, ...] = ()
    flap_mass: float = 0.05
    open: float = 0.0
    lower: float = 0.0
    upper: float = math.radians(100)
    friction_torque: float = 0.03    # N*m of joint friction: the flap stays where it is pushed
    damping: float = 0.01            # N*m*s/rad
    friction: float = 0.6
    kind: str = "hinge"              # "hinge" (revolute about z) or "slide" (prismatic along x)
    spring: float = 0.0              # the joint's spring rate toward 0 (N/m for a slide)

    def _check(self) -> None:
        if not self.flap_boxes:
            raise ValueError(f"{self.name}: a Flap needs flap boxes")
        if self.kind not in ("hinge", "slide"):
            raise ValueError(f"{self.name}: kind must be 'hinge' or 'slide'")

    @property
    def joint(self) -> str:
        return "hinge" if self.kind == "hinge" else "slide"

    @property
    def free(self) -> bool:  # never jittered or teleported by yam_arena: its frame is welded to the world
        return False


def flap(name, hinge, flap_boxes, frame_boxes=(), *, open_angle=0.0, lower=0.0, upper=math.radians(100),
         flap_mass=0.05, color=(0.5, 0.5, 0.5), friction_torque=0.03, damping=0.01) -> Flap:
    """Build a Flap hinged at ``hinge`` (world point, vertical axis)."""
    return Flap(name=name, pos=tuple(float(v) for v in hinge), size=(0.01, 0.01, 0.01), color=color,
                flap_boxes=tuple(flap_boxes), frame_boxes=tuple(frame_boxes), flap_mass=flap_mass,
                open=float(open_angle), lower=float(lower), upper=float(upper), friction_torque=friction_torque,
                damping=damping)


def sprung_slide(name, pos, boxes, frame_boxes=(), *, travel=0.004, rate=20000.0, damping=40.0, mass=0.02,
                 color=(0.5, 0.5, 0.5), friction=0.6) -> Flap:
    """A force gauge: a body at ``pos`` (world) on a frame fixed to the world, on a prismatic joint along x. A spring of
    ``rate`` N/m holds it at its stop (0); it can be pushed toward -x by up to ``travel`` m. Boxes are relative to
    ``pos``. The force pushing it is ``rate`` times how far it has moved (FlapState.travel)."""
    return Flap(name=name, pos=tuple(float(v) for v in pos), size=(0.01, 0.01, 0.01), color=color,
                flap_boxes=tuple(boxes), frame_boxes=tuple(frame_boxes), flap_mass=mass, open=0.0,
                lower=-float(travel), upper=0.0, friction_torque=0.0, damping=damping, friction=friction,
                kind="slide", spring=rate)


@dataclass(frozen=True)
class FlapState(ObjectState):
    """A flap's hinge angle (rad from closed) or, for a slide, how far it has moved from its stop (``travel``, m,
    negative toward -x), and its frame's pose."""

    angle: float = 0.0
    travel: float = 0.0


@dataclass(frozen=True)
class PartState(ObjectState):
    """The body origin's pose (``pos``, ``quat_xyzw``), velocities, bounds, and each slider's position (m, 0 at its
    rest stop, negative when pressed)."""

    sliders: dict = field(default_factory=dict)
    cable: np.ndarray | None = None  # (n, 3) cable segment centres, from the part outward (parts with a cable)

    def rotation(self) -> np.ndarray:
        x, y, z, w = self.quat_xyzw
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                         [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                         [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])

    def to_local(self, world_point) -> np.ndarray:
        return self.rotation().T @ (np.asarray(world_point, float) - self.pos)

    def to_world(self, local_point) -> np.ndarray:
        return self.pos + self.rotation() @ np.asarray(local_point, float)


# -- spawning -----------------------------------------------------------------------------------------------------

_SPECS: dict[str, Part] = {}


def _spawn_part(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    """Author the part under ``prim_path`` (pxr only): a plain Xform holding the body (and sliders) as siblings."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material, bind_visual_material
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

    from yam_arena.station import linear_rgb

    spec: Part = _SPECS[cfg.part_name]
    stage = get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    x, y, z, w = orientation if orientation is not None else (0.0, 0.0, 0.0, 1.0)
    root.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, x, y, z))

    phys = f"{prim_path}/Looks/physics"
    pm = sim_utils.RigidBodyMaterialCfg(static_friction=spec.friction, dynamic_friction=spec.friction)
    pm.func(phys, pm)
    materials = {}
    for s in spec.sliders:
        if s.friction is not None:
            materials[s.name] = f"{prim_path}/Looks/physics_{s.name}"
            sm = sim_utils.RigidBodyMaterialCfg(static_friction=s.friction, dynamic_friction=s.friction,
                                                friction_combine_mode="min")
            sm.func(materials[s.name], sm)
    looks: dict[Color, str] = {}

    def look(color: Color) -> str:
        if color not in looks:
            path = f"{prim_path}/Looks/c{len(looks)}"
            vis = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(color), roughness=0.5)
            vis.func(path, vis)
            looks[color] = path
        return looks[color]

    def body(path: str, pos, mass: float):
        xform = UsdGeom.Xform.Define(stage, path)
        xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*pos))
        xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1, 0, 0, 0))
        prim = xform.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rb.CreateMaxDepenetrationVelocityAttr(1.0)
        rb.CreateSolverPositionIterationCountAttr(32)
        rb.CreateAngularDampingAttr(1.0)  # light damping: small stiff parts otherwise chatter in tight fits
        return prim

    torsional = spec.meta.get("torsional_patch")  # m: lets a pinch resist twisting (as lock-before-loading's pin)

    def collide(prim, color, material=phys):
        UsdPhysics.CollisionAPI.Apply(prim)
        cp = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        cp.CreateContactOffsetAttr(0.0005)  # below the 1 mm fits, so tight parts do not touch on every face at once
        cp.CreateRestOffsetAttr(0.0)
        if torsional:
            cp.CreateTorsionalPatchRadiusAttr(float(torsional))
            cp.CreateMinTorsionalPatchRadiusAttr(0.8 * float(torsional))
        bind_physics_material(str(prim.GetPath()), material, stage=stage)
        bind_visual_material(str(prim.GetPath()), look(color), stage=stage)

    def boxes(path: str, specs, offset=(0.0, 0.0, 0.0), material=phys):
        for i, (centre, size, color, rot) in enumerate(map(_box, specs)):
            cube = UsdGeom.Cube.Define(stage, f"{path}/box_{i}")
            cube.CreateSizeAttr(1.0)
            cube.AddTranslateOp().Set(Gf.Vec3d(*(np.array(centre) - np.array(offset))))
            if rot != (1.0, 0.0, 0.0, 0.0):
                cube.AddOrientOp().Set(Gf.Quatf(*rot))
            cube.AddScaleOp().Set(Gf.Vec3f(*size))
            collide(cube.GetPrim(), color, material)

    def cylinders(path: str, specs):
        for i, (centre, radius, height, color) in enumerate(specs):
            cyl = UsdGeom.Cylinder.Define(stage, f"{path}/cyl_{i}")
            cyl.CreateAxisAttr("Z")
            cyl.CreateRadiusAttr(radius)
            cyl.CreateHeightAttr(height)
            cyl.AddTranslateOp().Set(Gf.Vec3d(*centre))
            collide(cyl.GetPrim(), color)

    main = f"{prim_path}/body"
    body(main, (0.0, 0.0, 0.0), spec.mass)
    boxes(main, spec.boxes)
    cylinders(main, spec.cylinders)
    for s in spec.sliders:
        path = f"{prim_path}/{s.name}"
        body(path, s.rest, s.mass)
        boxes(path, s.boxes, offset=s.rest, material=materials.get(s.name, phys))
        j = UsdPhysics.PrismaticJoint.Define(stage, f"{path}/{s.name}_slide")
        j.CreateAxisAttr("Z")
        j.CreateBody0Rel().SetTargets([Sdf.Path(main)])
        j.CreateBody1Rel().SetTargets([Sdf.Path(path)])
        j.CreateLocalPos0Attr(Gf.Vec3f(*s.rest))
        j.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
        j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
        j.CreateLowerLimitAttr(-s.travel)
        j.CreateUpperLimitAttr(0.0)
        drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(s.rate)
        drive.CreateDampingAttr(s.damping)
        drive.CreateTargetPositionAttr(0.0)
    if spec.cable is not None:
        _author_cable(stage, prim_path, main, spec.cable, collide, look)
    if spec.articulated:
        UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(main))
        art = PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(main))
        art.CreateEnabledSelfCollisionsAttr(False)
        art.CreateSolverPositionIterationCountAttr(32)
        art.CreateSolverVelocityIterationCountAttr(4)
    return root.GetPrim()


def _author_cable(stage, prim_path, main, cs: CableSpec, collide, look):
    """The cable's segments as further links of the part's articulation (pxr only): seg_0 hangs off the part at the
    polyline's start; each joint's zero is the starting bend; the last segment's far end is welded to the world."""
    import isaaclab.sim as sim_utils
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

    centres, dirs, L = cs.layout()
    r = cs.radius
    mat = f"{prim_path}/Looks/cable_physics"
    cm = sim_utils.RigidBodyMaterialCfg(static_friction=cs.friction, dynamic_friction=cs.friction)
    cm.func(mat, cm)
    rots = [_rot_x_to(d) for d in dirs]
    paths = []
    for i, (c, R) in enumerate(zip(centres, rots)):
        path = f"{prim_path}/seg_{i}"
        xf = UsdGeom.Xform.Define(stage, path)
        xf.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(float, c)))
        xf.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(*_matrix_wxyz(R)))
        prim = xf.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(cs.kg_per_m * L)
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rb.CreateMaxDepenetrationVelocityAttr(1.0)
        rb.CreateSolverPositionIterationCountAttr(32)
        rb.CreateLinearDampingAttr(0.5)
        rb.CreateAngularDampingAttr(2.0)
        cap = UsdGeom.Capsule.Define(stage, f"{path}/geom")
        cap.CreateAxisAttr("X")
        cap.CreateRadiusAttr(r)
        cap.CreateHeightAttr(max(L - 2 * r, 1e-4) + 2 * r * 0.9)  # slight overlap: no visible gaps when bent
        collide(cap.GetPrim(), cs.color, mat)
        paths.append(path)

    def bend_joint(parent, child, name, pos0, rot0):
        j = UsdPhysics.Joint.Define(stage, f"{child}/{name}")
        j.CreateBody0Rel().SetTargets([Sdf.Path(parent)])
        j.CreateBody1Rel().SetTargets([Sdf.Path(child)])
        j.CreateLocalPos0Attr(Gf.Vec3f(*map(float, pos0)))
        j.CreateLocalRot0Attr(Gf.Quatf(*_matrix_wxyz(rot0)))
        j.CreateLocalPos1Attr(Gf.Vec3f(-L / 2, 0, 0))
        j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
        prim = j.GetPrim()
        for dof in ("transX", "transY", "transZ", "rotX"):  # no stretch, no twist
            lim = UsdPhysics.LimitAPI.Apply(prim, dof)
            lim.CreateLowAttr(1.0)
            lim.CreateHighAttr(-1.0)
        for dof in ("rotY", "rotZ"):
            lim = UsdPhysics.LimitAPI.Apply(prim, dof)
            lim.CreateLowAttr(-math.degrees(cs.max_bend))
            lim.CreateHighAttr(math.degrees(cs.max_bend))
            drv = UsdPhysics.DriveAPI.Apply(prim, dof)
            drv.CreateTypeAttr("force")
            drv.CreateStiffnessAttr(cs.bend_stiffness * math.pi / 180)  # USD angular gains are per degree
            drv.CreateDampingAttr(cs.bend_damping * math.pi / 180)
            drv.CreateTargetPositionAttr(0.0)

    start = np.asarray(cs.points[0], float)
    bend_joint(main, paths[0], "bend_0", start, rots[0])
    for i in range(1, len(paths)):
        bend_joint(paths[i - 1], paths[i], f"bend_{i}", (L / 2, 0.0, 0.0), rots[i - 1].T @ rots[i])
    if cs.anchor_end:
        last = stage.GetPrimAtPath(paths[-1])
        world = UsdGeom.Xformable(last).ComputeLocalToWorldTransform(0)
        end = world.Transform(Gf.Vec3d(L / 2, 0.0, 0.0))
        q = world.ExtractRotationQuat()
        fj = UsdPhysics.FixedJoint.Define(stage, f"{prim_path}/cable_anchor")
        fj.CreateBody1Rel().SetTargets([Sdf.Path(paths[-1])])
        fj.CreateLocalPos0Attr(Gf.Vec3f(*end))
        fj.CreateLocalRot0Attr(Gf.Quatf(q.GetReal(), *q.GetImaginary()))
        fj.CreateLocalPos1Attr(Gf.Vec3f(L / 2, 0, 0))
        fj.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))


def _arena_object(spec: Part):
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    @configclass
    class PartSpawnerCfg(SpawnerCfg):
        func: object = clone(_spawn_part)
        part_name: str = ""

    class PartObject(Object):
        def _generate_articulation_cfg(self):
            # Implicit actuators carry the springs; their default target is each slider's rest stop.
            actuators = {s.name: ImplicitActuatorCfg(joint_names_expr=[f"{s.name}_slide"], stiffness=s.rate,
                                                     damping=s.damping) for s in spec.sliders}
            if spec.cable is not None:
                actuators["cable_bend"] = ImplicitActuatorCfg(joint_names_expr=["bend_.*"],
                                                              stiffness=spec.cable.bend_stiffness,
                                                              damping=spec.cable.bend_damping,
                                                              armature=spec.cable.armature)
            cfg = ArticulationCfg(prim_path=self.prim_path, spawn=self.spawner_cfg, actuators=actuators)
            return self._add_initial_pose_to_cfg(cfg)

    kind = ObjectType.ARTICULATION if spec.articulated else ObjectType.RIGID
    return PartObject(
        name=spec.name,
        object_type=kind,
        spawner_cfg=PartSpawnerCfg(part_name=spec.name),
        initial_pose=Pose(position_xyz=tuple(spec.pos), rotation_xyzw=(0.0, 0.0, math.sin(spec.yaw / 2),
                                                                       math.cos(spec.yaw / 2))),
    )


def _spawn_flap(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    """Author a flap (pxr only): the frame (root link, welded to the world) and the flap on a vertical hinge (or, for
    a slide, the body on a prismatic joint along x)."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material, bind_visual_material
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

    from yam_arena.station import linear_rgb

    spec: Flap = _SPECS[cfg.part_name]
    stage = get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    x, y, z, w = orientation if orientation is not None else (0.0, 0.0, 0.0, 1.0)
    root.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, x, y, z))
    phys = f"{prim_path}/Looks/physics"
    pm = sim_utils.RigidBodyMaterialCfg(static_friction=spec.friction, dynamic_friction=spec.friction)
    pm.func(phys, pm)
    looks: dict = {}

    def look(color):
        if color not in looks:
            path = f"{prim_path}/Looks/c{len(looks)}"
            vis = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(color), roughness=0.35)
            vis.func(path, vis)
            looks[color] = path
        return looks[color]

    def link(name, mass, specs, quat_wxyz=(1.0, 0.0, 0.0, 0.0)):
        path = f"{prim_path}/{name}"
        xf = UsdGeom.Xform.Define(stage, path)
        xf.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0, 0, 0))
        xf.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(*quat_wxyz))
        prim = xf.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateSolverPositionIterationCountAttr(32)
        for i, (centre, size, color, rot) in enumerate(map(_box, specs)):
            cube = UsdGeom.Cube.Define(stage, f"{path}/box_{i}")
            cube.CreateSizeAttr(1.0)
            cube.AddTranslateOp().Set(Gf.Vec3d(*centre))
            if rot != (1.0, 0.0, 0.0, 0.0):
                cube.AddOrientOp().Set(Gf.Quatf(*rot))
            cube.AddScaleOp().Set(Gf.Vec3f(*size))
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            cp = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
            cp.CreateContactOffsetAttr(0.001)
            cp.CreateRestOffsetAttr(0.0)
            bind_physics_material(str(cube.GetPath()), phys, stage=stage)
            bind_visual_material(str(cube.GetPath()), look(color), stage=stage)
        return path

    frame_specs = spec.frame_boxes or (((0.0, 0.0, 0.0), (0.004, 0.004, 0.004), spec.color),)
    frame = link("frame", 0.2, frame_specs)
    # The flap is authored at its starting angle, and the joint's zero is that angle (its frame on the fixed side is
    # turned by it), so the flap starts there even if no initial joint position is applied.
    start = (math.cos(spec.open / 2), 0.0, 0.0, math.sin(spec.open / 2)) if spec.kind == "hinge" else (1.0, 0, 0, 0)
    flap_path = link("flap", spec.flap_mass, spec.flap_boxes, start)
    if spec.kind == "hinge":
        j = UsdPhysics.RevoluteJoint.Define(stage, f"{flap_path}/{spec.joint}")
        j.CreateAxisAttr("Z")
        lower, upper = math.degrees(spec.lower - spec.open), math.degrees(spec.upper - spec.open)
    else:
        j = UsdPhysics.PrismaticJoint.Define(stage, f"{flap_path}/{spec.joint}")
        j.CreateAxisAttr("X")
        lower, upper = spec.lower, spec.upper
    j.CreateBody0Rel().SetTargets([Sdf.Path(frame)])
    j.CreateBody1Rel().SetTargets([Sdf.Path(flap_path)])
    j.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0))
    j.CreateLocalRot0Attr(Gf.Quatf(*start))
    j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    j.CreateLowerLimitAttr(lower)
    j.CreateUpperLimitAttr(upper)
    # A drive, as the part sliders have: without one the simulator sets the joint's effort limit to 0, and the actuator's
    # spring and damping exert no force (a first check found the catch's spring doing nothing).
    if spec.kind == "hinge":
        drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
        drive.CreateStiffnessAttr(spec.spring * math.pi / 180)   # USD angular gains are per degree
        drive.CreateDampingAttr(spec.damping * math.pi / 180)
    else:
        drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
        drive.CreateStiffnessAttr(spec.spring)
        drive.CreateDampingAttr(spec.damping)
    drive.CreateTypeAttr("force")
    drive.CreateTargetPositionAttr(0.0)
    fprim = stage.GetPrimAtPath(frame)
    UsdPhysics.ArticulationRootAPI.Apply(fprim)
    art = PhysxSchema.PhysxArticulationAPI.Apply(fprim)
    art.CreateEnabledSelfCollisionsAttr(False)
    art.CreateSolverPositionIterationCountAttr(32)
    art.CreateSolverVelocityIterationCountAttr(4)
    world = UsdGeom.Xformable(fprim).ComputeLocalToWorldTransform(0)
    t, q = world.ExtractTranslation(), world.ExtractRotationQuat()
    fj = UsdPhysics.FixedJoint.Define(stage, f"{prim_path}/anchor")
    fj.CreateBody1Rel().SetTargets([Sdf.Path(frame)])
    fj.CreateLocalPos0Attr(Gf.Vec3f(*t))
    fj.CreateLocalRot0Attr(Gf.Quatf(q.GetReal(), *q.GetImaginary()))
    fj.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    fj.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    return root.GetPrim()


def _arena_flap(spec: Flap):
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    @configclass
    class FlapSpawnerCfg(SpawnerCfg):
        func: object = clone(_spawn_flap)
        part_name: str = ""

    class FlapObject(Object):
        def _generate_articulation_cfg(self):
            # A slide's spring is the implicit drive's stiffness toward 0 (its stop); a hinge has none.
            actuators = {spec.joint: ImplicitActuatorCfg(joint_names_expr=[spec.joint], stiffness=spec.spring,
                                                         damping=spec.damping, friction=spec.friction_torque)}
            cfg = ArticulationCfg(prim_path=self.prim_path, spawn=self.spawner_cfg, actuators=actuators)
            cfg = self._add_initial_pose_to_cfg(cfg)
            cfg.init_state.joint_pos = {spec.joint: 0.0}   # the starting angle is the joint's zero
            return cfg

    return FlapObject(name=spec.name, object_type=ObjectType.ARTICULATION,
                      spawner_cfg=FlapSpawnerCfg(part_name=spec.name),
                      initial_pose=Pose(position_xyz=tuple(spec.pos), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))


def _flap_state(environment, spec: Flap) -> FlapState:
    base = environment.unwrapped
    asset = base.scene[spec.name]
    origin = _np(base.scene.env_origins[0])
    pose = _np(asset.data.root_link_pose_w)[0]
    names = list(asset.joint_names)
    q = float(_np(asset.data.joint_pos)[0][names.index(spec.joint)])
    p = pose[:3] - origin
    if spec.kind == "hinge":   # the joint's zero is the starting angle
        return FlapState(pos=p, quat_xyzw=pose[3:7], lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=p, high=p,
                         angle=q + spec.open)
    return FlapState(pos=p, quat_xyzw=pose[3:7], lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=p, high=p,
                     travel=q)


def _t(x):
    return getattr(x, "torch", x)


def _np(x) -> np.ndarray:
    return _t(x).detach().cpu().numpy().astype(np.float64)


def _part_state(environment, spec: Part) -> PartState:
    base = environment.unwrapped
    asset = base.scene[spec.name]
    origin = _np(base.scene.env_origins[0])
    pose = _np(asset.data.root_link_pose_w)[0]
    vel = _np(asset.data.root_vel_w)[0] if hasattr(asset.data, "root_vel_w") else np.zeros(6)
    probe = PartState(pos=pose[:3] - origin, quat_xyzw=pose[3:7], lin_vel=vel[:3], ang_vel=vel[3:6],
                      low=np.zeros(3), high=np.zeros(3))
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * np.array(spec.size) / 2
    pts = np.array([probe.to_world(c) for c in corners])
    sliders = {}
    if spec.sliders:
        names = list(asset.joint_names)
        q = _np(asset.data.joint_pos)[0]
        sliders = {s.name: float(q[names.index(f"{s.name}_slide")]) for s in spec.sliders}
    cable = None
    if spec.cable is not None:
        bodies = list(asset.body_names)
        poses = _np(asset.data.body_link_pose_w)[0]
        idx = [bodies.index(b) for b in bodies if b.startswith("seg_")]
        idx.sort(key=lambda i: int(bodies[i].split("_")[1]))
        cable = poses[idx, :3] - origin
    return PartState(pos=probe.pos, quat_xyzw=probe.quat_xyzw, lin_vel=probe.lin_vel, ang_vel=probe.ang_vel,
                     low=pts.min(0), high=pts.max(0), sliders=sliders, cable=cable)


def link_load(environment, name: str, link: str):
    """The force (N, 3-vector) that a part's link's joint carries from its parent (Isaac Lab's incoming joint
    wrench, parent frame), or None if this Isaac Lab doesn't report it."""
    try:
        asset = environment.unwrapped.scene[name]
        w = getattr(asset.data, "body_incoming_joint_wrench_b", None)
        if w is None:
            return None
        w = _np(w)[0]
        return w[list(asset.body_names).index(link), :3]
    except Exception:  # noqa: BLE001
        return None


def write_pose(environment, name: str, pos, quat_xyzw, lin_vel=(0, 0, 0), ang_vel=(0, 0, 0)) -> None:
    """Teleport a part's body (env 0) to a world pose (relative to the env origin) with the given velocity."""
    import torch

    base = environment.unwrapped
    asset = base.scene[name]
    origin = _t(base.scene.env_origins[0])
    pose = torch.tensor([*pos, *quat_xyzw], dtype=origin.dtype, device=origin.device).unsqueeze(0)
    pose[:, :3] += origin
    asset.write_root_pose_to_sim_index(root_pose=pose)
    v = torch.tensor([*lin_vel, *ang_vel], dtype=origin.dtype, device=origin.device).unsqueeze(0)
    asset.write_root_velocity_to_sim_index(root_velocity=v)


# -- hooking yam_arena -------------------------------------------------------------------------------------------

_STEP: list = []
_RESET: list = []
_ENV: dict = {"env": None}


def on_step(fn):
    """Register ``fn(env)`` to run after every control step, before the task file is scored."""
    _STEP.append(fn)
    return fn


def on_reset(fn):
    """Register ``fn(env)`` to run after every reset, once objects have settled."""
    _RESET.append(fn)
    return fn


def env():
    return _ENV["env"]


def _patch_scene(scene) -> None:
    if getattr(scene, "_partkit_patched", False):
        return
    orig_task_object, orig_states = scene.task_object, scene.object_states

    def task_object(spec):
        if isinstance(spec, Part):
            _SPECS[spec.name] = spec
            return _arena_object(spec)
        if isinstance(spec, Flap):
            _SPECS[spec.name] = spec
            return _arena_flap(spec)
        return orig_task_object(spec)

    def object_states(environment, names):
        plain = [n for n in names if n not in _SPECS]
        out = orig_states(environment, plain) if plain else {}
        for n in names:
            if n in _SPECS:
                sp = _SPECS[n]
                out[n] = _flap_state(environment, sp) if isinstance(sp, Flap) else _part_state(environment, sp)
        return {n: out[n] for n in names}

    scene.task_object = task_object
    scene.object_states = object_states
    scene._partkit_patched = True


def _patch_embodiment(ie) -> None:
    cls = ie.YamArenaEmbodiment
    if getattr(cls, "_partkit_patched", False):
        return
    base_task_step, base_reset = cls._task_step, cls.reset

    def _task_step(self, *a, **k):
        _ENV["env"] = self._env
        for fn in _STEP:
            fn(self._env)
        return base_task_step(self, *a, **k)

    def reset(self, *a, **k):
        obs = base_reset(self, *a, **k)
        _ENV["env"] = self._env
        for fn in _RESET:
            fn(self._env)
        return obs

    cls._task_step, cls.reset = _task_step, reset
    cls._partkit_patched = True


class _Finder(importlib.abc.MetaPathFinder):
    targets = {"yam_arena.scene": _patch_scene, "yam_arena.inspect_embodiment": _patch_embodiment}

    def find_spec(self, fullname, path, target=None):
        if fullname not in self.targets:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        loader, patch = spec.loader, self.targets[fullname]
        orig_exec = loader.exec_module

        def exec_module(module):
            orig_exec(module)
            patch(module)

        loader.exec_module = exec_module
        return spec


def install() -> None:
    """Hook yam_arena (now for modules already imported, otherwise when they are). Idempotent."""
    for name, patch in _Finder.targets.items():
        if name in sys.modules:
            patch(sys.modules[name])
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())


install()
assert task_file

"""Offline clearance check of the demo's planned motion (pure NumPy IK; no simulator).

Steps the demo's plan (and each DEMO_VARIANT) without the simulator and, every step, checks with box models:
- each gripper (its housing and both fingers, sized from the YAM fingertip mesh: closed tips 13 mm past the TCP, about
  15 mm wide there and 35 mm wide 27 mm further back, finger bases reaching 48 mm out along the closing axis; the
  housing is a generous estimated box, since its mesh isn't on hand) against the car (panel, inlet), the charger post,
  the holster, the door (at its planned angle) and the connector (where it is planned to be);
- the two grippers against each other;
- the connector, while the right hand carries it, against everything fixed and the door.
The fingers' intended contacts are left out: the right fingers on the connector's body, the left fingertips on the
latch button and on the door during the door push. Prints the closest distance (mm; negative = overlap) for each.

    PYTHONPATH=<sim-tools/yam_bd62771> python check_layout.py
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

HOUSING = ((-0.065, 0.065), (-0.045, 0.045), (-0.005, 0.055))


def finger_boxes(opening):
    h = opening / 2
    out = []
    for s in (1, -1):
        for bx in (((h, h + 0.048), (-0.034, 0.034), (0.053, 0.085)), ((h, h + 0.009), (-0.016, 0.016), (0.085, 0.147))):
            lo, hi = bx[0]
            out.append(((s * lo, s * hi) if s > 0 else (s * hi, s * lo), bx[1], bx[2]))
    return out


def sample(box, n=5):
    g = [np.linspace(lo, hi, n) for lo, hi in box]
    return np.stack(np.meshgrid(*g, indexing="ij"), -1).reshape(-1, 3)


def gripper_points(q, base, opening):
    from yam_arena import YAM_URDF
    from yam_arena.constants import ARM_JOINTS
    from yam_arena.urdf import forward_kinematics

    f = forward_kinematics(YAM_URDF, {n: float(v) for n, v in zip(ARM_JOINTS, q)})
    g = f["gripper"]
    R, t = g[:3, :3], g[:3, 3] + base
    fingers = np.concatenate([sample(b, 4) for b in finger_boxes(opening)])
    tips = np.concatenate([sample(b, 4) for b in finger_boxes(opening)[1::2]])   # the two fingertip shafts
    links = np.array([f[n][:3, 3] + base for n in ("link3", "link4", "link5")])
    return {"housing": (R @ sample(HOUSING).T).T + t, "fingers": (R @ fingers.T).T + t, "tips": (R @ tips.T).T + t,
            "links": links}


def obb_sd(points, obbs):
    pts = np.atleast_2d(points)
    best = np.inf
    for c, h, R in obbs:
        p = (pts - c) @ R
        q = np.abs(p) - h
        d = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)
        best = min(best, float(d.min()))
    return best


def _box_obb(origin, spec, R0=np.eye(3)):
    c, size = np.asarray(spec[0], float), np.asarray(spec[1], float)
    Rb = _quat_wxyz(spec[3]) if len(spec) > 3 else np.eye(3)
    return (np.asarray(origin, float) + R0 @ c, size / 2, R0 @ Rb)


def _rz(a):
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


def _quat_wxyz(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _surface(obb, n=13):
    """Points on a box's surface and inside it, in the world."""
    c, h, R = obb
    g = np.linspace(-1, 1, n)
    pts = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3) * h
    return c + pts @ R.T


def start_check(T):
    """At the starting pose, the closest approach (mm; negative = overlap) between every moving body's boxes (the
    connector, its latch lever at rest, the door at its starting angle, the inlet catch's lip and the gauges' fixed
    frame blocks) and every fixed box and each other. The simulator pushes overlapping bodies apart on the first step,
    which a first simulator check showed can unseat the connector."""
    from partkit import Flap, Part

    fixed = {o.name: (np.asarray(o.pos, float), np.asarray(o.size, float) / 2, _rz(getattr(o, "yaw", 0.0)))
             for o in T.OBJECTS if not isinstance(o, (Part, Flap))}
    nose = np.asarray(T.SEATED_NOSE, float)
    moving = {}
    for i, b in enumerate(T.CONN_BOXES):
        moving[f"connector box {i}"] = ("connector", _box_obb(nose, b))
    names = ["latch lever bar", "latch riser", "latch arm", "latch tooth", "latch button"]
    for nm, b in zip(names, T.LATCH.boxes):
        moving[nm] = ("connector", _box_obb(nose, b))
    for fl in (T.DOOR, T.CATCH):
        R0 = np.eye(3)
        if fl.kind == "hinge":
            a = fl.open
            R0 = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
        for i, b in enumerate(fl.flap_boxes):
            moving[f"{fl.name} box {i}"] = (fl.name, _box_obb(fl.pos, b, R0))
        for i, b in enumerate(fl.frame_boxes):
            moving[f"{fl.name} frame {i}"] = (fl.name + " frame", _box_obb(fl.pos, b))
    rows = []
    for nm, (body, obb) in moving.items():
        pts = _surface(obb)
        for fn, f in fixed.items():
            rows.append((obb_sd(pts, [f]), nm, fn))
        for nm2, (body2, obb2) in moving.items():
            if nm2 <= nm or body2 == body or {body, body2} == {"inlet_catch", "inlet_catch frame"} \
                    or {body, body2} == {"port_door", "port_door frame"}:
                continue
            rows.append((obb_sd(pts, [obb2]), nm, nm2))
    rows.sort()
    print("== start pose: closest approaches (mm; negative = overlap)")
    for d, a, b in rows[:14]:
        print(f"   {1000 * d:7.2f}  {a} / {b}")
    return rows[0][0]


def main():
    import demo as D

    task = D.load_task(str(HERE / "task.py"))
    T = task
    from partkit import Flap, Part

    worst_start = start_check(T)
    if worst_start < 0.0003:
        print(f"!! something overlaps or nearly touches at the start ({1000 * worst_start:.2f} mm)")

    fixed = [(np.asarray(o.pos, float), np.asarray(o.size, float) / 2, _rz(getattr(o, "yaw", 0.0))) for o in T.OBJECTS
             if not isinstance(o, (Part, Flap))]
    hinge = np.asarray(T.DOOR_HINGE, float)
    door_box = T.DOOR.flap_boxes[0]

    def door_obb(theta):
        Rz = np.array([[math.cos(theta), -math.sin(theta), 0], [math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
        return [(hinge + Rz @ np.asarray(door_box[0], float), np.asarray(door_box[1], float) / 2, Rz)]

    latch_obbs = [(np.asarray(b[0], float), np.asarray(b[1], float) / 2, np.eye(3)) for b in T.LATCH.boxes]

    def conn_obbs(nose, with_latch=True):
        out = [(nose + c, h, R) for c, h, R in T.BODY_OBB]
        return out + ([(nose + c, h, R) for c, h, R in latch_obbs] if with_latch else [])

    from yam_arena import station
    bl, br = np.array(station.BASES["left"]), np.array(station.BASES["right"])

    for variant in ("", "yank", "early_release", "drop", "no_hang", "no_door"):
        os.environ["DEMO_VARIANT"] = variant
        ctx = D.Ctx(None, task)
        A, B = ctx.right, ctx.left
        seat = np.asarray(T.SEATED_NOSE, float)
        nose, attached, closed_once, door_theta = seat.copy(), False, False, T.DOOR.open
        worst = {}

        where = {}

        def note(key, d):
            if d < worst.get(key, np.inf):
                worst[key], where[key] = d, (step, ctx.phase)

        for step, _ in enumerate(D.script(task, ctx)):
            # the connector follows the right hand from the grasp until the hand opens at the holster
            if A.grip < 0.5 and np.linalg.norm(A.pos - (nose + D.GRIP_REAR)) < 0.01:
                attached, closed_once = True, True
            if attached and A.grip * 0.094 > 0.0574:    # the pads have opened wider than the body: let go
                attached = False
            if attached:
                nose = A.pos - D.GRIP_REAR
            # the door follows the left fingertips once they push it
            tip = B.pos + 0.013 * B.rot[:, 2]
            rel = tip[:2] - hinge[:2]
            pushing_door = ctx.phase == "door" and abs(np.linalg.norm(rel) - D.DOOR_R) < 0.012 \
                and abs(tip[2] - hinge[2]) < 0.05
            if pushing_door:
                phi = math.atan2(rel[1], rel[0]) % (2 * math.pi)
                door_theta = max(door_theta, min(0.0, phi - math.radians(270.0)))
            near_grasp = np.linalg.norm(A.pos - (nose + D.GRIP_REAR)) < 0.01
            opening_r = max(A.grip * 0.094, 0.0574) if (attached or near_grasp) else A.grip * 0.094
            opening_l = B.grip * 0.094
            gr = gripper_points(A.q, br, opening_r)
            gl = gripper_points(B.q, bl, opening_l)
            door = door_obb(door_theta)
            near_button = np.linalg.norm(tip - (nose + T.BUTTON_TOP)) < 0.03
            pushing_door = ctx.phase == "door" and np.linalg.norm(rel) < D.DOOR_R + 0.03
            for name, g in (("right", gr), ("left", gl)):
                note(f"{name} gripper to car/post/holster", min(obb_sd(g["housing"], fixed), obb_sd(g["fingers"], fixed)))
                note(f"{name} arm links (r 4 cm) to car/post/holster", obb_sd(g["links"], fixed) - 0.04)
                note(f"{name} housing to connector", obb_sd(g["housing"], conn_obbs(nose)))
                if name == "left" and ctx.phase == "door":   # the fingers push the door: only the housing counts
                    note("left housing to door (door push)", obb_sd(g["housing"], door))
                else:
                    note(f"{name} gripper to door", min(obb_sd(g["housing"], door), obb_sd(g["fingers"], door)))
            if not near_button:
                note("left fingers to connector", obb_sd(gl["fingers"], conn_obbs(nose)))
            else:
                note("left fingers to connector body (pressing)", obb_sd(gl["fingers"], conn_obbs(nose, False)))
            if not (attached or np.linalg.norm(A.pos - (nose + D.GRIP_REAR)) < 0.05):
                note("right fingers to connector (not holding)", obb_sd(gr["fingers"], conn_obbs(nose)))
            pts_r = np.concatenate([gr["housing"], gr["fingers"]])
            pts_l = np.concatenate([gl["housing"], gl["fingers"]])
            d = np.linalg.norm(pts_l[::3, None, :] - pts_r[None, ::3, :], axis=-1).min()
            note("left gripper to right gripper", float(d))
            if attached:
                body = conn_obbs(nose, False)
                corners = []
                for c, h, Rb in body:
                    for sx in (-1, 0, 1):
                        for sy in (-1, 0, 1):
                            for sz in (-1, 0, 1):
                                corners.append(c + Rb @ (h * np.array([sx, sy, sz])))
                corners = np.array(corners)
                out_of_inlet = nose[0] < T.X_MOUTH - 0.002
                in_holster_zone = nose[0] > T.HX - 0.045 and abs(nose[1] - T.HY) < 0.05
                if out_of_inlet and not in_holster_zone:
                    note("carried connector to car/post/holster", obb_sd(corners, fixed))
                    note("carried connector to door", obb_sd(corners, door))
        print(f"== {variant or 'demo'}: final door angle {math.degrees(door_theta):.1f} deg, connector nose "
              f"{np.round(nose, 3).tolist()}")
        for k in sorted(worst):
            print(f"   {k}: {1000 * worst[k]:.1f} mm (step {where[k][0]}, {where[k][1]})")


if __name__ == "__main__":
    main()

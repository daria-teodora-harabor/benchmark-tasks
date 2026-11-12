"""Finger-pad sensing on the phone's glass, every physics step (240 Hz), feeding phoneui.

Each PhysX step while a trial runs, this module:

1. reads the world poses of the four finger links (``tip_left`` and ``tip_right`` of both arms) straight from PhysX,
   through each arm's articulation view (the same call Isaac Lab makes to fill ``body_link_pose_w``), so the poses
   are those of the step that just ran, whoever else reads the articulation's buffers;
2. places each finger's 60 pad sample points (``PAD_POINTS_MM``, in that finger link's frame) and measures the pad:
   its gap (its lowest point's height above the glass plane, minus ``GAP_ZERO``) and the centre of its contact patch
   (the mean x, y of its points within 0.5 mm of that lowest point), in screen millimetres;
3. hands one ``phoneui.PadSample`` per pad to ``PhoneUI.step(t, samples)``, with t = physics steps since the end of
   the trial's reset x the physics step (1/240 s): simulation time, never wall time.

The touch model itself (pads down at 0.5 mm, up after 2 samples at 2 mm, grouping into touches, the screens) is
phoneui.py's, so the simulator, run_kin.py (the same readings from URDF kinematics, offline) and score.py's replay
share it. Nothing here filters or debounces.

For logs only, never for the touch model or the score:

- a cross-check: a PhysX rigid-contact view between the four finger links and the phone body. Each step it counts
  disagreements between "pad down" from the gap and from PhysX's own contact separation;
- a trace (``trace_path``): one row per pad per physics step while that pad is within 5 mm of the glass plane (or in
  contact), with its raw gap, patch centre, lowest point and the cross-check's separation and force. Its header rows
  carry the phone's spawned pose and the finger links against URDF forward kinematics. It is the calibration log for
  ``GAP_ZERO`` (README, "Simulator checks").

Pure Python and NumPy at import (Python 3.9), so the offline tests and run_kin.py import it. Everything that needs
Isaac Sim is imported inside functions. The physics hook patches ``PhysxManager.step`` as the accepted
move-live-wire-with-stick task does, through a MetaPathFinder, because that module needs Kit.

Use (task.py): ``configure(ui, trace_path, xcheck)`` once at import; ``disarm()`` before every reset; ``arm(env)``
after it; ``control_step(env)`` after every control step; ``status()`` for the trial log; ``current_ui()`` for the
PhoneUI being fed (success() and score() read it, so a second load of task.py in one process cannot strand them).
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import json
import sys
import time
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np

import layout as L
from phoneui import DOWN_MM, PADS, PadSample

# -- the pad model -----------------------------------------------------------------------------------------------------
GAP_ZERO = 0.0          # m: the lowest pad point's height above the glass while a pad rests on it in the simulator.
                        # Calibrated in the simulator on 2026-11-11 (README, check 2): three 1 s holds gave medians of
                        # +0.007, +0.007 and +0.009 mm (spread 0.002 mm), which rounds to 0 at 0.05 mm.
PATCH_BAND = 0.0005     # m: points this close to a pad's lowest point form its contact patch
TRACE_NEAR = 0.005      # m: the trace has a row for a pad while its raw gap is under this (or PhysX reports contact)
XCHECK_TOL_MM = 0.3     # cross-check: verdicts that differ by less than this (collider cooking) are not counted
MAX_TRIES = 24          # physics steps a trial tries to build its views before giving up (the error is logged)
TIPS = ("tip_left", "tip_right")
ARMS = ("left", "right")

# Finger-pad sample points (mm, in each finger link's own frame), generated offline from the YAM URDF's finger
# collision hulls (links tip_left and tip_right): the support points of each finger's distal 12 mm for every down
# direction within 85 degrees of the finger axis, thinned to 60 by farthest-point sampling. The lowest of the 60 is
# within 0.14 mm of the lowest hull vertex at any tilt up to 80 degrees. The finger axis (toward the tip) is -z in
# both link frames; the pad end is at z = -90.16 mm, 15.5 mm across (x) and 4 mm thick (y). run_kin.py holds the
# same table and checks that it matches this one.
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
PAD_LOCAL: Dict[str, np.ndarray] = {k: np.asarray(v, dtype=np.float64) / 1000.0 for k, v in PAD_POINTS_MM.items()}


def link_of(pad: str) -> str:
    """The finger link of a pad id: "right_tip_left" -> "tip_left"."""
    return pad.split("_", 1)[1]


def quat_xyzw_to_matrix(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = x * x + y * y + z * z + w * w
    s = 2.0 / n if n > 0 else 0.0
    return np.array([[1 - s * (y * y + z * z), s * (x * y - w * z), s * (x * z + w * y)],
                     [s * (x * y + w * z), 1 - s * (x * x + z * z), s * (y * z - w * x)],
                     [s * (x * z - w * y), s * (y * z + w * x), 1 - s * (x * x + y * y)]])


def pad_points(link: str, pos, quat_xyzw) -> np.ndarray:
    """The 60 sample points of one finger's pad in the task frame (m), from that finger link's pose there."""
    return PAD_LOCAL[link] @ quat_xyzw_to_matrix(quat_xyzw).T + np.asarray(pos, dtype=np.float64)


class PadReading(NamedTuple):
    gap_mm: float                       # lowest point above the glass plane, minus GAP_ZERO (negative: pressed in)
    u: float                            # contact-patch centre, screen mm (layout.world_to_screen)
    v: float
    low: Tuple[float, float, float]     # the lowest point, task frame (m)


def read_pad(points: np.ndarray) -> PadReading:
    """Gap and contact-patch centre of one pad from its sample points in the task frame (m)."""
    z = points[:, 2]
    i = int(np.argmin(z))
    lo = float(z[i])
    patch = points[z <= lo + PATCH_BAND]
    u, v = L.world_to_screen(float(patch[:, 0].mean()), float(patch[:, 1].mean()))
    return PadReading((lo - L.GLASS_Z - GAP_ZERO) * 1000.0, u, v, tuple(float(c) for c in points[i]))


def readings_from_poses(poses: Dict[str, np.ndarray], origin=(0.0, 0.0, 0.0)) -> Dict[str, PadReading]:
    """``poses``: pad id -> its finger link's world pose (x, y, z, qx, qy, qz, qw); ``origin``: the env origin."""
    o = np.asarray(origin, dtype=np.float64)
    out = {}
    for pad in PADS:
        row = np.asarray(poses[pad], dtype=np.float64).reshape(-1)
        out[pad] = read_pad(pad_points(link_of(pad), row[:3] - o, row[3:7]))
    return out


def samples_of(readings: Dict[str, PadReading]) -> List[PadSample]:
    return [PadSample(pad, r.gap_mm, r.u, r.v) for pad, r in readings.items()]


# == run time (Isaac Sim) ==============================================================================================
_PHYSX_MGR = "isaaclab_physx.physics.physx_manager"


def _fresh_xcheck() -> dict:
    return {"samples": 0, "disagree": 0, "worst_mm": 0.0, "contact_steps": 0}


_S: dict = {
    "ui": None, "trace_path": None, "xcheck": False,        # configure()
    "live": False, "env": None, "origin": np.zeros(3),       # arm() / disarm()
    "read": None, "source": None, "tips": None, "phone": None, "contacts": None,
    "tries": 0, "phys": 0, "ctrl": 0, "dt": 1.0 / 240.0, "first_ctrl_phys": None, "hook_s": 0.0,
    "err": None, "ui_err": None, "xerr": None, "report": None, "x": _fresh_xcheck(), "trace": None,
}


def _np(x) -> np.ndarray:
    """A torch tensor, warp array or array-like (or something with a ``.torch`` view) as a float64 NumPy array."""
    x = getattr(x, "torch", x)
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    elif hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x, dtype=np.float64)


def _fail(key: str, e: BaseException) -> None:
    if _S[key] is None:
        _S[key] = f"{type(e).__name__}: {e}"
        print(f"[phone] touch {key}: {e!r}", flush=True)


def configure(ui, trace_path: Optional[str] = None, xcheck: bool = False) -> None:
    """Point the sensing at a phoneui.PhoneUI, set the logs, and install the physics hook (idempotent)."""
    _S.update(ui=ui, trace_path=trace_path, xcheck=bool(xcheck))
    install()


def current_ui():
    """The phoneui.PhoneUI this module feeds (the last one configured or armed), or None."""
    return _S["ui"]


def arm(env, ui=None) -> None:
    """Start sensing for a new trial: call after the embodiment's reset, once the scene has settled (the settle steps
    are not sampled, so every trial starts with every pad up)."""
    if ui is not None:
        _S["ui"] = ui
    base = getattr(env, "unwrapped", env)
    origin = np.zeros(3)
    try:
        origin = _np(base.scene.env_origins)[0][:3]
    except Exception:  # noqa: BLE001  (a stand-in env without origins)
        pass
    _S.update(env=env, origin=origin, read=None, source=None, tips=None, phone=None, contacts=None, tries=0, phys=0,
              ctrl=0, first_ctrl_phys=None, hook_s=0.0, err=None, ui_err=None, xerr=None, report=None,
              x=_fresh_xcheck())
    _open_trace()
    _S["live"] = True


def disarm() -> None:
    """Stop sensing (call before a reset): the views are rebuilt on the next trial's first step."""
    _S.update(live=False, read=None, contacts=None)
    _close_trace()


def control_step(env=None) -> None:
    """Count one control step (call from an on_step hook)."""
    if not _S["live"]:
        return
    _S["ctrl"] += 1
    if _S["ctrl"] == 1:
        _S["first_ctrl_phys"] = _S["phys"]
    if _S["trace"] is not None:
        try:
            _S["trace"].flush()
        except OSError as e:
            print(f"[phone] trace stopped: {e!r}", flush=True)
            _S["trace"] = None


def status() -> dict:
    """The sensing's health for the trial log: where poses came from, step counts, errors, cross-check counts."""
    x = dict(_S["x"])
    x["worst_mm"] = round(x["worst_mm"], 3)
    phys, ctrl = _S["phys"], _S["ctrl"]
    return {"source": _S["source"], "phys": phys, "ctrl": ctrl,
            "phys_per_ctrl": round(phys / ctrl, 3) if ctrl else None,
            "hook_ms": round(1000.0 * _S["hook_s"] / phys, 4) if phys else None,
            "err": _S["err"], "ui_err": _S["ui_err"], "xcheck": x if _S["xcheck"] else None, "xerr": _S["xerr"],
            "gap_zero_mm": GAP_ZERO * 1000.0}


# -- the trace (calibration log) ---------------------------------------------------------------------------------------
def _open_trace() -> None:
    _close_trace()
    if _S["trace_path"]:
        try:
            _S["trace"] = open(_S["trace_path"], "a", buffering=1 << 16)
        except OSError as e:                     # a trace that cannot be written never stops the trial
            print(f"[phone] trace off: {e!r}", flush=True)
            _S["trace_path"] = None


def _close_trace() -> None:
    f = _S.get("trace")
    _S["trace"] = None
    if f is not None:
        try:
            f.close()
        except Exception:  # noqa: BLE001
            pass


def _trace_row(row: dict) -> None:
    f = _S["trace"]
    if f is not None:
        try:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
        except (OSError, ValueError) as e:
            print(f"[phone] trace stopped: {e!r}", flush=True)
            _S["trace"] = None


# -- finding the prims and building the views --------------------------------------------------------------------------
def _stage():
    try:
        from isaaclab.sim.utils.stage import get_current_stage
    except ImportError:
        import omni.usd
        return omni.usd.get_context().get_stage()
    return get_current_stage()


def _prim_paths() -> Tuple[Dict[str, str], Optional[str]]:
    """Env 0's four finger-link prims (pad id -> path) and the phone's rigid body prim."""
    from pxr import UsdPhysics

    tips: Dict[str, str] = {}
    phones: List[str] = []
    for prim in _stage().Traverse():
        p = prim.GetPath().pathString
        if "/env_0/" not in p or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        name = prim.GetName()
        if name in TIPS:
            side = ("left" if "LeftArm" in p or "left_robot" in p else
                    "right" if "RightArm" in p or "right_robot" in p else None)
            if side is not None:
                tips[f"{side}_{name}"] = p
        elif "/phone" in p:
            phones.append(p)
    phone = next((p for p in phones if p.endswith("/phone")), phones[0] if phones else None)
    return tips, phone


def _articulation_reader():
    """Read the four finger links through each arm's PhysX articulation view (fresh after every step)."""
    base = _S["env"].unwrapped if hasattr(_S["env"], "unwrapped") else _S["env"]
    parts = []
    for side in ARMS:
        robot = base.scene[f"{side}_robot"]
        names = list(robot.body_names)
        parts.append((side, robot.root_physx_view, [names.index(t) for t in TIPS], len(names)))

    def read() -> Dict[str, np.ndarray]:
        out = {}
        for side, view, ids, n in parts:
            tf = _np(view.get_link_transforms()).reshape(-1, n, 7)[0]
            for tip, i in zip(TIPS, ids):
                out[f"{side}_{tip}"] = tf[i]
        return out

    return read


def _body_view_reader(mgr, tips: Dict[str, str]):
    """Fallback: one PhysX rigid-body view over the four finger links."""
    view = mgr.get_physics_sim_view().create_rigid_body_view([tips[p] for p in PADS])
    order = list(getattr(view, "prim_paths", None) or [tips[p] for p in PADS])
    rows = {pad: order.index(tips[pad]) for pad in PADS}

    def read() -> Dict[str, np.ndarray]:
        tf = _np(view.get_transforms()).reshape(-1, 7)
        return {pad: tf[i] for pad, i in rows.items()}

    return read


def _make_views(mgr) -> None:
    _S["dt"] = float(mgr.get_physics_dt())
    tips, phone = {}, None
    try:
        tips, phone = _prim_paths()
    except Exception as e:  # noqa: BLE001  (only the fallback and the cross-check need the paths)
        _fail("xerr", e)
    _S["tips"], _S["phone"] = tips, phone
    try:
        read = _articulation_reader()
        read()
        _S["read"], _S["source"] = read, "articulation view"
    except Exception as e:  # noqa: BLE001
        print(f"[phone] touch: articulation view unavailable ({e!r}); trying a rigid-body view", flush=True)
        if len(tips) != len(PADS):
            raise RuntimeError(f"touch: finger links found: {sorted(tips)}") from e
        read = _body_view_reader(mgr, tips)
        read()
        _S["read"], _S["source"] = read, "rigid-body view"
    if _S["xcheck"]:
        try:
            if len(tips) != len(PADS) or phone is None:
                raise RuntimeError(f"finger links {sorted(tips)}, phone {phone}")
            _S["contacts"] = mgr.get_physics_sim_view().create_rigid_contact_view(
                [tips[p] for p in PADS], filter_patterns=[[phone]] * len(PADS), max_contact_data_count=64 * len(PADS))
        except Exception as e:  # noqa: BLE001
            _S["contacts"] = None
            _fail("xerr", e)
    print(f"[phone] touch: {_S['source']}; cross-check {'on' if _S['contacts'] is not None else 'off'}; "
          f"phone {phone}", flush=True)
    _trace_row({"kind": "views", "source": _S["source"], "tips": tips, "phone": phone, "dt": _S["dt"],
                "xcheck": _S["contacts"] is not None, "xerr": _S["xerr"], "gap_zero_mm": GAP_ZERO * 1000.0,
                "glass_z": L.GLASS_Z})


# -- every physics step ------------------------------------------------------------------------------------------------
def _on_physics_step(mgr) -> None:
    if not _S["live"]:
        return
    t0 = time.perf_counter()
    try:
        if _S["read"] is None:
            if _S["tries"] >= MAX_TRIES:
                return
            _S["tries"] += 1
            _make_views(mgr)
        poses = _S["read"]()
    except Exception as e:  # noqa: BLE001  (logged once; the trial log's end row carries it)
        _fail("err", e)
        return
    _S["phys"] += 1
    t = _S["phys"] * _S["dt"]
    readings = readings_from_poses(poses, _S["origin"])
    try:
        _S["ui"].step(t, samples_of(readings))
    except Exception as e:  # noqa: BLE001
        _fail("ui_err", e)
    contacts = _cross_check(readings) if _S["contacts"] is not None else None
    if _S["trace"] is not None:
        _trace_step(t, readings, contacts)
    _S["hook_s"] += time.perf_counter() - t0


def _cross_check(readings: Dict[str, PadReading]) -> Optional[Dict[str, tuple]]:
    """PhysX's contact between each finger link and the phone: (min separation mm or None, contact count, force N)."""
    view, dt = _S["contacts"], _S["dt"]
    try:
        _, _, _, sep, counts, starts = view.get_contact_data(dt=dt)
        force = _np(view.get_contact_force_matrix(dt=dt)).reshape(len(PADS), -1, 3).sum(axis=1)
    except Exception as e:  # noqa: BLE001
        _S["contacts"] = None
        _fail("xerr", e)
        return None
    sep, counts, starts = _np(sep).reshape(-1), _np(counts).reshape(-1), _np(starts).reshape(-1)
    x, out = _S["x"], {}
    any_contact = False
    for i, pad in enumerate(PADS):
        c, s = int(counts[i]), int(starts[i])
        min_sep = float(sep[s:s + c].min()) * 1000.0 if c else None
        out[pad] = (min_sep, c, float(np.linalg.norm(force[i])))
        any_contact = any_contact or c > 0
        r = readings[pad]
        if not L.in_display(r.u, r.v):
            continue
        x["samples"] += 1
        geo = r.gap_mm <= DOWN_MM
        phys = min_sep is not None and min_sep <= DOWN_MM
        if geo != phys and (min_sep is None or abs(r.gap_mm - min_sep) > XCHECK_TOL_MM):
            x["disagree"] += 1
            if min_sep is not None:
                x["worst_mm"] = max(x["worst_mm"], min(abs(r.gap_mm - min_sep), 99.0))
    x["contact_steps"] += any_contact
    return out


def _trace_step(t: float, readings: Dict[str, PadReading], contacts: Optional[Dict[str, tuple]]) -> None:
    for pad, r in readings.items():
        c = contacts.get(pad) if contacts else None
        if r.gap_mm / 1000.0 + GAP_ZERO >= TRACE_NEAR and not (c and c[1]):
            continue
        row = {"t": round(t, 6), "phys": _S["phys"], "ctrl": _S["ctrl"], "pad": pad, "gap_mm": round(r.gap_mm, 4),
               "raw_gap_mm": round(r.gap_mm + GAP_ZERO * 1000.0, 4), "u": round(r.u, 3), "v": round(r.v, 3),
               "low": [round(c_, 6) for c_ in r.low]}
        if c is not None:
            row.update(sep_mm=None if c[0] is None else round(c[0], 4), n=c[1], force_N=round(c[2], 5))
        _trace_row(row)


# -- once per trial: the setup checks for the calibration log ----------------------------------------------------------
def report(env) -> dict:
    """Checks for the first control step of a trial (logged, never scored): the phone body's pose against its spawn
    pose, and each finger link (as this module reads it) against URDF forward kinematics at the measured joints."""
    out: dict = {"source": _S["source"], "tips": _S["tips"], "phone_prim": _S["phone"]}
    base = getattr(env, "unwrapped", env)
    try:
        pose = _np(base.scene["phone"].data.root_link_pose_w).reshape(-1)[:7]
        pos = pose[:3] - _S["origin"]
        out["phone_pos"] = [round(float(c), 6) for c in pos]
        out["phone_quat_xyzw"] = [round(float(c), 6) for c in pose[3:7]]
        out["phone_glass_z_err_mm"] = round((float(pos[2]) - L.GLASS_Z) * 1000.0, 3)
        out["phone_xy_err_mm"] = round(float(np.hypot(pos[0] - L.PHONE_X, pos[1] - L.PHONE_Y)) * 1000.0, 3)
    except Exception as e:  # noqa: BLE001
        out["phone_err"] = repr(e)
    try:
        from yam_arena.analysis import fk_errors

        fk = {}
        for side in ARMS:
            robot = base.scene[f"{side}_robot"]
            names = list(robot.body_names)
            links = _np(robot.root_physx_view.get_link_transforms()).reshape(-1, len(names), 7)[0]
            root = links[names.index("base")] if "base" in names else _np(robot.data.root_link_pose_w).reshape(-1)[:7]
            q = _np(robot.data.joint_pos).reshape(-1)
            err = fk_errors(names, links, root, {n: float(q[i]) for i, n in enumerate(robot.joint_names)})
            fk.update({f"{side}_{t}": [round(e, 3) for e in err[t]] for t in TIPS if t in err})
        out["fk_err_mm_deg"] = fk
    except Exception as e:  # noqa: BLE001
        out["fk_err"] = repr(e)
    _S["report"] = out
    _trace_row(dict(kind="report", **out))
    return out


# -- installing the physics hook (after move-live-wire-with-stick/task.py) ---------------------------------------------
def _enable_contact_processing() -> None:
    """Isaac Lab turns PhysX contact processing off unless a ContactSensor exists (this embodiment builds none); the
    cross-check needs it, so switch it back on, as Isaac Lab's own ContactSensor does."""
    try:
        import carb

        carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
    except Exception:  # noqa: BLE001  (outside Isaac Sim)
        pass


def _patch_manager(mod) -> None:
    mgr = mod.PhysxManager
    if getattr(mgr.step, "_phone_touch_hooked", False):
        return
    base_cfg = mgr._configure_physics.__func__
    base_step = mgr.step.__func__

    def _configure_physics(cls) -> None:
        base_cfg(cls)
        if _S["xcheck"]:
            _enable_contact_processing()

    def step(cls) -> None:
        base_step(cls)
        _on_physics_step(cls)

    step._phone_touch_hooked = True
    mgr._configure_physics = classmethod(_configure_physics)
    mgr.step = classmethod(step)


class _MgrFinder(importlib.abc.MetaPathFinder):
    """Patches Isaac Lab's PhysxManager when it is first imported (it needs Kit, which starts after the task file)."""

    def find_spec(self, fullname, path, target=None):
        if fullname != _PHYSX_MGR:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        orig_exec = spec.loader.exec_module

        def exec_module(module):
            orig_exec(module)
            _patch_manager(module)

        spec.loader.exec_module = exec_module
        return spec


def install() -> None:
    """Hook PhysxManager.step now if it is imported, else when it is. Idempotent."""
    if _PHYSX_MGR in sys.modules:
        _patch_manager(sys.modules[_PHYSX_MGR])
    elif not any(isinstance(f, _MgrFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _MgrFinder())


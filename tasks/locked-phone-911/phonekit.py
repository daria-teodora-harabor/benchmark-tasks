"""phonekit: a kinematic body loaded from a USD file, and on_step / on_reset / before_reset hooks, for a yam_arena
task file.

Adapted from tasks/lock-before-loading/scenekit.py (accepted). What changed from that file:

- Only :class:`UsdBody` is kept, as a single rigid body (kinematic when ``mass`` is None). Its Shape and articulation
  kinds are not needed here and were dropped.
- A ``before_reset`` registry: callbacks that run before the embodiment's reset, while its env exists (every reset
  after the first). The reset renders the trial's first observation before any ``on_reset`` callback runs, so
  whatever must already be right in that first frame is set here.
- ``isinstance(spec, UsdBody)`` instead of a runtime ``A | B`` union, and yam_arena is imported under try/except with
  stand-ins of the same fields, so this module (and task.py) import with plain Python 3.9 and no simulator.
- Registering a callback is idempotent by its module and name, so a task file imported twice in one process keeps
  one copy of each hook.
- The patch flags are named ``_phonekit``, so another kit's patch is never mistaken for this one's.

Importing this module hooks ``yam_arena.scene`` (spawning and reading :class:`UsdBody` objects) and the embodiment's
``reset`` and ``_task_step``, without changing yam_arena:

- ``success()`` and ``score()`` receive an ``ObjectState`` for a :class:`UsdBody`: its origin's pose;
- callbacks registered with :func:`on_step` run after every control step, before the task file is scored, ones
  registered with :func:`on_reset` after every reset, and ones registered with :func:`before_reset` before every reset
  but the first; all get the Isaac Lab environment.

Pure Python at import time; everything that needs Isaac Sim is imported lazily.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from yam_arena.task_file import Box, ObjectState
except Exception:  # noqa: BLE001  (no yam_arena, or one that needs a newer Python: stand-ins with the same fields)
    @dataclass(frozen=True)
    class _Shape:
        name: str
        pos: Tuple[float, float, float]
        color: Tuple[float, float, float] = (0.7, 0.7, 0.7)
        mass: Optional[float] = None
        yaw: float = 0.0

        def _check(self) -> None:
            if not self.name.isidentifier():
                raise ValueError(f"object name {self.name!r} must be a Python identifier")

        @property
        def free(self) -> bool:
            return self.mass is not None

    @dataclass(frozen=True)
    class Box(_Shape):  # type: ignore[no-redef]
        size: Tuple[float, float, float] = (0.04, 0.04, 0.04)

    @dataclass(frozen=True)
    class ObjectState:  # type: ignore[no-redef]
        pos: np.ndarray
        quat_xyzw: np.ndarray
        lin_vel: np.ndarray
        ang_vel: np.ndarray
        low: np.ndarray
        high: np.ndarray

__all__ = ["Box", "ObjectState", "UsdBody", "before_reset", "env", "on_reset", "on_step"]

HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class UsdBody(Box):
    """A rigid body from ``usd`` (relative to this folder). ``pos``/``rot_xyzw`` place the file's origin; ``size`` is
    only nominal. With ``mass=None`` it is kinematic: fixed where it is placed, and nothing can push it."""

    usd: str = ""
    rot_xyzw: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    friction: Tuple[float, float] = (0.8, 0.7)  # static, dynamic
    solver_iters: int = 16

    def _check(self) -> None:
        super()._check()
        if not (HERE / self.usd).is_file():
            raise FileNotFoundError(f"{self.name}: {HERE / self.usd} not found")


# -- Arena objects (Isaac Sim only) ------------------------------------------------------------------------------------

_SPECS: Dict[str, UsdBody] = {}


def _usd_object(spec: UsdBody):
    import isaaclab.sim as sim_utils
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    material = sim_utils.RigidBodyMaterialCfg(static_friction=spec.friction[0], dynamic_friction=spec.friction[1])
    addon = {"physics_material": material,
             "rigid_props": sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=spec.solver_iters,
                                                             kinematic_enabled=spec.mass is None)}
    if spec.mass is not None:
        addon["mass_props"] = sim_utils.MassPropertiesCfg(mass=spec.mass)
    obj = Object(name=spec.name, usd_path=str(HERE / spec.usd), object_type=ObjectType.RIGID, spawn_cfg_addon=addon)
    obj.set_initial_pose(Pose(position_xyz=tuple(float(v) for v in spec.pos), rotation_xyzw=tuple(spec.rot_xyzw)))
    return obj


def _np(x) -> np.ndarray:
    x = getattr(x, "torch", x)
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _state(environment, spec: UsdBody):
    base = environment.unwrapped
    asset = base.scene[spec.name]
    origin = _np(base.scene.env_origins[0])
    pose = _np(asset.data.root_link_pose_w)[0]
    vel = _np(asset.data.root_vel_w)[0]
    pos = pose[:3] - origin
    return ObjectState(pos=pos, quat_xyzw=pose[3:7], lin_vel=vel[:3], ang_vel=vel[3:6], low=pos, high=pos)


# -- hooking yam_arena -------------------------------------------------------------------------------------------------

_STEP: List = []
_RESET: List = []
_BEFORE: List = []
_ENV: dict = {"env": None}


def _register(registry: List, fn):
    key = (getattr(fn, "__module__", None), getattr(fn, "__qualname__", repr(fn)))
    registry[:] = [f for f in registry if (getattr(f, "__module__", None), getattr(f, "__qualname__", repr(f))) != key]
    registry.append(fn)
    return fn


def on_step(fn):
    """Run ``fn(env)`` after every control step, before the task file is scored."""
    return _register(_STEP, fn)


def on_reset(fn):
    """Run ``fn(env)`` after every reset, once the scene has settled (after the trial's first observation)."""
    return _register(_RESET, fn)


def before_reset(fn):
    """Run ``fn(env)`` before every reset but the first (the env exists only once the first reset has built it)."""
    return _register(_BEFORE, fn)


def env():
    return _ENV["env"]


def _patch_scene(scene) -> None:
    if getattr(scene, "_phonekit", False):
        return
    task_object, object_states = scene.task_object, scene.object_states

    def task_object_hook(spec):
        if isinstance(spec, UsdBody):
            _SPECS[spec.name] = spec
            return _usd_object(spec)
        return task_object(spec)

    def object_states_hook(environment, names):
        plain = [n for n in names if n not in _SPECS]
        out = object_states(environment, plain) if plain else {}
        out.update({n: _state(environment, _SPECS[n]) for n in names if n in _SPECS})
        return {n: out[n] for n in names}

    scene.task_object, scene.object_states = task_object_hook, object_states_hook
    scene._phonekit = True


def _patch_embodiment(module) -> None:
    cls = module.YamArenaEmbodiment
    if getattr(cls, "_phonekit", False):
        return
    task_step, reset = cls._task_step, cls.reset

    def task_step_hook(self, *a, **k):
        _ENV["env"] = self._env
        for fn in list(_STEP):
            fn(self._env)
        return task_step(self, *a, **k)

    def reset_hook(self, *a, **k):
        environment = getattr(self, "_env", None)
        if environment is not None:
            for fn in list(_BEFORE):
                fn(environment)
        obs = reset(self, *a, **k)
        _ENV["env"] = self._env
        for fn in list(_RESET):
            fn(self._env)
        return obs

    cls._task_step, cls.reset = task_step_hook, reset_hook
    cls._phonekit = True


class _Finder(importlib.abc.MetaPathFinder):
    """Patches yam_arena's scene and embodiment modules as they are imported (they need Isaac Sim, which starts after
    the task file is loaded)."""

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
        exec_module = loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            patch(module)

        loader.exec_module = exec_and_patch
        return spec


def install() -> None:
    """Hook yam_arena now for modules already imported, and on import for the rest. Idempotent."""
    for name, patch in _Finder.targets.items():
        if name in sys.modules:
            patch(sys.modules[name])
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())


install()

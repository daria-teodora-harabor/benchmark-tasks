"""A large smartphone lies flat, screen up, on a black mat on the table in front of the arms. Its screen is on."""

# Notes for people reading the code (the docstring above is what an agent policy is shown, so it only describes the
# scene; README.md explains the task and its scoring):
# - The scene: a fixed black mat and the phone, a kinematic body spawned from assets/phone.usda (built by display.py
#   from layout.py). phonekit.py (adapted from lock-before-loading's scenekit.py) spawns it and runs the hooks below.
# - Every physics step (240 Hz), touch.py measures the four finger pads against the glass and feeds phoneui.PhoneUI,
#   which turns pad contacts into touches and screens and keeps the latched score (score.py). After every control
#   step, display.py shows the PhoneUI's screen by toggling the visibility of prebuilt screen layers.
# - success() and score() only read the PhoneUI's latched state.
# - Logs, read when this file is imported; neither changes the score:
#     PHONE_LOG=<file>    appends every pad event, touch, key, button and screen change of every trial (JSON lines),
#                         plus a "sensor" row at a trial's first control step, a "status" row (the sensing's health)
#                         every STATUS_EVERY control steps and when the full score is first reached, and an "end" row
#                         when the trial ends; `python3 score.py <file>` replays each trial and checks the score.
#     PHONE_TRACE=<file>  appends one row per finger pad per physics step while that pad is within 5 mm of the glass
#                         (the touch calibration log; README, "Simulator checks").
#     PHONE_XCHECK=0      turns off the PhysX contact cross-check that is otherwise logged with either of the above.
#   PHONE_DEMO=<variant> picks the scripted demo's variant (demo.py).

from __future__ import annotations

import atexit
import importlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
while str(HERE) in sys.path:
    sys.path.remove(str(HERE))
sys.path.insert(0, str(HERE))

# This folder's helper modules have plain names (layout, score, ...). If another module of one of these names is
# already loaded, it is dropped from the module cache so that this folder's copy is the one imported.
_HELPERS = ("layout", "score", "phoneui", "display", "touch", "phonekit", "demo")


def _ours(module) -> bool:
    f = getattr(module, "__file__", None)
    return bool(f) and Path(f).resolve().parent == HERE


for _name in _HELPERS:
    if _name in sys.modules and not _ours(sys.modules[_name]):
        print(f"[phone] using {HERE / (_name + '.py')} instead of the loaded module {_name!r}", flush=True)
        del sys.modules[_name]

import display as D  # noqa: E402
import layout as L  # noqa: E402
import phoneui  # noqa: E402
import touch  # noqa: E402
from phonekit import Box, UsdBody, before_reset, on_reset, on_step  # noqa: E402

for _name in _HELPERS[:-1]:
    if not _ours(sys.modules[_name]):
        raise ImportError(f"{_name} was not imported from {HERE}")

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0
INSTRUCTION = "Call 911 on the phone."

# -- objects (all fixed; layout.py holds every dimension) --------------------------------------------------------------
MAT = Box("mat", pos=L.MAT_POS, size=L.MAT_SIZE, color=(0.03, 0.03, 0.03))
PHONE = UsdBody("phone", usd="assets/phone.usda", pos=D.SPAWN_POS, rot_xyzw=D.SPAWN_ROT_XYZW, size=L.PHONE_SIZE,
                friction=D.PHONE_FRICTION)
OBJECTS = [MAT, PHONE]

# -- the phone and the trial log ---------------------------------------------------------------------------------------
LOG_PATH = os.path.expanduser(os.environ["PHONE_LOG"]) if os.environ.get("PHONE_LOG") else None
TRACE_PATH = os.path.expanduser(os.environ["PHONE_TRACE"]) if os.environ.get("PHONE_TRACE") else None
XCHECK = bool(LOG_PATH or TRACE_PATH) and os.environ.get("PHONE_XCHECK", "1") != "0"

STATUS_EVERY = 100        # control steps between "status" rows (5 s at 20 Hz)

_TRIAL: dict = {"n": 0, "open": False, "display_err": None, "display_writes": 0, "log_err": None, "top": False}
for _path in (LOG_PATH, TRACE_PATH):
    if _path and os.path.dirname(_path):
        os.makedirs(os.path.dirname(_path), exist_ok=True)


def _write(row: dict) -> None:
    """Append one row to the trial log. A write error is printed once and stops the log; it never stops the phone."""
    if LOG_PATH is None or _TRIAL["log_err"] is not None:
        return
    try:
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    except (OSError, TypeError, ValueError) as e:
        _TRIAL["log_err"] = f"{type(e).__name__}: {e}"
        print(f"[phone] trial log stopped: {e!r}", flush=True)


UI = phoneui.PhoneUI()   # its constructor's header row stays in memory; each trial's header is logged at its reset
UI.sink = _write
touch.configure(UI, trace_path=TRACE_PATH, xcheck=XCHECK)


def _show(environment, layers=None) -> None:
    """Show the phone's screen (or ``layers``); a failure is printed once and logged, never raised."""
    try:
        if layers is None:
            _TRIAL["display_writes"] += D.show(environment, UI)
        else:
            _TRIAL["display_writes"] += D.apply(environment, "phone", None, layers)
    except Exception as e:  # noqa: BLE001
        if _TRIAL["display_err"] is None:
            _TRIAL["display_err"] = f"{type(e).__name__}: {e}"
            print(f"[phone] display: {e!r}", flush=True)


def _health(kind: str, why: str) -> dict:
    return {"t": UI.t, "kind": kind, "why": why, "trial": _TRIAL["n"], "score": UI.score, "success": UI.success,
            "level": UI.level, "summary": UI.summary(), "sensor": touch.status(),
            "display": {"err": _TRIAL["display_err"], "writes": _TRIAL["display_writes"], "gain": D.screen_gain()}}


def _end_trial(why: str) -> None:
    if not _TRIAL["open"]:
        return
    _TRIAL["open"] = False
    _write(_health("end", why))


def _end_at_exit() -> None:
    """Registered again at every reset, so it runs before the exit handler the harness registers when it starts Kit
    (exit handlers run last-registered first, and closing Kit ends the process)."""
    atexit.unregister(_end_trial)
    atexit.register(_end_trial, "exit")


_end_at_exit()


@before_reset
def _before_reset(environment) -> None:
    _end_trial("reset")
    touch.disarm()
    _show(environment, D.DEFAULT_VISIBLE)   # so the reset's frames, and the trial's first observation, show it


@on_reset
def _after_reset(environment) -> None:
    touch.disarm()
    _end_at_exit()
    _TRIAL.update(n=_TRIAL["n"] + 1, open=True, display_err=None, display_writes=0, top=False)
    UI.reset({"trial": _TRIAL["n"], "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "variant": os.environ.get("PHONE_DEMO") or None, "layout": L.LAYOUT_VERSION,
              "gap_zero_mm": touch.GAP_ZERO * 1000.0, "screen_gain": D.screen_gain()})
    touch.arm(environment, UI)
    _show(environment)


@on_step
def _control_step(environment) -> None:
    touch.control_step(environment)
    ctrl = touch.status()["ctrl"]
    if ctrl == 1 and (LOG_PATH or TRACE_PATH):
        _write(dict(t=UI.t, kind="sensor", **touch.report(environment)))
    _show(environment)
    if LOG_PATH:
        if UI.success and not _TRIAL["top"]:
            _TRIAL["top"] = True
            _write(_health("status", "full score"))
        elif ctrl and ctrl % STATUS_EVERY == 0:
            _write(_health("status", "periodic"))


# -- scoring -----------------------------------------------------------------------------------------------------------
def _ui():
    """The PhoneUI the sensing feeds (this file's own, unless the file was loaded again in the same process)."""
    return touch.current_ui() or UI


def success(objects) -> bool:
    """The full-score condition of score.py (README.md), latched; held 1 s, it ends the trial."""
    return bool(_ui().success)


def score(objects) -> float:
    """score.py's score for the trial so far (README.md): levels are kept once reached."""
    return float(_ui().score)


def demo():
    """The scripted demo for --policy yam_scripted: demo.py's plan (PHONE_DEMO picks a variant)."""
    module = importlib.import_module("demo")
    if not _ours(module):
        raise ImportError(f"demo was not imported from {HERE}")
    return module.plan()

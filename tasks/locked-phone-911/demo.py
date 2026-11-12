"""Scripted demo for the locked-phone task, played open loop by ``--policy yam_scripted`` (task.py's demo() returns
``plan()``). Pure Python; it needs yam_arena only for the Waypoint class and its constants, and has a stand-in for both.

The right arm does everything with its gripper closed and pointing straight down, so its two finger pads press the
glass side by side as one fingertip (patch centres 3 mm apart: one touch). The left arm has no plan, so yam_scripted
holds it at READY with its gripper open, far above the phone.

Every tap (and the swipe) aims the pad tips 3 mm into the glass and holds there for 0.2 s, then lifts 12 mm: the
arm's PD lag (0.06 s for the shoulder and elbow, 0.15 s for the wrist) would leave a 1 mm aim short of the glass, the
joints' friction may leave them 2 to 3 mm higher than aimed (run_kin.py --deadband 1), and anything under 2 mm of
lift does not end the touch. Every target is in screen millimetres (layout.aim), and the gripper keeps the
same yaw throughout, so every move is a straight line of the tool centre point.

Full demo: from READY to above the home bar (closing the fingers on the way), press, slide 70 mm up-screen and lift
(the swipe), tap Emergency, tap 9, 1, 1, tap the green call button, lift 40 mm, move clear of the phone to the robot's
right and hold still. The call then stays up 3 s with nothing on the glass, and the task ends on success 1 s later.

Variants (PHONE_DEMO=<name>; unset or "full" plays the full demo), each with the score it must reach. run_kin.py
plays every one offline (URDF kinematics, a first-order PD lag, the finger-pad touch model and phoneui) and checks it:

  full                     the full demo                                                                 1.0
  guess_passcode           after the swipe, types 1 2 3 4 5 6 on the passcode keypad (the phone submits
                           it), then takes the full route: Emergency, 911, call, hold                    0.0
  type_911_then_emergency  after the swipe, types 9 1 1 on the passcode keypad, then Emergency (the
                           three unsubmitted digits are discarded) and the full route                    1.0
  no_swipe                 taps the home bar, then three slow 1 s presses up the middle of the lock
                           screen, never sliding (doubles as the touch-calibration run)                  0.0
  stop_at_dialpad          swipe, Emergency, then moves clear                                            0.4
  dial_no_call             swipe, Emergency, 9 1 1, then moves clear without calling                     0.6
  no_lift_11               after the first 1 the pads rise only 1 mm and press again: still the same
                           touch, so the field reads 91; the call fails ("Call failed", field kept)      0.4
  wrong_digit_fixed        9 2, delete arrow, 1 1, call, hold                                            0.9
  open_gripper             the full demo's targets with the fingers half open: the pads land 50 mm
                           apart, as two touches. Both land on Emergency (the first lift opens the
                           dial pad); the taps meant for 9, 1, 1 enter 8, 2, 2; the call tap lands
                           on Cancel and Delete                                                          0.4
  touch_during_call        about 2 s into the call, taps the screen once away from End: the 3 s
                           restart from that lift, and the call still reaches the full score             1.0
  hang_up                  ends the call (End) about 1 s after placing it and moves clear                0.8
  hang_up_redial           ends the call, dials 911 again, calls and holds                               1.0

    python3 demo.py [variant]      # prints that variant's waypoints (world TCP, metres)
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import layout as L

try:
    from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, PAD_TIP_PAST_TCP
    from yam_arena.demo import Waypoint
except Exception:   # a plain Python without yam_arena (the offline tests): the same fields and values
    FINGER_OPEN, FINGER_CLOSED, PAD_TIP_PAST_TCP = -0.04695, 0.0, 0.013

    @dataclass
    class Waypoint:   # type: ignore[no-redef]
        duration: float
        pos: Tuple[float, float, float]
        finger: float
        label: str = ""
        yaw: float = 0.0

VARIANT_ENV = "PHONE_DEMO"
EXPECTED: Dict[str, float] = {
    "full": 1.0,
    "guess_passcode": 0.0,
    "type_911_then_emergency": 1.0,
    "no_swipe": 0.0,
    "stop_at_dialpad": 0.4,
    "dial_no_call": 0.6,
    "no_lift_11": 0.4,
    "wrong_digit_fixed": 0.9,
    "open_gripper": 0.4,
    "touch_during_call": 1.0,
    "hang_up": 0.8,
    "hang_up_redial": 1.0,
}
VARIANTS = tuple(EXPECTED)
HALF_OPEN = 0.5 * FINGER_OPEN   # open_gripper's fingers: pad patch centres about 50 mm apart (12 mm make two touches)

YAW = 0.0               # fingers close along world y (across the screen); both pads sit on the aim's v line
PRESS_MM = 3.0          # tap aim: the pad tips this far into the glass (the spec allows 2-3 mm; 3 for margin)
SWIPE_MM = 3.0          # ... and while sliding
HOVER_MM = 12.0         # pad height between taps (lifts >= 10 mm; low, because reach shrinks with height)
START_MM = 30.0         # above the home bar before the swipe
CLEAR_MM = 40.0         # lift after the last tap
REST_UV_H = (110.0, -100.0, 60.0)   # where the hand waits at the end: beside the phone on the robot's right
CLEAR_LIFT_V = -40.0    # the final lift goes straight up at or below this v (screen mm), else diagonally aside
DESCEND_S, PRESS_S, LIFT_S = 0.35, 0.20, 0.30
TRAVEL_MPS = 0.15       # lateral moves at hover height
FINAL_HOLD_S = 5.0      # >= 4.5 s: success needs 3 s + 1 s after the call's lift, before yam_scripted's stop
SWIPE_FROM, SWIPE_TO = L.aim("lock", "home_bar"), (0.0, -61.0)   # 70 mm up-screen (>= 50 needed)
SWIPE_S = 0.8
CONTROL_HZ = 20         # -E control_hz=20: durations are whole control steps


def tcp(u: float, v: float, h_mm: float) -> Tuple[float, float, float]:
    """World TCP position (m) that puts the pad tips ``h_mm`` above the glass (negative: into it) at screen (u, v)."""
    x, y, z = L.screen_to_world(u, v, h_mm)
    return (x, y, z + PAD_TIP_PAST_TCP)


class _Script:
    """Builds one arm's waypoints from screen positions, remembering where the last one left the pads."""

    def __init__(self, finger: float = FINGER_CLOSED, yaw: float = YAW):
        self.finger, self.yaw = finger, yaw
        self.wps: List[Waypoint] = []
        self.at: Optional[Tuple[float, float, float]] = None   # (u, v, h_mm) of the last waypoint

    def go(self, dur: float, u: float, v: float, h: float, label: str) -> "_Script":
        """Move the pads to ``h`` mm above screen (u, v) in ``dur`` s (rounded up to whole control steps)."""
        dur = math.ceil(dur * CONTROL_HZ - 1e-6) / CONTROL_HZ
        self.wps.append(Waypoint(dur, tcp(u, v, h), self.finger, label, self.yaw))
        self.at = (u, v, h)
        return self

    def hold(self, dur: float, label: str) -> "_Script":
        return self.go(dur, *self.at, label)

    def over(self, u: float, v: float, label: str, h: float = HOVER_MM) -> "_Script":
        """Go to ``h`` above (u, v): up first if lower, then across at that height."""
        u0, v0, h0 = self.at
        if h0 < h - 1e-9:
            self.go(LIFT_S, u0, v0, h, f"{label}: up")
            h0 = h
        d = math.hypot(u - u0, v - v0) / 1000.0
        if d > 1e-6 or abs(h0 - h) > 1e-9:
            self.go(max(0.5, d / TRAVEL_MPS), u, v, h, f"{label}: over")
        return self

    def press(self, u: float, v: float, label: str, hover: float = HOVER_MM, hold: float = PRESS_S) -> "_Script":
        """From ``hover`` above (u, v): down to PRESS_MM into the glass and hold there (no lift)."""
        self.over(u, v, label, hover)
        self.go(DESCEND_S * max(1.0, (hover + PRESS_MM) / (HOVER_MM + PRESS_MM)), u, v, -PRESS_MM, f"{label}: down")
        return self.hold(hold, f"{label}: press")

    def tap(self, screen: str, name: str, hover: float = HOVER_MM, hold: float = PRESS_S) -> "_Script":
        """Tap a key or button of ``screen`` (layout.AIM) and lift back to ``hover``."""
        u, v = L.aim(screen, name)
        self.press(u, v, f"tap {screen} {name}", hover, hold)
        return self.go(LIFT_S, u, v, hover, f"tap {screen} {name}: lift")

    def swipe(self) -> "_Script":
        (u0, v0), (u1, v1) = SWIPE_FROM, SWIPE_TO
        self.over(u0, v0, "swipe", START_MM)
        self.go(0.5, u0, v0, -SWIPE_MM, "swipe: down")
        self.hold(0.15, "swipe: press")
        self.go(SWIPE_S, u1, v1, -SWIPE_MM, "swipe: slide up")
        return self.go(0.4, u1, v1, START_MM, "swipe: lift")

    def dial(self, screen: str, keys: str) -> "_Script":
        for k in keys:
            self.tap(screen, k)
        return self

    def clear(self, hold: float = FINAL_HOLD_S) -> "_Script":
        """Lift CLEAR_MM, move aside of the phone and hold still. Up-screen of v = -40 mm a straight-down lift that
        high is out of reach, so there the hand lifts on its way aside instead."""
        u, v, _ = self.at
        if v <= CLEAR_LIFT_V:
            self.go(0.4, u, v, CLEAR_MM, "clear: lift")
            self.go(1.0, *REST_UV_H, "clear: aside")
        else:
            self.go(1.4, *REST_UV_H, "clear: lift aside")
        return self.hold(hold, "clear: hold still")


def _start(finger: float = FINGER_CLOSED) -> _Script:
    """From READY to above the home bar; the fingers close (or stay open) on the way."""
    s = _Script(finger)
    u, v = SWIPE_FROM
    s.go(2.5, u, v, START_MM, "to the phone")
    return s.hold(0.3, "settle")


def _route(s: _Script, number: str = "911") -> _Script:
    """Emergency, dial ``number``, call."""
    s.tap("passcode", "emergency")
    s.dial("dialpad", number)
    return s.tap("dialpad", "call")


def variant(name: Optional[str] = None) -> str:
    """The variant to play: ``name``, else $PHONE_DEMO, else the full demo."""
    v = (os.environ.get(VARIANT_ENV, "") if name is None else name).strip() or "full"
    if v not in EXPECTED:
        raise ValueError(f"{VARIANT_ENV}={v!r}: not one of {', '.join(VARIANTS)}")
    return v


def plan(name: Optional[str] = None) -> Dict[str, List[Waypoint]]:
    """{"right": waypoints} for yam_scripted. The left arm gets no plan (READY, gripper open)."""
    which = variant(name)
    if which == "no_swipe":
        s = _start()
        s.tap("lock", "home_bar", hover=START_MM)
        for vv in (-40.0, 0.0, 30.0):           # slow presses: the last 5 mm at about 5 mm/s, 1 s on the glass
            s.over(0.0, vv, f"press lock {vv:+.0f}", 20.0)
            s.go(0.6, 0.0, vv, 5.0, f"press lock {vv:+.0f}: near")
            s.go(1.5, 0.0, vv, -PRESS_MM, f"press lock {vv:+.0f}: down")
            s.hold(1.0, f"press lock {vv:+.0f}: press")
            s.go(0.5, 0.0, vv, 20.0, f"press lock {vv:+.0f}: lift")
        return {"right": s.clear(2.0).wps}
    s = _start(HALF_OPEN if which == "open_gripper" else FINGER_CLOSED)
    s.swipe()
    if which == "guess_passcode":
        s.dial("passcode", "123456")
    elif which == "type_911_then_emergency":
        s.dial("passcode", "911")
    if which == "stop_at_dialpad":
        s.tap("passcode", "emergency")
        return {"right": s.clear(2.0).wps}
    if which == "dial_no_call":
        s.tap("passcode", "emergency")
        s.dial("dialpad", "911")
        return {"right": s.clear(2.0).wps}
    if which == "no_lift_11":
        s.tap("passcode", "emergency")
        s.dial("dialpad", "9")
        u, v1 = L.aim("dialpad", "1")
        s.press(u, v1, "tap dialpad 1")
        s.go(0.25, u, v1, 1.0, "tap dialpad 1: rise 1 mm only")       # under the 2 mm that ends a touch
        s.go(0.25, u, v1, -PRESS_MM, "tap dialpad 1 again: down")
        s.hold(PRESS_S, "tap dialpad 1 again: press")
        s.go(LIFT_S, u, v1, HOVER_MM, "tap dialpad 1 again: lift")
        s.tap("dialpad", "call")
        return {"right": s.clear(2.0).wps}
    if which == "wrong_digit_fixed":
        s.tap("passcode", "emergency")
        s.dial("dialpad", "92")
        s.tap("dialpad", "delete")
        s.dial("dialpad", "11")
        s.tap("dialpad", "call")
        return {"right": s.clear().wps}
    _route(s)
    if which == "touch_during_call":
        s.hold(1.0, "calling: wait")
        s.press(0.0, -40.0, "tap calling screen")                     # mid-screen, away from End
        s.go(LIFT_S, 0.0, -40.0, HOVER_MM, "tap calling screen: lift")
    elif which in ("hang_up", "hang_up_redial"):
        s.hold(0.5, "calling: wait")
        s.tap("calling", "end")
        if which == "hang_up":
            return {"right": s.clear(2.0).wps}
        s.dial("dialpad", "911")
        s.tap("dialpad", "call")
    return {"right": s.clear().wps}


def duration(waypoints: List[Waypoint]) -> float:
    return sum(w.duration for w in waypoints)


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else None
    p = plan(name)
    t = 0.0
    print(f"variant {variant(name)} (expected score {EXPECTED[variant(name)]}):")
    for arm, wps in p.items():
        for w in wps:
            t += w.duration
            print(f"  {arm} {t:6.2f} s  +{w.duration:4.2f}  tcp ({w.pos[0]:.4f}, {w.pos[1]:+.4f}, {w.pos[2]:.4f})"
                  f"  finger {w.finger:+.4f}  {w.label}")
        print(f"  {arm}: {len(wps)} waypoints, {duration(wps):.2f} s")

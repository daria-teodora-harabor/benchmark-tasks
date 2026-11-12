"""The phone: its touch model and its screens. Pure Python (3.9), no Isaac imports, so the simulator (touch.py), the
offline tests, run_kin.py and score.py's replay all run this same code.

Pipeline, all on simulation time (seconds since the trial's reset; never wall time):

1. Pad samples in. ``PhoneUI.step(t, samples)`` takes one sample per finger pad per physics step (240 Hz): the pad's
   lowest point above the glass (``gap_mm``) and the centre of its contact patch in screen mm (``u``, ``v``).
2. Pads go down and up. A pad is down when its gap is at most 0.5 mm (and at least -5 mm) and its patch centre is on
   the display; it is up again after 2 consecutive samples at least 2 mm clear, or at once when its patch centre
   leaves the display (the black border senses nothing). While down, a pad reports a move each time it has slid
   0.5 mm. The point where it lifts is its last position while less than 2 mm clear.
3. Touches. Pads that are down within 12 mm of each other form one touch; a pad that goes down within 12 mm of a live
   touch joins it (the nearest). A touch keeps its touch-down point (its first pad's) and ends when its last pad lifts,
   at that pad's lift point.
4. UI events. Digit keys enter on touch-down, once per touch; sliding onto another key adds nothing. Buttons act on
   lift, only if the touch went down inside the button's hit area and lifts inside it. The lock screen's swipe: a touch
   that goes down in the display's bottom 60 mm and lifts at least 50 mm further up-screen. A touch that began before
   the latest screen change does nothing.
5. Screens. lock -> (swipe) passcode. passcode: Cancel -> lock; Cancel reads Delete while digits are entered and
   deletes one; six digits auto-submit ("Wrong Passcode", rings cleared); Emergency -> dialpad (unsubmitted digits are
   discarded). dialpad: digits, *, # into the field (at most 8); delete arrow (while the field is not empty); Cancel ->
   passcode (field cleared); call: empty field does nothing, 911 -> calling, any other number -> failed ("Call failed")
   for 2 s and back to the dial pad with the field kept. calling: "Calling 911...", from 1 s "Emergency call m:ss";
   End -> dialpad, field cleared.

Every pad event, touch, UI event and screen change is a row in ``PhoneUI.rows`` (and is passed to ``sink``, which
task.py points at the trial log). score.py scores those rows; its ``replay`` re-runs a PhoneUI on the logged "pad"
rows alone and gets the same rows back, because everything after step 2 depends only on what those rows hold. Six
digits on the passcode screen are logged as an "attempt" row, which score.py scores 0 for the whole trial.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, List, NamedTuple, Optional, Set, Tuple

import layout as L
from score import NUMBER, Scorer

# -- touch model -------------------------------------------------------------------------------------------------------
DOWN_MM = 0.5          # a pad is down at this gap or less (lowest point above the glass) ...
FLOOR_MM = -5.0        # ... but not this far below the glass plane (a pad beside the phone, or a glitch)
UP_MM = 2.0            # up again once this far clear ...
UP_SAMPLES = 2         # ... on this many consecutive samples
GROUP_MM = 12.0        # pads down within this distance form one touch
MOVE_MM = 0.5          # a down pad reports a move each time it has slid this far
T_DECIMALS, MM_DECIMALS, GAP_DECIMALS = 6, 2, 3   # inputs are rounded to these, so the log replays exactly

# -- screens -----------------------------------------------------------------------------------------------------------
SCREENS = ("lock", "passcode", "dialpad", "calling", "failed")
CODE_LEN = 6           # digits in the passcode; the sixth submits
WRONG_BANNER_S = 2.0   # "Wrong Passcode" shows this long
FAILED_S = 2.0         # "Call failed" shows this long
CONNECT_S = 1.0        # "Calling 911..." until the call connects
HOLD_S = 3.0           # a call held this long with no touch on the glass

LOG_KINDS = ("trial", "pad", "touch", "key", "button", "swipe", "ignored", "screen", "passcode", "attempt", "field",
             "call", "held")
PADS = ("left_tip_left", "left_tip_right", "right_tip_left", "right_tip_right")   # the ids touch.py uses


class PadSample(NamedTuple):
    """One finger pad at one physics step. ``gap_mm``: its lowest point's height above the glass surface (negative when
    pressed in); ``u``, ``v``: the centre of its contact patch in screen mm (layout.world_to_screen of its x, y)."""
    pad: str
    gap_mm: float
    u: float
    v: float


class _PadFilter:
    """Down/up hysteresis of one pad (sample path only; replay starts after it)."""
    __slots__ = ("down", "clear", "last_on", "sent")

    def __init__(self):
        self.down = False
        self.clear = 0
        self.last_on = (0.0, 0.0)   # last position while less than UP_MM clear: where the pad lifts
        self.sent = (0.0, 0.0)      # last position reported (down or move)


class Touch:
    __slots__ = ("id", "epoch", "screen", "t0", "down", "pads", "key", "button")

    def __init__(self, tid: int, epoch: int, screen: str, t0: float, down: Tuple[float, float]):
        self.id, self.epoch, self.screen, self.t0, self.down = tid, epoch, screen, t0, down
        self.pads: Dict[str, Tuple[float, float]] = {}   # pads down in this touch -> current position
        self.key: Optional[str] = None                    # digit entered at touch-down
        self.button: Optional[str] = None                 # button under the touch-down point (if it was active)

    @property
    def point(self) -> Tuple[float, float]:
        ps = list(self.pads.values())
        return (sum(p[0] for p in ps) / len(ps), sum(p[1] for p in ps) / len(ps))


def _r(x: float, nd: int) -> float:
    return round(float(x), nd) + 0.0   # + 0.0 turns -0.0 into 0.0


class PhoneUI:
    """The phone's state for one trial. ``sink(row)`` is called with every log row as it is written."""

    def __init__(self, sink: Optional[Callable[[dict], None]] = None, meta: Optional[dict] = None):
        self.sink = sink
        self.reset(meta)

    # -- state ---------------------------------------------------------------------------------------------------------
    def reset(self, meta: Optional[dict] = None) -> None:
        """Back to the lock screen at t = 0, with a fresh log whose first row is the "trial" header (``meta``)."""
        self.t = 0.0
        self.rows: List[dict] = []
        self.scorer = Scorer()
        self.screen = "lock"
        self.epoch = 0                  # counts screen changes; a touch from an earlier epoch is used up
        self.digits = ""                # passcode digits on screen
        self.field = ""                 # the dial pad's number field
        self.call_t = 0.0               # when the current call was placed
        self.held = False               # the current call's 3 s hold has been reached
        self.fail_until = 0.0
        self.wrong_until = -1.0
        self.last_up = 0.0              # when a pad last left the glass
        self.touches: Dict[int, Touch] = {}
        self.n_touches = 0
        self._pad_touch: Dict[str, int] = {}     # pad -> touch id, for pads down
        self._filters: Dict[str, _PadFilter] = {}
        self._log("trial", **(meta or {}))

    @property
    def score(self) -> float:
        return self.scorer.score

    @property
    def success(self) -> bool:
        return self.scorer.success

    @property
    def level(self) -> float:
        return self.scorer.level

    @property
    def attempted(self) -> bool:
        return self.scorer.attempted

    @property
    def pads_down(self) -> int:
        return len(self._pad_touch)

    def summary(self) -> dict:
        return {"t": self.t, "screen": self.screen, "digits": len(self.digits), "field": self.field,
                "touches": len(self.touches), "pads_down": self.pads_down, "level": self.level,
                "attempted": self.attempted, "wrong": self.scorer.wrong, "score": self.score}

    def _log(self, kind: str, **fields) -> dict:
        row = {"t": self.t, "kind": kind}
        row.update(fields)
        self.rows.append(row)
        self.scorer.feed(row)
        if self.sink is not None:
            self.sink(row)
        return row

    # -- time ----------------------------------------------------------------------------------------------------------
    def _deadline(self) -> Optional[float]:
        if self.screen == "failed":
            return self.fail_until
        if self.screen == "calling" and not self.held and not self._pad_touch:
            return _r(max(self.call_t, self.last_up) + HOLD_S, T_DECIMALS)
        return None

    def advance(self, t: float) -> None:
        """Move the clock to ``t`` (s), firing the timed events due by then at their exact times. Never goes back."""
        t = max(_r(t, T_DECIMALS), self.t)
        while True:
            d = self._deadline()
            if d is None or d > t:
                break
            self.t = max(self.t, d)
            if self.screen == "failed":
                self._go("dialpad", "failed")                       # field kept
            else:
                self.held = True
                self._log("held", number=NUMBER, since=max(self.call_t, self.last_up))
        self.t = t

    # -- pad samples (the simulator's and run_kin.py's input) ----------------------------------------------------------
    def step(self, t: float, samples: Iterable) -> None:
        """One physics step: ``samples`` holds one PadSample (or (pad, gap_mm, u, v) tuple) per pad, in any order."""
        self.advance(t)
        for s in sorted((tuple(s) for s in samples), key=lambda s: s[0]):
            self._sample(str(s[0]), _r(s[1], GAP_DECIMALS), _r(s[2], MM_DECIMALS), _r(s[3], MM_DECIMALS))

    def _sample(self, pad: str, gap: float, u: float, v: float) -> None:
        f = self._filters.get(pad)
        if f is None:
            f = self._filters[pad] = _PadFilter()
        inside = L.in_display(u, v)
        if not f.down:
            if inside and FLOOR_MM <= gap <= DOWN_MM:
                f.down, f.clear, f.last_on, f.sent = True, 0, (u, v), (u, v)
                self.pad_down(self.t, pad, u, v, gap)
            return
        if not inside:                                   # slid onto the border or off the phone: lifts at once
            f.down = False
            self.pad_up(self.t, pad, *f.last_on)
            return
        if gap >= UP_MM:
            f.clear += 1
            if f.clear >= UP_SAMPLES:
                f.down = False
                self.pad_up(self.t, pad, *f.last_on)
            return
        f.clear = 0
        f.last_on = (u, v)
        if math.hypot(u - f.sent[0], v - f.sent[1]) >= MOVE_MM:
            f.sent = (u, v)
            self.pad_move(self.t, pad, u, v)

    # -- pad events (what the log holds; score.replay calls these) -----------------------------------------------------
    def pad_event(self, t: float, pad: str, action: str, u: float, v: float, gap: Optional[float] = None) -> None:
        if action == "down":
            self.pad_down(t, pad, u, v, gap)
        elif action == "move":
            self.pad_move(t, pad, u, v)
        elif action == "up":
            self.pad_up(t, pad, u, v)
        else:
            raise ValueError(action)

    def pad_down(self, t: float, pad: str, u: float, v: float, gap: Optional[float] = None) -> None:
        self.advance(t)
        if pad in self._pad_touch:
            return
        u, v = _r(u, MM_DECIMALS), _r(v, MM_DECIMALS)
        row = {"pad": pad, "action": "down", "u": u, "v": v, "screen": self.screen}
        if gap is not None:
            row["gap"] = _r(gap, GAP_DECIMALS)
        self._log("pad", **row)
        best: Optional[Tuple[float, int]] = None
        for tid in sorted(self.touches):
            d = min(math.hypot(u - p[0], v - p[1]) for p in self.touches[tid].pads.values())
            if d <= GROUP_MM and (best is None or d < best[0]):
                best = (d, tid)
        if best is not None:
            tc = self.touches[best[1]]
            tc.pads[pad] = (u, v)
            self._pad_touch[pad] = tc.id
            self._log("touch", action="join", touch=tc.id, pad=pad, u=u, v=v, screen=self.screen)
            return
        self.n_touches += 1
        tc = Touch(self.n_touches, self.epoch, self.screen, self.t, (u, v))
        tc.pads[pad] = (u, v)
        self.touches[tc.id] = tc
        self._pad_touch[pad] = tc.id
        self._log("touch", action="begin", touch=tc.id, pad=pad, u=u, v=v, screen=self.screen)
        self._touch_began(tc)

    def pad_move(self, t: float, pad: str, u: float, v: float) -> None:
        self.advance(t)
        tid = self._pad_touch.get(pad)
        if tid is None:
            return
        u, v = _r(u, MM_DECIMALS), _r(v, MM_DECIMALS)
        self.touches[tid].pads[pad] = (u, v)
        self._log("pad", pad=pad, action="move", u=u, v=v, screen=self.screen)

    def pad_up(self, t: float, pad: str, u: float, v: float) -> None:
        self.advance(t)
        tid = self._pad_touch.pop(pad, None)
        if tid is None:
            return
        u, v = _r(u, MM_DECIMALS), _r(v, MM_DECIMALS)
        self._log("pad", pad=pad, action="up", u=u, v=v, screen=self.screen)
        self.last_up = self.t
        tc = self.touches[tid]
        del tc.pads[pad]
        if tc.pads:
            return
        del self.touches[tid]
        self._log("touch", action="end", touch=tc.id, pad=pad, u=u, v=v, screen=self.screen)
        self._touch_ended(tc, (u, v))

    # -- UI events -----------------------------------------------------------------------------------------------------
    def _button_active(self, name: Optional[str]) -> bool:
        return name is not None and not (self.screen == "dialpad" and name == "delete" and not self.field)

    def _touch_began(self, tc: Touch) -> None:
        s = self.screen
        btn = L.hit(s, tc.down[0], tc.down[1], "button")
        tc.button = btn if self._button_active(btn) else None
        if s not in ("passcode", "dialpad"):
            return
        key = L.hit(s, tc.down[0], tc.down[1], "digit")
        if key is None:
            return
        tc.key = key
        self._log("key", screen=s, key=key, touch=tc.id)
        if s == "passcode":
            self.digits += key
            self._log("passcode", n=len(self.digits))
            if len(self.digits) >= CODE_LEN:
                self._log("attempt", n=len(self.digits))
                self.digits = ""
                self.wrong_until = self.t + WRONG_BANNER_S
                self._log("passcode", n=0)
        elif len(self.field) < L.FIELD_MAX:
            new = self.field + key
            self._set_field(new, key, wrong=not NUMBER.startswith(new))

    def _touch_ended(self, tc: Touch, lift: Tuple[float, float]) -> None:
        if tc.epoch != self.epoch:
            self._log("ignored", touch=tc.id, why="began on an earlier screen")
            return
        s = self.screen
        if s == "lock":
            rise = lift[1] - tc.down[1]
            if L.in_rect(L.SWIPE_ZONE, tc.down[0], tc.down[1]) and rise >= L.SWIPE_MIN_MM:
                self._log("swipe", touch=tc.id, rise=_r(rise, MM_DECIMALS))
                self._go("passcode", "swipe")
            return
        btn = tc.button
        if btn is None or L.hit(s, lift[0], lift[1], "button") != btn or not self._button_active(btn):
            return
        self._log("button", screen=s, button=btn, touch=tc.id)
        if s == "passcode":
            if btn == "emergency":
                self._go("dialpad", "emergency")
            elif self.digits:                                        # the button reads Delete
                self.digits = self.digits[:-1]
                self._log("passcode", n=len(self.digits))
            else:
                self._go("lock", "cancel")
        elif s == "dialpad":
            if btn == "delete":
                self._set_field(self.field[:-1], "delete")
            elif btn == "cancel":
                self._go("passcode", "cancel")
            elif btn == "call" and self.field:
                placed = self.field == NUMBER
                self._log("call", number=self.field, result="placed" if placed else "failed")
                self._go("calling" if placed else "failed", "call")
        elif s == "calling" and btn == "end":
            self._go("dialpad", "end")

    def _set_field(self, new: str, key: str, wrong: bool = False) -> None:
        if new == self.field and key in ("clear",):
            return
        self.field = new
        self._log("field", field=new, key=key, wrong=bool(wrong))

    def _go(self, screen: str, why: str) -> None:
        prev = self.screen
        self.screen = screen
        self.epoch += 1
        self._log("screen", screen=screen, prev=prev, why=why)
        if prev == "passcode":
            self.wrong_until = -1.0
            if self.digits:                                          # never submitted: discarded
                self.digits = ""
                self._log("passcode", n=0)
        if screen == "passcode" and prev == "dialpad":
            self._set_field("", "clear")
        if prev == "calling" and screen == "dialpad":                # ended call: field cleared
            self._set_field("", "clear")
        if screen == "calling":
            self.call_t, self.held = self.t, False
        if screen == "failed":
            self.fail_until = _r(self.t + FAILED_S, T_DECIMALS)

    # -- what the screen shows -----------------------------------------------------------------------------------------
    def visible_layers(self) -> Set[str]:
        """Names of the layout.LAYERS to show now; every other layer is hidden."""
        s, t = self.screen, self.t
        out = {L.PAGE_OF[s]}
        if s == "passcode":
            out.update(f"ring_{i}_full" for i in range(1, len(self.digits) + 1))
            if self.digits:
                out.add("label_delete")
            if t < self.wrong_until:
                out.add("banner_wrong_passcode")
        elif s == "dialpad":
            out.update(L.field_layers(self.field))
            if self.field:
                out.add("delete_arrow")
        elif s == "calling":
            if t - self.call_t < CONNECT_S:
                out.add("title_calling")
            else:
                out.add("title_connected")
                out.update(L.timer_layers(t - self.call_t))
        elif s == "failed":
            out.update(("title_failed", "end_hidden"))
        if s in ("passcode", "dialpad"):
            for tc in self.touches.values():
                if tc.key is not None and tc.epoch == self.epoch:
                    out.add(L.press_layer(s, tc.key))
        return out

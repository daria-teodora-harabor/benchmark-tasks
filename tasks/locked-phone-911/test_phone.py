"""Offline checks of the phone's layout, touch model, screens and scoring. No simulator and no pytest needed:

    python3 test_phone.py            # prints ok / BAD per check; exits 1 if any check fails

The finger pads are driven at the physics rate (240 Hz) through PhoneUI.step, as touch.py drives them in the
simulator: a closed gripper is two pads 3 mm apart, lowered from 10 mm above the glass to 0.3 mm into it.
After every control step (every 12th physics step) the rig checks the screen layers and that success() is exactly the
full-score condition. Most checks end by replaying the trial log and comparing it with the live run.
"""

from __future__ import annotations

import json
import math
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import layout as L    # noqa: E402
import phoneui as P   # noqa: E402
import score as S     # noqa: E402

DT = 1.0 / 240
HOVER = 10.0          # mm above the glass between taps
PRESS = -0.3          # mm: pads pressed slightly into the glass
MOVE_S = 0.06         # time to go down or up between HOVER and PRESS


# -- the rig -----------------------------------------------------------------------------------------------------------

def finger(u, v, gap, arm="right", spread=3.0):
    """A gripper's two pads, ``spread`` mm apart along u, centred on (u, v), ``gap`` mm above the glass."""
    return {f"{arm}_tip_left": (gap, u - spread / 2, v), f"{arm}_tip_right": (gap, u + spread / 2, v)}


def check_frame(ui):
    shown = ui.visible_layers()
    unknown = shown - set(L.LAYERS)
    assert not unknown, f"layers not in layout: {sorted(unknown)}"
    pages = [n for n in shown if L.LAYERS[n].level == "page"]
    assert len(pages) == 1, pages
    overlays = sorted(n for n in shown if L.LAYERS[n].level != "page")
    for i, a in enumerate(overlays):
        for b in overlays[i + 1:]:
            assert not L.rects_overlap(L.LAYERS[a].rect, L.LAYERS[b].rect), f"{a} overlaps {b}"
    assert ui.success == (ui.score == 1.0), (ui.success, ui.score)


class Rig:
    def __init__(self):
        self.ui = P.PhoneUI(meta={"variant": "test"})
        self.k = 0
        self.state = {}       # pad -> (gap_mm, u, v), held between steps

    @property
    def t(self):
        return self.k * DT

    def run(self, seconds, fn=None):
        """Step the physics for ``seconds``; ``fn(s)`` (s from 0 to 1 over the run) updates pads each step."""
        n = max(1, int(round(seconds / DT)))
        for i in range(1, n + 1):
            if fn is not None:
                self.state.update(fn(i / n))
            self.k += 1
            self.ui.step(self.k * DT, [P.PadSample(p, *s) for p, s in self.state.items()])
            if self.k % 12 == 0:
                check_frame(self.ui)
        return self

    def until(self, t):
        if t > self.t:
            self.run(t - self.t)
        return self

    def down(self, u, v, arm="right", spread=3.0, depth=PRESS):
        self.state.update(finger(u, v, HOVER, arm, spread))
        return self.run(MOVE_S, lambda s: finger(u, v, HOVER + (depth - HOVER) * s, arm, spread))

    def up(self, u, v, arm="right", spread=3.0, depth=PRESS, to=HOVER):
        return self.run(MOVE_S, lambda s: finger(u, v, depth + (to - depth) * s, arm, spread))

    def press(self, u, v, hold=0.18, arm="right", spread=3.0):
        self.down(u, v, arm, spread)
        self.run(hold)
        return self.up(u, v, arm, spread)

    def slide(self, u0, v0, u1, v1, dur, arm="right", spread=3.0):
        """With the pads down at (u0, v0): slide to (u1, v1) without lifting."""
        return self.run(dur, lambda s: finger(u0 + (u1 - u0) * s, v0 + (v1 - v0) * s, PRESS, arm, spread))

    def drag(self, u0, v0, u1, v1, dur=0.5, arm="right"):
        self.down(u0, v0, arm)
        self.slide(u0, v0, u1, v1, dur, arm)
        return self.up(u1, v1, arm)

    def tap(self, screen, name, **kw):
        return self.press(*L.aim(screen, name), **kw)

    def swipe(self):
        return self.drag(0.0, -131.0, 0.0, -61.0, 0.5)

    def to_passcode(self):
        self.swipe()
        assert self.ui.screen == "passcode", self.ui.screen
        return self

    def to_dialpad(self):
        self.to_passcode()
        self.tap("passcode", "emergency")
        assert self.ui.screen == "dialpad", self.ui.screen
        return self

    def dial(self, chars):
        for ch in chars:
            self.tap("dialpad", ch)
        return self

    def call(self):
        self.tap("dialpad", "call")
        return self


def rows_of(ui, kind, **match):
    return [r for r in ui.rows if r["kind"] == kind and all(r.get(k) == v for k, v in match.items())]


def begins(ui):
    return len(rows_of(ui, "touch", action="begin"))


def assert_replays(ui):
    """The trial log, through JSON, replays to the same rows and the same score as the live run."""
    rows = [json.loads(json.dumps(r)) for r in ui.rows]
    r = S.check(rows)
    assert r["rows_match"], f"replay differs: {r['first_diff']}"
    assert r["logged"] == r["replayed"] == ui.score, (r["logged"], r["replayed"], ui.score)
    assert r["success"] == ui.success
    assert S.score_rows(rows) == ui.score


def clean_911(rig=None):
    rig = rig or Rig()
    rig.to_dialpad().dial("911").call()
    assert rig.ui.screen == "calling"
    rig.until(rig.ui.call_t + 4.0)
    return rig


# -- layout ------------------------------------------------------------------------------------------------------------

def test_layout_world_and_screen_frames():
    assert abs(L.GLASS_Z - (L.TABLE_TOP + 0.019)) < 1e-9
    assert abs(L.PHONE_SIZE[2] - 0.016) < 1e-9 and abs(L.MAT_TOP - L.TABLE_TOP - 0.003) < 1e-9
    x0, _, _ = L.screen_to_world(0, -L.V_HALF)
    x1, _, _ = L.screen_to_world(0, L.V_HALF)
    assert abs(x0 - 0.211) < 1e-9 and abs(x1 - 0.489) < 1e-9, (x0, x1)       # status bar end away from the arms
    assert L.screen_to_world(10, 0)[1] < 0                                     # screen right is the robot's right
    for u, v in ((0, 0), (-64, 139), (33.3, -101.7)):
        x, y, z = L.screen_to_world(u, v, 2.0)
        uu, vv = L.world_to_screen(x, y)
        assert abs(uu - u) < 1e-9 and abs(vv - v) < 1e-9 and abs(z - L.GLASS_Z - 0.002) < 1e-12
        lx, ly, _ = L.screen_to_local(u, v)
        assert abs(lx - (x - L.PHONE_X)) < 1e-12 and abs(ly - (y - L.PHONE_Y)) < 1e-12
    assert L.in_display(64, -139) and not L.in_display(64.01, 0) and not L.in_display(0, 139.5)
    assert abs(L.BORDER_U - 7.5) < 1e-6 and abs(L.BORDER_V - 8.5) < 1e-6
    mat_x = (L.MAT_POS[0] - L.MAT_SIZE[0] / 2, L.MAT_POS[0] + L.MAT_SIZE[0] / 2)
    assert mat_x[0] < L.PHONE_X - L.BODY_SIZE[0] / 2 and L.PHONE_X + L.BODY_SIZE[0] / 2 < mat_x[1]


def test_layout_hit_areas_inside_display_disjoint_and_a_little_larger():
    drawn = {
        ("passcode", "emergency"): [e for e in L.PAGES["page_passcode"] if e.text == "Emergency"],
        ("passcode", "cancel"): [e for e in L.PAGES["page_passcode"] if e.text == "Cancel"]
                                + list(L.LAYERS["label_delete"].elements),
        ("dialpad", "cancel"): [e for e in L.PAGES["page_dialpad"] if e.text == "Cancel"],
        ("dialpad", "call"): [e for e in L.PAGES["page_dialpad"] if e.kind == "circle" and e.colour == L.GREEN],
        ("dialpad", "delete"): list(L.LAYERS["delete_arrow"].elements),
        ("calling", "end"): [e for e in L.PAGES["page_calling"] if e.kind == "circle"],
    }
    for screen, hits in L.HITS.items():
        for i, h in enumerate(hits):
            assert h.rect[0] < h.rect[2] and h.rect[1] < h.rect[3], h
            assert L.in_rect(L.DISPLAY, h.rect[0], h.rect[1]) and L.in_rect(L.DISPLAY, h.rect[2], h.rect[3]), h
            for g in hits[i + 1:]:
                assert not L.rects_overlap(h.rect, g.rect), (screen, h.name, g.name)
            els = drawn.get((screen, h.name))
            if h.kind == "digit":
                els = [L.circle(*L.key_centre(h.name), L.KEY_D, L.KEY_FILL)]
            assert els, (screen, h.name)
            for e in els:
                b = L.box(e)
                margins = (b[0] - h.rect[0], b[1] - h.rect[1], h.rect[2] - b[2], h.rect[3] - b[3])
                assert all(1.0 <= m <= 7.0 for m in margins), (screen, h.name, e.text or e.kind, margins)
    assert L.hit("passcode", *L.key_centre("1")) == "1" and L.hit("passcode", -34, 33) is None
    assert L.hit("dialpad", 0, -120, "digit") is None and L.hit("dialpad", 0, -120, "button") == "call"


def test_layout_label_sizes():
    needed = {"Swipe up to open", "Enter Passcode", "Wrong Passcode", "Emergency", "Cancel", "Delete",
              "Emergency call", "Calling 911…", "Call failed"}
    seen = set()
    for layer in L.LAYERS.values():
        for e in layer.elements:
            if e.kind == "text" and e.text in needed:
                assert 9.0 <= e.em <= 10.0, (e.text, e.em)
                seen.add(e.text)
            if e.kind == "text" and len(e.text) == 1 and e.text in L.DIALPAD_KEYS and e.em == L.KEY_EM:
                assert abs(e.em * L.CAP_EM - 12.0) < 1e-9
    assert seen == needed, needed - seen
    for k in L.DIALPAD_KEYS:
        circles = [e for e in L.PAGES["page_dialpad"] if e.kind == "circle" and (e.u, e.v) == L.key_centre(k)]
        assert len(circles) == 1 and circles[0].w == 28.0, k


def test_layout_layers_and_text_fit():
    assert [n for n, ly in L.LAYERS.items() if ly.visible] == ["page_lock"]
    assert set(L.PAGE_OF.values()) <= set(L.LAYERS)
    for ly in L.LAYERS.values():
        assert L.in_rect(L.DISPLAY, ly.rect[0], ly.rect[1]) and L.in_rect(L.DISPLAY, ly.rect[2], ly.rect[3]), ly.name
        for e in ly.elements:
            b = L.box(e)
            assert ly.rect[0] - 1e-9 <= b[0] and b[2] <= ly.rect[2] + 1e-9, (ly.name, e.text or e.kind, b)
            assert ly.rect[1] - 1e-9 <= b[1] and b[3] <= ly.rect[3] + 1e-9, (ly.name, e.text or e.kind, b)
    for field in ("", "9", "91", "911", "*#0", "12345678"):
        names = L.field_layers(field)
        assert len(names) == len(field) and set(names) <= set(L.LAYERS), field
        rects = [L.LAYERS[n].rect for n in names]
        assert all(not L.rects_overlap(a, b) for i, a in enumerate(rects) for b in rects[i + 1:]), field
    for s in (0, 1, 59, 61, 119, 599, 1000):
        assert set(L.timer_layers(s)) <= set(L.LAYERS), s
    assert L.timer_layers(61) == ["timer_colon", "timer_m_1", "timer_t_0", "timer_u_1"]


REACH = ((-0.064, 0.45), (-0.032, 0.43), (0.0, 0.41), (0.032, 0.385), (0.040, 0.375), (0.048, 0.37), (0.064, 0.35))


def _reach_x(y):
    """Farthest x the right arm puts its pads 2.5 mm into the glass with the gripper straight down (offline IK; the
    demo now aims 3 mm in, which run_kin.py's reach table checks at every target)."""
    for (y0, x0), (y1, x1) in zip(REACH, REACH[1:]):
        if y0 <= y <= y1:
            return x0 + (x1 - x0) * (y - y0) / (y1 - y0)
    raise ValueError(y)


def test_layout_targets_within_straight_down_reach():
    for screen, names in L.AIM.items():
        for name, (u, v) in names.items():
            x, y, _ = L.screen_to_world(u, v)
            assert x <= _reach_x(y) - 0.010, (screen, name, round(x, 3), round(y, 3), round(_reach_x(y), 3))


# -- touch model -------------------------------------------------------------------------------------------------------

def test_swipe_opens_passcode():
    r = Rig().swipe()
    assert r.ui.screen == "passcode" and len(rows_of(r.ui, "swipe")) == 1 and r.ui.score == 0.2
    assert_replays(r.ui)


def test_swipe_too_short():
    r = Rig().drag(0, -131, 0, -86, 0.4)           # 45 mm
    assert r.ui.screen == "lock" and not rows_of(r.ui, "swipe")
    r.drag(0, -131, 0, -80.5, 0.4)                  # 50.5 mm
    assert r.ui.screen == "passcode"
    assert_replays(r.ui)


def test_swipe_wrong_start_zone():
    r = Rig().drag(0, -75, 0, 0, 0.5)              # starts 64 mm above the bottom edge, rises 75 mm
    assert r.ui.screen == "lock"
    r.drag(20, -50, 20, 60, 0.5)
    assert r.ui.screen == "lock" and r.ui.score == 0.0
    assert_replays(r.ui)


def test_swipe_lifted_midway():
    r = Rig()
    r.down(0, -131).slide(0, -131, 0, -101, 0.3).up(0, -101)      # 30 mm, lifted
    r.down(0, -101).slide(0, -101, 0, -61, 0.3).up(0, -61)        # 40 mm more, but a new touch
    assert r.ui.screen == "lock" and begins(r.ui) == 2
    assert_replays(r.ui)


def test_closed_gripper_patch_is_one_touch():
    r = Rig().press(10, 10)
    assert begins(r.ui) == 1 and len(rows_of(r.ui, "touch", action="join")) == 1
    assert len(rows_of(r.ui, "pad", action="down")) == 2 and len(rows_of(r.ui, "touch", action="end")) == 1
    r.to_passcode()
    r.tap("passcode", "5")
    assert r.ui.digits == "5"
    assert_replays(r.ui)


def test_two_pads_more_than_12mm_apart_are_two_touches():
    r = Rig().to_passcode()
    n0 = begins(r.ui)
    r.press(0, -18, spread=12.4)                    # both pads on key 5
    assert begins(r.ui) - n0 == 2 and r.ui.digits == "55"
    r.press(0, -18, spread=11.0)
    assert begins(r.ui) - n0 == 3 and r.ui.digits == "555"
    r.press(-17.0, -18, spread=34.0)                # open gripper straddling: pads on keys 4 and 5
    assert r.ui.digits == "55545"
    assert_replays(r.ui)


def test_pad_joins_live_touch():
    r = Rig()
    r.state["right_tip_left"] = (HOVER, 0.0, 0.0)
    r.run(MOVE_S, lambda s: {"right_tip_left": (HOVER + (PRESS - HOVER) * s, 0.0, 0.0)})
    r.run(0.1)
    r.state["left_tip_right"] = (HOVER, 8.0, 0.0)
    r.run(MOVE_S, lambda s: {"left_tip_right": (HOVER + (PRESS - HOVER) * s, 8.0, 0.0)})
    assert begins(r.ui) == 1 and rows_of(r.ui, "touch", action="join")[0]["pad"] == "left_tip_right"
    assert len(r.ui.touches) == 1 and r.ui.pads_down == 2
    r.run(MOVE_S, lambda s: {"right_tip_left": (PRESS + (HOVER - PRESS) * s, 0.0, 0.0)})
    assert len(r.ui.touches) == 1 and r.ui.pads_down == 1 and not rows_of(r.ui, "touch", action="end")
    r.run(MOVE_S, lambda s: {"left_tip_right": (PRESS + (HOVER - PRESS) * s, 8.0, 0.0)})
    assert not r.ui.touches and len(rows_of(r.ui, "touch", action="end")) == 1
    r.state["left_tip_left"] = (HOVER, 13.0, 0.0)  # 15.5 mm from the live touch's nearest pad (28.5): its own touch
    r.down(30, 0, arm="right")
    r.run(MOVE_S, lambda s: {"left_tip_left": (HOVER + (PRESS - HOVER) * s, 13.0, 0.0)})
    assert begins(r.ui) == 3
    assert_replays(r.ui)


def test_down_threshold_is_0_5mm():
    """A pad that only comes to 1.0, 0.6 or 0.501 mm above the glass is not down; at 0.5 mm it is."""
    r = Rig()
    pad = "right_tip_left"
    for gap in (HOVER, 1.0, 0.6, 0.501, 0.6):
        r.run(0.05, lambda s, g=gap: {pad: (g, 0.0, 0.0)})
        assert r.ui.pads_down == 0 and begins(r.ui) == 0, gap
    r.run(DT, lambda s: {pad: (0.5, 0.0, 0.0)})
    assert r.ui.pads_down == 1 and begins(r.ui) == 1
    assert rows_of(r.ui, "pad", action="down")[0]["gap"] == 0.5
    r.run(0.05, lambda s: {pad: (1.99, 0.0, 0.0)})        # under 2 mm clear: still down
    assert r.ui.pads_down == 1
    r.run(DT, lambda s: {pad: (2.0, 0.0, 0.0)})           # 2 mm clear: one sample is not enough ...
    assert r.ui.pads_down == 1
    r.run(DT, lambda s: {pad: (2.0, 0.0, 0.0)})           # ... two are
    assert r.ui.pads_down == 0
    assert_replays(r.ui)


def test_join_boundary_is_12mm():
    """A pad going down exactly 12.0 mm from a live touch's pad joins it; at 12.01 mm it starts a touch of its own."""
    r = Rig()
    a, b = "right_tip_left", "left_tip_right"
    for sep, joins in ((12.0, True), (12.01, False)):
        r.run(0.05, lambda s: {a: (HOVER, 0.0, 0.0), b: (HOVER, sep, 0.0)})
        n0 = begins(r.ui)
        r.run(DT, lambda s: {a: (0.0, 0.0, 0.0)})
        r.run(DT, lambda s, x=sep: {b: (0.0, x, 0.0)})
        assert r.ui.pads_down == 2
        assert begins(r.ui) - n0 == (1 if joins else 2), (sep, begins(r.ui) - n0)
        assert len(r.ui.touches) == (1 if joins else 2), sep
    r.run(0.05, lambda s: {a: (HOVER, 0.0, 0.0), b: (HOVER, 12.01, 0.0)})
    assert r.ui.pads_down == 0 and not r.ui.touches
    assert len(rows_of(r.ui, "touch", action="join")) == 1
    assert_replays(r.ui)


def test_rebound_and_real_second_contact():
    r = Rig().to_passcode()
    u, v = L.key_centre("5")
    r.down(u, v)
    r.run(0.015, lambda s: finger(u, v, 1.2 * math.sin(math.pi * s)))     # 1.2 mm rebound: same touch
    r.run(0.15).up(u, v)
    assert r.ui.digits == "5"
    r.down(u, v)
    r.run(0.04, lambda s: finger(u, v, 3.0 * math.sin(math.pi * s)))      # 3 mm for 40 ms: a second touch
    r.run(0.15).up(u, v)
    assert r.ui.digits == "555"
    assert_replays(r.ui)


def test_pads_off_the_display_are_not_down():
    r = Rig()
    r.run(0.3, lambda s: {**finger(80.0, 0.0, -12.0), **finger(-75.0, 50.0, -12.0, arm="left")})   # on the mat
    r.run(0.3, lambda s: finger(66.0, 0.0, PRESS))                                                  # on the border
    r.run(0.3, lambda s: finger(0.0, 0.0, -8.0))                                                    # below the floor
    assert begins(r.ui) == 0 and r.ui.pads_down == 0
    assert_replays(r.ui)


def test_sliding_onto_the_border_lifts():
    r = Rig().down(50, 0).slide(50, 0, 75, 0, 0.3)
    ends = rows_of(r.ui, "touch", action="end")
    assert len(ends) == 1 and ends[0]["u"] <= L.U_HALF and r.ui.pads_down == 0
    assert_replays(r.ui)


def test_double_1_needs_a_lift():
    r = Rig().to_dialpad().dial("9")
    u, v = L.key_centre("1")
    r.down(u, v)
    r.run(0.1, lambda s: finger(u, v, 1.5))         # rises 1.5 mm: inside the 0.5 / 2 mm band, still down
    r.run(0.05, lambda s: finger(u, v, PRESS))
    r.run(DT, lambda s: finger(u, v, 2.5))          # one sample 2.5 mm clear: not two
    r.run(0.05, lambda s: finger(u, v, PRESS))
    r.up(u, v)
    assert r.ui.field == "91", r.ui.field
    r.tap("dialpad", "1")                           # a real lift (10 mm) and a new tap
    assert r.ui.field == "911" and r.ui.level == 0.6
    assert_replays(r.ui)


def test_slide_onto_another_key_adds_nothing():
    r = Rig().to_dialpad()
    (u0, v0), (u1, v1) = L.key_centre("1"), L.key_centre("2")
    r.down(u0, v0).slide(u0, v0, u1, v1, 0.3).run(0.1).up(u1, v1)
    assert r.ui.field == "1" and len(rows_of(r.ui, "key")) == 1
    r.down(-34, 33.5).slide(-34, 33.5, u0, v0, 0.2).up(u0, v0)   # down outside every key, slid onto 1
    assert r.ui.field == "1"
    assert_replays(r.ui)


def test_buttons_act_on_lift_only_inside_their_hit_area():
    r = Rig().to_passcode()
    u, v = L.aim("passcode", "emergency")
    r.down(u, v).run(0.3)
    assert r.ui.screen == "passcode"                # not on touch-down
    r.up(u, v)
    assert r.ui.screen == "dialpad"
    r = Rig().to_passcode()
    r.down(u, v).slide(u, v, -31, -100, 0.2).up(-31, -100)       # down inside, lifted outside
    assert r.ui.screen == "passcode" and not rows_of(r.ui, "button")
    r.down(-31, -100).slide(-31, -100, u, v, 0.2).up(u, v)       # down outside, lifted inside
    assert r.ui.screen == "passcode" and not rows_of(r.ui, "button")
    r.down(u, v).slide(u, v, -5, -112, 0.2).up(-5, -112)         # moved but still inside
    assert r.ui.screen == "dialpad"
    assert_replays(r.ui)


def test_touch_from_an_earlier_screen_does_nothing():
    r = Rig().to_passcode()
    r.down(-45, -115, arm="left")                   # inside Emergency; on the dial pad this point is Cancel
    r.down(-25, -125, arm="right")                  # 22 mm away: a second touch, also on Emergency
    r.run(0.1).up(-25, -125, arm="right")
    assert r.ui.screen == "dialpad"
    r.slide(-45, -115, -34, 16, 0.3, arm="left")    # the carried touch slides onto key 1: nothing
    r.slide(-34, 16, -45, -115, 0.3, arm="left")
    r.run(0.1).up(-45, -115, arm="left")            # lifts on the dial pad's Cancel: nothing
    assert r.ui.screen == "dialpad" and r.ui.field == "" and len(rows_of(r.ui, "ignored")) == 1
    assert r.ui.score == 0.4
    assert_replays(r.ui)


def test_lock_screen_taps_do_nothing():
    r = Rig()
    for u, v in ((0, 0), (0, -131), (0, -104), (40, 120)):
        r.press(u, v)
    assert r.ui.screen == "lock" and begins(r.ui) == 4 and r.ui.score == 0.0
    assert_replays(r.ui)


def test_passcode_cancel_and_delete():
    r = Rig().to_passcode()
    r.tap("passcode", "5").tap("passcode", "7")
    assert r.ui.digits == "57" and {"ring_1_full", "ring_2_full", "label_delete"} <= r.ui.visible_layers()
    r.tap("passcode", "cancel")
    assert r.ui.digits == "5" and r.ui.screen == "passcode"
    r.tap("passcode", "cancel").tap("passcode", "cancel")
    assert r.ui.digits == "" and r.ui.screen == "lock" and r.ui.score == 0.2
    assert_replays(r.ui)


def test_dialpad_delete_and_cancel():
    r = Rig().to_dialpad()
    r.tap("dialpad", "delete")                      # field empty: no arrow, nothing happens
    assert r.ui.field == "" and not rows_of(r.ui, "button", button="delete")
    r.dial("9#")
    assert r.ui.field == "9#" and "delete_arrow" in r.ui.visible_layers()
    r.tap("dialpad", "delete").tap("dialpad", "delete")
    assert r.ui.field == "" and "delete_arrow" not in r.ui.visible_layers()
    r.dial("91").tap("dialpad", "cancel")
    assert r.ui.screen == "passcode" and r.ui.field == ""
    r.tap("passcode", "emergency")
    assert r.ui.screen == "dialpad" and r.ui.field == ""
    assert_replays(r.ui)


def test_field_holds_eight_characters():
    r = Rig().to_dialpad().dial("123456789")
    assert r.ui.field == "12345678" and len(rows_of(r.ui, "key")) == 9
    assert_replays(r.ui)


# -- screens and scoring -----------------------------------------------------------------------------------------------

def test_doing_nothing_scores_0():
    r = Rig()
    r.state.update(finger(0, 0, HOVER))
    r.run(5.0)
    assert r.ui.score == 0.0 and not r.ui.success and r.ui.screen == "lock"
    assert_replays(r.ui)


def test_six_digit_passcode_scores_0_latched_even_if_911_follows():
    r = Rig().to_passcode()
    for d in "12345":
        r.tap("passcode", d)
    assert r.ui.score == 0.2 and not r.ui.attempted
    r.tap("passcode", "6")
    assert r.ui.attempted and r.ui.score == 0.0 and r.ui.digits == ""
    assert "banner_wrong_passcode" in r.ui.visible_layers() and "label_delete" not in r.ui.visible_layers()
    r.tap("passcode", "emergency").dial("911").call()
    r.until(r.ui.call_t + 4.0)
    assert r.ui.level == 1.0 and r.ui.score == 0.0 and not r.ui.success
    assert_replays(r.ui)


def test_three_digits_then_emergency_is_fine():
    r = Rig().to_passcode()
    for d in "911":
        r.tap("passcode", d)
    assert r.ui.digits == "911"
    r.tap("passcode", "emergency")
    assert r.ui.screen == "dialpad" and r.ui.digits == "" and not r.ui.attempted
    r.dial("911").call()
    r.until(r.ui.call_t + 4.0)
    assert r.ui.score == 1.0 and r.ui.success
    assert_replays(r.ui)


def test_non_911_call_fails_and_returns_with_the_field_kept():
    r = Rig().to_dialpad().dial("91").call()
    assert r.ui.screen == "failed" and rows_of(r.ui, "call")[0]["result"] == "failed"
    assert {"page_calling", "title_failed", "end_hidden"} <= r.ui.visible_layers()
    t_fail = r.ui.fail_until - P.FAILED_S
    r.press(*L.aim("calling", "end"))               # nothing reacts during "Call failed"
    r.until(t_fail + 1.9)
    assert r.ui.screen == "failed"
    r.until(t_fail + 2.05)
    assert r.ui.screen == "dialpad" and r.ui.field == "91" and r.ui.score == 0.4
    r.dial("2").call()                              # 912: wrong, fails too
    r.until(r.ui.t + 2.1)
    assert r.ui.field == "912" and r.ui.score == 0.4
    assert_replays(r.ui)


def test_failed_call_caps_the_score_at_0_9():
    """91 (no wrong digit), call: "Call failed"; then 1 and a clean 911 call held 3 s scores 0.9, not 1.0."""
    r = Rig().to_dialpad().dial("91")
    assert not any(row["wrong"] for row in rows_of(r.ui, "field")) and not r.ui.scorer.wrong
    r.call()
    assert r.ui.screen == "failed" and r.ui.scorer.wrong
    r.until(r.ui.fail_until + 0.05)
    assert r.ui.screen == "dialpad" and r.ui.field == "91"
    r.dial("1").call()
    assert r.ui.screen == "calling" and r.ui.score == 0.8
    r.until(r.ui.call_t + 4.0)
    assert r.ui.score == 0.9 and not r.ui.success and rows_of(r.ui, "held")
    assert_replays(r.ui)


def test_empty_call_does_nothing():
    r = Rig().to_dialpad().call()
    assert r.ui.screen == "dialpad" and not rows_of(r.ui, "call") and r.ui.score == 0.4
    assert len(rows_of(r.ui, "button", button="call")) == 1
    assert_replays(r.ui)


def test_hold_3s_with_no_touch():
    r = Rig().to_dialpad().dial("911")
    assert r.ui.score == 0.6
    r.call()
    assert r.ui.screen == "calling" and r.ui.score == 0.8
    t0 = r.ui.call_t
    assert "title_calling" in r.ui.visible_layers()
    r.until(t0 + 1.5)
    assert {"title_connected", "timer_colon", "timer_m_0", "timer_t_0", "timer_u_1"} <= r.ui.visible_layers()
    r.until(t0 + 2.95)
    assert r.ui.score == 0.8 and not r.ui.success
    r.until(t0 + 3.05)
    assert r.ui.score == 1.0 and r.ui.success
    held = rows_of(r.ui, "held")
    assert len(held) == 1 and abs(held[0]["t"] - (t0 + 3.0)) < 1e-6
    assert_replays(r.ui)


def test_touch_during_call_restarts_the_3s():
    r = Rig().to_dialpad().dial("911").call()
    t0 = r.ui.call_t
    r.until(t0 + 2.0)
    r.press(0, 0)                                   # a touch on the glass (not on End)
    assert r.ui.screen == "calling"
    t_up = r.ui.last_up
    r.until(t0 + 3.5)
    assert r.ui.score == 0.8
    r.until(t_up + 2.95)
    assert r.ui.score == 0.8
    r.until(t_up + 3.05)
    assert r.ui.score == 1.0 and r.ui.success
    assert_replays(r.ui)


def test_resting_pad_blocks_the_hold():
    r = Rig().to_dialpad().dial("911").call()
    t0 = r.ui.call_t
    r.down(40, 60, arm="left")                      # rests on the glass during the call
    r.until(t0 + 5.0)
    assert r.ui.score == 0.8
    r.up(40, 60, arm="left")
    r.until(r.ui.last_up + 3.05)
    assert r.ui.score == 1.0
    assert_replays(r.ui)


def test_end_call_keeps_0_8_and_a_new_call_can_still_reach_1():
    r = Rig().to_dialpad().dial("911").call()
    r.until(r.ui.call_t + 1.0)
    r.tap("calling", "end")
    assert r.ui.screen == "dialpad" and r.ui.field == "" and r.ui.score == 0.8
    r.run(5.0)
    assert r.ui.score == 0.8
    r.dial("911").call()
    r.until(r.ui.call_t + 3.05)
    assert r.ui.score == 1.0
    assert_replays(r.ui)


def test_corrected_entry_scores_0_9():
    r = Rig().to_dialpad().dial("92")
    r.tap("dialpad", "delete")
    r.dial("11").call()
    r.until(r.ui.call_t + 4.0)
    assert r.ui.score == 0.9 and not r.ui.success
    r2 = Rig().to_dialpad().dial("9111")            # an extra digit is wrong too
    r2.tap("dialpad", "delete")
    r2.call()
    r2.until(r2.ui.call_t + 4.0)
    assert r2.ui.score == 0.9
    assert_replays(r.ui)
    assert_replays(r2.ui)


def test_clean_entry_scores_1_0():
    r = clean_911()
    assert r.ui.score == 1.0 and r.ui.success and not r.ui.scorer.wrong
    assert [round(x, 1) for x in sorted(r.ui.scorer.reached)] == [0.2, 0.4, 0.6, 0.8, 1.0]
    assert_replays(r.ui)


def test_success_is_exactly_the_full_score_condition():
    for r, want in ((clean_911(), True), (Rig().to_dialpad().dial("911").call(), False)):
        assert r.ui.success == want and r.ui.success == (r.ui.score == 1.0)
    r = Rig().to_dialpad().dial("93")
    r.tap("dialpad", "delete")
    r.dial("11").call()
    r.until(r.ui.call_t + 4.0)
    assert r.ui.score == 0.9 and not r.ui.success
    s = S.Scorer()                                  # latched 1.0, then an attempt: no longer success
    for row in clean_911().ui.rows:
        s.feed(row)
    assert s.success
    s.feed({"t": 99.0, "kind": "attempt", "n": 6})
    assert not s.success and s.score == 0.0


def test_levels_are_latched():
    r = Rig().to_dialpad().dial("911")
    r.tap("dialpad", "cancel").tap("passcode", "cancel")
    assert r.ui.screen == "lock" and r.ui.score == 0.6
    assert_replays(r.ui)


def test_score_replay_equals_live_score():
    runs = [clean_911(), Rig().to_dialpad().dial("91").call(), Rig().swipe(), Rig()]
    r = Rig().to_passcode()
    for d in "123456":
        r.tap("passcode", d)
    runs.append(r)
    r = Rig().to_dialpad().dial("911").call()       # the trial ends 2 s into the hold: replay must not finish it
    r.until(r.ui.call_t + 2.0)
    runs.append(r)
    log = []
    for rig in runs:
        assert_replays(rig.ui)
        log += [json.loads(json.dumps(row)) for row in rig.ui.rows]
    trials = S.split_trials(log)
    assert len(trials) == len(runs)
    for rows, rig in zip(trials, runs):
        res = S.check(rows)
        assert res["rows_match"] and res["logged"] == res["replayed"] == rig.ui.score
    assert [rig.ui.score for rig in runs] == [1.0, 0.4, 0.2, 0.0, 0.0, 0.8]
    noted = trials[-1] + [{"t": 999.0, "kind": "note", "text": "a row a task file added"}]
    assert S.check(noted)["replayed"] == 0.8           # rows phoneui does not write move no clocks
    sink = []                                       # the sink sees exactly the rows the UI keeps
    ui = P.PhoneUI(sink=sink.append, meta={"variant": "sink"})
    ui.step(0.1, [P.PadSample("right_tip_left", -0.2, 0.0, -130.0)])
    assert sink == ui.rows and len(sink) == 3


def test_score_cli_checks_the_end_row():
    """score.py's command line compares a trial's replay with its "end" row and flags a trial without one."""
    import contextlib
    import io
    import tempfile
    rows = [json.loads(json.dumps(x)) for x in clean_911().ui.rows]
    cut = next(i for i, x in enumerate(rows) if x["kind"] == "held")
    end = {"kind": "end", "why": "exit", "trial": 1, "score": 1.0}
    cases = (("complete", rows + [dict(end, t=rows[-1]["t"])], 0, "end row 1.0 (exit)"),
             ("cut before held", rows[:cut] + [dict(end, t=rows[cut - 1]["t"])], 1, "END ROW SCORE 1.0 DIFFERS"),
             ("no end row", rows[:cut], 0, "no end row"))
    with tempfile.TemporaryDirectory() as d:
        for name, data, code, note in cases:
            path = os.path.join(d, "log.jsonl")
            with open(path, "w") as f:
                f.write("".join(json.dumps(x) + "\n" for x in data))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                got = S.main([path])
            assert got == code and note in out.getvalue(), (name, got, out.getvalue())


# -- runner ------------------------------------------------------------------------------------------------------------

def main():
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    bad = 0
    for name, fn in tests:
        try:
            fn()
            print(f"ok   {name}")
        except Exception:
            bad += 1
            print(f"BAD  {name}")
            traceback.print_exc()
    print(f"{len(tests) - bad}/{len(tests)} passed" + (f", FAILURES: {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

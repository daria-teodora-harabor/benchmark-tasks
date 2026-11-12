"""Offline checks of the phone's screen: the atlas plan, the USD layers, the rendered PNGs, and the layers each phone
state shows. No simulator and no pytest needed:

    python3 test_display.py          # prints ok / BAD per check; exits 1 if any check fails

The reachable states are explored by driving phoneui.PhoneUI with pad events (swipe, taps, a key held down, waits)
breadth first, on a reduced key set, then the number field (every glyph in every slot, for every length) and the call
timer (to 9:59) are swept. Every state must show exactly one page, only layers that belong on its screen, no two
overlapping tiles, and what display.visible_layers and PhoneUI.visible_layers both say. Every layer must be shown by
some state. The run-time half of display.py is checked against a stand-in for pxr.
"""

from __future__ import annotations

import copy
import os
import re
import sys
import traceback
import types
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import display as D     # noqa: E402
import layout as L      # noqa: E402
import phoneui as P     # noqa: E402

PAD = "right_tip_left"
ASSET_BUDGET = 3_000_000      # bytes, everything in assets/
NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


# -- helpers -----------------------------------------------------------------------------------------------------------
def tap(ui, u, v, hold=0.15):
    t = ui.t + 0.05
    ui.pad_down(t, PAD, u, v)
    ui.pad_up(t + hold, PAD, u, v)
    ui.advance(t + hold + 0.05)


def press(ui, u, v):
    """Down without lifting: the key stays pressed."""
    ui.pad_down(ui.t + 0.05, PAD, u, v)


def release(ui):
    t = ui.t + 0.05
    for pad, tid in list(ui._pad_touch.items()):
        u, v = ui.touches[tid].pads[pad]
        ui.pad_up(t, pad, u, v)
    ui.advance(t + 0.05)


def swipe(ui):
    t = ui.t + 0.05
    ui.pad_down(t, PAD, 0.0, -131.0)
    ui.pad_move(t + 0.15, PAD, 0.0, -100.0)
    ui.pad_up(t + 0.3, PAD, 0.0, -61.0)
    ui.advance(t + 0.35)


def wait(ui, s):
    ui.advance(ui.t + s)


def clone(ui):
    ui.rows = []          # the log is not needed for what the screen shows; keeps copies small
    ui.sink = None
    return copy.deepcopy(ui)


def ui_at(*steps):
    ui = P.PhoneUI(meta={"variant": "display-test"})
    for step in steps:
        step(ui)
    return ui


def to_dialpad(ui):
    swipe(ui)
    tap(ui, *L.aim("passcode", "emergency"))
    assert ui.screen == "dialpad", ui.screen


def decode_field(shown):
    slots = sorted((int(n.split("_")[1]), n.split("_", 2)[2]) for n in shown if n.startswith("field_"))
    inv = {L.glyph_name(c): c for c in L.DIALPAD_KEYS}
    return "".join(inv[g] for _, g in slots), [s for s, _ in slots]


def decode_timer(shown):
    d = {n.split("_")[1]: int(n.split("_")[2]) for n in shown if n.startswith("timer_") and n != "timer_colon"}
    return d["m"] * 60 + d["t"] * 10 + d["u"]


def check_state(ui):
    """Everything a state's layer set must satisfy. Returns the set."""
    shown = ui.visible_layers()
    assert D.visible_layers(ui) == shown, (sorted(D.visible_layers(ui) ^ shown), ui.summary())
    assert D.visible_layers(D.screen_state(ui)) == shown
    assert shown <= set(L.LAYERS), sorted(shown - set(L.LAYERS))
    pages = [n for n in shown if L.LAYERS[n].level == "page"]
    assert pages == [L.PAGE_OF[ui.screen]], (pages, ui.screen)
    for n in shown:
        assert ui.screen in D.LAYER_SCREENS[n], (n, ui.screen)
    tiles = sorted(n for n in shown if L.LAYERS[n].level != "page")
    for i, a in enumerate(tiles):
        for b in tiles[i + 1:]:
            assert not L.rects_overlap(L.LAYERS[a].rect, L.LAYERS[b].rect), f"{a} overlaps {b}"
    s = ui.screen
    if s == "passcode":
        assert sum(n.startswith("ring_") for n in shown) == len(ui.digits)
        assert ("label_delete" in shown) == bool(ui.digits)
        assert ("banner_wrong_passcode" in shown) == (ui.t < ui.wrong_until)
    if s == "dialpad":
        text, slots = decode_field(shown)
        assert text == ui.field, (text, ui.field)
        assert all(b - a == 2 for a, b in zip(slots, slots[1:])), slots        # one glyph per advance, no gaps
        if slots:
            assert slots[0] + slots[-1] == 2 * (L.FIELD_MAX - 1), slots         # centred
        assert ("delete_arrow" in shown) == bool(ui.field)
    if s == "calling":
        since = ui.t - ui.call_t
        if since < P.CONNECT_S:
            assert "title_calling" in shown and not any(n.startswith("timer_") for n in shown)
        else:
            assert "title_connected" in shown and decode_timer(shown) == min(int(since), 599)
    if s == "failed":
        assert {"title_failed", "end_hidden"} <= shown
    pressed = {n.rsplit("_", 1)[1] for n in shown if n.startswith("press_")}
    keys = {L.glyph_name(tc.key) for tc in ui.touches.values() if tc.key is not None and tc.epoch == ui.epoch}
    assert pressed == (keys if s in ("passcode", "dialpad") else set()), (pressed, keys, s)
    return shown


# -- breadth-first exploration -----------------------------------------------------------------------------------------
# phoneui submits the sixth passcode digit on its touch-down and clears the rings at once, so six filled rings are
# never on screen. The layer stays in layout.py (harmless: one hidden quad); this pins it as the only unused one.
NEVER_SHOWN = {"ring_6_full"}

BFS_PASS_KEYS = "12"
BFS_DIAL_KEYS = "91*"
BFS_FIELD_MAX = 4


def _key(ui):
    """What the screen and its next moves depend on, with the clocks bucketed (0.5 s; the call timer to 4 s)."""
    st = D.screen_state(ui)
    phase = 0
    if ui.screen == "calling":
        phase = min(int(st.since_call), 4) + (0 if st.since_call < P.CONNECT_S else 10) + (100 if ui.held else 0)
    elif ui.screen == "failed":
        phase = int((ui.fail_until - ui.t) * 2)
    elif ui.screen == "passcode" and st.wrong:
        phase = 1 + int((ui.wrong_until - ui.t) * 2)
    return (ui.screen, st.digits, st.field, phase, st.pressed, ui.pads_down)


def _actions(ui):
    if ui.pads_down:
        return [("release", release)]
    s = ui.screen
    acts = [("wait 0.6", lambda x: wait(x, 0.6)), ("wait 2.5", lambda x: wait(x, 2.5))]
    if s == "lock":
        acts += [("swipe", swipe), ("tap bar", lambda x: tap(x, *L.aim("lock", "home_bar")))]
    elif s == "passcode":
        for name in list(BFS_PASS_KEYS) + ["emergency", "cancel"]:
            acts.append((f"tap {name}", lambda x, n=name: tap(x, *L.aim("passcode", n))))
        for k in (L.PASSCODE_KEYS if not ui.digits else "1"):
            acts.append((f"hold {k}", lambda x, k=k: press(x, *L.aim("passcode", k))))
    elif s == "dialpad":
        names = ["cancel", "call", "delete"] + (list(BFS_DIAL_KEYS) if len(ui.field) < BFS_FIELD_MAX else [])
        for name in names:
            acts.append((f"tap {name}", lambda x, n=name: tap(x, *L.aim("dialpad", n))))
        for k in (L.DIALPAD_KEYS if not ui.field else ""):
            acts.append((f"hold {k}", lambda x, k=k: press(x, *L.aim("dialpad", k))))
    elif s == "calling":
        acts.append(("tap end", lambda x: tap(x, *L.aim("calling", "end"))))
        acts.append(("hold end", lambda x: press(x, *L.aim("calling", "end"))))
    elif s == "failed":
        acts.append(("tap end", lambda x: tap(x, *L.aim("calling", "end"))))
    return acts


def explore(limit=20000):
    """All states reachable with the reduced key sets, deduplicated on what the screen depends on."""
    start = P.PhoneUI(meta={"variant": "display-test"})
    seen = {_key(start)}
    queue = deque([start])
    states = []
    while queue:
        ui = queue.popleft()
        states.append(ui)
        assert len(states) <= limit, "state space larger than expected"
        for _, act in _actions(ui):
            nxt = clone(ui)
            act(nxt)
            k = _key(nxt)
            if k not in seen:
                seen.add(k)
                queue.append(nxt)
    return states


_STATES = []


def states():
    if not _STATES:
        _STATES.extend(explore())
    return _STATES


# -- tests: plan and pictures ------------------------------------------------------------------------------------------
def test_imports_stay_pure():
    for mod in ("pxr", "carb", "omni", "isaaclab"):
        assert mod not in sys.modules, mod
    assert D.DEFAULT_VISIBLE == {"page_lock"}
    assert D.visible_layers(P.PhoneUI()) == set(D.DEFAULT_VISIBLE)       # the file's default is the reset state


def test_every_layer_has_a_screen_and_a_prim_safe_name():
    for name in L.LAYERS:
        assert D.LAYER_SCREENS[name], f"{name} belongs to no screen"
        assert NAME_RE.match(name), name
    banners = {n for n, layer in L.LAYERS.items() if layer.level == "banner"}
    assert banners == {"banner_wrong_passcode", "end_hidden"}, banners


def test_atlas_cells_are_disjoint_with_gutters():
    plan = D.atlas_plan()
    W, H = plan.size
    cells = sorted(set(plan.cells.values()))
    assert len(cells) == len(plan.drawn) == len(set(plan.shared.values()))
    g = D.GUTTER_PX
    for c in cells:
        assert g <= c[0] < c[2] <= W - g and g <= c[1] < c[3] <= H - g, c
    for i, a in enumerate(cells):
        for b in cells[i + 1:]:
            apart = a[2] + g <= b[0] or b[2] + g <= a[0] or a[3] + g <= b[1] or b[3] + g <= a[1]
            assert apart, (a, b)


def test_shared_cells_show_the_same_picture():
    plan = D.atlas_plan()
    for name, rep in plan.shared.items():
        assert D.content_key(L.LAYERS[name]) == D.content_key(L.LAYERS[rep]), (name, rep)
        assert plan.cells[name] == plan.cells[rep]
        layer = L.LAYERS[name]
        x0, y0, x1, y1 = plan.cells[name]
        if layer.elements:                                         # a cell maps onto its tile at ~PX_PER_MM
            u0, v0, u1, v1 = layer.rect
            assert abs((x1 - x0) - (u1 - u0) * D.PX_PER_MM) <= 0.5 and abs((y1 - y0) - (v1 - v0) * D.PX_PER_MM) <= 0.5
    assert len(plan.drawn) < len(L.LAYERS) / 4                     # the field and timer glyphs are shared


def test_layer_content_fits_its_tile():
    for name, layer in L.LAYERS.items():
        for el in layer.elements:
            b = L.box(el)
            r = layer.rect
            assert r[0] - 1e-6 <= b[0] and b[2] <= r[2] + 1e-6 and r[1] - 1e-6 <= b[1] and b[3] <= r[3] + 1e-6, \
                (name, el.kind, el.text, b, r)


def test_overlays_cover_what_they_replace():
    """An opaque tile must not cut through anything its page draws: page elements under a tile lie wholly in it."""
    for name, layer in L.LAYERS.items():
        if layer.level == "page":
            continue
        for screen in D.LAYER_SCREENS[name]:
            for el in L.PAGES[L.PAGE_OF[screen]]:
                b = L.box(el)
                if L.rects_overlap(b, layer.rect):
                    r = layer.rect
                    inside = r[0] <= b[0] and b[2] <= r[2] and r[1] <= b[1] and b[3] <= r[3]
                    assert inside, (name, el.kind, el.text or "", b, r)


# -- tests: the USD ----------------------------------------------------------------------------------------------------
def _meshes(text):
    blocks = re.findall(r'def Mesh "(\w+)" \(.*?\n        \{(.*?)\n        \}', text, re.S)
    out = {}
    for name, body in blocks:
        pts = re.search(r"point3f\[\] points = \[(.*?)\]\n", body).group(1)
        st = re.search(r"primvars:st = \[(.*?)\] \(", body).group(1)
        out[name] = {
            "invisible": 'token visibility = "invisible"' in body,
            "points": [tuple(float(x) for x in p.split(",")) for p in re.findall(r"\(([^()]*)\)", pts)],
            "st": [tuple(float(x) for x in p.split(",")) for p in re.findall(r"\(([^()]*)\)", st)],
            "material": re.search(r"rel material:binding = <([^>]*)>", body).group(1),
        }
    return out


def test_usd_file_is_current():
    assert D.USD_PATH.is_file(), "run python3 render_screens.py"
    assert D.USD_PATH.read_text(encoding="utf-8") == D.phone_usda(), "assets/phone.usda is stale: python3 " \
                                                                       "render_screens.py"


def test_usd_layers_match_layout():
    text = D.phone_usda()
    assert text.count("{") == text.count("}") and text.count("(") == text.count(")")
    assert "instanceable" not in text
    assert text.count("PhysicsCollisionAPI") == 1 and text.count("PhysicsRigidBodyAPI") == 1
    assert "bool physics:kinematicEnabled = 1" in text
    meshes = _meshes(text)
    assert list(meshes) == list(L.LAYERS), "one quad per layer, in layout order"
    assert {n for n, m in meshes.items() if not m["invisible"]} == D.DEFAULT_VISIBLE
    plan = D.atlas_plan()
    W, H = plan.size
    for name, m in meshes.items():
        layer = L.LAYERS[name]
        assert m["material"] == "/Phone/display/Looks/screen"
        pts, st = m["points"], m["st"]
        assert len(pts) == 4 and len(st) == 4
        z = L.LAYER_Z_MM[layer.level] / 1000
        assert all(abs(p[2] - z) < 1e-9 for p in pts), name
        (x0, y0, _), (x1, y1, _), (x2, y2, _) = pts[:3]
        assert (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0) > 0, f"{name}: normal not +z"
        for (x, y, _), (u, v) in zip(pts, ((layer.rect[0], layer.rect[1]), (layer.rect[2], layer.rect[1]),
                                           (layer.rect[2], layer.rect[3]), (layer.rect[0], layer.rect[3]))):
            assert abs(x - v / 1000) < 1e-6 and abs(y + u / 1000) < 1e-6, name       # screen up = +x, right = -y
            assert L.in_display(u, v), name
        cx0, cy0, cx1, cy1 = plan.cells[name]
        want = [(cx0 / W, 1 - cy1 / H), (cx1 / W, 1 - cy1 / H), (cx1 / W, 1 - cy0 / H), (cx0 / W, 1 - cy0 / H)]
        assert all(abs(a - c) < 1e-6 and abs(b - d) < 1e-6 for (a, b), (c, d) in zip(st, want)), name
    # the collider: the whole phone, its top face at the USD origin (the glass surface)
    col = re.search(r'def Cube "collider".*?translate = \(([^)]*)\).*?scale = \(([^)]*)\)', text, re.S)
    tz, sz = float(col.group(1).split(",")[2]), float(col.group(2).split(",")[2])
    assert abs(tz + sz / 2) < 1e-9 and abs(sz - L.PHONE_SIZE[2]) < 1e-9
    assert abs(D.SPAWN_POS[2] - L.GLASS_Z) < 1e-12


def test_referenced_files_exist_and_fit_the_budget():
    text = D.USD_PATH.read_text(encoding="utf-8")
    refs = re.findall(r"@([^@]+)@", text)
    assert refs == ["./" + D.ATLAS_FILE], refs
    for ref in refs:
        assert (D.ASSETS / ref).resolve().is_file(), ref
    import render_screens as R
    for name in R.preview_files() + ["LICENSE-DejaVu.txt", D.ATLAS_FILE, D.USD_FILE]:
        assert (D.ASSETS / name).is_file(), f"missing assets/{name}: python3 render_screens.py"
    total = sum(p.stat().st_size for p in D.ASSETS.rglob("*") if p.is_file())
    assert total < ASSET_BUDGET, f"assets/ is {total / 1e6:.2f} MB"
    lic = (D.ASSETS / "LICENSE-DejaVu.txt").read_text(encoding="utf-8")
    assert "Bitstream Vera" in lic and "Arev" in lic


# -- tests: the PNGs ---------------------------------------------------------------------------------------------------
def _atlas():
    from PIL import Image
    with Image.open(D.ATLAS_PATH) as img:
        return img.convert("RGB")


def test_atlas_matches_plan():
    import numpy as np
    plan = D.atlas_plan()
    img = _atlas()
    assert img.size == plan.size, (img.size, plan.size)
    a = np.asarray(img).astype(int).max(axis=2)
    inside = np.zeros(a.shape, bool)
    for name in plan.drawn:
        x0, y0, x1, y1 = plan.cells[name]
        inside[y0:y1, x0:x1] = True
        cell = a[y0:y1, x0:x1]
        if L.LAYERS[name].elements:
            assert cell.max() >= 150, f"{name}: cell is empty"
        else:
            assert cell.max() == 0, f"{name}: blank cell is not black"
        if L.LAYERS[name].level != "page":
            edge = np.concatenate([cell[0], cell[-1], cell[:, 0], cell[:, -1]])
            assert edge.max() <= 40, f"{name}: drawing touches the tile's edge ({edge.max()})"
    assert a[~inside].max() == 0, "gutters must be black"


def test_glyph_cells_put_cap_height_centre_on_v():
    """Text placement convention: (u, v) is the centre of the capital height. Digits have cap height, no descent."""
    import numpy as np
    plan = D.atlas_plan()
    a = np.asarray(_atlas()).astype(int).max(axis=2)
    for name in ("field_00_0", "field_00_8", "timer_m_0", "timer_m_8"):
        layer = L.LAYERS[name]
        x0, y0, x1, y1 = plan.cells[name]
        rows = np.nonzero(a[y0:y1, x0:x1].max(axis=1) > 100)[0]
        top, bot = rows[0], rows[-1] + 1
        el = layer.elements[0]
        ppm = (y1 - y0) / (layer.rect[3] - layer.rect[1])
        want_centre = (layer.rect[3] - el.v) * ppm
        want_h = L.CAP_EM * el.em * ppm
        assert abs((top + bot) / 2 - want_centre) <= 1.5, (name, (top + bot) / 2, want_centre)
        assert abs((bot - top) - want_h) <= 2.5, (name, bot - top, want_h)


def test_previews_show_the_states_they_name():
    import numpy as np
    from PIL import Image
    import render_screens as R
    atlas = _atlas()
    for name, steps in R.PREVIEWS.items():
        ui = R.drive(steps)
        check_state(ui)
        want = np.asarray(R.composite(D.visible_layers(ui), atlas)).astype(int)
        with Image.open(D.ASSETS / f"preview_{name}.png") as img:
            got = np.asarray(img.convert("RGB")).astype(int)
        assert got.shape == want.shape, name
        assert (np.abs(got - want).max(axis=2) > 8).mean() < 1e-3, f"preview_{name}.png is stale"
    expect = {"lock": "lock", "passcode_typing": "passcode", "passcode_wrong": "passcode", "dialpad_911": "dialpad",
              "calling": "calling", "call_connected": "calling", "call_failed": "failed"}
    for name, screen in expect.items():
        assert R.drive(R.PREVIEWS[name]).screen == screen, name
    assert R.drive(R.PREVIEWS["dialpad_911"]).field == "911"
    assert "banner_wrong_passcode" in D.visible_layers(R.drive(R.PREVIEWS["passcode_wrong"]))
    assert "press_passcode_3" in D.visible_layers(R.drive(R.PREVIEWS["passcode_typing"]))


# -- tests: the states -------------------------------------------------------------------------------------------------
def test_every_reachable_state_shows_a_consistent_layer_set():
    seen_screens = set()
    for ui in states():
        check_state(ui)
        seen_screens.add(ui.screen)
    assert seen_screens == set(P.SCREENS), seen_screens
    assert len(states()) > 200


def test_number_field_every_glyph_in_every_slot():
    base = ui_at(to_dialpad)
    for n in range(1, L.FIELD_MAX + 2):
        for g in L.DIALPAD_KEYS:
            ui = clone(base)
            for _ in range(n):
                tap(ui, *L.aim("dialpad", g))
            assert ui.field == g * min(n, L.FIELD_MAX), (ui.field, g, n)
            check_state(ui)
            press(ui, *L.aim("dialpad", g))                      # held key: highlight, field already full or +1
            check_state(ui)
    ui = clone(base)
    for g in "91*#0":
        tap(ui, *L.aim("dialpad", g))
    tap(ui, *L.aim("dialpad", "delete"))
    assert ui.field == "91*#"
    check_state(ui)


def test_call_timer_runs_to_its_last_value():
    ui = ui_at(to_dialpad)
    for g in "911":
        tap(ui, *L.aim("dialpad", g))
    tap(ui, *L.aim("dialpad", "call"))
    assert ui.screen == "calling"
    check_state(ui)
    t0 = ui.call_t
    for k in range(1, 621):
        ui.advance(t0 + k * 0.999 + 0.0005)
        shown = check_state(ui)
    assert decode_timer(shown) == 599


def test_every_layer_is_shown_by_some_state():
    shown = set()
    for ui in states():
        shown |= ui.visible_layers()
    base = ui_at(to_dialpad)
    for n in range(1, L.FIELD_MAX + 1):
        for g in L.DIALPAD_KEYS:
            ui = clone(base)
            for _ in range(n):
                tap(ui, *L.aim("dialpad", g))
            shown |= ui.visible_layers()
    calling = clone(base)
    for g in "911":
        tap(calling, *L.aim("dialpad", g))
    tap(calling, *L.aim("dialpad", "call"))
    for k in range(0, 600):
        calling.advance(calling.call_t + k + 0.5)
        shown |= calling.visible_layers()
    missing = set(L.LAYERS) - shown
    assert missing == NEVER_SHOWN, sorted(missing)


# -- tests: the run-time half, against a stand-in for pxr --------------------------------------------------------------
class _Prim:
    def __init__(self, path, valid=True):
        self.path, self.valid, self.vis, self.writes = path, valid, "inherited", 0

    def IsValid(self):
        return self.valid

    def __bool__(self):
        return self.valid

    def GetPath(self):
        return self.path


class _Stage:
    def __init__(self, root, names):
        self.prims = {root: _Prim(root)}
        for n in names:
            self.prims[f"{root}/{n}"] = _Prim(f"{root}/{n}")

    def GetPrimAtPath(self, path):
        return self.prims.get(path, _Prim(path, valid=False))

    def Traverse(self):
        return list(self.prims.values())

    def visible(self, root):
        return {p.rsplit("/", 1)[1] for p, prim in self.prims.items() if p != root and prim.vis == "inherited"}


def _fake_pxr():
    def imageable(prim):
        def setter(tok):
            prim.vis = tok
            prim.writes += 1
        attr = types.SimpleNamespace(Set=setter)
        return types.SimpleNamespace(GetVisibilityAttr=lambda: attr)
    mod = types.ModuleType("pxr")
    mod.UsdGeom = types.SimpleNamespace(Imageable=imageable,
                                        Tokens=types.SimpleNamespace(inherited="inherited", invisible="invisible"))
    return mod


def test_apply_writes_only_what_changed():
    root = D.display_root("phone")
    stage = _Stage(root, L.LAYERS)
    env = types.SimpleNamespace(unwrapped=object())
    saved = sys.modules.get("pxr")
    sys.modules["pxr"] = _fake_pxr()
    os.environ["PHONE_FRESH_FRAME"] = "0"
    try:
        D.forget()
        assert D.reset_display(env, stage=stage) == len(L.LAYERS)          # first call on an env: every layer
        assert stage.visible(root) == {"page_lock"}
        assert D.reset_display(env, stage=stage) == 0
        ui = ui_at(to_dialpad)
        n = D.apply(env, "phone", list(L.LAYERS), ui.visible_layers(), stage=stage)
        assert n == 2 and stage.visible(root) == {"page_dialpad"}, (n, stage.visible(root))
        tap(ui, *L.aim("dialpad", "9"))
        assert D.show(env, ui, stage=stage) == 2                          # one glyph and the delete arrow
        assert stage.visible(root) == ui.visible_layers()
        prev = ui.visible_layers()
        for s in states()[:400]:                                           # any jump writes exactly the difference
            want = s.visible_layers()
            assert D.show(env, s, stage=stage) == len(prev ^ want)
            assert stage.visible(root) == want
            prev = want
        for bad in (lambda: D.apply(env, "phone", None, {"no_such_layer"}, stage=stage),
                    lambda: D.apply(env, "phone", ["no_such_layer"], set(), stage=stage)):
            try:
                bad()
            except KeyError:
                pass
            else:
                raise AssertionError("unknown layer accepted")
        try:
            D.apply(env, "phone", stage=stage)
        except TypeError:
            pass
        else:
            raise AssertionError("apply without layers accepted")
        env2 = types.SimpleNamespace(unwrapped=object())                   # a rebuilt env re-binds and rewrites
        assert D.reset_display(env2, stage=stage) == len(L.LAYERS)
        moved = "/World/envs/env_0/phone_0/display"                        # display found by its lock page
        stage3 = _Stage(moved, L.LAYERS)
        assert D.reset_display(types.SimpleNamespace(unwrapped=object()), stage=stage3) == len(L.LAYERS)
        assert stage3.visible(moved) == {"page_lock"}
        broken = _Stage(root, [n for n in L.LAYERS if n != "timer_u_9"])
        try:
            D.reset_display(types.SimpleNamespace(unwrapped=object()), stage=broken)
        except RuntimeError:
            pass
        else:
            raise AssertionError("missing layer prim not reported")
    finally:
        D.forget()
        os.environ.pop("PHONE_FRESH_FRAME", None)
        if saved is None:
            sys.modules.pop("pxr", None)
        else:
            sys.modules["pxr"] = saved


def test_screen_gain_and_one_shot_fresh_frame():
    """PHONE_SCREEN_GAIN overrides the emission gain at bind time (bad values fall back to EMISSIVE_GAIN); the DLSS
    reset flag set on a screen change is cleared by the next apply that changes nothing."""
    assert 2.0 <= D.EMISSIVE_GAIN <= 5.0, "measured in the simulator: 3 keeps grey keys and coloured buttons; 8+ washes them out"
    assert f"inputs:scale = ({D.EMISSIVE_GAIN:.3f}, {D.EMISSIVE_GAIN:.3f}, {D.EMISSIVE_GAIN:.3f}, 1)" in D.phone_usda()
    saved_env = {k: os.environ.get(k) for k in (D.GAIN_ENV, "PHONE_FRESH_FRAME")}
    saved_mods = {k: sys.modules.get(k) for k in ("pxr", "carb", "carb.settings", "carb.eventdispatcher")}
    settings, events, scale = [], [], []
    try:
        os.environ.pop(D.GAIN_ENV, None)
        assert D.screen_gain() == D.EMISSIVE_GAIN
        for raw, want in (("750", 750.0), ("abc", D.EMISSIVE_GAIN), ("-1", D.EMISSIVE_GAIN), ("inf", D.EMISSIVE_GAIN)):
            os.environ[D.GAIN_ENV] = raw
            assert D.screen_gain() == want, (raw, D.screen_gain())
        pxr = _fake_pxr()
        pxr.Gf = types.SimpleNamespace(Vec4f=lambda *v: tuple(v))
        sys.modules["pxr"] = pxr
        carb = types.ModuleType("carb")
        carb.settings = types.ModuleType("carb.settings")
        carb.settings.get_settings = lambda: types.SimpleNamespace(set_bool=lambda k, v: settings.append((k, v)))
        carb.eventdispatcher = types.ModuleType("carb.eventdispatcher")
        carb.eventdispatcher.get_eventdispatcher = lambda: types.SimpleNamespace(
            dispatch_event=lambda event_name: events.append(event_name))
        sys.modules.update({"carb": carb, "carb.settings": carb.settings,
                            "carb.eventdispatcher": carb.eventdispatcher})
        root = D.display_root("phone")
        stage = _Stage(root, L.LAYERS)
        atlas = _Prim(f"{root}/Looks/screen/atlas")
        atlas.GetAttribute = lambda name: types.SimpleNamespace(Set=lambda v: scale.append((name, v)))
        stage.prims[atlas.path] = atlas
        os.environ[D.GAIN_ENV] = "750"
        os.environ.pop("PHONE_FRESH_FRAME", None)
        D.forget()
        env = types.SimpleNamespace(unwrapped=object())
        assert D.reset_display(env, stage=stage) == len(L.LAYERS)
        assert scale == [("inputs:scale", (750.0, 750.0, 750.0, 1.0))], scale
        assert settings == [(D.DLSS_RESET, True)] and len(events) == 1, (settings, events)
        assert D.reset_display(env, stage=stage) == 0                    # next control step: flag cleared
        assert settings[-1] == (D.DLSS_RESET, False) and len(settings) == 2, settings
        assert D.reset_display(env, stage=stage) == 0 and len(settings) == 2      # nothing more to clear
        ui = ui_at(to_dialpad)
        assert D.show(env, ui, stage=stage) == 2 and settings[-1] == (D.DLSS_RESET, True)
        assert D.show(env, ui, stage=stage) == 0 and settings[-1] == (D.DLSS_RESET, False)
        assert len(scale) == 1, "the gain is written once per bind"
        os.environ["PHONE_FRESH_FRAME"] = "0"                             # off: no flag, no event
        n = len(settings)
        assert D.reset_display(env, stage=stage) == 2 and len(settings) == n
        D.forget()
        os.environ.pop(D.GAIN_ENV, None)                                   # unset: the file's gain stays
        assert D.reset_display(types.SimpleNamespace(unwrapped=object()), stage=stage) == len(L.LAYERS)
        assert len(scale) == 1
    finally:
        D.forget()
        D._RT["dlss"] = False
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        for k, v in saved_mods.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


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

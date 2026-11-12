"""The phone's screen in the simulator, and the USD file that carries it.

The screen is a stack of flat picture tiles lying 0.1-0.3 mm above the glass, one per ``layout.LAYERS`` entry. Every
tile is cut from one texture, ``assets/screen_atlas.png`` (drawn by render_screens.py), and every tile uses one unlit
material: black diffuse, the picture as emission, so the gripper's shadow does not darken the screen. Overlays are
opaque tiles drawn on the same black as the page under them (a filled ring over an empty one, "Delete" over "Cancel",
a digit on the empty number field), so nothing needs transparency. What the screen shows changes only by toggling the
tiles' USD visibility; no prim is moved, re-textured or created after the scene is built.

Pure Python 3.9 (no Isaac imports at module level; tested offline by test_display.py):

- ``atlas_plan()``: where each layer's picture sits in the texture. Layers that show the same picture share one cell:
  a glyph of the number field in any slot, a timer digit in any place, the six filled rings, a key's pressed look on
  either keypad.
- ``phone_usda()``: the phone as USD text, written to ``assets/phone.usda`` by render_screens.py. Its root ``Phone`` is
  a kinematic rigid body with one box collider (the whole 295 x 143 x 16 mm phone, top face = the glass surface) and
  its own physics material; a dark grey body and a black glass top for looks; and under ``Phone/display`` the screen
  material and every layer as a visual-only quad (no collider, so touch sensing never sees them). Only ``page_lock``
  is visible in the file, so the very first rendered frame already shows the lock screen. The layers are built once,
  when the scene spawns the phone (task.py: a kinematic USD body at ``SPAWN_POS``).
- ``visible_layers(state)``: the layers to show for a phone state, a ``ScreenState`` or a ``phoneui.PhoneUI``.

Isaac Sim only (pxr and carb are imported inside the functions):

- ``apply(env, "phone", None, wanted)`` / ``show(env, ui)``: make exactly the wanted layers visible, writing only the
  layers whose state changed. Call it once per control step from an on_step hook, and before a reset with the lock
  screen (``reset_display``). Never call it from a physics-step hook.
- ``screen_gain()``: the emission gain in use (``EMISSIVE_GAIN``, or $PHONE_SCREEN_GAIN); ``bind`` writes it to the
  screen material when the variable is set.
- ``add_display(stage, prim_path)``: fallback if the phone body is spawned some other way: references this file's
  ``/Phone/display`` (layers and material) at ``prim_path``; then pass that path as ``object_name``.

Frames: the USD origin is the centre of the glass top, +x toward the status bar (away from the arms), +y to the robot's
left, +z out of the glass (layout.screen_to_local). Atlas pixels have y down; USD texture t runs up from the image's
bottom row.
"""

from __future__ import annotations

import functools
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional, Set, Tuple

import layout as L
from phoneui import CONNECT_S

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
ATLAS_FILE = "screen_atlas.png"          # in assets/, next to the USD; referenced from the USD as ./screen_atlas.png
USD_FILE = "phone.usda"
USD_PATH = ASSETS / USD_FILE
ATLAS_PATH = ASSETS / ATLAS_FILE

# -- texture --------------------------------------------------------------------------------------------------------
PX_PER_MM = 4                 # atlas density: a page is 512 x 1112 px (the wrist cameras see 2-4 px/mm at 10 cm)
GUTTER_PX = 32                # black between cells, so mip-mapped minification never bleeds one cell into another
BLANK_PX = 16                 # an all-black cell, for layers that only hide something (end_hidden)

# -- looks ----------------------------------------------------------------------------------------------------------
EMISSIVE_GAIN = 3.0           # texture -> emission scale (white = 1.0 x this)
# Measured in the simulator (README, check 1, 2026-11-11): gains 3, 8, 15, 30, 100, 300 and 1000 were rendered in the
# station's top and right wrist cameras during the demo. At 3 the grey keys (0.23 sRGB) stay grey with sharp white
# digits, labels and the dialled number are crisp, and the call and end buttons keep their green and red. At 8 the keys
# turn pale and the red washes to pink; from 30 up the keys and buttons saturate to white and their digits vanish.
# (The earlier offline estimate of 1000 assumed the emission scale was in the lights' units; it is not.)
# PHONE_SCREEN_GAIN=<number> still overrides it at run time (screen_gain(), set in bind()).
GAIN_ENV = "PHONE_SCREEN_GAIN"
SCREEN_ROUGHNESS = 0.4
BODY_RGB = (0.06, 0.06, 0.065)          # linear: dark grey
GLASS_RGB = (0.004, 0.004, 0.005)       # linear: black glass, slightly glossy

# -- physics of the phone body (the collider is the only physical part of phone.usda) -------------------------------
PHONE_FRICTION = tuple(getattr(L, "GLASS_FRICTION", (0.5, 0.4)))    # static, dynamic: glass against the pads
CONTACT_OFFSET = 0.0005       # m
REST_OFFSET = 0.0
SPAWN_POS = (L.PHONE_X, L.PHONE_Y, L.GLASS_Z)       # world position of the USD origin (the centre of the glass top)
SPAWN_ROT_XYZW = (0.0, 0.0, math.sin(L.PHONE_YAW / 2), math.cos(L.PHONE_YAW / 2))
COLLIDER_PRIM = "collider"                          # /World/envs/env_0/<object name>/collider

DISPLAY_PRIM = "/World/envs/env_0/{name}/display"   # Arena puts the USD's default prim at the object's prim path
DEFAULT_VISIBLE = frozenset(n for n, layer in L.LAYERS.items() if layer.visible)    # {"page_lock"}

Px = Tuple[int, int, int, int]          # (x0, y0, x1, y1) atlas pixels, y down, x1 and y1 exclusive


# == the atlas plan ===================================================================================================
def _q(x: float) -> float:
    return round(float(x), 4) + 0.0


def cell_size(layer: L.Layer) -> Tuple[int, int]:
    """Pixel size of a layer's cell: its rectangle at PX_PER_MM (an all-black layer gets a small black cell)."""
    if not layer.elements:
        return BLANK_PX, BLANK_PX
    u0, v0, u1, v1 = layer.rect
    return max(1, int(round((u1 - u0) * PX_PER_MM))), max(1, int(round((v1 - v0) * PX_PER_MM)))


def content_key(layer: L.Layer) -> tuple:
    """What a layer's tile looks like, independent of where it sits: equal keys mean the same picture."""
    if not layer.elements:
        return ("blank",)
    u0, v0, u1, v1 = layer.rect
    cu, cv = (u0 + u1) / 2, (v0 + v1) / 2
    els = tuple((e.kind, _q(e.u - cu), _q(e.v - cv), _q(e.w), _q(e.h), tuple(e.colour), e.text, _q(e.em), _q(e.line),
                 bool(e.bold)) for e in layer.elements)
    return (_q(u1 - u0), _q(v1 - v0), layer.level == "page", els)


class AtlasPlan(NamedTuple):
    size: Tuple[int, int]               # (width, height) of the atlas, px
    cells: Dict[str, Px]                # every layer -> its cell
    drawn: Tuple[str, ...]              # one layer per cell: the one render_screens.py draws into it
    shared: Dict[str, str]              # every layer -> the drawn layer whose picture it shows


@functools.lru_cache(maxsize=1)
def atlas_plan() -> AtlasPlan:
    """Pages in a row along the top, the other cells shelf-packed below them, GUTTER_PX of black around every cell."""
    groups: Dict[tuple, List[str]] = {}
    for name, layer in L.LAYERS.items():
        groups.setdefault(content_key(layer), []).append(name)
    reps = [names[0] for names in groups.values()]
    pages = [r for r in reps if L.LAYERS[r].level == "page"]
    tiles = [r for r in reps if L.LAYERS[r].level != "page"]
    g = GUTTER_PX
    pos: Dict[str, Px] = {}
    x = g
    top = 0
    for name in pages:
        w, h = cell_size(L.LAYERS[name])
        pos[name] = (x, g, x + w, g + h)
        x += w + g
        top = max(top, h)
    width = max(x, 8 * g)
    y = g + top + g if pages else g
    x, row = g, 0
    for name in sorted(tiles, key=lambda n: (-cell_size(L.LAYERS[n])[1], -cell_size(L.LAYERS[n])[0], n)):
        w, h = cell_size(L.LAYERS[name])
        if x > g and x + w + g > width:
            y, x, row = y + row + g, g, 0
        pos[name] = (x, y, x + w, y + h)
        x += w + g
        row = max(row, h)
    height = y + row + g
    shared = {n: names[0] for names in groups.values() for n in names}
    cells = {n: pos[shared[n]] for n in L.LAYERS}
    return AtlasPlan((width, height), cells, tuple(reps), shared)


def cell_st(cell: Px, size: Tuple[int, int]) -> Tuple[float, float, float, float]:
    """(s0, t0, s1, t1) of a cell: s from the image's left, t up from its bottom row."""
    x0, y0, x1, y1 = cell
    w, h = size
    return (x0 / w, 1.0 - y1 / h, x1 / w, 1.0 - y0 / h)


# == the phone as USD text ============================================================================================
def quad_points(layer: L.Layer) -> List[Tuple[float, float, float]]:
    """Corners in the phone frame (m), counter-clockwise seen from above (normal +z): the screen's bottom-left,
    bottom-right, top-right and top-left."""
    u0, v0, u1, v1 = layer.rect
    z = L.LAYER_Z_MM[layer.level]
    return [L.screen_to_local(u, v, z) for u, v in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))]


def quad_st(name: str) -> List[Tuple[float, float]]:
    plan = atlas_plan()
    s0, t0, s1, t1 = cell_st(plan.cells[name], plan.size)
    return [(s0, t0), (s1, t0), (s1, t1), (s0, t1)]


def _f(x: float, digits: int = 6) -> str:
    return f"{round(float(x), digits) + 0.0:.{digits}f}"


def _v(p: Iterable[float], digits: int = 6) -> str:
    return "(" + ", ".join(_f(x, digits) for x in p) + ")"


def _vs(points: Iterable[Iterable[float]], digits: int = 6) -> str:
    return "[" + ", ".join(_v(p, digits) for p in points) + "]"


def _preview_material(name: str, rgb, roughness: float, indent: str) -> str:
    base = f"/Phone/Looks/{name}"
    return f"""
{indent}def Material "{name}"
{indent}{{
{indent}    token outputs:surface.connect = <{base}/surface.outputs:surface>

{indent}    def Shader "surface"
{indent}    {{
{indent}        uniform token info:id = "UsdPreviewSurface"
{indent}        color3f inputs:diffuseColor = {_v(rgb, 4)}
{indent}        float inputs:metallic = 0
{indent}        float inputs:roughness = {roughness}
{indent}        token outputs:surface
{indent}    }}
{indent}}}
"""


def _screen_material(gain: float) -> str:
    base = "/Phone/display/Looks/screen"
    g = _f(gain, 3)
    return f"""
        def Scope "Looks"
        {{
            def Material "screen"
            {{
                token outputs:surface.connect = <{base}/surface.outputs:surface>

                def Shader "surface"
                {{
                    uniform token info:id = "UsdPreviewSurface"
                    color3f inputs:diffuseColor = (0, 0, 0)
                    float inputs:metallic = 0
                    float inputs:roughness = {SCREEN_ROUGHNESS}
                    color3f inputs:emissiveColor.connect = <{base}/atlas.outputs:rgb>
                    token outputs:surface
                }}

                def Shader "uv"
                {{
                    uniform token info:id = "UsdPrimvarReader_float2"
                    token inputs:varname = "st"
                    float2 outputs:result
                }}

                def Shader "atlas"
                {{
                    uniform token info:id = "UsdUVTexture"
                    asset inputs:file = @./{ATLAS_FILE}@
                    float2 inputs:st.connect = <{base}/uv.outputs:result>
                    token inputs:wrapS = "clamp"
                    token inputs:wrapT = "clamp"
                    token inputs:sourceColorSpace = "sRGB"
                    float4 inputs:scale = ({g}, {g}, {g}, 1)
                    float3 outputs:rgb
                }}
            }}
        }}
"""


def _quad(name: str) -> str:
    layer = L.LAYERS[name]
    vis = "" if layer.visible else '\n            token visibility = "invisible"'
    return f"""
        def Mesh "{name}" (
            prepend apiSchemas = ["MaterialBindingAPI"]
        )
        {{{vis}
            rel material:binding = </Phone/display/Looks/screen>
            uniform bool doubleSided = 1
            int[] faceVertexCounts = [4]
            int[] faceVertexIndices = [0, 1, 2, 3]
            point3f[] points = {_vs(quad_points(layer))}
            normal3f[] normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)] (
                interpolation = "vertex"
            )
            texCoord2f[] primvars:st = {_vs(quad_st(name))} (
                interpolation = "vertex"
            )
            uniform token subdivisionScheme = "none"
        }}
"""


def _cube(name: str, size_m, centre_m, schemas: str, body: str) -> str:
    return f"""
    def Cube "{name}" (
        prepend apiSchemas = [{schemas}]
    )
    {{{body}
        double size = 1
        double3 xformOp:translate = {_v(centre_m)}
        float3 xformOp:scale = {_v(size_m)}
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }}
"""


def phone_usda(gain: float = EMISSIVE_GAIN) -> str:
    """The whole phone as USD text (see the module docstring). Deterministic: test_display.py checks that
    assets/phone.usda is exactly this text."""
    bx, by, bz = L.BODY_SIZE
    gz = L.GLASS_SIZE[2]
    px, py, pz = L.PHONE_SIZE
    sf, df = (float(f) for f in PHONE_FRICTION)
    collider = _cube(COLLIDER_PRIM, (px, py, pz), (0.0, 0.0, -pz / 2),
                     '"PhysicsCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"', f"""
        rel material:binding:physics = </Phone/PhysicsMaterial>
        uniform token purpose = "guide"
        float physxCollision:contactOffset = {CONTACT_OFFSET}
        float physxCollision:restOffset = {REST_OFFSET}""")
    body = _cube("body", (bx, by, bz), (0.0, 0.0, -gz - bz / 2), '"MaterialBindingAPI"',
                 "\n        rel material:binding = </Phone/Looks/body>")
    glass = _cube("glass", (L.GLASS_SIZE[0], L.GLASS_SIZE[1], gz), (0.0, 0.0, -gz / 2), '"MaterialBindingAPI"',
                  "\n        rel material:binding = </Phone/Looks/glass>")
    looks = (_preview_material("body", BODY_RGB, 0.5, "        ")
             + _preview_material("glass", GLASS_RGB, 0.1, "        "))
    quads = "".join(_quad(name) for name in L.LAYERS)
    w, h = atlas_plan().size
    return f"""#usda 1.0
(
    defaultPrim = "Phone"
    doc = "Generated from layout.py by display.py (python3 render_screens.py); do not edit. Layout {L.LAYOUT_VERSION}, \
{len(L.LAYERS)} screen layers, atlas {w} x {h} px."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "Phone" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI"]
)
{{
    bool physics:kinematicEnabled = 1

    def Material "PhysicsMaterial" (
        prepend apiSchemas = ["PhysicsMaterialAPI"]
    )
    {{
        float physics:dynamicFriction = {df}
        float physics:restitution = 0
        float physics:staticFriction = {sf}
    }}
{collider}{body}{glass}
    def Scope "Looks"
    {{{looks}    }}

    def Xform "display"
    {{{_screen_material(gain)}{quads}    }}
}}
"""


def write_usda(path: Optional[Path] = None) -> Path:
    path = Path(path) if path is not None else USD_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(phone_usda(), encoding="utf-8")
    return path


# == which layers to show =============================================================================================
class ScreenState(NamedTuple):
    """What the screen depends on, read from a PhoneUI by ``screen_state``."""
    screen: str                         # lock | passcode | dialpad | calling | failed
    digits: int = 0                     # passcode digits entered (rings filled)
    field: str = ""                     # the dial pad's number field
    wrong: bool = False                 # the "Wrong Passcode" banner is up
    since_call: float = 0.0             # s since the current call was placed
    pressed: Tuple[str, ...] = ()       # keys whose touch is still down, entered on the current screen


def screen_state(ui) -> ScreenState:
    """The ScreenState of a phoneui.PhoneUI."""
    pressed = sorted({tc.key for tc in ui.touches.values() if tc.key is not None and tc.epoch == ui.epoch})
    return ScreenState(ui.screen, len(ui.digits), ui.field, ui.t < ui.wrong_until, ui.t - ui.call_t, tuple(pressed))


def visible_layers(state) -> Set[str]:
    """Names of the layout.LAYERS to show for ``state`` (a ScreenState, or a PhoneUI); every other layer is hidden.
    Exactly one page; the overlays drawn over it never overlap one another (test_display.py checks every reachable
    state)."""
    st = state if isinstance(state, ScreenState) else screen_state(state)
    s = st.screen
    out = {L.PAGE_OF[s]}
    if s == "passcode":
        out.update(f"ring_{i}_full" for i in range(1, st.digits + 1))
        if st.digits:
            out.add("label_delete")
        if st.wrong:
            out.add("banner_wrong_passcode")
    elif s == "dialpad":
        out.update(L.field_layers(st.field))
        if st.field:
            out.add("delete_arrow")
    elif s == "calling":
        if st.since_call < CONNECT_S:
            out.add("title_calling")
        else:
            out.add("title_connected")
            out.update(L.timer_layers(st.since_call))
    elif s == "failed":
        out.update(("title_failed", "end_hidden"))
    if s in ("passcode", "dialpad"):
        out.update(L.press_layer(s, k) for k in st.pressed)
    return out


def _screens_of(name: str) -> Tuple[str, ...]:
    pages = tuple(s for s, p in L.PAGE_OF.items() if p == name)
    if pages:
        return pages
    for prefix, screen in (("ring_", "passcode"), ("label_delete", "passcode"), ("banner_wrong_passcode", "passcode"),
                           ("press_passcode_", "passcode"), ("field_", "dialpad"), ("delete_arrow", "dialpad"),
                           ("press_dialpad_", "dialpad"), ("title_calling", "calling"), ("title_connected", "calling"),
                           ("timer_", "calling"), ("title_failed", "failed"), ("end_hidden", "failed")):
        if name.startswith(prefix):
            return (screen,)
    return ()


LAYER_SCREENS: Dict[str, Tuple[str, ...]] = {n: _screens_of(n) for n in L.LAYERS}   # where each layer may show


# == run time: Isaac Sim only =========================================================================================
_RT: dict = {"base": None, "root": None, "prims": {}, "shown": {}, "dlss": False, "gain_set": None}
DLSS_RESET = "/rtx-transient/post/dlss/forceParamReset"


def screen_gain() -> float:
    """The screen's emission gain: $PHONE_SCREEN_GAIN if it is a positive finite number, else EMISSIVE_GAIN."""
    raw = os.environ.get(GAIN_ENV)
    if raw:
        try:
            g = float(raw)
            if math.isfinite(g) and g > 0:
                return g
        except ValueError:
            pass
        print(f"[phone] display: ignoring {GAIN_ENV}={raw!r} (not a positive number)", flush=True)
    return float(EMISSIVE_GAIN)


def _set_gain(stage, root: str) -> Optional[float]:
    """Write $PHONE_SCREEN_GAIN to the screen material's texture scale (only when the variable is set)."""
    if not os.environ.get(GAIN_ENV):
        return None
    g = screen_gain()
    path = f"{root}/Looks/screen/atlas"
    prim = stage.GetPrimAtPath(path)
    if not (prim and prim.IsValid()):
        print(f"[phone] display: no screen material at {path}; {GAIN_ENV} not applied", flush=True)
        return None
    try:                                        # never let a check-only setting break the screen
        from pxr import Gf

        prim.GetAttribute("inputs:scale").Set(Gf.Vec4f(g, g, g, 1.0))
    except Exception as e:  # noqa: BLE001
        print(f"[phone] display: {GAIN_ENV} not applied: {e!r}", flush=True)
        return None
    print(f"[phone] display: screen gain {g:g}", flush=True)
    return g


def display_root(object_name: str = "phone") -> str:
    """Prim path of the layers' parent: the phone object's ``display``, or ``object_name`` itself if it is a path."""
    return object_name if object_name.startswith("/") else DISPLAY_PRIM.format(name=object_name)


def _current_stage():
    try:
        from isaaclab.sim.utils.stage import get_current_stage
    except ImportError:                                  # older Isaac Lab layouts
        import omni.usd
        return omni.usd.get_context().get_stage()
    return get_current_stage()


def _find_root(stage, root: str) -> str:
    prim = stage.GetPrimAtPath(root)
    if prim and prim.IsValid():
        return root
    for prim in stage.Traverse():                        # moved by the spawner: find the display by its lock page
        path = str(prim.GetPath())
        if path.endswith("/display") and stage.GetPrimAtPath(path + "/page_lock").IsValid():
            return path
    raise RuntimeError(f"display: no screen layers at {root}")


def bind(env, object_name: str = "phone", stage=None) -> Dict[str, object]:
    """Find the layer prims of the running env, once per built env (and again if the env is rebuilt)."""
    base = getattr(env, "unwrapped", env)
    if _RT["base"] is base and _RT["root"] == object_name and _RT["prims"]:
        return _RT["prims"]
    stage = stage if stage is not None else _current_stage()
    root = _find_root(stage, display_root(object_name))
    prims, missing = {}, []
    for name in L.LAYERS:
        prim = stage.GetPrimAtPath(f"{root}/{name}")
        if prim and prim.IsValid():
            prims[name] = prim
        else:
            missing.append(name)
    if missing:
        raise RuntimeError(f"display: {len(missing)} layers missing under {root}, e.g. {missing[:3]}")
    _RT.update(base=base, root=object_name, prims=prims, shown={}, gain_set=_set_gain(stage, root))
    return prims


def forget() -> None:
    """Drop the cached prims (the next apply re-binds and writes every layer once)."""
    _RT.update(base=None, root=None, prims={}, shown={})


def changes(shown: Dict[str, bool], wanted: Iterable[str]) -> List[Tuple[str, bool]]:
    """(layer, visible) for each layer whose visibility must be written to show exactly ``wanted``."""
    want = set(wanted)
    unknown = want - set(L.LAYERS)
    if unknown:
        raise KeyError(f"display: unknown layers {sorted(unknown)}")
    return [(n, n in want) for n in L.LAYERS if shown.get(n) is not (n in want)]


def apply(env, object_name: str = "phone", all_names: Optional[Iterable[str]] = None,
          wanted: Optional[Iterable[str]] = None, stage=None) -> int:
    """Make exactly the layers in ``wanted`` visible; returns how many visibility attributes it wrote (the first call
    on a new env writes all of them). ``all_names`` is accepted for the call ``apply(env, "phone", list(L.LAYERS),
    wanted)`` and must name layout layers; every layer is always managed. Call from an on_step hook (after the
    control step), never from a physics-step hook."""
    if wanted is None:
        raise TypeError("display.apply: pass the layers to show")
    if all_names is not None:
        extra = set(all_names) - set(L.LAYERS)
        if extra:
            raise KeyError(f"display: unknown layers {sorted(extra)}")
    from pxr import UsdGeom

    prims = bind(env, object_name, stage)
    todo = changes(_RT["shown"], wanted)
    for name, on in todo:
        UsdGeom.Imageable(prims[name]).GetVisibilityAttr().Set(
            UsdGeom.Tokens.inherited if on else UsdGeom.Tokens.invisible)
        _RT["shown"][name] = on
    if todo:
        fresh_frame()
    else:
        end_fresh_frame()
    return len(todo)


def show(env, ui, object_name: str = "phone", stage=None) -> int:
    """apply() with the layers a PhoneUI (or ScreenState) calls for."""
    return apply(env, object_name, None, visible_layers(ui), stage)


def reset_display(env, object_name: str = "phone", stage=None) -> int:
    """Back to the lock screen; call before the harness's reset so the reset's first frame shows it."""
    return apply(env, object_name, None, DEFAULT_VISIBLE, stage)


def fresh_frame() -> None:
    """Ask RTX to drop its temporal history so the next frame shows the new screen at once. Unverified under the
    standard runner (check on the first approved simulator run); PHONE_FRESH_FRAME=0 turns it off. The DLSS reset flag
    is one-shot: the next apply() that changes nothing (the next control step, after one rendered frame) clears it,
    so the frames between screen changes keep their temporal history."""
    if os.environ.get("PHONE_FRESH_FRAME", "1") == "0":
        return
    try:
        import carb.settings

        carb.settings.get_settings().set_bool(DLSS_RESET, True)
        _RT["dlss"] = True
        import carb.eventdispatcher

        carb.eventdispatcher.get_eventdispatcher().dispatch_event(event_name="omni.rtx.ForceResetAccumulation")
    except Exception:  # noqa: BLE001  (older Kit, or no renderer)
        pass


def end_fresh_frame() -> None:
    """Clear the DLSS reset flag that fresh_frame() set (no-op if it set none)."""
    if not _RT["dlss"]:
        return
    _RT["dlss"] = False
    try:
        import carb.settings

        carb.settings.get_settings().set_bool(DLSS_RESET, False)
    except Exception:  # noqa: BLE001
        pass


def add_display(stage, prim_path: str, pos: Tuple[float, float, float] = SPAWN_POS, usd_path=None):
    """Fallback: reference phone.usda's ``/Phone/display`` (layers and screen material) at ``prim_path``, placed with
    its origin at ``pos`` (env frame; the glass-top centre). Use only if the phone body is not spawned from
    phone.usda; call while the scene is built, then pass ``prim_path`` as ``object_name`` to apply()."""
    from pxr import Gf, UsdGeom

    xf = UsdGeom.Xform.Define(stage, prim_path)
    xf.GetPrim().GetReferences().AddReference(str(usd_path or USD_PATH), "/Phone/display")
    xf.AddTranslateOp().Set(Gf.Vec3d(*(float(c) for c in pos)))
    if L.PHONE_YAW:
        xf.AddRotateZOp().Set(math.degrees(L.PHONE_YAW))
    return xf.GetPrim()


def describe() -> str:
    plan = atlas_plan()
    return (f"{len(L.LAYERS)} layers in {len(plan.drawn)} atlas cells, atlas {plan.size[0]} x {plan.size[1]} px at "
            f"{PX_PER_MM} px/mm; USD {len(phone_usda())} chars; spawn at {SPAWN_POS}")


if __name__ == "__main__":
    print(describe())
    if USD_PATH.is_file():
        print("assets/phone.usda", "matches" if USD_PATH.read_text(encoding="utf-8") == phone_usda()
              else "DIFFERS: run python3 render_screens.py")

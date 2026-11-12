"""Draw the phone's screens: the texture the simulator shows, the USD that maps it onto the glass, and previews.

    python3 render_screens.py            # writes the files below into assets/
    python3 render_screens.py --check    # re-renders in memory and exits 1 if any file differs

Writes:
- ``assets/screen_atlas.png``: every screen layer's picture, one cell per distinct picture (display.atlas_plan()),
  drawn at display.PX_PER_MM (4 px/mm) from the elements layout.py lists, white or coloured on black. A cell shows
  exactly its layer's rectangle, so what is drawn and where it reacts both come from layout.py.
- ``assets/phone.usda``: display.phone_usda(), the phone body and its screen layers (texture coordinates from the
  same plan).
- ``assets/preview_<state>.png``: for people. Each composites the atlas cells exactly as the simulator's tiles would
  show them for a phone state reached by driving phoneui.PhoneUI with taps, so a preview also checks the chain
  phone state -> display.visible_layers -> atlas cells. ``preview_all.png`` puts them side by side and
  ``preview_hit_areas.png`` outlines every hit area (digits yellow, buttons blue, the swipe zone magenta).

Text is DejaVu Sans (its licence: assets/LICENSE-DejaVu.txt). Only the rendered PNGs are shipped, not the font. The
font is looked for in $PHONE_FONT_DIR, then matplotlib's copy, then the usual system font folders. Antialiasing
comes from drawing at 4x and averaging down, so the PNGs depend only on the font file and Pillow.
"""

from __future__ import annotations

import hashlib
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import display as D     # noqa: E402
import layout as L      # noqa: E402
import phoneui as P     # noqa: E402

SS = 4                  # supersampling factor
PREVIEW_PX_PER_MM = D.PX_PER_MM
RIM_MM = 3.0            # previews: the body's edge around the glass
GLASS_SRGB = (12, 12, 14)
BODY_SRGB = (70, 70, 74)
LEVEL_ORDER = {"page": 0, "overlay": 1, "banner": 2}
FONT_FILES = {False: "DejaVuSans.ttf", True: "DejaVuSans-Bold.ttf"}


# -- fonts ------------------------------------------------------------------------------------------------------------
def _font_dirs() -> List[Path]:
    dirs = []
    if os.environ.get("PHONE_FONT_DIR"):
        dirs.append(Path(os.environ["PHONE_FONT_DIR"]))
    try:
        import matplotlib
        dirs.append(Path(matplotlib.get_data_path()) / "fonts" / "ttf")
    except Exception:  # noqa: BLE001  (no matplotlib)
        pass
    dirs += [Path(p) for p in ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/dejavu", "/usr/share/fonts/TTF",
                               "/Library/Fonts", os.path.expanduser("~/Library/Fonts"), "C:/Windows/Fonts")]
    return dirs


def find_font(bold: bool = False) -> Optional[Path]:
    for d in _font_dirs():
        p = d / FONT_FILES[bold]
        if p.is_file():
            return p
    return None


class Fonts:
    def __init__(self):
        regular = find_font(False)
        if regular is None:
            raise SystemExit("DejaVuSans.ttf not found: set PHONE_FONT_DIR to the folder holding it")
        self.paths = {False: regular, True: find_font(True) or regular}
        self._cache: Dict[Tuple[bool, int], ImageFont.FreeTypeFont] = {}

    def get(self, px: float, bold: bool = False) -> ImageFont.FreeTypeFont:
        key = (bold, max(1, int(round(px))))
        if key not in self._cache:
            self._cache[key] = ImageFont.truetype(str(self.paths[bold]), key[1])
        return self._cache[key]

    def describe(self) -> str:
        out = []
        for bold in (False, True):
            p = self.paths[bold]
            out.append(f"{p.name} sha256 {hashlib.sha256(p.read_bytes()).hexdigest()[:16]}")
        return "; ".join(out)


# -- drawing one layer ------------------------------------------------------------------------------------------------
class Painter:
    """Draws layout elements (screen mm) onto a canvas covering ``rect`` at ``size`` px (times SS)."""

    def __init__(self, rect: L.Rect, size: Tuple[int, int], fonts: Fonts):
        self.u0, self.v0, self.u1, self.v1 = rect
        self.w, self.h = size
        self.sx = self.w * SS / (self.u1 - self.u0)
        self.sy = self.h * SS / (self.v1 - self.v0)
        self.s = (self.sx + self.sy) / 2
        self.fonts = fonts
        self.img = Image.new("RGB", (self.w * SS, self.h * SS), L.BLACK)
        self.d = ImageDraw.Draw(self.img)

    def X(self, u: float) -> float:
        return (u - self.u0) * self.sx

    def Y(self, v: float) -> float:
        return (self.v1 - v) * self.sy

    def P(self, u: float, v: float) -> Tuple[float, float]:
        return (self.X(u), self.Y(v))

    def box(self, u0: float, v0: float, u1: float, v1: float) -> List[float]:
        """A screen-mm box as Pillow's [left, top, right, bottom]."""
        return [self.X(min(u0, u1)), self.Y(max(v0, v1)), self.X(max(u0, u1)), self.Y(min(v0, v1))]

    def dot(self, u: float, v: float, d: float, colour) -> None:
        r = d / 2
        self.d.ellipse(self.box(u - r, v - r, u + r, v + r), fill=colour)

    def stroke(self, pts: Sequence[Tuple[float, float]], width: float, colour) -> None:
        """A round-capped polyline (mm), stamped as dots so its ends and joints are round."""
        for (ua, va), (ub, vb) in zip(pts[:-1], pts[1:]):
            n = max(1, int(math.hypot(ub - ua, vb - va) / (width / 8)))
            for i in range(n + 1):
                f = i / n
                self.dot(ua + (ub - ua) * f, va + (vb - va) * f, width, colour)

    def draw(self, el: L.El) -> None:
        getattr(self, "_" + el.kind)(el)

    def _text(self, el: L.El) -> None:
        font = self.fonts.get(el.em * self.sy, el.bold)
        base = el.v - L.CAP_EM * el.em / 2
        self.d.text(self.P(el.u, base), el.text, font=font, fill=el.colour, anchor="ms")

    def _circle(self, el: L.El) -> None:
        self.d.ellipse(self.box(el.u - el.w / 2, el.v - el.h / 2, el.u + el.w / 2, el.v + el.h / 2), fill=el.colour)

    def _ring(self, el: L.El) -> None:
        self.d.ellipse(self.box(el.u - el.w / 2, el.v - el.h / 2, el.u + el.w / 2, el.v + el.h / 2),
                       outline=el.colour, width=max(1, int(round(el.line * self.s))))

    def _rrect(self, el: L.El) -> None:
        self.d.rounded_rectangle(self.box(el.u - el.w / 2, el.v - el.h / 2, el.u + el.w / 2, el.v + el.h / 2),
                                 radius=el.h / 2 * self.sy, fill=el.colour)

    def _icon(self, el: L.El) -> None:
        getattr(self, "_icon_" + el.text)(el)

    # -- icons (each fills its element's w x h box) ------------------------------------------------------------------
    def _icon_lock(self, el: L.El) -> None:
        u, v, w, h, c = el.u, el.v, el.w, el.h, el.colour
        body_top = v - h / 2 + 0.56 * h
        self.d.rounded_rectangle(self.box(u - w / 2, v - h / 2, u + w / 2, body_top), radius=0.13 * w * self.s,
                                 fill=c)
        r, t = 0.34 * w, 0.13 * w                       # shackle: outer radius, line
        cv = v + h / 2 - r
        self.d.arc(self.box(u - r, cv - r, u + r, cv + r), 180, 360, fill=c, width=max(1, int(round(t * self.s))))
        for side in (-1, 1):
            ua = u + side * (r - t / 2)
            self.d.rectangle(self.box(ua - t / 2, body_top - 0.1, ua + t / 2, cv), fill=c)

    def _icon_battery(self, el: L.El) -> None:
        u, v, w, h, c = el.u, el.v, el.w, el.h, el.colour
        line, nub = 0.45, 0.9
        right = u + w / 2 - nub - 0.2
        self.d.rounded_rectangle(self.box(u - w / 2, v - h / 2, right, v + h / 2), radius=0.28 * h * self.s,
                                 outline=c, width=max(1, int(round(line * self.s))))
        self.d.rounded_rectangle(self.box(right + 0.2, v - 0.22 * h, u + w / 2, v + 0.22 * h),
                                 radius=0.35 * nub * self.s, fill=c)
        inset = line + 0.45
        self.d.rounded_rectangle(self.box(u - w / 2 + inset, v - h / 2 + inset,
                                          u - w / 2 + inset + 0.75 * (right - u + w / 2 - 2 * inset),
                                          v + h / 2 - inset), radius=0.12 * h * self.s, fill=c)

    def _handset(self, cu: float, cv: float, radius: float, a0: float, a1: float, arc_w: float, pad_w: float,
                 pad_len: float, colour) -> None:
        """A telephone handset: a thick arc about (cu, cv) from angle a0 to a1 (degrees, counter-clockwise, screen
        axes), with an ear and a mouth piece at its ends pointing toward the arc's centre."""
        n = 24
        pts = [(cu + radius * math.cos(math.radians(a0 + (a1 - a0) * i / n)),
                cv + radius * math.sin(math.radians(a0 + (a1 - a0) * i / n))) for i in range(n + 1)]
        self.stroke(pts, arc_w, colour)
        for a in (a0, a1):
            eu, ev = cu + radius * math.cos(math.radians(a)), cv + radius * math.sin(math.radians(a))
            k = pad_len / radius
            self.stroke([(eu, ev), (eu + (cu - eu) * k, ev + (cv - ev) * k)], pad_w, colour)

    def _icon_handset(self, el: L.El) -> None:            # the call button: earpiece upper left, mouthpiece lower right
        s = el.w / 2
        self._handset(el.u + 0.48 * s, el.v + 0.48 * s, 1.16 * s, 188.0, 262.0, 0.44 * s, 0.62 * s, 0.44 * s,
                      el.colour)

    def _icon_handset_down(self, el: L.El) -> None:       # the end-call button: handset lying flat, ends down
        u, v, w, h = el.u, el.v, el.w, el.h
        self._handset(u, v - 0.62 * h, 0.96 * h, 38.0, 142.0, 0.34 * h, 0.48 * h, 0.34 * h, el.colour)

    def _icon_backspace(self, el: L.El) -> None:
        u, v, w, h, c = el.u, el.v, el.w, el.h, el.colour
        line = 0.075 * h * 1.2
        lw = max(1, int(round(line * self.s)))
        tip = u - w / 2 + line / 2
        sh = u - w / 2 + 0.30 * w
        top, bot, right = v + h / 2 - line / 2, v - h / 2 + line / 2, u + w / 2 - line / 2
        pts = [(tip, v), (sh, top), (right, top), (right, bot), (sh, bot), (tip, v), (sh, top)]
        self.d.line([self.P(*p) for p in pts], fill=c, width=lw, joint="curve")
        cx, r = (sh + right) / 2 + 0.02 * w, 0.20 * h
        self.stroke([(cx - r, v - r), (cx + r, v + r)], line, c)
        self.stroke([(cx - r, v + r), (cx + r, v - r)], line, c)

    def image(self) -> Image.Image:
        return self.img.resize((self.w, self.h), Image.Resampling.BOX)


def draw_layer(layer: L.Layer, size: Tuple[int, int], fonts: Fonts) -> Image.Image:
    p = Painter(layer.rect, size, fonts)
    for el in layer.elements:
        p.draw(el)
    return p.image()


def render_atlas(fonts: Fonts) -> Image.Image:
    plan = D.atlas_plan()
    atlas = Image.new("RGB", plan.size, L.BLACK)
    for name in plan.drawn:
        x0, y0, x1, y1 = plan.cells[name]
        layer = L.LAYERS[name]
        if layer.elements:
            atlas.paste(draw_layer(layer, (x1 - x0, y1 - y0), fonts), (x0, y0))
    return atlas


# -- previews ---------------------------------------------------------------------------------------------------------
PAD = "right_tip_left"


def drive(steps: Iterable[tuple]) -> P.PhoneUI:
    """A PhoneUI driven by pad events: ("swipe",), ("tap", screen, name), ("hold", screen, name), ("wait", s)."""
    ui = P.PhoneUI(meta={"variant": "preview"})
    t = 0.5
    for step in steps:
        kind = step[0]
        if kind == "swipe":
            ui.pad_down(t, PAD, 0.0, -131.0)
            ui.pad_move(t + 0.15, PAD, 0.0, -100.0)
            ui.pad_up(t + 0.3, PAD, 0.0, -61.0)
            t += 0.5
        elif kind in ("tap", "hold"):
            u, v = L.aim(step[1], step[2])
            ui.pad_down(t, PAD, u, v)
            if kind == "tap":
                ui.pad_up(t + 0.15, PAD, u, v)
            t += 0.4
        elif kind == "wait":
            t += step[1]
        else:
            raise ValueError(kind)
        ui.advance(t)
    return ui


def _taps(screen: str, names: Iterable[str]) -> List[tuple]:
    return [("tap", screen, n) for n in names]


TO_DIALPAD = [("swipe",), ("tap", "passcode", "emergency")]
PREVIEWS: Dict[str, List[tuple]] = {
    "lock": [],
    "passcode": [("swipe",)],
    "passcode_typing": [("swipe",)] + _taps("passcode", "12") + [("hold", "passcode", "3")],
    "passcode_wrong": [("swipe",)] + _taps("passcode", "123456"),
    "dialpad": TO_DIALPAD,
    "dialpad_911": TO_DIALPAD + _taps("dialpad", "911"),
    "calling": TO_DIALPAD + _taps("dialpad", "911") + _taps("dialpad", ["call"]),
    "call_connected": TO_DIALPAD + _taps("dialpad", "911") + _taps("dialpad", ["call"]) + [("wait", 7.0)],
    "call_failed": TO_DIALPAD + _taps("dialpad", "912") + _taps("dialpad", ["call"]),
}


def preview_files() -> List[str]:
    return [f"preview_{n}.png" for n in PREVIEWS] + ["preview_all.png", "preview_hit_areas.png"]


def _glass_xy(u: float, v: float, ppm: float) -> Tuple[float, float]:
    """Screen mm to preview pixels (the glass plus its rim)."""
    return ((u + L.U_HALF + L.BORDER_U + RIM_MM) * ppm, (L.V_HALF + L.BORDER_V + RIM_MM - v) * ppm)


def composite(visible: Iterable[str], atlas: Image.Image, ppm: float = PREVIEW_PX_PER_MM) -> Image.Image:
    """The phone face as the simulator's tiles would show ``visible``: each layer's atlas cell over its rectangle."""
    plan = D.atlas_plan()
    gw = L.GLASS_SIZE[1] * 1000 + 2 * RIM_MM
    gh = L.GLASS_SIZE[0] * 1000 + 2 * RIM_MM
    img = Image.new("RGB", (int(round(gw * ppm)), int(round(gh * ppm))), BODY_SRGB)
    ImageDraw.Draw(img).rounded_rectangle([RIM_MM * ppm, RIM_MM * ppm, (gw - RIM_MM) * ppm - 1,
                                           (gh - RIM_MM) * ppm - 1], radius=10 * ppm, fill=GLASS_SRGB)
    for name in sorted(visible, key=lambda n: (LEVEL_ORDER[L.LAYERS[n].level], n)):
        u0, v0, u1, v1 = L.LAYERS[name].rect
        x0, y0 = _glass_xy(u0, v1, ppm)
        x1, y1 = _glass_xy(u1, v0, ppm)
        x0, y0 = int(round(x0)), int(round(y0))
        size = (max(1, int(round(x1)) - x0), max(1, int(round(y1)) - y0))
        tile = atlas.crop(plan.cells[name])
        if tile.size != size:
            tile = tile.resize(size, Image.Resampling.BILINEAR)
        img.paste(tile, (x0, y0))
    return img


def _caption(img: Image.Image, text: str, fonts: Fonts, px: int = 22) -> Image.Image:
    out = Image.new("RGB", (img.width, img.height + px * 2), (255, 255, 255))
    out.paste(img, (0, 0))
    ImageDraw.Draw(out).text((img.width / 2, img.height + px), text, font=fonts.get(px), fill=(0, 0, 0), anchor="mm")
    return out


def _sheet(images: Sequence[Image.Image], gap: int = 16) -> Image.Image:
    w = sum(i.width for i in images) + gap * (len(images) + 1)
    h = max(i.height for i in images) + 2 * gap
    out = Image.new("RGB", (w, h), (255, 255, 255))
    x = gap
    for i in images:
        out.paste(i, (x, gap))
        x += i.width + gap
    return out


def hit_area_panel(screen: str, visible: Iterable[str], atlas: Image.Image, fonts: Fonts) -> Image.Image:
    ppm = PREVIEW_PX_PER_MM
    base = composite(visible, atlas, ppm).convert("RGBA")
    over = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)

    def rect(r, colour):
        x0, y0 = _glass_xy(r[0], r[3], ppm)
        x1, y1 = _glass_xy(r[2], r[1], ppm)
        d.rectangle([x0, y0, x1, y1], fill=colour + (48,), outline=colour + (255,), width=2)

    rect(L.DISPLAY, (128, 128, 128))
    if screen == "lock":
        rect(L.SWIPE_ZONE, (255, 64, 255))
    for h in L.HITS[screen]:
        rect(h.rect, (255, 214, 10) if h.kind == "digit" else (64, 170, 255))
    return _caption(Image.alpha_composite(base, over).convert("RGB"), screen, fonts)


def render_all(fonts: Fonts) -> Dict[str, object]:
    """Every output file: name -> PIL image or text."""
    atlas = render_atlas(fonts)
    out: Dict[str, object] = {D.ATLAS_FILE: atlas, D.USD_FILE: D.phone_usda()}
    shots = []
    for name, steps in PREVIEWS.items():
        img = composite(D.visible_layers(drive(steps)), atlas)
        out[f"preview_{name}.png"] = img
        half = img.resize((img.width // 2, img.height // 2), Image.Resampling.LANCZOS)
        shots.append(_caption(half, name.replace("_", " "), fonts, 18))
    out["preview_all.png"] = _sheet(shots)
    panels = [hit_area_panel(s, D.visible_layers(drive(steps)), atlas, fonts) for s, steps in (
        ("lock", PREVIEWS["lock"]), ("passcode", PREVIEWS["passcode_typing"]), ("dialpad", PREVIEWS["dialpad_911"]),
        ("calling", PREVIEWS["call_connected"]))]
    out["preview_hit_areas.png"] = _sheet(panels)
    return out


def _same(path: Path, obj) -> bool:
    if not path.is_file():
        return False
    if isinstance(obj, str):
        return path.read_text(encoding="utf-8") == obj
    with Image.open(path) as old:
        return old.size == obj.size and old.convert("RGB").tobytes() == obj.tobytes()


def main(argv: Sequence[str] = ()) -> int:
    check = "--check" in argv
    fonts = Fonts()
    files = render_all(fonts)
    D.ASSETS.mkdir(parents=True, exist_ok=True)
    bad = 0
    for name, obj in files.items():
        path = D.ASSETS / name
        if check:
            same = _same(path, obj)
            bad += not same
            print(("same     " if same else "DIFFERS  ") + name)
        elif isinstance(obj, str):
            path.write_text(obj, encoding="utf-8")
        else:
            obj.save(path, optimize=True)
    total = sum((D.ASSETS / n).stat().st_size for n in files if (D.ASSETS / n).is_file())
    plan = D.atlas_plan()
    print(f"font: {fonts.describe()}")
    print(f"atlas {plan.size[0]} x {plan.size[1]} px, {len(plan.drawn)} cells for {len(L.LAYERS)} layers; "
          f"{len(files)} files, {total / 1e6:.2f} MB")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

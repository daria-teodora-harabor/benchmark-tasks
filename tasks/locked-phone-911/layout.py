"""Geometry of the phone scene: where everything stands on the table, and what every screen draws and where it reacts.

Pure Python, so the task file, the offline tests and the screen renderer can all import it before Isaac Sim starts.
Every dimension is in millimetres unless a name says otherwise (``*_M`` and the world poses are in metres).

World (task) frame: the station's, metres. x points away from the arm bases, y to the robot's left, z up; the arm
bases stand at (0, -0.305) and (0, +0.305).

Screen frame: millimetres from the centre of the display, in the plane of the glass, as the robot reads the screen.
The phone lies with its status-bar end away from the arms, so the text reads upright from the robot:
  u: to the right on the screen (world -y)
  v: up the screen (world +x)
The display is 128 mm wide (u from -64 to 64) and 278 mm tall (v from -139 to 139).

Phone frame (for the USD layers): metres, origin at the centre of the glass top, +x toward the status bar, +y to the
robot's left, +z out of the glass. ``screen_to_local`` converts.

Text elements are placed by (u, v) = the horizontal centre of the text and the vertical centre of its capital height.
Their baseline is at ``v - CAP_EM * em / 2``. ``em`` is the font size (DejaVu Sans, as render_screens.py draws it).
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Tuple

LAYOUT_VERSION = "1"

Rect = Tuple[float, float, float, float]  # (u0, v0, u1, v1), screen mm, u0 < u1 and v0 < v1

# -- the table, the mat and the phone (world frame, metres) -----------------------------------------------------------
TABLE_TOP = 0.7475                      # station table top (yam_arena.task_file.TABLE_TOP)

MAT_SIZE = (0.360, 0.200, 0.003)        # black rubber mat, long side along x
MAT_POS = (0.350, 0.0, TABLE_TOP + MAT_SIZE[2] / 2)
MAT_TOP = TABLE_TOP + MAT_SIZE[2]

PHONE_X, PHONE_Y = 0.350, 0.0           # centre of the phone (and of its display) on the table; centred on the mat
PHONE_YAW = 0.0                         # status-bar end toward +x, away from the arms
BODY_SIZE = (0.295, 0.143, 0.015)       # dark grey body, 2x a 6.1-inch phone
GLASS_SIZE = (0.295, 0.143, 0.001)      # black glass top over the whole face
BODY_POS = (PHONE_X, PHONE_Y, MAT_TOP + BODY_SIZE[2] / 2)
GLASS_POS = (PHONE_X, PHONE_Y, MAT_TOP + BODY_SIZE[2] + GLASS_SIZE[2] / 2)
GLASS_Z = MAT_TOP + BODY_SIZE[2] + GLASS_SIZE[2]          # the glass surface, 19 mm above the table: 0.7665
PHONE_SIZE = (BODY_SIZE[0], BODY_SIZE[1], BODY_SIZE[2] + GLASS_SIZE[2])   # body and glass as one box
PHONE_POS = (PHONE_X, PHONE_Y, MAT_TOP + PHONE_SIZE[2] / 2)               # that box's centre

# -- the display (screen frame, mm) -----------------------------------------------------------------------------------
DISPLAY_W, DISPLAY_H = 128.0, 278.0     # the sensing, lit area: the central part of the glass
U_HALF, V_HALF = DISPLAY_W / 2, DISPLAY_H / 2
DISPLAY: Rect = (-U_HALF, -V_HALF, U_HALF, V_HALF)
BORDER_U = (GLASS_SIZE[1] * 1000 - DISPLAY_W) / 2    # black border that senses nothing: 7.5 mm at the sides
BORDER_V = (GLASS_SIZE[0] * 1000 - DISPLAY_H) / 2    # and 8.5 mm at the ends

LAYER_Z_MM = {"page": 0.10, "overlay": 0.20, "banner": 0.30}   # visual-only layers, this far above the glass


def screen_to_world(u: float, v: float, h_mm: float = 0.0) -> Tuple[float, float, float]:
    """Screen point (mm) to the world (task) frame (m), ``h_mm`` above the glass."""
    return (PHONE_X + v / 1000.0, PHONE_Y - u / 1000.0, GLASS_Z + h_mm / 1000.0)


def world_to_screen(x: float, y: float) -> Tuple[float, float]:
    """World (task) frame x, y (m) to screen (u, v) in mm. Height is ignored."""
    return ((PHONE_Y - y) * 1000.0, (x - PHONE_X) * 1000.0)


def screen_to_local(u: float, v: float, z_mm: float = 0.0) -> Tuple[float, float, float]:
    """Screen point (mm) to the phone frame (m): origin at the centre of the glass top, +x toward the status bar."""
    return (v / 1000.0, -u / 1000.0, z_mm / 1000.0)


def in_rect(rect: Rect, u: float, v: float) -> bool:
    """Edges count as inside."""
    return rect[0] <= u <= rect[2] and rect[1] <= v <= rect[3]


def in_display(u: float, v: float) -> bool:
    """True over the lit, touch-sensing display; False over the black border and anywhere off the phone."""
    return in_rect(DISPLAY, u, v)


def rect_at(u: float, v: float, w: float, h: float) -> Rect:
    return (u - w / 2, v - h / 2, u + w / 2, v + h / 2)


def rect_centre(rect: Rect) -> Tuple[float, float]:
    return ((rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2)


def rects_overlap(a: Rect, b: Rect) -> bool:
    """True when the two rectangles share area (touching edges do not count)."""
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


# -- colours (sRGB) ---------------------------------------------------------------------------------------------------
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
SOFT = (200, 200, 200)        # secondary text
KEY_FILL = (64, 64, 64)       # keypad circles
KEY_PRESSED = (175, 175, 175)
GREEN = (48, 209, 88)         # call button
RED = (255, 69, 58)           # end-call button, emergency header

# -- text -------------------------------------------------------------------------------------------------------------
FONT = "DejaVu Sans"
CAP_EM, ASC_EM, DESC_EM = 0.729, 0.760, 0.208   # capital height, ascender and descender depth, in em
# Advance widths (em) of the DejaVu Sans glyphs the screens use, measured with Pillow. Used only to size hit areas and
# check that text fits; render_screens.py measures the real font.
ADV = {
    " ": 0.3179, "#": 0.8379, "*": 0.5000, ",": 0.3179, ":": 0.3369, "…": 1.0000,
    "0": 0.6362, "1": 0.6362, "2": 0.6362, "3": 0.6362, "4": 0.6362, "5": 0.6362, "6": 0.6362, "7": 0.6362,
    "8": 0.6362, "9": 0.6362,
    "C": 0.6982, "D": 0.7700, "E": 0.6318, "J": 0.2949, "P": 0.6030, "S": 0.6348, "T": 0.6108, "W": 0.9888,
    "a": 0.6128, "c": 0.5498, "d": 0.6348, "e": 0.6152, "f": 0.3521, "g": 0.6348, "i": 0.2778, "l": 0.2778,
    "m": 0.9741, "n": 0.6338, "o": 0.6118, "p": 0.6348, "r": 0.4111, "s": 0.5210, "t": 0.3921, "u": 0.6338,
    "w": 0.8179, "y": 0.5918,
}
DIGIT_ADV = ADV["0"]

LABEL_EM = 10.0                         # labels the policy needs: Swipe up to open, Enter Passcode, Emergency,
                                        # Cancel / Delete, Emergency call, Calling 911..., Call failed (10 mm type:
                                        # capitals 7.3 mm, ascender to descender 9.7 mm)
KEY_D = 28.0                            # keypad circles
KEY_DIGIT_MM = 12.0                     # height of a key's digit
KEY_EM = KEY_DIGIT_MM / CAP_EM          # 16.5 mm type gives 12 mm digits
FIELD_EM = 15.0                         # dial pad number field (10.9 mm digits)
FIELD_ADV = 13.0                        # one character slot of the number field
FIELD_MAX = 8                           # characters the field holds; further key presses are ignored
TIMER_EM = 8.0
SMALL_EM = 7.0                          # the lock screen's date


def text_width(text: str, em: float) -> float:
    return sum(ADV.get(c, 0.65) for c in text) * em


class El(NamedTuple):
    """One drawn element. (u, v) is its centre (text: see the module docstring); w, h its size in mm (a circle's w is
    its diameter; for text, the estimated ink box). ``text`` is the string, or the icon's name for kind "icon"."""
    kind: str               # "text" | "circle" | "ring" | "rrect" | "icon"
    u: float
    v: float
    w: float
    h: float
    colour: Tuple[int, int, int]
    text: str = ""
    em: float = 0.0
    line: float = 0.0       # ring line width
    bold: bool = False


def text(s: str, u: float, v: float, em: float, colour=WHITE, bold: bool = False) -> El:
    return El("text", u, v, text_width(s, em), (ASC_EM + DESC_EM) * em, colour, s, em, 0.0, bold)


def circle(u: float, v: float, d: float, colour) -> El:
    return El("circle", u, v, d, d, colour)


def ring(u: float, v: float, d: float, colour, line: float) -> El:
    return El("ring", u, v, d, d, colour, line=line)


def rrect(u: float, v: float, w: float, h: float, colour) -> El:
    """A rounded bar (corner radius h / 2)."""
    return El("rrect", u, v, w, h, colour)


def icon(name: str, u: float, v: float, w: float, h: float, colour=WHITE) -> El:
    """Icons render_screens.py draws: lock, battery, handset (call), handset_down (end call), backspace."""
    return El("icon", u, v, w, h, colour, name)


def box(el: El) -> Rect:
    """The rectangle an element covers on the screen."""
    if el.kind != "text":
        return rect_at(el.u, el.v, el.w, el.h)
    base = el.v - CAP_EM * el.em / 2
    return (el.u - el.w / 2, base - DESC_EM * el.em, el.u + el.w / 2, base + ASC_EM * el.em)


# -- keypads (shared by the passcode screen and the emergency dial pad) -----------------------------------------------
KEY_COLS = (-34.0, 0.0, 34.0)           # u of the three columns
KEY_ROWS = (16.0, -18.0, -52.0, -86.0)  # v of the four rows (1 2 3 / 4 5 6 / 7 8 9 / * 0 #)
KEY_HIT = 32.0                          # square hit area around each 28 mm key (2 mm all round, 2 mm dead gaps)
BOTTOM_V = -120.0                       # the row under the keypad: Emergency / Cancel, call, end, delete
KEY_GRID = ("123", "456", "789", "*0#")
GLYPH_NAME = {"*": "star", "#": "hash"}  # prim-safe names; digits keep their own


def glyph_name(ch: str) -> str:
    return GLYPH_NAME.get(ch, ch)


def key_centre(key: str) -> Tuple[float, float]:
    for r, row in enumerate(KEY_GRID):
        c = row.find(key)
        if c >= 0:
            return KEY_COLS[c], KEY_ROWS[r]
    raise KeyError(key)


PASSCODE_KEYS = "1234567890"
DIALPAD_KEYS = "123456789*0#"


def _key_elements(keys: str) -> List[El]:
    out = []
    for k in keys:
        u, v = key_centre(k)
        out += [circle(u, v, KEY_D, KEY_FILL), text(k, u, v, KEY_EM, WHITE)]
    return out


# -- the four screens: what each page draws ---------------------------------------------------------------------------
BATTERY = icon("battery", 52.0, 132.0, 10.0, 4.6)

TITLE_V = 72.0                          # passcode screen: "Enter Passcode" (and "Wrong Passcode" over it)
RING_V, RING_D, RING_LINE, RING_PITCH = 50.0, 7.0, 0.8, 13.0
RING_U = tuple((i - 3.5) * RING_PITCH for i in range(1, 7))
HEADER_V = 100.0                        # dial pad and calling screen titles
FIELD_V = 60.0                          # dial pad number field
TIMER_V = 84.0                          # calling screen timer, under the title

EMERGENCY_LABEL = (-31.0, BOTTOM_V)     # passcode screen, bottom-left
PASS_CANCEL_LABEL = (34.0, BOTTOM_V)    # passcode screen, bottom-right: Cancel, or Delete while digits are entered
DIAL_CANCEL_LABEL = (-38.0, BOTTOM_V)   # dial pad, bottom-left
CALL_CENTRE = (0.0, BOTTOM_V)           # green call button (dial pad) and red end button (calling screen)
DELETE_CENTRE = (36.0, BOTTOM_V)        # dial pad's delete arrow, shown while the field is not empty

PAGES: Dict[str, List[El]] = {
    "page_lock": [
        BATTERY,
        icon("lock", 0.0, 122.0, 7.0, 9.0),
        text("Tuesday, January 9", 0.0, 106.0, SMALL_EM, SOFT),
        text("9:41", 0.0, 80.0, 30.0, WHITE),
        text("Swipe up to open", 0.0, -104.0, LABEL_EM, WHITE),
        rrect(0.0, -131.0, 46.0, 2.6, WHITE),                       # home bar
    ],
    "page_passcode": [
        BATTERY,
        text("Enter Passcode", 0.0, TITLE_V, LABEL_EM, WHITE),
        *[ring(u, RING_V, RING_D, WHITE, RING_LINE) for u in RING_U],
        *_key_elements(PASSCODE_KEYS),
        text("Emergency", EMERGENCY_LABEL[0], EMERGENCY_LABEL[1], LABEL_EM, WHITE),
        text("Cancel", PASS_CANCEL_LABEL[0], PASS_CANCEL_LABEL[1], LABEL_EM, WHITE),
    ],
    "page_dialpad": [
        BATTERY,
        text("Emergency call", 0.0, HEADER_V, LABEL_EM, RED),
        *_key_elements(DIALPAD_KEYS),
        circle(CALL_CENTRE[0], CALL_CENTRE[1], KEY_D, GREEN),
        icon("handset", CALL_CENTRE[0], CALL_CENTRE[1], 14.0, 14.0, WHITE),
        text("Cancel", DIAL_CANCEL_LABEL[0], DIAL_CANCEL_LABEL[1], LABEL_EM, WHITE),
    ],
    "page_calling": [
        BATTERY,
        circle(CALL_CENTRE[0], CALL_CENTRE[1], KEY_D, RED),
        icon("handset_down", CALL_CENTRE[0], CALL_CENTRE[1], 16.0, 7.0, WHITE),
    ],
}

# -- hit areas (screen mm). Digits act on touch-down, buttons on lift ------------------------------------------------
SWIPE_ZONE: Rect = (-U_HALF, -V_HALF, U_HALF, -V_HALF + 60.0)   # the lock screen's bottom 60 mm: a swipe starts here
SWIPE_MIN_MM = 50.0                                             # ... and rises at least this far up-screen


class Hit(NamedTuple):
    name: str               # digit "0".."9", "*", "#", or a button name
    kind: str               # "digit" | "button"
    rect: Rect


def _key_hits(keys: str) -> List[Hit]:
    return [Hit(k, "digit", rect_at(*key_centre(k), KEY_HIT, KEY_HIT)) for k in keys]


HITS: Dict[str, List[Hit]] = {
    "lock": [],                                                  # taps do nothing; only the swipe (SWIPE_ZONE)
    "passcode": _key_hits(PASSCODE_KEYS) + [
        Hit("emergency", "button", (-62.0, -130.0, 0.0, -110.0)),
        Hit("cancel", "button", (14.0, -130.0, 54.0, -110.0)),  # shows Cancel, or Delete while digits are entered
    ],
    "dialpad": _key_hits(DIALPAD_KEYS) + [
        Hit("cancel", "button", (-58.0, -130.0, -18.0, -110.0)),
        Hit("call", "button", rect_at(CALL_CENTRE[0], CALL_CENTRE[1], KEY_HIT, KEY_HIT)),
        Hit("delete", "button", (23.0, -130.0, 49.0, -110.0)),  # only while the field is not empty
    ],
    "calling": [Hit("end", "button", rect_at(CALL_CENTRE[0], CALL_CENTRE[1], KEY_HIT, KEY_HIT))],
    "failed": [],                                                # "Call failed": nothing reacts for its 2 s
}


def hit(screen: str, u: float, v: float, kind: Optional[str] = None) -> Optional[str]:
    """Name of the hit area of ``screen`` that contains (u, v), optionally only of one kind ("digit" / "button")."""
    for h in HITS.get(screen, ()):
        if (kind is None or h.kind == kind) and in_rect(h.rect, u, v):
            return h.name
    return None


def hit_rect(screen: str, name: str) -> Rect:
    for h in HITS[screen]:
        if h.name == name:
            return h.rect
    raise KeyError((screen, name))


AIM: Dict[str, Dict[str, Tuple[float, float]]] = {
    # where to tap: the centre of the drawn key or label (not of its hit area)
    "lock": {"home_bar": (0.0, -131.0)},
    "passcode": {**{k: key_centre(k) for k in PASSCODE_KEYS}, "emergency": EMERGENCY_LABEL,
                 "cancel": PASS_CANCEL_LABEL},
    "dialpad": {**{k: key_centre(k) for k in DIALPAD_KEYS}, "cancel": DIAL_CANCEL_LABEL, "call": CALL_CENTRE,
                "delete": DELETE_CENTRE},
    "calling": {"end": CALL_CENTRE},
}


def aim(screen: str, name: str) -> Tuple[float, float]:
    return AIM[screen][name]


# -- layers: what can be shown, each a flat tile over the glass -------------------------------------------------------
class Layer(NamedTuple):
    """One visual-only screen layer. ``rect`` is the tile it covers; ``level`` its height class (LAYER_Z_MM);
    ``elements`` what it draws on a black tile (pages draw over the whole display). Only ``page_lock`` is shown when
    the scene loads."""
    name: str
    rect: Rect
    level: str
    visible: bool
    elements: Tuple[El, ...]


PAGE_OF = {"lock": "page_lock", "passcode": "page_passcode", "dialpad": "page_dialpad", "calling": "page_calling",
           "failed": "page_calling"}

FIELD_SLOTS = 2 * FIELD_MAX - 1          # centred text: slot k sits at u = (k - FIELD_MAX + 1) * FIELD_ADV / 2
TIMER_ADV = DIGIT_ADV * TIMER_EM
TIMER_COLON_ADV = ADV[":"] * TIMER_EM
_E0 = -(3 * TIMER_ADV + TIMER_COLON_ADV) / 2       # m:ss, centred: edges of the four glyph tiles
_E1 = _E0 + TIMER_ADV
_E2 = _E1 + TIMER_COLON_ADV
_E3 = _E2 + TIMER_ADV
_E4 = _E3 + TIMER_ADV
TIMER_H = 11.0
TIMER_RECT: Dict[str, Rect] = {
    pos: (a, TIMER_V - TIMER_H / 2, b, TIMER_V + TIMER_H / 2)
    for pos, a, b in (("m", _E0, _E1), ("colon", _E1, _E2), ("t", _E2, _E3), ("u", _E3, _E4))}
TIMER_U = {pos: (r[0] + r[2]) / 2 for pos, r in TIMER_RECT.items()}
TITLE_RECT: Rect = (-62.0, HEADER_V - 10.0, 62.0, HEADER_V + 10.0)


def field_slot_u(slot: int) -> float:
    return (slot - (FIELD_MAX - 1)) * FIELD_ADV / 2


def field_layers(field: str) -> List[str]:
    """Names of the layers that show ``field`` centred in the number field (at most FIELD_MAX characters)."""
    n = len(field)
    return [f"field_{2 * i - (n - 1) + FIELD_MAX - 1:02d}_{glyph_name(ch)}" for i, ch in enumerate(field)]


def timer_layers(seconds: float) -> List[str]:
    """Names of the layers that show the call timer m:ss for ``seconds`` (whole seconds, at most 9:59)."""
    s = max(0, min(int(seconds), 599))
    return ["timer_colon", f"timer_m_{s // 60}", f"timer_t_{(s % 60) // 10}", f"timer_u_{s % 10}"]


def press_layer(screen: str, key: str) -> str:
    """Highlight of a pressed key on the passcode screen or the dial pad."""
    return f"press_{screen}_{glyph_name(key)}"


def _layers() -> Dict[str, Layer]:
    out: Dict[str, Layer] = {}

    def add(name, rect, level, elements, visible=False):
        assert name not in out, name
        out[name] = Layer(name, tuple(float(x) for x in rect), level, visible, tuple(elements))

    for page, els in PAGES.items():
        add(page, DISPLAY, "page", els, visible=(page == "page_lock"))
    # passcode screen
    for i, u in enumerate(RING_U, start=1):
        add(f"ring_{i}_full", rect_at(u, RING_V, 10.0, 10.0), "overlay", [circle(u, RING_V, RING_D, WHITE)])
    add("label_delete", hit_rect("passcode", "cancel"), "overlay",
        [text("Delete", PASS_CANCEL_LABEL[0], PASS_CANCEL_LABEL[1], LABEL_EM, WHITE)])
    add("banner_wrong_passcode", (-62.0, TITLE_V - 10.0, 62.0, TITLE_V + 10.0), "banner",
        [text("Wrong Passcode", 0.0, TITLE_V, LABEL_EM, WHITE)])
    for screen, keys in (("passcode", PASSCODE_KEYS), ("dialpad", DIALPAD_KEYS)):
        for k in keys:
            u, v = key_centre(k)
            add(press_layer(screen, k), rect_at(u, v, 30.0, 30.0), "overlay",
                [circle(u, v, KEY_D, KEY_PRESSED), text(k, u, v, KEY_EM, BLACK)])
    # dial pad
    for slot in range(FIELD_SLOTS):
        u = field_slot_u(slot)
        for ch in DIALPAD_KEYS:
            add(f"field_{slot:02d}_{glyph_name(ch)}", rect_at(u, FIELD_V, FIELD_ADV, 20.0), "overlay",
                [text(ch, u, FIELD_V, FIELD_EM, WHITE)])
    add("delete_arrow", hit_rect("dialpad", "delete"), "overlay",
        [icon("backspace", DELETE_CENTRE[0], DELETE_CENTRE[1], 18.0, 12.0, WHITE)])
    # calling screen (also shows "Call failed")
    add("title_calling", TITLE_RECT, "overlay", [text("Calling 911…", 0.0, HEADER_V, LABEL_EM, WHITE)])
    add("title_connected", TITLE_RECT, "overlay", [text("Emergency call", 0.0, HEADER_V, LABEL_EM, WHITE)])
    add("title_failed", TITLE_RECT, "overlay", [text("Call failed", 0.0, HEADER_V, LABEL_EM, WHITE)])
    add("end_hidden", hit_rect("calling", "end"), "banner", [])
    add("timer_colon", TIMER_RECT["colon"], "overlay",
        [text(":", TIMER_U["colon"], TIMER_V, TIMER_EM, SOFT)])
    for pos, top in (("m", 10), ("t", 6), ("u", 10)):
        for d in range(top):
            add(f"timer_{pos}_{d}", TIMER_RECT[pos], "overlay",
                [text(str(d), TIMER_U[pos], TIMER_V, TIMER_EM, SOFT)])
    return out


LAYERS: Dict[str, Layer] = _layers()


def layer_names() -> List[str]:
    return list(LAYERS)


def describe() -> str:
    """A plain-text summary of the layout (python3 layout.py)."""
    lines = [f"layout {LAYOUT_VERSION}: phone centre ({PHONE_X:.3f}, {PHONE_Y:.3f}) m, glass top z {GLASS_Z:.4f} m",
             f"display {DISPLAY_W:.0f} x {DISPLAY_H:.0f} mm: world x {screen_to_world(0, -V_HALF)[0]:.3f}.."
             f"{screen_to_world(0, V_HALF)[0]:.3f}, y {-U_HALF / 1000:+.3f}..{U_HALF / 1000:+.3f}",
             f"swipe zone {SWIPE_ZONE}, rise >= {SWIPE_MIN_MM:.0f} mm", f"{len(LAYERS)} layers"]
    for screen, hits in HITS.items():
        for h in hits:
            x, y, _ = screen_to_world(*rect_centre(h.rect))
            lines.append(f"  {screen:9s} {h.kind:6s} {h.name:9s} rect {h.rect}  centre world ({x:.3f}, {y:+.3f})")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())

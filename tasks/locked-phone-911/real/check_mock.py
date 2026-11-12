"""Checks of the real-trial mock dialler (mock_dialler.html) and of score_real.py. Plain python3 (3.9), no pytest:

    python3 check_mock.py            # prints ok / BAD per check; exits 1 if any check fails
    python3 check_mock.py --fix      # first rewrite the page's DATA block (from layout.py, phoneui.py, score.py) and
                                     # its CSP script hash, then check
    python3 check_mock.py --no-node  # skip the JavaScript cross-check

Static checks of the HTML (nothing is executed):
  - one inline script and no external resource of any kind; the Content-Security-Policy meta comes right after the
    charset, has exactly the expected directives (default-src 'none', script-src only the script's sha256 hash,
    style-src 'unsafe-inline', img-src data:, connect-src 'none', everything else 'none') and its hash matches the
    inline script byte for byte;
  - no forbidden strings anywhere: call and messaging link schemes (tel:, sip:, callto:, facetime: ...), network APIs
    (fetch(, XMLHttpRequest, WebSocket, EventSource, sendBeacon ...), WebRTC and media capture (RTCPeerConnection,
    getUserMedia ...), audio (<audio, AudioContext, speechSynthesis ...), workers, eval, frames, forms, links; no
    http:// or https:// in a src or href, nor anywhere else in the file; no inline event-handler attributes;
  - no navigation: a CSP does not govern navigations (location changes, link clicks, window.open), so the script may
    mention ``location`` only to read ``window.location.hash`` (the calibration), may set exactly one ``.href`` and call
    exactly one ``.click()``, both in exportLog on a blob URL from URL.createObjectURL (the local download of the
    log), and must not touch ``top.``, ``parent.``, ``opener`` or call ``open(``;
  - the touch guards (touch-action: none, no selection, no callout, no pinch zoom, no context menu) and the visible
    "Training dialler" header are present;
  - the page's DATA block equals what layout.py, phoneui.py and score.py hold now (geometry, hit areas, layers,
    timings), so the page cannot drift from the simulator.
score_real.py: imports only the standard-library modules it needs and the task's own modules (no network modules).
Scenarios: pointer-event fixtures (the demo route, a passcode attempt, a corrected digit, a failed call, hang-up and
redial, a touch during the call, pads in pairs, sliding onto the border, a cancelled pointer, a mouse, a resize, exact
rounding ties, every rotation) scored by score_real.py against the expected score. With node on PATH (only for this
check; the page itself needs nothing), the page's script is run under node on the same fixtures and must produce the
same phoneui rows, the same (u, v) for every event and the same visible layers after every event as the Python side.
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
from typing import Callable, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.dirname(HERE)
HTML = os.path.join(HERE, "mock_dialler.html")
SCORE_REAL = os.path.join(HERE, "score_real.py")
sys.path.insert(0, HERE)
sys.path.insert(0, TASK)

import layout as L        # noqa: E402
import phoneui as P       # noqa: E402
import score as S         # noqa: E402
import score_real as R    # noqa: E402

HEADER = "Training dialler — cannot place calls"
DATA_RE = re.compile(r"(/\*DATA:BEGIN\*/const DATA = )(.*?)(;/\*DATA:END\*/)", re.S)
SCRIPT_RE = re.compile(r"<script>(.*?)</script>", re.S)
CSP_RE = re.compile(r'<meta http-equiv="Content-Security-Policy" content="([^"]*)">')
HASH_RE = re.compile(r"script-src 'sha256-[A-Za-z0-9+/=]*'")

EXPECTED_CSP = {
    "default-src": ["'none'"], "style-src": ["'unsafe-inline'"], "img-src": ["data:"], "connect-src": ["'none'"],
    "font-src": ["'none'"], "media-src": ["'none'"], "object-src": ["'none'"], "frame-src": ["'none'"],
    "child-src": ["'none'"], "worker-src": ["'none'"], "manifest-src": ["'none'"], "base-uri": ["'none'"],
    "form-action": ["'none'"],
}
FORBIDDEN = (
    # call, message and app-launch link schemes
    "tel:", "sip:", "sips:", "callto:", "facetime:", "facetime-audio:", "skype:", "sms:", "mms:", "mailto:",
    "intent:", "wtai:", "whatsapp:", "msteams:", "zoomphonecall:", "javascript:",
    # network
    "fetch(", "XMLHttpRequest", "WebSocket", "EventSource", "sendBeacon", "WebTransport", "ws://", "wss://", "ftp://",
    "http://", "https://", "import(", "importScripts", "serviceWorker", "new Worker", "SharedWorker", "postMessage",
    "window.open", "location.href", "location.assign", "location.replace", "document.write", "Notification",
    "navigator.share", "navigator.bluetooth", "navigator.usb", "navigator.serial", "navigator.hid", "requestMIDIAccess",
    # WebRTC, capture, audio
    "RTCPeerConnection", "RTCDataChannel", "getUserMedia", "getDisplayMedia", "mediaDevices", "<audio", "<video",
    "AudioContext", "new Audio", "speechSynthesis", "SpeechRecognition", "geolocation",
    # code loading and embedding
    "eval(", "new Function", "<iframe", "<frame", "<object", "<embed", "<form", "<link", "<base", "<img", "<source",
    "<track", "<a ", "srcdoc", "http-equiv=\"refresh\"",
    # navigation (the CSP does not cover it; see check_navigation for location, href and click)
    "navigation.navigate", "opener", "document.domain", "setAttribute(\"href\"", "setAttribute('href'",
    "requestSubmit", ".submit(",
)
REQUIRED = (
    ("touch-action: none", "touch-action: none"), ("user-select: none", "no text selection"),
    ("-webkit-touch-callout: none", "no long-press callout"), ("user-scalable=no", "no pinch zoom (viewport)"),
    ('"contextmenu"', "context menu blocked"), ('"gesturestart"', "Safari pinch gestures blocked"),
    ('"selectstart"', "selection blocked"), ('"touchmove"', "touch scrolling blocked"),
    ('"wheel"', "ctrl-wheel zoom blocked"), ("pointerdown", "Pointer Events used"),
    ("telephone=no", "no automatic phone-number links"),
)
SCORE_REAL_IMPORTS = {"__future__", "argparse", "json", "os", "sys", "decimal", "typing", "layout", "phoneui", "score"}


# -- the DATA block ----------------------------------------------------------------------------------------------------

def layout_data() -> dict:
    """Everything the page takes from the Python modules. Floats are exact (json writes the shortest repr that reads
    back to the same double), so the page's hit tests use the very same numbers as phoneui."""
    def el(e):
        return [e.kind, e.u, e.v, e.w, e.h, list(e.colour), e.text, e.em, e.line, e.bold]
    return {
        "version": L.LAYOUT_VERSION,
        "display": list(L.DISPLAY),
        "glass_mm": [L.GLASS_SIZE[1] * 1000.0, L.GLASS_SIZE[0] * 1000.0],
        "cap_em": L.CAP_EM,
        "font": L.FONT,
        "swipe_zone": list(L.SWIPE_ZONE),
        "swipe_min_mm": L.SWIPE_MIN_MM,
        "field_max": L.FIELD_MAX,
        "glyph_name": dict(L.GLYPH_NAME),
        "hits": {s: [[h.name, h.kind, list(h.rect)] for h in hs] for s, hs in L.HITS.items()},
        "page_of": dict(L.PAGE_OF),
        "layers": [[ly.name, list(ly.rect), ly.level, ly.visible, [el(e) for e in ly.elements]]
                   for ly in L.LAYERS.values()],
        "phoneui": {"GROUP_MM": P.GROUP_MM, "MOVE_MM": P.MOVE_MM, "CODE_LEN": P.CODE_LEN,
                    "WRONG_BANNER_S": P.WRONG_BANNER_S, "FAILED_S": P.FAILED_S, "CONNECT_S": P.CONNECT_S,
                    "HOLD_S": P.HOLD_S, "T_DECIMALS": P.T_DECIMALS, "MM_DECIMALS": P.MM_DECIMALS,
                    "LOG_KINDS": list(P.LOG_KINDS), "SCREENS": list(P.SCREENS)},
        "score": {"NUMBER": S.NUMBER, "LEVELS": [list(x) for x in S.LEVELS]},
    }


def data_json() -> str:
    return json.dumps(layout_data(), ensure_ascii=True, separators=(",", ":"), allow_nan=False)


def script_hash(script: str) -> str:
    return "sha256-" + base64.b64encode(hashlib.sha256(script.encode("utf-8")).digest()).decode("ascii")


def fix(path: str = HTML) -> None:
    with open(path, encoding="utf-8", newline="") as f:
        html = f.read()
    html, n = DATA_RE.subn(lambda m: m.group(1) + data_json() + m.group(3), html)
    assert n == 1, "DATA block markers not found exactly once"
    scripts = SCRIPT_RE.findall(html)
    assert len(scripts) == 1, "expected exactly one inline <script>"
    html, n = HASH_RE.subn("script-src '" + script_hash(scripts[0]) + "'", html)
    assert n == 1, "script-src hash not found exactly once"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(html)
    print(f"fixed: DATA block and CSP hash in {path}")


# -- static checks -----------------------------------------------------------------------------------------------------

def read_html() -> str:
    with open(HTML, "rb") as f:
        raw = f.read()
    assert b"\r" not in raw, "carriage returns in the file (the browser hashes LF-normalised text)"
    return raw.decode("utf-8")


def markup_only(html: str) -> str:
    return SCRIPT_RE.sub("<script></script>", html)


def check_structure():
    html = read_html()
    assert html.lower().count("<script") == 1, "exactly one <script> element"
    assert "<script>" in html, "the script has no attributes (no src)"
    head = html[:1024]
    assert '<meta charset="utf-8">' in head, "charset meta in the first 1024 bytes"
    i_charset = html.index('<meta charset="utf-8">')
    m = CSP_RE.search(html)
    assert m, "CSP meta present"
    between = html[i_charset + len('<meta charset="utf-8">'):m.start()]
    assert between.strip() == "", "the CSP meta comes right after the charset meta"
    assert m.start() < html.index("<script>") and m.start() < html.index("<style>"), "CSP before style and script"
    assert html.count("http-equiv") == 1, "one http-equiv meta (the CSP)"
    assert html.count("Content-Security-Policy") == 1, "one CSP"


def parse_csp(content: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for part in content.split(";"):
        toks = part.split()
        if toks:
            assert toks[0] not in out, f"directive {toks[0]} repeated"
            out[toks[0]] = toks[1:]
    return out


def check_csp():
    html = read_html()
    csp = parse_csp(CSP_RE.search(html).group(1))
    script = SCRIPT_RE.search(html).group(1)
    want = dict(EXPECTED_CSP)
    want["script-src"] = ["'" + script_hash(script) + "'"]
    for k in sorted(set(csp) | set(want)):
        assert csp.get(k) == want.get(k), f"CSP {k}: {csp.get(k)} (want {want.get(k)})"


def check_hash():
    html = read_html()
    script = SCRIPT_RE.search(html).group(1)
    got = re.search(r"script-src '(sha256-[A-Za-z0-9+/=]*)'", html).group(1)
    assert got == script_hash(script), f"CSP hash {got} but the script hashes to {script_hash(script)}"
    assert "</script" not in script.lower() and "<!--" not in script, "script text closes or comments itself"


def check_forbidden():
    html = read_html()
    low = html.lower()
    found = [s for s in FORBIDDEN if s.lower() in low]
    assert not found, f"forbidden strings: {found}"
    bad = re.findall(r"""(?:src|href)\s*=\s*["']?\s*(?:https?:|wss?:|ftp:|//)""", html, re.I)
    assert not bad, f"remote src/href: {bad}"
    mk = markup_only(html)
    handlers = re.findall(r"<[^>]*\son[a-z]+\s*=", mk, re.I)
    assert not handlers, f"inline event handlers: {handlers}"
    assert not re.search(r"\b(?:src|href)\s*=", mk, re.I), "src/href attributes in the markup"


NAV_RULES = (
    (re.compile(r"\blocation\b"), 1, "``location`` (only the read of window.location.hash)"),
    (re.compile(r"\.href\b"), 1, "``.href`` (only the blob URL in exportLog)"),
    (re.compile(r"\.click\s*\("), 1, "``.click()`` (only the download link in exportLog)"),
    (re.compile(r"(?<!\w)open\s*\("), 0, "``open(`` (window.open and the like)"),
    (re.compile(r"\b(?:top|parent|self|frames)\s*\.\s*(?!style\b)\w"), 0,
     "``top.``, ``parent.``, ``self.`` or ``frames.``"),
    (re.compile(r"\bwindow\s*\.\s*(?:top|parent|frames|name)\b"), 0, "window.top / parent / frames / name"),
)


def check_navigation(script: Optional[str] = None):
    """The page cannot navigate anywhere: the CSP blocks fetches and subresources but not navigations."""
    if script is None:
        script = SCRIPT_RE.search(read_html()).group(1)
    for rx, allowed, what in NAV_RULES:
        n = len(rx.findall(script))
        assert n == allowed, f"{what}: {n} found, {allowed} allowed"
    assert len(re.findall(r"window\.location\.hash\b", script)) == 1, "the one location use reads window.location.hash"
    assert not re.search(r"location\.hash\s*=(?!=)", script), "window.location.hash is only read"
    m = re.search(r"\nfunction exportLog\b.*?\n}\n", script, re.S)
    assert m, "exportLog() found"
    body = m.group(0)
    assert re.search(r"\bconst url = URL\.createObjectURL\(blob\);", body), "exportLog's URL is a blob URL"
    assert re.search(r"\ba\.href = url;", body) and re.search(r"\ba\.click\(\);", body), \
        "the one .href and .click() are exportLog's blob download"
    assert re.search(r"\ba\.download = ", body), "the link is a download, not a navigation"


def check_required():
    html = read_html()
    missing = [why for s, why in REQUIRED if s not in html]
    assert not missing, f"missing: {missing}"
    mk = markup_only(html)
    assert f"<title>{HEADER}</title>" in mk, "title is the training-dialler header"
    assert mk.count(f">{HEADER}</text>") >= 1, "header drawn in the SVG margin"
    assert "HEADER" in SCRIPT_RE.search(html).group(1), "header shown in the operator panel"


def check_data():
    html = read_html()
    m = DATA_RE.search(html)
    assert m, "DATA block present"
    got = json.loads(m.group(2))
    want = json.loads(data_json())
    if got != want:
        diff = sorted(k for k in set(got) | set(want) if got.get(k) != want.get(k))
        raise AssertionError(f"DATA differs from layout.py / phoneui.py / score.py in {diff}; run check_mock.py --fix")


NAV_MUTANTS = (
    'document.location = "te" + "l:911";',
    'location = "sip" + ":x";',
    'const b = document.createElement("a"); b.href = ["tel", "911"].join(":"); b.click();',
    'top.location = "x";',
    'window.open("x");',
    'parent.postMessage;',
    'const w = self.open;',
)


def check_navigation_catches():
    """Each way the review found to build a call URL at run time, added to the page's script, fails the check."""
    script = SCRIPT_RE.search(read_html()).group(1)
    for m in NAV_MUTANTS:
        try:
            check_navigation(script + "\n" + m + "\n")
        except AssertionError:
            continue
        raise AssertionError(f"not caught: {m}")
    return f"{len(NAV_MUTANTS)} mutants caught"


def check_score_real_imports():
    with open(SCORE_REAL, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            mods.add((node.module or "").split(".")[0])
    extra = mods - SCORE_REAL_IMPORTS
    assert not extra, f"score_real.py imports {sorted(extra)}"
    with open(SCORE_REAL, encoding="utf-8") as f:
        text = f.read().lower()
    found = [s for s in ("tel:", "sip:", "callto:", "socket", "urllib", "requests", "subprocess", "webbrowser")
             if s in text]
    assert not found, f"score_real.py mentions {found}"


# -- pointer fixtures --------------------------------------------------------------------------------------------------
FRAME = 1.0 / 60   # Chrome delivers pointer moves once per frame

CALS = {
    "landscape90": {"ppm": 6.53846, "rot": 90, "cx": 960.0, "cy": 540.0},
    "portrait0": {"ppm": 6.21934, "rot": 0, "cx": 540.5, "cy": 960.25},
    "rot180": {"ppm": 5.5, "rot": 180, "cx": 600.0, "cy": 900.0},
    "rot270": {"ppm": 6.53846, "rot": 270, "cx": 959.5, "cy": 539.5},
    "tie8": {"ppm": 8.0, "rot": 0, "cx": 400.0, "cy": 1200.0},
}


class Script:
    """Builds one trial's pointer log the way the page records it: aims in screen mm, events in CSS px."""

    def __init__(self, cal: dict, t0: float = 0.5, off: Tuple[float, float] = (0.0, 0.0)):
        self.cal0 = dict(cal)
        self.cal = dict(cal)
        self.t = t0
        self.off = off
        self.pointer: List[dict] = []
        self._next = 1
        self.end_at: Optional[float] = None           # the trial's end time, when a scenario sets it exactly

    def xy(self, u: float, v: float) -> Tuple[float, float]:
        u, v = u + self.off[0], v + self.off[1]
        rot = int(self.cal["rot"]) % 360
        dx, dy = {0: (u, -v), 90: (v, u), 180: (-u, v), 270: (-v, -u)}[rot]
        return round(self.cal["cx"] + dx * self.cal["ppm"], 4), round(self.cal["cy"] + dy * self.cal["ppm"], 4)

    def ev(self, kind: str, pid: int, u: float, v: float, pt: str = "touch") -> None:
        x, y = self.xy(u, v)
        self.pointer.append({"t": round(self.t, 6), "type": kind, "id": pid, "pt": pt, "x": x, "y": y,
                             "w": 8.0, "h": 8.0, "p": 0.5})

    def wait(self, s: float) -> "Script":
        self.t += s
        return self

    def down(self, u, v, pt="touch") -> int:
        pid = 1 if pt == "mouse" else self._next     # a mouse keeps pointerId 1
        self._next += pt != "mouse"
        self.ev("down", pid, u, v, pt)
        return pid

    def move(self, pid, u, v, pt="touch"):
        self.t += FRAME
        self.ev("move", pid, u, v, pt)

    def up(self, pid, u, v, pt="touch", kind="up"):
        self.t += FRAME
        self.ev(kind, pid, u, v, pt)

    def tap(self, u, v, hold=0.15, pt="touch", after=0.25, wobble=0.2):
        pid = self.down(u, v, pt)
        for i in range(max(1, int(round(hold / FRAME)))):
            self.move(pid, u + wobble * (i % 2), v, pt)          # sub-threshold jitter: no move rows
        self.up(pid, u, v, pt)
        self.wait(after)
        return pid

    def pair_tap(self, u, v, sep=4.0, hold=0.15, after=0.25):
        """A closed gripper: two pads ``sep`` mm apart landing one frame apart."""
        a = self.down(u - sep / 2, v)
        self.t += FRAME
        b = self.down(u + sep / 2, v)
        self.wait(hold)
        self.up(b, u + sep / 2, v)
        self.up(a, u - sep / 2, v)
        self.wait(after)

    def press(self, screen, name, **kw):
        return self.tap(*L.aim(screen, name), **kw)

    def drag(self, u0, v0, u1, v1, dur=0.4, pt="touch", kind="up", after=0.3):
        pid = self.down(u0, v0, pt)
        n = max(2, int(round(dur / FRAME)))
        for i in range(1, n + 1):
            s = i / n
            self.move(pid, u0 + (u1 - u0) * s, v0 + (v1 - v0) * s, pt)
        self.up(pid, u1, v1, pt, kind)
        self.wait(after)
        return pid

    def swipe(self, **kw):
        return self.drag(0.0, -131.0, 0.0, -61.0, **kw)

    def dial(self, keys, **kw):
        for k in keys:
            self.press("dialpad", k, **kw)
        return self

    def recal(self, **changes):
        self.cal.update(changes)
        self.pointer.append({"t": round(self.t, 6), "type": "cal", "cal": dict(self.cal)})

    def trial(self, n: int, name: str, tail: float = 0.5) -> dict:
        t_end = self.end_at if self.end_at is not None else round(self.t + tail, 6)
        return {"trial": n, "started": f"fixture {name}", "meta": {"variant": "fixture", "trial": n, "started": name},
                "cal": dict(self.cal0), "t_end": t_end, "ended": True, "pointer": self.pointer}


def _to_dialpad(s: Script, pt="touch") -> Script:
    s.swipe(pt=pt)
    s.press("passcode", "emergency", pt=pt)
    return s


def _call_and_hold(s: Script, hold=4.5, pt="touch") -> Script:
    s.press("dialpad", "call", pt=pt)
    return s.wait(hold)


def sc_demo(s, pt="touch"):
    _to_dialpad(s, pt).dial("911", pt=pt)
    _call_and_hold(s, pt=pt)


def sc_guess_passcode(s):
    s.swipe()
    for k in "123456":
        s.press("passcode", k)
    s.press("passcode", "emergency")
    s.dial("911")
    _call_and_hold(s)


def sc_type_911_then_emergency(s):
    s.swipe()
    for k in "911":
        s.press("passcode", k)
    s.press("passcode", "emergency")
    s.dial("911")
    _call_and_hold(s)


def sc_no_swipe(s):
    s.press("lock", "home_bar")
    s.tap(0.0, 0.0)
    s.wait(1.0)


def sc_short_swipe(s):
    s.drag(0.0, -131.0, 0.0, -86.0)
    s.wait(1.0)


def sc_stop_at_dialpad(s):
    _to_dialpad(s).wait(2.0)


def sc_dial_no_call(s):
    _to_dialpad(s).dial("911").wait(3.0)


def sc_no_lift_11(s):
    _to_dialpad(s).dial("9")
    u, v = L.aim("dialpad", "1")
    pid = s.down(u, v)
    for i in range(36):                               # 0.6 s on the key, wobbling, never lifting
        s.move(pid, u + 0.3 * (i % 3), v)
    s.up(pid, u, v)
    s.wait(0.25)
    _call_and_hold(s, hold=3.0)


def sc_wrong_digit_fixed(s):
    _to_dialpad(s).dial("92")
    s.press("dialpad", "delete")
    s.dial("11")
    _call_and_hold(s)


def sc_touch_during_call(s):
    _to_dialpad(s).dial("911")
    s.press("dialpad", "call")
    s.wait(1.0)
    s.tap(0.0, 30.0, hold=0.2)                        # nothing there; restarts the 3 s
    s.wait(2.0)
    s.wait(2.5)


def sc_hang_up(s):
    _to_dialpad(s).dial("911")
    s.press("dialpad", "call")
    s.wait(1.5)
    s.press("calling", "end")
    s.wait(1.0)


def sc_hang_up_redial(s):
    sc_hang_up(s)
    s.dial("911")
    _call_and_hold(s)


def sc_pair_closed(s):
    s.swipe()
    s.pair_tap(*L.aim("passcode", "emergency"))
    for k in "911":
        s.pair_tap(*L.aim("dialpad", k))
    s.pair_tap(*L.aim("dialpad", "call"))
    s.wait(4.5)


def sc_pair_wide(s):
    _to_dialpad(s)
    (u1, v1), (u2, v2) = L.aim("dialpad", "1"), L.aim("dialpad", "2")
    a = s.down(u1, v1)
    b = s.down(u2, v2)                                # 34 mm apart: two touches, "12"
    s.wait(0.15)
    s.up(a, u1, v1)
    s.up(b, u2, v2)
    s.wait(0.25)
    s.press("dialpad", "delete")
    s.press("dialpad", "delete")
    s.dial("911")
    _call_and_hold(s)


def sc_slide_off_border(s):
    _to_dialpad(s).dial("911")
    s.drag(0.0, -120.0, 0.0, -150.0, dur=0.3)         # off the display while still in contact: no call
    s.wait(2.0)


def sc_cancel_swipe(s):
    s.swipe(kind="cancel")                            # a cancelled pointer lifts at its last point
    s.wait(1.0)


def sc_pre_trial_pointer(s):
    s.ev("move", 77, 0.0, -131.0)                     # went down before the trial: ignored
    s.ev("up", 77, 0.0, -50.0)
    sc_demo(s)


def sc_recal(s):
    s.swipe()
    s.press("passcode", "emergency")
    s.recal(cx=s.cal["cx"] + 37.5, cy=s.cal["cy"] - 12.25)   # window resized mid-trial
    s.dial("911")
    _call_and_hold(s)


def sc_failed_then_fix(s):
    _to_dialpad(s).dial("912")
    s.press("dialpad", "call")                        # "Call failed" for 2 s
    s.wait(0.3)
    s.press("dialpad", "5")                           # ignored while failed
    s.wait(2.0)
    s.press("dialpad", "delete")
    s.dial("1")
    _call_and_hold(s)


def sc_failed_91_then_911(s):
    _to_dialpad(s).dial("91")                         # no wrong digit: 91 is the start of 911
    s.press("dialpad", "call")                        # "Call failed" for 2 s, field kept
    s.wait(2.2)
    s.dial("1")
    _call_and_hold(s)                                 # a clean 911 call held 3 s, after a failed call: 0.9


def sc_empty_call(s):
    _to_dialpad(s)
    s.press("dialpad", "call")
    s.wait(2.0)


def sc_swipe_rise(rise, v0=-131.0, steps_mm=None):
    def build(s):
        if steps_mm:                                  # slow: exactly steps_mm per frame, so moves land on MOVE_MM
            s.drag(0.0, v0, 0.0, v0 + rise, dur=rise / steps_mm * FRAME)
        else:
            s.drag(0.0, v0, 0.0, v0 + rise)
        s.wait(1.0)
    return build


def sc_pair_sep(sep):
    def build(s):
        _to_dialpad(s)
        for k in "911":
            s.pair_tap(*L.aim("dialpad", k), sep=sep)
        _call_and_hold(s)
    return build


def sc_edge_tap(du):
    def build(s):
        s.swipe()
        r = L.hit_rect("passcode", "emergency")
        s.tap(r[2] + du, r[3], wobble=0.0)            # the hit area's top-right corner (edges count as inside)
        s.wait(1.0)
    return build


def sc_hold_exact(extra):
    def build(s):
        _to_dialpad(s).dial("911")
        s.press("dialpad", "call", after=0.0)
        s.end_at = round(s.pointer[-1]["t"] + P.HOLD_S + extra, 6)
    return build


def sc_carried_touch(s):
    """A pad rests in the bottom zone on the passcode screen while another taps Cancel (back to the lock screen); the
    resting pad then slides 60 mm up and lifts. It began on an earlier screen, so it is no swipe."""
    s.swipe()
    a = s.down(20.0, -131.0)
    s.press("passcode", "cancel")
    for i in range(1, 25):
        s.move(a, 20.0, -131.0 + 2.5 * i)
    s.up(a, 20.0, -71.0)
    s.wait(1.0)


def sc_passcode_cancel(s):
    s.swipe()
    s.press("passcode", "1")
    s.press("passcode", "2")
    s.press("passcode", "cancel")                     # reads Delete
    s.press("passcode", "cancel")
    s.press("passcode", "cancel")                     # reads Cancel: back to the lock screen
    s.wait(1.0)


# name -> (calibration, offset, builder, expected score, expected score with --touch-only)
SCENARIOS: Dict[str, Tuple[str, Tuple[float, float], Callable, float, Optional[float]]] = {
    "demo": ("landscape90", (0, 0), sc_demo, 1.0, None),
    "demo_portrait": ("portrait0", (0, 0), sc_demo, 1.0, None),
    "demo_rot180": ("rot180", (0, 0), sc_demo, 1.0, None),
    "demo_rot270": ("rot270", (0, 0), sc_demo, 1.0, None),
    "demo_ties": ("tie8", (0.125, -0.375), sc_demo, 1.0, None),
    "demo_mouse": ("landscape90", (0, 0), lambda s: sc_demo(s, pt="mouse"), 1.0, 0.0),
    "guess_passcode": ("landscape90", (0, 0), sc_guess_passcode, 0.0, None),
    "type_911_then_emergency": ("landscape90", (0, 0), sc_type_911_then_emergency, 1.0, None),
    "no_swipe": ("landscape90", (0, 0), sc_no_swipe, 0.0, None),
    "short_swipe": ("portrait0", (0, 0), sc_short_swipe, 0.0, None),
    "stop_at_dialpad": ("landscape90", (0, 0), sc_stop_at_dialpad, 0.4, None),
    "dial_no_call": ("landscape90", (0, 0), sc_dial_no_call, 0.6, None),
    "no_lift_11": ("landscape90", (0, 0), sc_no_lift_11, 0.4, None),
    "wrong_digit_fixed": ("landscape90", (0, 0), sc_wrong_digit_fixed, 0.9, None),
    "touch_during_call": ("landscape90", (0, 0), sc_touch_during_call, 1.0, None),
    "hang_up": ("landscape90", (0, 0), sc_hang_up, 0.8, None),
    "hang_up_redial": ("landscape90", (0, 0), sc_hang_up_redial, 1.0, None),
    "pair_closed": ("portrait0", (0, 0), sc_pair_closed, 1.0, None),
    "pair_wide": ("landscape90", (0, 0), sc_pair_wide, 0.9, None),
    "slide_off_border": ("landscape90", (0, 0), sc_slide_off_border, 0.6, None),
    "cancel_swipe": ("landscape90", (0, 0), sc_cancel_swipe, 0.2, None),
    "pre_trial_pointer": ("landscape90", (0, 0), sc_pre_trial_pointer, 1.0, None),
    "recal_mid_trial": ("landscape90", (0, 0), sc_recal, 1.0, None),
    "failed_then_fix": ("landscape90", (0, 0), sc_failed_then_fix, 0.9, None),
    "failed_91_then_911": ("landscape90", (0, 0), sc_failed_91_then_911, 0.9, None),
    "empty_call": ("landscape90", (0, 0), sc_empty_call, 0.4, None),
    "passcode_cancel": ("landscape90", (0, 0), sc_passcode_cancel, 0.2, None),
    # boundaries: each rule's threshold, met exactly and just missed
    "swipe_50mm": ("portrait0", (0, 0), sc_swipe_rise(50.0), 0.2, None),
    "swipe_49_99mm": ("portrait0", (0, 0), sc_swipe_rise(49.99), 0.0, None),
    "swipe_from_zone_edge": ("landscape90", (0, 0), sc_swipe_rise(50.0, v0=L.SWIPE_ZONE[3]), 0.2, None),
    "swipe_from_above_zone": ("landscape90", (0, 0), sc_swipe_rise(60.0, v0=L.SWIPE_ZONE[3] + 0.01), 0.0, None),
    "swipe_slow_0_5mm_steps": ("landscape90", (0, 0), sc_swipe_rise(60.0, steps_mm=0.5), 0.2, None),
    "pair_12mm_one_touch": ("portrait0", (0, 0), sc_pair_sep(12.0), 1.0, None),
    "pair_12_02mm_two_touches": ("portrait0", (0, 0), sc_pair_sep(12.02), 0.4, None),
    "tap_on_hit_corner": ("rot270", (0, 0), sc_edge_tap(0.0), 0.4, None),
    "tap_just_off_hit_corner": ("rot270", (0, 0), sc_edge_tap(0.01), 0.2, None),
    "hold_exactly_3s": ("landscape90", (0, 0), sc_hold_exact(0.0), 1.0, None),
    "hold_1us_short": ("landscape90", (0, 0), sc_hold_exact(-0.000001), 0.8, None),
    "carried_touch_ignored": ("landscape90", (0, 0), sc_carried_touch, 0.2, None),
}


def fixture_trials() -> List[Tuple[str, dict, float, Optional[float]]]:
    out = []
    for i, (name, (cal, off, build, want, want_touch)) in enumerate(SCENARIOS.items(), start=1):
        s = Script(CALS[cal], off=off)
        build(s)
        out.append((name, s.trial(i, name), want, want_touch))
    return out


def py_run(trial: dict) -> dict:
    vis: List[List[str]] = []
    ui, info = R.run_trial(trial, after_event=lambda u, e: vis.append(sorted(u.visible_layers())))
    vis.append(sorted(ui.visible_layers()))
    return {"ui": ui, "rows": ui.rows, "vis": vis, "conv": [list(uv) for uv in info["conv"]]}


def check_scenarios(fx):
    bad = []
    for name, trial, want, want_touch in fx:
        r = R.score_trial(trial)
        if r["score"] != want or not r["replay_ok"] or r["success"] != (want == 1.0):
            bad.append(f"{name}: score {r['score']} (want {want}), replay ok {r['replay_ok']}")
        if want_touch is not None:
            rt = R.score_trial(trial, touch_only=True)
            if rt["score"] != want_touch:
                bad.append(f"{name} --touch-only: score {rt['score']} (want {want_touch})")
    assert not bad, "; ".join(bad)


def check_scenario_details(fx):
    by = {name: trial for name, trial, _, _ in fx}
    # the touch during the call restarts the 3 s: held exactly 3 s after that touch's lift
    rows = py_run(by["touch_during_call"])["rows"]
    call_t = next(r["t"] for r in rows if r["kind"] == "screen" and r["screen"] == "calling")
    held = [r for r in rows if r["kind"] == "held"]
    assert len(held) == 1 and held[0]["since"] > call_t + 1.0, held
    assert abs(held[0]["t"] - held[0]["since"] - P.HOLD_S) < 1e-9, held
    # the slide onto the border lifts the pad at once, short of the call button
    rows = py_run(by["slide_off_border"])["rows"]
    assert not any(r["kind"] == "call" for r in rows), "slide off the display placed a call"
    # the wide pair gives two touches and two digits
    rows = py_run(by["pair_wide"])["rows"]
    assert [r["field"] for r in rows if r["kind"] == "field"][:2] == ["1", "12"], "wide pair should enter 1 then 2"
    # the closed pair gives one touch per key
    rows = py_run(by["pair_closed"])["rows"]
    assert sum(1 for r in rows if r["kind"] == "touch" and r["action"] == "join") == 5, "closed pair joins"
    # pads 12.02 mm apart are two touches: every key entered twice
    rows = py_run(by["pair_12_02mm_two_touches"])["rows"]
    assert [r["field"] for r in rows if r["kind"] == "field"][:2] == ["9", "99"], "12.02 mm pair should enter 9 twice"
    # a touch carried across a screen change is used up: no swipe from the lock screen
    pr = py_run(by["carried_touch_ignored"])
    assert pr["ui"].screen == "lock" and any(r["kind"] == "ignored" for r in pr["rows"]), "carried touch acted"
    # 0.5 mm steps report a move every frame
    rows = py_run(by["swipe_slow_0_5mm_steps"])["rows"]
    assert sum(1 for r in rows if r["kind"] == "pad" and r["action"] == "move") == 120, "0.5 mm steps are moves"
    # the exact ties round half away from zero, as the page does
    assert R.round_half_up(0.125, 2) == 0.13 and R.round_half_up(-33.875, 2) == -33.88
    assert R.round_half_up(-0.001, 2) == 0.0 and str(R.round_half_up(-0.001, 2)) == "0.0"
    # a multi-trial export scores trial by trial
    data = {"format": R.FORMAT, "trials": [by["demo"], by["guess_passcode"], by["hang_up"]]}
    assert [r["score"] for r in R.score_export(data)] == [1.0, 0.0, 0.8]


def find_node() -> Optional[str]:
    return shutil.which("node") or next((p for p in ("/opt/homebrew/bin/node", "/usr/local/bin/node")
                                         if os.access(p, os.X_OK)), None)


NODE_HARNESS = r"""
const vm = require("vm");
let buf = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (d) => { buf += d; });
process.stdin.on("end", () => {
  const inp = JSON.parse(buf);
  const ctx = vm.createContext({});
  vm.runInContext(inp.script, ctx, {filename: "mock_dialler.html#script"});
  const api = ctx.MockDialler;
  const out = inp.trials.map((tr) => JSON.parse(JSON.stringify(api.runTrial(JSON.parse(JSON.stringify(tr))))));
  process.stdout.write(JSON.stringify(out));
});
"""


def check_node(fx, node: str):
    script = SCRIPT_RE.search(read_html()).group(1)
    payload = json.dumps({"script": script, "trials": [t for _, t, _, _ in fx]})
    res = subprocess.run([node, "-e", NODE_HARNESS], input=payload.encode("utf-8"), capture_output=True, timeout=120)
    assert res.returncode == 0, f"node failed: {res.stderr.decode('utf-8', 'replace')[-2000:]}"
    js = json.loads(res.stdout.decode("utf-8"))
    bad = []
    for (name, trial, _, _), jr in zip(fx, js):
        pr = py_run(trial)
        match, diff = R.compare_rows(pr["rows"], jr["rows"])
        if not match:
            bad.append(f"{name}: rows differ {json.dumps(diff)}")
        if pr["conv"] != jr["conv"]:
            i = next(i for i, (a, b) in enumerate(zip(pr["conv"], jr["conv"])) if a != b) \
                if len(pr["conv"]) == len(jr["conv"]) else -1
            bad.append(f"{name}: (u, v) differ at event {i}")
        if pr["vis"] != jr["vis"]:
            i = next((i for i, (a, b) in enumerate(zip(pr["vis"], jr["vis"])) if a != b), -1)
            bad.append(f"{name}: visible layers differ after event {i}: python {pr['vis'][i]} page {jr['vis'][i]}")
        if pr["ui"].score != jr["score"] or pr["ui"].success != jr["success"]:
            bad.append(f"{name}: score python {pr['ui'].score} page {jr['score']}")
    assert len(js) == len(fx), "node returned a different number of trials"
    assert not bad, "; ".join(bad[:6])
    return sum(len(t["pointer"]) for _, t, _, _ in fx), sum(len(j["rows"]) for j in js)


# -- runner ------------------------------------------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Check mock_dialler.html and score_real.py.")
    ap.add_argument("--fix", action="store_true", help="rewrite the DATA block and the CSP hash first")
    ap.add_argument("--no-node", action="store_true", help="skip the JavaScript cross-check under node")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)
    if args.fix:
        fix()
    fails = 0

    def run(name, fn, *a):
        nonlocal fails
        try:
            out = fn(*a)
            print(f"ok   {name}" + (f" ({out})" if isinstance(out, str) else ""))
        except Exception as e:   # noqa: BLE001 - report every check
            fails += 1
            print(f"BAD  {name}: {e}")
            if not isinstance(e, AssertionError):
                traceback.print_exc()

    run("structure: one inline script, CSP meta right after the charset", check_structure)
    run("CSP directives exactly as expected", check_csp)
    run("CSP sha256 matches the inline script", check_hash)
    run("no forbidden strings, no remote src/href, no inline handlers", check_forbidden)
    run("no navigation: one location.hash read, one blob download link, no open/top/parent", check_navigation)
    run("the navigation check catches built-up call URLs", check_navigation_catches)
    run("touch guards and the training-dialler header present", check_required)
    run("DATA block equals layout.py / phoneui.py / score.py", check_data)
    run("score_real.py imports only what it needs", check_score_real_imports)
    fx = fixture_trials()
    run(f"{len(fx)} pointer scenarios score as expected (score_real.py + score.py replay)", check_scenarios, fx)
    run("scenario details (hold restart, border lift, pad pairs, ties, multi-trial)", check_scenario_details, fx)
    node = None if args.no_node else find_node()
    if node:
        def node_check():
            n_ev, n_rows = check_node(fx, node)
            return f"{len(fx)} trials, {n_ev} pointer events, {n_rows} rows identical"
        run("page script under node == phoneui (rows, (u, v), visible layers, score)", node_check)
    else:
        print("skip page script under node (" + ("--no-node" if args.no_node else "node not found") + ")")
    print(f"{'all checks passed' if not fails else f'{fails} check(s) failed'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

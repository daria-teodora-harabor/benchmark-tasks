"""Score real trials of the locked-phone task from the log the mock dialler (mock_dialler.html) exports.

The mock runs on a capacitive touch monitor driven by a host with no radios and keeps every Pointer Event in memory.
On the operator's X key it ends the trial and downloads them as JSON, with the calibration each trial used and the
page's own live phoneui rows. This script turns the pointer events into phoneui pad events by the same rules the page
uses, runs them through phoneui.PhoneUI (the simulator's state machine) and scores the rows with score.py. The Python
result is the trial's score.

Pointer events to pad events (the same in mock_dialler.html's PointerFeed):
  - Each pointer (pointerId) is one pad, named "p<id>". The monitor's own touch-down and lift replace the simulator's
    gap thresholds (down within 0.5 mm, up once 2 mm clear for 2 samples).
  - (u, v) in screen mm come from the CSS pixel position and the trial's calibration (px per mm, the display centre in
    the viewport, the drawing's rotation), rounded half away from zero to 0.01 mm, exactly as the page rounds them.
  - A pointer is down while it is in contact and its point is on the 128 x 278 mm display. Sliding onto the border
    lifts it at once, at its last point on the display; sliding back on puts it down again. While down it reports a
    move each time it has slid 0.5 mm. A cancelled pointer (pointercancel) lifts at its last point.
  - Moves and lifts of a pointer that went down before the trial began do nothing.
  - Time is the page's clock in seconds since the trial began (1 us steps), never wall time.

Checks reported per trial (exit status 1 if any fails):
  - the page's live rows equal the rows replayed here, row for row (the JS port and phoneui agree);
  - the page's (u, v) for every event equal this script's conversion;
  - score.check: phoneui re-run on the replayed pad rows alone writes the same rows and score.

    python3 score_real.py phone911-real-20261111-153000.json
    python3 score_real.py export.json --jsonl rows.jsonl     # also write the rows; then: python3 ../score.py rows.jsonl
    python3 score_real.py export.json --touch-only           # drop mouse and pen pointers (the page fed them)

Pure Python 3.9, standard library only, no network.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from decimal import ROUND_HALF_UP, Decimal
from typing import Callable, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.dirname(HERE)
sys.path.insert(0, TASK)

import layout as L   # noqa: E402
import phoneui as P  # noqa: E402
import score as S    # noqa: E402

for _m in (L, P, S):   # an Isaac or site-packages module of the same name must not shadow the task's own
    if os.path.dirname(os.path.abspath(_m.__file__)) != TASK:
        raise ImportError(f"{_m.__name__} was imported from {_m.__file__}, not from {TASK}")

FORMAT = "locked-phone-911 mock dialler log"
TOUCH_TYPES = ("touch",)


def round_half_up(x: float, nd: int) -> float:
    """Round the exact binary value of ``x`` to ``nd`` decimals, ties away from zero, as JavaScript's
    Number(x.toFixed(nd)) does."""
    q = Decimal(1).scaleb(-nd)
    return float(Decimal(float(x)).quantize(q, rounding=ROUND_HALF_UP)) + 0.0


def to_screen(x: float, y: float, cal: dict) -> Tuple[float, float]:
    """CSS pixels (clientX, clientY) to screen mm (u, v). ``cal``: ppm (CSS px per mm), cx, cy (display centre in the
    viewport, CSS px), rot (clockwise turn of the drawing on the monitor, degrees, a multiple of 90)."""
    ppm = float(cal["ppm"])
    dx = (float(x) - float(cal["cx"])) / ppm
    dy = (float(y) - float(cal["cy"])) / ppm
    rot = int(cal["rot"]) % 360
    if rot == 0:
        u, v = dx, -dy
    elif rot == 90:
        u, v = dy, dx
    elif rot == 180:
        u, v = -dx, dy
    elif rot == 270:
        u, v = -dy, -dx
    else:
        raise ValueError(f"rotation must be a multiple of 90, not {cal['rot']}")
    return round_half_up(u, P.MM_DECIMALS), round_half_up(v, P.MM_DECIMALS)


class _Contact:
    __slots__ = ("down", "last_on", "sent")

    def __init__(self):
        self.down = False
        self.last_on = (0.0, 0.0)
        self.sent = (0.0, 0.0)


class PointerFeed:
    """Pointer log entries in, phoneui pad events out (mock_dialler.html's PointerFeed, line for line)."""

    def __init__(self, ui: "P.PhoneUI", cal: dict):
        self.ui = ui
        self.cal = dict(cal)
        self.contact: Dict[int, _Contact] = {}

    def event(self, ent: dict) -> None:
        kind = ent["type"]
        if kind == "cal":
            self.cal = dict(ent["cal"])
            return
        pid, t = ent["id"], ent["t"]
        pad = f"p{pid}"
        if kind == "down":
            if pid not in self.contact:
                self.contact[pid] = _Contact()
            self._sample(t, pid, pad, ent["x"], ent["y"])
            return
        if pid not in self.contact:
            return
        if kind == "move":
            self._sample(t, pid, pad, ent["x"], ent["y"])
            return
        if kind == "up":
            self._sample(t, pid, pad, ent["x"], ent["y"])
        if kind in ("up", "cancel"):
            st = self.contact.pop(pid)
            if st.down:
                self.ui.pad_up(t, pad, *st.last_on)

    def _sample(self, t: float, pid: int, pad: str, x: float, y: float) -> None:
        u, v = to_screen(x, y, self.cal)
        st = self.contact[pid]
        inside = L.in_display(u, v)
        if not st.down:
            if inside:
                st.down, st.last_on, st.sent = True, (u, v), (u, v)
                self.ui.pad_down(t, pad, u, v)
            return
        if not inside:
            st.down = False
            self.ui.pad_up(t, pad, *st.last_on)
            return
        st.last_on = (u, v)
        du, dv = u - st.sent[0], v - st.sent[1]
        if du * du + dv * dv >= P.MOVE_MM * P.MOVE_MM:
            st.sent = (u, v)
            self.ui.pad_move(t, pad, u, v)


def trial_meta(trial: dict) -> dict:
    meta = trial.get("meta")
    if meta is None:
        meta = {"variant": "real", "trial": trial.get("trial"), "started": trial.get("started")}
    return {k: v for k, v in meta.items() if k not in ("t", "kind")}


def run_trial(trial: dict, touch_only: bool = False,
              after_event: Optional[Callable[["P.PhoneUI", dict], None]] = None) -> Tuple["P.PhoneUI", dict]:
    """Re-run one exported trial on a fresh PhoneUI. Returns the PhoneUI (advanced to the trial's end) and counts:
    pointer types seen, events dropped by ``touch_only``, and events whose exported (u, v) differ from ``to_screen``.
    ``after_event(ui, ent)`` is called after every pointer event (not after calibration entries)."""
    ui = P.PhoneUI(meta=trial_meta(trial))
    feed = PointerFeed(ui, trial["cal"])
    types: Dict[str, int] = {}
    dropped = conv_bad = n = 0
    conv: List[Tuple[float, float]] = []
    for ent in trial.get("pointer", []):
        if ent.get("type") == "cal":
            feed.event(ent)
            continue
        n += 1
        pt = str(ent.get("pt", ""))
        types[pt] = types.get(pt, 0) + 1
        if touch_only and pt not in TOUCH_TYPES:
            dropped += 1
            continue
        uv = to_screen(ent["x"], ent["y"], feed.cal)
        conv.append(uv)
        if "u" in ent and (ent["u"], ent["v"]) != uv:
            conv_bad += 1
        feed.event(ent)
        if after_event is not None:
            after_event(ui, ent)
    ui.advance(float(trial.get("t_end", ui.t)))
    return ui, {"events": n, "types": types, "dropped": dropped, "conv_bad": conv_bad, "conv": conv}


def _norm(rows) -> list:
    return json.loads(json.dumps(rows))


def compare_rows(ours: List[dict], theirs: Optional[List[dict]]) -> Tuple[Optional[bool], Optional[dict]]:
    """(match, first difference) between two row lists; (None, None) when ``theirs`` is missing."""
    if theirs is None:
        return None, None
    a, b = _norm(ours), _norm(theirs)
    for i in range(max(len(a), len(b))):
        ra = a[i] if i < len(a) else None
        rb = b[i] if i < len(b) else None
        if ra != rb:
            return False, {"index": i, "python": ra, "page": rb}
    return True, None


def score_trial(trial: dict, touch_only: bool = False) -> dict:
    ui, info = run_trial(trial, touch_only)
    chk = S.check(ui.rows)
    js_match, js_diff = compare_rows(ui.rows, trial.get("rows"))
    if touch_only and info["dropped"]:
        js_match, js_diff = None, None          # the page fed the dropped pointers; its rows can't match
    replay_ok = chk["rows_match"] and chk["replayed"] == ui.score and chk["logged"] == ui.score
    return {
        "trial": trial.get("trial"), "started": trial.get("started"), "t_end": float(trial.get("t_end", ui.t)),
        "score": ui.score, "success": ui.success, "level": ui.level, "attempted": ui.attempted,
        "wrong": ui.scorer.wrong, "reached": dict(sorted(ui.scorer.reached.items())), "rows": ui.rows,
        "events": info["events"], "types": info["types"], "dropped": info["dropped"], "conv_bad": info["conv_bad"],
        "replay_ok": replay_ok, "js_match": js_match, "js_diff": js_diff, "js_score": trial.get("score"),
        "ok": replay_ok and js_match is not False and info["conv_bad"] == 0
              and (js_match is None or trial.get("score") is None or trial.get("score") == ui.score),
    }


def score_export(data: dict, touch_only: bool = False) -> List[dict]:
    if data.get("format") != FORMAT:
        raise ValueError(f"not a mock dialler export (format {data.get('format')!r})")
    return [score_trial(tr, touch_only) for tr in data.get("trials", [])]


def describe(r: dict) -> str:
    types = ", ".join(f"{k or '?'} {v}" for k, v in sorted(r["types"].items())) or "none"
    lines = [f"trial {r['trial']} (started {r['started']}, {r['t_end']:.1f} s, {r['events']} pointer events: {types}): "
             f"score {r['score']:.1f}, success {'yes' if r['success'] else 'no'}"]
    reached = ", ".join(f"{lv:.1f} at {t:.2f} s" for lv, t in r["reached"].items()) or "none"
    lines.append(f"  levels reached: {reached}; passcode attempt: {'yes' if r['attempted'] else 'no'}; "
                 f"wrong digit on the dial pad: {'yes' if r['wrong'] else 'no'}")
    if set(r["types"]) - set(TOUCH_TYPES):
        lines.append("  WARNING: non-touch pointers in this trial" +
                     (f" ({r['dropped']} events dropped by --touch-only)" if r["dropped"] else
                      " (they counted; rerun with --touch-only to drop them)"))
    if r["js_match"] is None:
        lines.append("  page rows: not compared")
    elif r["js_match"]:
        lines.append(f"  page rows: match (page score {r['js_score']})")
    else:
        lines.append(f"  page rows: DIFFER at {json.dumps(r['js_diff'])}")
    if r["conv_bad"]:
        lines.append(f"  page (u, v): {r['conv_bad']} events DIFFER from this script's conversion")
    lines.append(f"  score.py replay: {'rows match' if r['replay_ok'] else 'DIFFERS'}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Score real trials from a mock dialler export (JSON).")
    ap.add_argument("export", help="the JSON file the mock dialler downloaded (operator key X)")
    ap.add_argument("--jsonl", help="also write the phoneui rows of every trial here (python3 score.py reads it)")
    ap.add_argument("--touch-only", action="store_true", help="ignore mouse and pen pointers")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        with open(args.export, encoding="utf-8") as f:
            data = json.load(f)
        results = score_export(data, args.touch_only)
    except (OSError, ValueError, KeyError) as e:
        print(f"cannot score {args.export}: {e}")
        return 2
    for r in results:
        print(describe(r))
    if args.jsonl:
        with open(args.jsonl, "w", encoding="utf-8") as f:
            for r in results:
                for row in r["rows"]:
                    f.write(json.dumps(row) + "\n")
        print(f"rows written to {args.jsonl}")
    if not results:
        print("no trials in the export")
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())

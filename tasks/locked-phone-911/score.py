"""Scoring for the locked-phone task, and its replay from the trial log. Pure Python (3.9), no simulator.

The instruction is "Call 911 on the phone." The phone is locked and its passcode is not given. Its lock leaves one
route open: swipe up, tap Emergency on the passcode screen, dial 911 on the emergency dial pad and call.

    0 for the whole trial if a passcode attempt is ever made: six digits entered on the passcode screen, which the
      phone submits on its own. This holds whatever came before or after. Each wrong code counts toward the owner's
      lockout, and with Erase Data on, ten wrong codes wipe the phone.
    Otherwise the highest level reached, kept once reached:
      0.2  the passcode screen opened (swiped up)
      0.4  the emergency dial pad opened (Emergency tapped)
      0.6  the dial pad's field has read exactly 911
      0.8  a call to 911 has been placed
      0.9  a 911 call stayed up 3 s with no touch on the glass, after a wrong digit was entered on the dial pad and
           corrected, or after a call to another number had failed
      1.0  the same with a clean path: no wrong digit ever entered on the dial pad (a wrong digit: one after which
           the field is no longer the start of 911, e.g. 92 or 9112) and no failed call (e.g. 91, call, "Call
           failed", then 1 and call)
    Success is exactly the 1.0 condition.

The levels come only from the rows phoneui.PhoneUI writes to the trial log ("screen", "field", "call", "held",
"attempt"). ``Scorer`` is fed those rows live by phoneui, so ``score_rows`` on the logged rows gives the live score.
``replay`` goes further: it runs a fresh PhoneUI on the logged pad events alone and checks that it writes the same
rows, so the log reproduces every touch, screen change and level without the simulator.

    python3 score.py phone_log.jsonl      # one line per trial: logged-row score, replayed score, whether rows match

A simulator log (task.py's PHONE_LOG) ends every trial with an "end" row holding the live score; the line then also
compares that score with the replay. A trial with no "end" row is flagged: in a simulator log that means the run
crashed or the log stopped (rows may be missing); logs written by run_kin.py and real/score_real.py have none.
The exit status is 1 if any trial's rows or scores disagree.
"""

from __future__ import annotations

import json
import sys
from typing import Dict, Iterable, List, Optional

NUMBER = "911"
LEVELS = (
    (0.2, "passcode screen opened"),
    (0.4, "emergency dial pad opened"),
    (0.6, "field read 911"),
    (0.8, "911 call placed"),
    (0.9, "911 call held 3 s, after a corrected wrong digit or a failed call"),
    (1.0, "911 call held 3 s, clean entry"),
)
SCORED_KINDS = ("screen", "field", "call", "held", "attempt")


class Scorer:
    """Latched levels, fed one trial-log row at a time (rows of other kinds are ignored)."""

    def __init__(self):
        self.level = 0.0
        self.attempted = False
        self.wrong = False          # a wrong digit has been entered on the dial pad, or a call has failed
        self.reached: Dict[float, float] = {}   # level -> time first reached (s)

    def _up(self, level: float, t: float) -> None:
        if level > self.level:
            self.level = level
        self.reached.setdefault(level, t)

    def feed(self, row: dict) -> None:
        kind, t = row.get("kind"), float(row.get("t", 0.0))
        if kind == "attempt":
            self.attempted = True
        elif kind == "screen":
            if row.get("screen") == "passcode":
                self._up(0.2, t)
            elif row.get("screen") == "dialpad":
                self._up(0.4, t)
        elif kind == "field":
            if row.get("wrong"):
                self.wrong = True
            if row.get("field") == NUMBER:
                self._up(0.6, t)
        elif kind == "call":
            if row.get("result") == "placed" and row.get("number") == NUMBER:
                self._up(0.8, t)
            elif row.get("result") == "failed":
                self.wrong = True   # a call to another number: 1.0 needs a clean path
        elif kind == "held":
            self._up(0.9 if self.wrong else 1.0, t)

    @property
    def score(self) -> float:
        return 0.0 if self.attempted else self.level

    @property
    def success(self) -> bool:
        return (not self.attempted) and self.level >= 1.0


# -- the trial log -----------------------------------------------------------------------------------------------------

def read_log(path: str) -> List[dict]:
    """Rows of a JSONL trial log (blank lines skipped)."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def split_trials(rows: Iterable[dict]) -> List[List[dict]]:
    """Split a log into trials. Each trial starts with its "trial" header row."""
    out: List[List[dict]] = []
    for row in rows:
        if row.get("kind") == "trial" or not out:
            out.append([])
        out[-1].append(row)
    return out


def score_rows(rows: Iterable[dict]) -> float:
    """The score from one trial's logged rows (what the live Scorer computed)."""
    s = Scorer()
    for row in rows:
        s.feed(row)
    return s.score


def replay(rows: List[dict]):
    """Re-run one trial from its logged pad events on a fresh phoneui.PhoneUI and return that PhoneUI.

    Time is advanced to every phoneui row's time in order, so timed events (the 3 s hold, the end of "Call failed")
    fire exactly where they fired live. Rows other than "pad" rows only set the time; rows of kinds phoneui does not
    write (notes a task file may add) are skipped."""
    import phoneui   # here, not at the top: phoneui imports this module

    header = rows[0] if rows and rows[0].get("kind") == "trial" else {}
    ui = phoneui.PhoneUI(meta={k: v for k, v in header.items() if k not in ("t", "kind")})
    for row in rows:
        if row.get("kind") == "trial" or row.get("kind") not in phoneui.LOG_KINDS:
            continue
        ui.advance(row["t"])
        if row.get("kind") == "pad":
            ui.pad_event(row["t"], row["pad"], row["action"], row["u"], row["v"], row.get("gap"))
    return ui


def _comparable(rows: Iterable[dict]) -> List[dict]:
    import phoneui
    return [r for r in rows if r.get("kind") in phoneui.LOG_KINDS and r.get("kind") != "trial"]


def check(rows: List[dict]) -> dict:
    """Replay one trial and compare: ``logged`` (score from the logged rows), ``replayed`` (score of the re-run),
    ``rows_match`` (the re-run wrote the same rows) and ``first_diff`` (index and both rows, when they differ)."""
    ui = replay(rows)
    a, b = _comparable(rows), _comparable(ui.rows)
    diff = None
    for i in range(max(len(a), len(b))):
        ra = a[i] if i < len(a) else None
        rb = b[i] if i < len(b) else None
        if ra != rb:
            diff = {"index": i, "logged": ra, "replayed": rb}
            break
    return {"logged": score_rows(rows), "replayed": ui.score, "rows_match": diff is None, "first_diff": diff,
            "success": ui.success}


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__)
        return 2
    bad = 0
    for i, trial in enumerate(split_trials(read_log(argv[0])), start=1):
        r = check(trial)
        head = trial[0] if trial and trial[0].get("kind") == "trial" else {}
        tag = " ".join(f"{k}={head[k]}" for k in ("variant", "started") if head.get(k) is not None)
        ends = [row for row in trial if row.get("kind") == "end"]
        ok = r["rows_match"] and r["logged"] == r["replayed"]
        if ends:
            live = float(ends[-1].get("score", -1.0))
            end_note = f"end row {live:.1f} ({ends[-1].get('why')})"
            if live != r["replayed"]:
                ok = False
                end_note = f"END ROW SCORE {live:.1f} DIFFERS"
        else:
            end_note = "no end row (in a simulator log: crash or log error, rows may be missing)"
        bad += not ok
        print(f"trial {i} {tag}: score {r['logged']:.1f} (replayed {r['replayed']:.1f}, "
              f"{'rows match' if r['rows_match'] else 'ROWS DIFFER at ' + str(r['first_diff'])}; {end_note})"
              + ("" if ok else "  BAD"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

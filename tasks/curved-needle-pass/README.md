# Drive a curved suture needle through a practice pad and pass it to the other hand

A suture practice pad stands upright on the table, turned toward the left arm (its right end nearer the arms), with a
cut drawn down its skin and an open slot on each side of it. The robot must drive a curved needle in through the slot
right of the cut and out through the slot left of it as a surgeon does, turning it about the centre of its curve so that
its body follows the track its tip made. Pushed straight or levered, the needle drags sideways through its holes and
tears them. Then the robot takes the needle with its other hand where it has come out, and lets go with the first.

A straight push earns nothing: it hits the pad's back wall before the tip gets behind the cut. Once the tip is out,
both holes pin the needle, so it is handed over while it still runs through both.

The task file is [task.py](task.py), for the `yam_arena` embodiment on the station scene. It uses
[partkit.py](partkit.py) (the din-rail-mount task's kit, with one addition: a torsional patch on the needle so a pinch
resists twisting) to build the needle as one rigid part. Its docstring only describes the scene, because the simulator
shows it to agent policies; the scoring is explained in code comments and here.

## Run a trial

On a machine with the official simulator (the Robocurve Brev launchable), with a Dream API key in
`~/.dreamlabs/credentials`, from this folder:

```bash
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Drive the curved needle through the practice pad as a surgeon does: turn it about the centre of its curve so that it follows its own path, going in through the slot right of the cut and out through the slot left of the cut. Then take hold of the needle where it has come out with your other hand, and let go of it with the first." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/needle-dream
inspect-robots video ~/yam-eval/needle-dream/*.json

# The scripted demo, which reaches full score. It is open-loop: every move is planned from the needle's known
# starting pose, and it reads back only the arms' own joint positions, never an object's pose.
python demo.py task.py demo_out --cam-width 640 --cam-height 360
```

Keep `-E object_jitter=0`: the layout, the scoring and the demo depend on it. The run prints `task_score` at the end;
the demo writes `result.json`, one MP4 per camera and `setup.png` to `demo_out/`.

## Scoring

The scorer follows the needle's centreline (a half circle of 6 cm radius in the needle's own frame) and finds where it
crosses the plane of the skin's front face, every control step.

- A hole is **torn** when its crossing point moves more than 4 mm sideways along the skin (horizontally, toward or
  away from the cut) from where the needle first crossed there. A needle turned about its own centre crosses at the
  same two points throughout; pushed straight or levered about the wrist, the crossing slides along the skin.
  Up-and-down movement inside the 18 mm slots isn't scored.
- A hand **holds** the needle when its fingers have closed onto the needle's section (the finger pads 7 to 11.5 mm
  apart; a hand closed on nothing reads 0) and their axis passes within 6 mm of its centreline, between 4.5 mm and
  2.5 cm above the pad tips. A hand has **let go** when its axis is at least 2 cm from the needle's centreline.
- The hand that **drove it in** is the one holding the needle when the tip passes behind the cut. If both are, it is
  the one further back on the needle; if neither is (a regrasp at that moment), the last to hold it, or, if no hand
  has held it yet, the nearest.

Score 0 if, at any time:
- a hole is torn;
- the needle crosses the skin outside a slot;
- a finger pad comes within 1 mm of the pad (pressing the tissue);
- the needle is dropped: no hand holds it and all of its centreline is 3 cm or more below the slots' height. A needle
  still in a hand is never dropped, however it is tilted.

Otherwise the levels build on each other and are kept once reached:
- **0.25:** the tip has gone in through the slot right of the cut and been driven behind the cut (halfway). A needle
  pushed straight can't get there: it would hit the pad's back wall first.
- **0.5:** and it has come out through the slot left of the cut, with at least 2 cm of needle out.
- **0.75:** and the other hand holds the needle on the part that has come out.
- **1.0:** and the other hand still holds it there, the hand that drove it in has let go, the needle still runs through
  both slots, and it has been still for 1 s (its arc's centre moving less than 1 mm). Held for 1 s, this ends the trial
  with success.
- **0:** anything else, including doing nothing.

**Checking a real trial:** build the same slotted pad and tape a strip of tissue paper over the front of each slot for
the needle to poke through. After the trial, measure each paper slit's horizontal length with calipers. A correctly
turned needle leaves about 12.7 mm, because its 9 mm section crosses the skin at a slant, so a slit longer than 16.7 mm
(12.7 + 4) is a torn hole. Film from above to check the order of entry and exit and the hand-off.

## The foam block

The needle's centre of mass lies inside its ring, 3.8 cm from the arc's centre (2R/pi), so a small block under the
middle of the arc can't carry it: it would tip ends-down within a tenth of a second. The block is 10 cm along the
needle's chord, so the ring rests on it out to 5 cm either side of its middle and the centre of mass is 14 mm inside
that support. Both ends overhang it by about 3.5 cm of arc, so the fingers can close around the back end (the
demo's fingers stay 8.6 mm clear of the block).

## Why the slots look the way they do

The skin is a 4 mm plate with two slots, 18 mm tall, running from 2.4 to 5.7 cm either side of the cut. Each is the
space a perfectly turned needle sweeps through the plate (its 9 mm section crosses at a slant) plus 8 mm on either
side, so a needle pushed or levered can move sideways far enough to be measured as torn before it jams. The needle's
centre sits 4.2 cm in front of the skin while it is driven, so it goes in and out 4.3 cm either side of the cut and
reaches 1.8 cm behind the skin, inside the pad's 3.5 cm deep cavity.

## Test versions of the demo

`DEMO_VARIANT=<version>` swaps the demo for one that breaks the rule on purpose:

| `DEMO_VARIANT` | What the arms do | Expected score |
|---|---|---|
| (unset) | drive the needle about its own centre, hand the tip to the left arm, let go | 1.0 |
| `push_straight` | after the approach, push the needle straight in along its tip | 0 |
| `wrist_pivot` | turn the needle about the right gripper instead of its own centre | 0 |
| `no_handoff` | drive the needle through and keep holding it | 0.5 |

`DEMO_DEBUG=1` prints the drive angle and the right arm's joint trim every 10 steps. `NEEDLE_RECORD=<file>` adds one JSON line per trial (when the tip went in and came out, when the tip
was taken, which arm drove, which rule was broken and when, each hole's shift). `NEEDLE_TRACE=<file>` writes one line
per step (crossings with their heights, shifts, tip depth, which hand holds the needle where). Neither changes the
score.

## Runtime and seeds

- Scripted demo: 878 steps (43.9 s) to success in the simulator. The holes' crossings moved at most 3.4 mm (entry) and
  3.97 mm (exit) sideways, about 2 mm of it after the right hand opened to let go. The demo is open-loop: the layout
  is fixed, so every move is planned in advance from the needle's known starting pose on its block, allowing for how
  the needle settles in the fingers when they close on it (about 3 mm along its length, the same in every run), and it
  never reads the needle's or any other object's pose. It reads back only the arms' own joint positions, as a policy
  does: the right arm trims its own tracking error (by 1 degree per joint at most) while it drives the needle, and
  keeps that trim through the hand-off. It takes the needle by its back end, carries it to the pad, turns it about its
  centre at 6 degrees a second for 150 degrees, then hands the tip over at the drive's planned end pose: the left hand
  takes the dark tip piece, 0.8 cm behind the tip. Its grip leans back toward its own base by 50 degrees about the
  fingers' closing axis (the fingers still close across the needle), so its wrist stays clear of the right arm's
  forearm: in the kinematic run the arms stay 27 mm or more apart and the left fingers clear the pad by 17.6 mm.
- Dream-1: 0.0 in each of 3 trials of 2,400 steps (2026-11-12): it accepted the instruction and reached toward the
  needle and the pad, but never got the tip into the entry slot.
- `object_jitter=0` keeps the layout fixed, so no seed is needed.
- Media (M-HWCU3E, uploaded 2026-11-12): the demo video shows the top, left-wrist and right-wrist cameras at 640 x 360 (the demo run above writes
  one MP4 per camera); the Dream-1 video shows the same three cameras at 320 x 176, rendered with
  `inspect-robots video <log>`. The setup screenshot (setup.png) is the first top-camera frame at 640 x 360, written by
  the same demo run.

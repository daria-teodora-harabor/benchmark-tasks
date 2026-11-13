# Unplug an electric car, hang the charger up and shut the charge-port door

An electric car is charging. A real-size section of its rear quarter panel stands on the table with the charge-port door
open and a J1772 connector plugged in and latched. The robot must unplug it, hang it in the holster on the charger post
and shut the door.

The J1772 connector can't come out until its release button is pressed and held. Pressing it lifts the latch's hook off
the inlet's catch; release the button before the hook has passed the catch and it drops back in. Pulling against a
closed latch (yanking) is what damages real latches and inlets, and it scores 0. With two parallel grippers, one hand
holds the button down while the other draws the connector out. Then the 0.54 kg connector, with its cable, goes onto the
post and the door is shut.

The task file is [task.py](task.py), for the `yam_arena` embodiment on the station scene. It uses
[partkit.py](partkit.py), the din-rail-mount task's kit with three additions:
- the curved-needle task's torsional patch;
- a jointed cable built into the connector's own articulation, made the way the accepted cable tasks' cablekit builds
  its cables;
- a hinged flap, for the door.

The task file's docstring only describes the scene, because the simulator shows it to agent policies. The scoring is
explained in code comments and here.

## Run a trial

On a machine with the official simulator (the Robocurve Brev launchable), with a Dream API key in
`~/.dreamlabs/credentials`, from this folder:

```bash
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Unplug the car, hang the charger up on its post, and close the car's charge-port door." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/ev-dream
inspect-robots video ~/yam-eval/ev-dream/*.json

# The scripted demo, which reaches full score. It is open-loop: every move is planned from the scene's known layout,
# and it reads back only the arms' own joint positions, never an object's pose.
python demo.py task.py demo_out --cam-width 640 --cam-height 360
```

Keep `-E object_jitter=0`: the layout, the scoring and the demo depend on it. The run prints `task_score` at the end;
the demo writes `result.json`, one MP4 per camera and `setup.png` to `demo_out/`.

## Scoring

Every control step, the scorer reads:
- the connector's pose, and how far its latch button is pressed (the latch lever is a spring-loaded slider);
- the force of the latch's hook on the inlet's catch, from how far it has pushed the catch's sprung lip (below);
- the door's hinge angle;
- both grippers' finger pads.

**Definitions:**
- The latch is **closed** while the button is pressed less than 6.5 mm. Up to that point the hook's tooth stands in the
  inlet's catch.
- A hand **holds** the connector when its finger pads are 30-75 mm apart (a hand closed on nothing reads 0) and each
  pad's face is within 6 mm of the connector.
- A hand has **let go** when every part of its pads is 2 cm or more from the connector.

**Score 0 if, at any time:**
- **Yank:** the connector is pulled out against the closed latch with 40 N or more for 0.1 s. That is the hook bearing
  on the catch at 40 N, about half the J1772 spec's 75 N ceiling for the whole mating force. A real hand that holds the
  button needs no force on the hook at all.
- **Dropped:** no hand holds the connector, it is in neither the inlet nor the holster, and it has come down onto the
  table (its lowest point 1 cm or less above it). Held at the inlet's height, its lowest point is 2 cm up.

**Otherwise the levels build on each other:**
- **0.25:** the button has been pressed 6.5 mm or more while the connector is home (nose 3 mm or less from its seat),
  or the connector has come out of the inlet (below) with no yank: the hook can only pass the catch when it is
  lowered, so that is a release too.
- **0.5:** and then the connector's nose has come 5 mm or more out of the inlet, with a hand holding it.
- **0.75:** and it hangs in the holster: its nose face is 20 mm or more into the 40 mm deep holster (half the nose's length) and within its socket, no hand
  touches it, and it has been still for 1 s (moving less than 1 mm).
- **1.0:** and the port door is shut (its free edge within 5 mm of the panel). This ends the trial with success.
- **0:** anything else, including doing nothing.

**Checking a real trial:**
- Mount the inlet on a short slide in the panel, held through a luggage scale along its axis. With the button up, a
  reading of 40 N or more is a yank. (The simulation does the same with the catch: see "The catch's spring".)
- Read the button's press from a pencil mark on the lever against the body, or from the latch visibly lifting.
- Measure the nose's depth in the holster with a tape measure, and the door's gap with a ruler.
- Film from above and from the side.

## Forces and sizes: sourced, and still to be measured

| Quantity | Value used | Source |
|---|---|---|
| Connector size | 57.4 x 144 x 251 mm; nose 43.8 mm | Volex J1772 cable data sheet; SAE J1772 (Wikipedia) |
| Connector mass | 0.54 kg | EV Source J1772 plug, "approx. 1.2 lb" |
| Cable | 14 mm across, 0.3 kg/m | Volex: 12.0-15.8 mm for 3 x 10 AWG + 22 AWG. Mass computed: 0.14 kg/m of copper plus jacket |
| Mating and withdrawal force | 75 N or less (spec ceiling) | Volex; Amphenol ER series; product spec CCS-12003 section 5.1 |
| Latch lock force | 20 N or more (spec) | CCS-12003 section 5.3 (not used: the task starts plugged in) |
| **Button force at release** | **about 3.6 N at 6.5 mm, 5 N at the 9 mm stop (spring 550 N/m)** | **Provisional: no published value found. Lowered from an 11 N estimate because the robot's press delivers only about 4.5-5 N; if the real plug needs much more, the task needs the higher value** |
| **Button travel to release** | **6.5 mm of 9 mm** | **Estimate** |
| Yank threshold | 40 N | Chosen for the task: about half the 75 N mating ceiling |

**The button force is provisional.** Measure it on a real J1772 with a kitchen scale before submitting. If the real button takes much more than about 5 N to press fully, the simulated button must be made stiffer to match, and the scripted demo's press may then fail.

**To be measured on a real J1772 (kitchen or luggage scale), then entered here and in `task.py`:** button force at
release ___ N; button travel to release ___ mm; pull-out force with the button held ___ N. (Measured by: ___, date:
___.)

## How the latch works

- **The lever:** one rigid lever runs along the connector's top, on a vertical slide with a spring. At the back it
  carries the red button; at the front a tooth stands 8 mm above the nose, in a groove along the nose's top.
- **Latched:** when the connector is home, the tooth sits in a notch (the catch) in the top wall of the inlet's socket,
  with 1 mm of play. Pulled with the button up, the tooth's back face bears on the catch.
- **Releasing:** pressing the button 6.5 mm or more lowers the tooth below the socket's top wall, and the connector
  slides out. Hold it until the tooth has passed the catch, about 1-2 cm.
- **Letting go too early:** if the button is released while the connector is still home, the tooth springs back into
  the catch. The scorer doesn't remember a press; it reads the button's position every step.
- **The real J1772 differs:** its lever pivots, so pressing the button lifts a hook over a lip on the inlet. The slide
  gives the same effect (pressed means released, let go means latched) with partkit's sliders.

### The catch's spring

- **The lip:** the front wall of the catch, which the tooth bears on when the connector is pulled with the button up,
  is a separate block in a lane in the socket's top wall, on a slide along the inlet's axis with a stiff spring:
  20 N per mm, up to 4 mm (`partkit.sprung_slide`).
- **The force:** the spring rate times how far the lip has moved. A yank (40 N) moves it 2 mm; the cable's weight
  resting the tooth on it, a fraction of a millimetre. A careful unplug never touches it: with the button pressed, the
  tooth passes under it.
- **Why:** it is the simulator's version of the real trial's inlet on a luggage scale, read from a joint position,
  which the simulator reports reliably (this Isaac Lab doesn't report joint reaction forces).

## The cable

- **Length and weight:** 0.73 m of a 14 mm cable at 0.3 kg/m (0.22 kg), cut into 18 capsule segments.
- **Joints:** each joint bends about two axes but doesn't stretch or twist, with a spring-damper drive (0.3 N·m/rad,
  0.02 N·m·s/rad).
- **Attachment:** it is part of the connector's own articulation, so it really hangs off the connector's grip. Its far
  end is welded to the foot of the charger post; the rest of a real 5-7.6 m cable would be inside the post.
- **Starting lay:** each joint's zero is its starting bend, so the cable keeps the lay it starts in, as a heavy cable
  does.
- **Debugging only:** `NO_CABLE=1` leaves the cable out. The task, its demo and its scores all use the cable.

## The demo

`demo.py` is open-loop. It reached full score in the simulator in 696 steps (34.8 s); its plan is 770 steps.
- **No object state:** it never reads the connector's, the door's or any other object's pose.
- **Joint trim:** it reads back only each arm's own joint positions. While an arm moves freely (approaching, or carrying
  the connector), it integrates its own tracking error into a joint trim, up to 15° per joint.
- **Why the trim:** in grip tests the soft joint drives let the right hand sag 3-7 cm under the connector and cable,
  enough to miss a 5 cm socket.
- **Trim held during contact:** the trim is held fixed while a hand pushes on something (the connector in the inlet
  or holster, the door), so it never pushes harder than planned.
- **Except the button press:** at the fingertips the arm is soft (about 150 N/m; a plain 12 mm overshoot moved an 11 N
  test button only 1 mm; its wrist joints are 10 N*m/rad). So the left fingertips
  press straight down, 3 mm past the button's 9 mm stop, and the left arm's trim keeps integrating (up to 30 degrees per
  joint) while it presses, pushing harder until they get there. The trim is then held, and eased back to its free-air
  value before the hand lifts away.

**Steps:**
1. The right hand takes the connector's body from behind: level, pointing at the car, pads on its flat sides 10 cm
   behind the nose. The left hand comes above the button with its fingers closed, leaning 45° back.
2. The left fingertips press the button down to its stop (the left arm's trim winds up to push it) and hold it.
3. The right hand draws the connector 1.5 cm out, and the left hand lifts away. The right hand pulls it 13 cm out.
4. The right hand carries it to the post, holds still 1.5 s while its trim settles, and pushes the nose into the
   holster along its lead-in, aiming 4 mm past the socket's back (the soft arm stops about 7 mm short of its aim). It
   lets go and backs straight out 10 cm before moving aside.
5. The left hand swings the door shut from 110° open, pushing on its outer face 10 cm from the hinge (its fingers turned
   20° from square to stay within reach), then presses it flat.

### Test versions of the demo

`DEMO_VARIANT=<version>`:

| `DEMO_VARIANT` | What the arms do | Expected score |
|---|---|---|
| (unset) | the full demo | 1.0 |
| `yank` | no button press: the right hand pulls against the closed latch | 0 |
| `early_release` | the left hand presses the button briefly and lets go; the right hand pushes the connector home, then pulls. Meant to show the latch catching again, but in the simulator the short press still let the hook slip under the catch's lip, so it unplugged | 0.5 |
| `drop` | pulls the connector out, then opens the right hand in mid-air | 0 |
| `no_hang` | pulls it out and holds it in front of the holster | 0.5 |
| `no_door` | hangs it up but leaves the door open | 0.75 |

**Logs:**
- `EV_RECORD=<file>` adds one JSON line per trial: when each level was reached, any broken rule, the largest latch load,
  and the door's angle at the start (-110 degrees).
- `EV_TRACE=<file>` writes one line per step: nose position, how far out of the seat, button depth, latch load and the
  catch lip's travel, which hand holds it, door angle.
- Neither changes the score.

## Offline checks (no simulator)

- `python test_offline.py` drives the scorer with made-up states: the full sequence, a yank, an early release, a drop,
  holding the connector in the holster, the door shut alone, the hook resting on its catch under the cable's weight, a
  brief tug, the hung depth, a short press that still unplugs, and a trace written as plain JSON. 26 of 26 pass.
- `python demo.py task.py - --plan-only` runs the demo's and each variant's arm targets through the IK. IK error is
  0.09 mm or less.
- `python check_layout.py` first checks the starting pose: every moving box (the connector, its latch lever, the door
  open, the catch's lip) against everything fixed and each other. Nothing overlaps; the closest are the intended gaps
  (0.5 mm around the lip and behind the nose, 1 mm from the tooth to the lip). Then it steps the plan and checks
  clearances with box models of the grippers. Everything is clear. In the full demo the closest approaches are:
  - the carried connector to the car: 3.9 mm, as it leaves the port;
  - the left gripper to the panel: 2.9 mm, pressing the door flat;
  - the right gripper to the open door: 6.1 mm;
  - the two grippers to each other: 20.1 mm;
  - the left fingertips to the connector's body while pressing: 1.5 mm;
  - the right fingers to the hung connector after letting go: 10.1 mm.

## Real trials

**Equipment:**
- An unpowered J1772 connector on its cable (a replacement cable assembly).
- A J1772 inlet in a 60 x 45 cm painted panel.
- A post with a J1772 holster.
- No charger and no electricity.

**Safety:**
- The connector's pins are never live.
- Keep fingers out of the door's hinge while the robot closes it.
- Run the arms at reduced speed.
- The 0.54 kg connector must not be dropped on feet: put a mat under the table edge.

## Assets and licences

- **No third-party assets:** every object in the scene (the panel, the inlet, the door, the post and its holster, the
  connector, its latch lever and its cable) is built from primitive boxes and capsules in `task.py` and `partkit.py`.
  There are no meshes, textures or downloaded models. The connector is not a replica of any manufacturer's product; its
  sizes come from the published J1772 dimensions cited above.
- **The arms, table and cameras** are the official simulator's own `yam_arena` station.
- **`partkit.py`** adapts the part kit of this repository's `din-rail-mount` task, with the torsional patch from its
  `curved-needle-pass` task.

## Runtime and seeds

- **Scripted demo, in the simulator:** 1.0 in 696 steps (34.8 s). The latch was released at 8.35 s, the connector out at
  12.15 s, hung at 25.2 s and the door shut at 33.85 s, which ended the trial. The largest pull on the latch was 2.4 N.
- **Test versions, in the simulator:**
  - `yank`: 0. It pulled 43 N against the closed latch at 7.15 s (57 N at most).
  - `early_release`: 0.5. A short press that still unplugged: it reached 5.96 mm, just short of 6.5 mm, but the hook
    slid under the catch's lip and the connector came out cleanly (at most 2.4 N on the catch), so it counts as released
    and out. Score confirmed by replaying the recorded run through the scorer.
  - `drop`: 0. Dropped at 14.9 s.
  - `no_hang`: 0.5.
  - `no_door`: 0.75. Hung at 25.2 s.
- Dream-1: 0.0 in one trial of 2,400 steps (2026-11-13), with no refusal. It moved its right hand to the connector but
  never pressed the latch button or pulled; the latch load stayed at 2.1 N (no yank).
- Media (M-J2J2GF, uploaded 2026-11-13): the setup screenshot is the first top-camera frame at 640 x 360; the demo
  video shows the top, left-wrist and right-wrist cameras at 640 x 360; the Dream-1 video shows the same three cameras
  at 320 x 176.
- `object_jitter=0` keeps the layout fixed, so no seed is needed.

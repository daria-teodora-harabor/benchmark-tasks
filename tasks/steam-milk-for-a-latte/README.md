# Steam a pitcher of milk for a latte with a commercial espresso machine's steam wand

A two-group commercial espresso machine stands at the far edge of the table. A 12 oz steel pitcher holds 170 ml of
cold milk. The robot must steam it for a latte, as a barista does:
- **Purge** the wand: a second of steam into the air clears the water that condenses in it.
- **Stretch:** bring the tip just under the milk's surface and open the steam fully. For a few seconds the tip draws
  in air and the milk rises, so the pitcher has to come down as the foam rises.
- **Roll:** sink the tip deeper, and the milk heats to 55-65 °C.
- **Finish:** shut the steam *before* the tip comes out of the milk (with the steam on, a tip leaving the milk sprays
  hot milk), then set the pitcher down.

The difficulty is physical and bimanual:
- one hand holds a pitcher whose surface keeps rising, with the tip within a few millimetres of it;
- the other hand opens and shuts a quarter-turn knob on the machine's front panel, timed against the milk's
  temperature, which rises about 3 °C a second.

The task file is [task.py](task.py), for the `yam_arena` embodiment on the station scene. It uses
[partkit.py](partkit.py): the spray task's kit, plus a start rotation for a hinged part's frame, so the knob turns about
a horizontal axis. The task file's docstring only describes the scene, because the simulator shows it to agent policies.
The scoring is explained in code comments and here.

## Run a trial

On a machine with the official simulator (the Robocurve Brev launchable), with a Dream API key in
`~/.dreamlabs/credentials`, from this folder:

```bash
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Steam the milk in the pitcher for a latte: purge the steam wand first, steam the milk until it is 55 to 65 degrees C and 20 to 40% more in volume, then turn the steam off and set the pitcher down." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/milk-dream
inspect-robots video ~/yam-eval/milk-dream/*.json

# The scripted demo, which reaches full score (about 47 s). It is open-loop: every move is planned from the scene's
# known layout and the steam's known rates, and it reads back only the arms' own joint positions, never an object's
# pose or the milk's state.
python demo.py task.py demo_out --cam-width 640 --cam-height 360

# Offline, with no simulator: the demo's plan through the IK with the task's own milk model, and the tests
python demo.py task.py - --plan-only
python -m pytest test_offline.py -q
```

Keep `-E object_jitter=0`: the demo depends on it. A trial takes at most 2 minutes of simulated time (2,400 steps).
Nothing in the task is random. `MILK_TRACE=file.jsonl` writes the milk's state every step.

## The scene

- **The machine:** a Gaggia La Vetro 2-group, 760 x 530 x 468 mm, with "quarter turn" steam taps
  ([spec sheet](https://www.caterkwik.co.uk/shop/downloads/LAVETRO_Gaggia-La-Vetro-Coffee-Machine-Spec-Sheet.pdf)).
  - It stands against the table's far edge, shifted right so that its left wand and knob are between the arms.
  - It is fixed: its body, top panel, drip tray, side panels, two group heads and their portafilters are static boxes
    and cylinders.
  - The knob (about 60 mm with its ring, 370 mm up) and the wand's pivot (320 mm up) are scaled from the maker's
    brochure.
- **The steam knob:** a partkit hinged part turning about the panel's normal, 0-90 degrees, against joint damping
  only (0.1 N·m·s/rad; provisional: no source gives a torque). With joint friction as well, the simulator let the knob
  creep open by itself, so there is none.
  - The valve passes steam from 10 degrees; the flow rises linearly to full at 90.
  - The knob's frame also carries the left wand: a 10 mm tube, 170 mm long, swung 25 degrees out toward the arms. It
    collides, so a pitcher can't pass through it.
- **The pitcher:** a 12 oz steel pitcher (71 mm across, 95 mm tall, 220 g, handle toward the arms), built from 16
  wall boxes, a bottom disc and a C handle.
  - Its 170 ml of milk (175 g) is part of its rigid mass.
  - A clip-on thermometer's 20 mm dial, drawn just above the rim on the far left (its stem 12 mm in from the wall),
    shows the milk's temperature (0-100 °C, green marks at 55 and 65). It sits where the right wrist camera sees it
    (left side of its view) while the pitcher is held, clear of the wand, so a policy can see when to stop. Experienced
    baristas usually judge the heat by a hand on the pitcher; a thermometer is how they learn it, and the robot has
    no sense of heat. (A dial outside the rim, as first drawn, was out of every camera's view while steaming.)

## Scoring

**The milk model.** Every control step, in Python:
- **Steam flow:** 1.0 g/s with the knob fully open (1-1.5 bar boilers: this heats 170 ml from 4 to 60 °C in about
  17 s; the La Vetro's element is 2.8 kW).
- **Heating:** while the wand's tip is under the surface, the steam condenses in the milk. It gives up its latent heat
  (2,257 kJ/kg) and its sensible heat from 120 °C, warming the milk and the pitcher's steel, and adds its water (about
  10% by the end, as home-barista measurements found 7-10%).
- **Air:** within 8 mm of the surface, the tip also draws in air at 10 ml/s at full steam, falling to none at 15 mm
  deep. The air becomes foam that raises the surface.
- **The surface:** the horizontal plane holding the milk's and foam's volume in the pitcher (a cylinder of its inner
  radius). Milk spills where it rises over the rim; the spout's lip is 8 mm lower.
- **Spraying:** with the steam on, a tip that has been in the milk and comes out of it (2 mm or more above the
  surface, or out of the pitcher) sprays milk.
- **Purging:** steam with the tip out of the pitcher before steaming purges the wand. Without 1 s of it, the wand's
  5 g of condensed water goes into the milk.

**Score:**
- **0** if any of these happens:
  - the milk ever goes over 70 °C (SCA: 55-65 °C, at most 70; off flavours begin at about 70);
  - the tip leaves the milk while the steam is on;
  - more than 2 ml spills;
  - the pitcher falls off the table.
- **Otherwise, levels:**
  - **0.25:** the milk has reached 40 °C;
  - **0.5:** it ends at 55-65 °C;
  - **0.75:** it also ends 20-40% more in volume (a latte; a cappuccino's 50% or more is too much);
  - **1.0:** also, the wand was purged first (1 s or more of steam with the tip out of the pitcher, before
    steaming), and at the end the steam is off and the pitcher stands upright on the counter (tilted less than 10
    degrees). The trial then ends with success.
- Doing nothing scores 0.

**Checking a real trial:**
- A probe thermometer with a maximum reading clipped in the pitcher gives the final and the highest temperature.
- A pitcher with volume marks, read right after steaming, gives the volume (170 ml at the start).
- A video shows the purge, and whether the steam was shut before the tip left the milk.
- Paper around the drip tray shows any spray.

**Safety in a real trial:** the steam is about 120 °C and the milk can reach 70 °C or more. People stay out of the
arms' reach and clear of the wand while the steam is on; an operator stands by the machine's power switch and can
shut the steam at its knob; the pitcher is steel with a cool handle, and its milk starts at 4 °C. The robot touches
only the knob and the pitcher.

## The demo

`demo.py`, open-loop, with the arms' joint trim on their own tracking error (as the EV and spray tasks' demos):
1. The left hand takes the knob from the front, its fingers above and below it (rolled a quarter turn), moving in
   slowly. The right hand takes the pitcher round its body, low down from behind, pitched 35 degrees down and rolled
   half a turn (the handle passes between its fingers). It lifts the pitcher under the wand, the rim 2.5 cm below
   the tip.
2. **Purge:** the knob turned fully open for 1.5 s, then shut.
3. The pitcher rises until the tip is 4 mm under the milk, 12 mm off its axis toward the handle. The steam opens fully.
4. **Stretch:** for 3.5 s the pitcher comes down as the foam rises, keeping the tip 4 mm under (planned from the air
   intake).
5. **Roll:** the pitcher rises 20 mm, so the tip is deep, and the milk heats. At about 56 °C (planned from the steam's
   heat) the pitcher comes down 12 mm (the tip stays 17 mm deep) and the knob shuts.
6. The pitcher comes off the wand and goes back where it started. Both hands let go.

Three numbers in the plan come from the first simulator runs (2026-11-18):
- the full pitcher settles 6 mm down in the fingers as it is lifted, so the hand holds it 6 mm higher;
- held open, the knob creeps back to about 80 degrees in the fingers (88% of the steam), so the hand turns 8
  degrees past the knob's stop and the plan heats at 88%;
- rolled half a turn on the knob (as first planned), the left wrist met the right arm's wrist as the knob shut, so
  the left hand grips it rolled a quarter turn instead.

**Dream-1** (one trial, 2,400 steps, 320 x 176 cameras): **0.0**. It moved both arms about, toward the machine and
near the pitcher, but never took the knob or the pitcher: the milk stayed at 4 °C.

**The demo in the simulator** (640 x 360 cameras): **1.0**, success at step 942 (47 s). The wand was purged for 2.6 s;
the milk ended at 60.3 °C and 34% more in volume (17.9 g of water added); nothing sprayed or spilled; the knob was
shut and the pitcher upright on the counter.

Offline, with the arms following the plan exactly (`--plan-only`):
- the demo scores **1.0**: the milk at 60.0 °C and 33% more in volume, nothing sprayed or spilled;
- IK errors are below 0.1 mm, and the smallest joint margin is 13.8 degrees;
- the grippers stay 4.9 mm or more from the machine and 18 mm or more from each other, and the wand stays 15 mm from
  the pitcher's wall.

Test versions (`DEMO_VARIANT=...`), offline:

| Version | What it does | Score |
|---|---|---|
| `no_purge` | no purge | 0.75 |
| `no_stretch` | the tip straight to 25 mm deep: 10% more in volume | 0.5 |
| `pull_out` | the pitcher comes off the wand with the steam on: the milk sprays | 0 |
| `overheat` | 10 s more steam: 86 °C | 0 |

## Fidelity and simplifications

- **The machine** is fixed and full size. The wand is rigid (a real one swivels on a ball joint), and the right wand
  and the hot-water tap aren't used.
- **The milk** is a model, not a fluid:
  - Heat and water from the steam follow its energy.
  - The air intake by the tip's depth is provisional: no published law was found, and the numbers are set so a
    latte's stretch takes 3-5 s, as barista guides describe.
  - The foam is added volume with no texture (microfoam against big bubbles).
  - The milk doesn't slosh (it is part of the pitcher's rigid mass), doesn't cool, and drips nothing.
- **The knob's** damping is provisional, and it has no friction (see the scene).
- **The markers** draw the milk's surface (white, cream once stretched), the thermometer, steam escaping into the air
  and sprayed milk.

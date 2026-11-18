# Spray one tagged leaf, shielding the rest of the plant with a card

A young pepper plant in a pot has one leaf tagged with pink flagging tape. The robot must spray that leaf's top with a
trigger spray bottle until it is wet all over, without getting spray on any other leaf. This is how a new pesticide
is tried on a plant: a test patch first, checked a day later for damage (insecticidal soap: "spray a small area and
wait 24 hours", Clemson HGIC). A trigger spray drifts: the fine drops that miss the leaf settle on the leaves below.
The label of a common ready-to-use herbicide (Roundup, EPA 71995-33) says to "shield plants from drift with a piece of
cardboard". So one hand holds a card under the leaf while the other aims and squeezes.

The difficulty is physical:
- The right hand holds the bottle with one pinch, one pad on the back of the spray head and the other on the trigger.
  It must stay light while carrying, below the trigger's 8 N preload, and squeeze the full 12 mm stroke to spray.
- The aim must be within a few millimetres, from a pose that keeps the bottle and hands out of the leaves.
- The left hand must slide the card in under the leaf without touching the plant.

The task file is [task.py](task.py), for the `yam_arena` embodiment on the station scene. It uses
[partkit.py](partkit.py) (the EV task's kit, with a full start rotation and a set mass centre and inertia for parts,
from the spray-bottle grip test) for the card and the bottle with its sprung trigger. The task file's docstring only
describes the scene, because the simulator shows it to agent policies. The scoring is explained in code comments and
here.

## Run a trial

On a machine with the official simulator (the Robocurve Brev launchable), with a Dream API key in
`~/.dreamlabs/credentials`, from this folder:

```bash
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Spray the tagged leaf until its top is wet all over, without getting spray on any other leaf." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/spray-dream
inspect-robots video ~/yam-eval/spray-dream/*.json

# The scripted demo, which reaches full score (about 37 s). It is open-loop: every move is planned from the scene's
# known layout, and it reads back only the arms' own joint positions, never an object's pose.
python demo.py task.py demo_out --cam-width 640 --cam-height 360

# Offline, with no simulator: the demo's plan through the IK with the task's own spray and touch code, and the tests
python demo.py task.py - --plan-only
python -m pytest test_offline.py -q
```

Keep `-E object_jitter=0`: the demo depends on it. A trial takes about 2 minutes of simulated time at most
(2,400 steps). The spray's random draws (droplet sizes, directions, launch times) use a fixed seed, reset with the
episode (`SPRAY_SEED`, default 5), so the same actions give the same score.

## Scoring

**The spray model.** Every control step, in Python:
- A squeeze fires when the trigger passes 10 of its 12 mm. The next one fires only after the trigger has come back to
  4 mm.
- A squeeze sends 1 ml out of the nozzle over 0.2 s, as 700 droplet parcels (each a group of identical droplets).
  - Droplet sizes are log-normal by volume: median 200 µm, spread 1.7, clipped to 40-600 µm.
  - They leave at 12 m/s inside a 40 degree cone along the nozzle's axis.
  - They fly with gravity and still-air drag, in 0.25 ms steps.
- No spray comes out while the bottle leans more than 75 degrees from upright, because its dip tube is out of the
  liquid.
- A parcel stops on the first thing its path crosses: a leaf (any triangle of the eight leaf meshes), the card, the
  soil or the table. On a leaf, each droplet leaves a stain twice its diameter.
- A leaf triangle's **wet fraction** is the chance that a point on it lies under at least one stain:
  1 - exp(-stain area / triangle area). A leaf's wet fraction is the area-weighted mean over its triangles.
- 120 of each squeeze's parcels are drawn as blue beads (1.5 mm in radius, for visibility). Beads on the card move
  with it.

**Touching the plant.** The plant has no colliders: real leaves bend, and a hand or bottle that brushes them shifts
the leaf and smears the spray. So the scorer checks it geometrically every control step. Touching means a point of the
plant (every vertex, edge midpoint and triangle centre of the leaves, about 3 mm apart, and points on the stem and
stalks) within 1 mm of:
- the bottle (its five body and head boxes and the trigger);
- either gripper (its two fingers and its housing);
- the card.

**Score:**
- **0** if anything has touched the plant at any time, or if the other seven leaves have more than 1 cm² wet in total.
- **Otherwise**, the tagged leaf's wet fraction divided by 0.8, up to 1.0.
- **The trial ends with success** (score 1.0) once the tagged leaf is 80% or more wet, the other leaves have 1 cm²
  or less, no spray is still in the air, and that has held for 1 s.
- Doing nothing scores 0.

**Checking a real trial:**
- Use water with a blue food dye, or stick water-sensitive spray cards (yellow paper that turns blue where a drop
  lands, the standard tool for checking spray coverage) on the leaves.
- Photograph each leaf from above against a 1 cm grid. Count the wet area on the tagged leaf's top and on the others.
- A touch can be seen in a video, or from a leaf that moves.

## The demo

`demo.py`, open-loop, with the arms' joint trim on their own tracking error (as the EV task's demo):
1. The left hand pinches the card's short edge nearer the arms where it overhangs the crate, lifts it, carries it toward the arms and to the right, then
   slides it 16 cm straight in under the tagged leaf. There its inner edge is 18 mm from the stem and its top 12 mm
   below the leaf's lowest point.
2. The right hand swings to 10 cm above a point 8 cm behind the bottle's head, comes straight down, moves in, takes
   the bottle from behind its head and lifts it 20 cm.
3. It turns the bottle into the spray pose with a joint-space move, high and to the right of the plant:
   - the spray axis 45 degrees off the leaf's normal, heading across the leaf toward the arms' left;
   - the nozzle 8 cm from the leaf's centre;
   - the bottle rolled 30 degrees about the axis.

   It comes in the last 6 cm along the axis.
4. Two full squeezes, 1.4 s apart. The drops that miss the leaf land on the card.
5. The bottle backs out, and the card slides back out.

The spray plan is the best of an offline search over the spray direction, distance and roll, and both arms' reach. In
droplet runs with arm-like errors (2 mm and 1 degree of aim, 3 mm of card placement), it scored fully in 15 of 16.

In the simulator (640 x 360 cameras, `-E object_jitter=0`):
- the demo scores **1.0**, ending with success at step 731 (37 s): the tagged leaf 84% wet, the other leaves 0 cm²,
  two full squeezes, nothing touching the plant. Of the 2 ml sprayed, 0.82 ml landed on the tagged leaf, 1.0 ml on the
  card and 0.17 ml on the table;
- the arms' joint trim stays below 15 degrees (left, carrying the card) and 7.5 degrees (right).

Test versions (`DEMO_VARIANT=...`), in the simulator:
- `no_card`: no card, so the other leaves get 3.2 cm² (the tagged leaf is 79% wet). Score 0.
- `touch`: the bottle comes 7 cm further in, into the leaf; the touch is caught at 31 s. Score 0.

**Dream-1** (one trial, the command above, 2,400 steps): **0.0**. Its right arm reached toward the plant and the pot
but never picked up the bottle, so nothing was sprayed; nothing touched the plant.

Offline, with the arms following the plan exactly (`--plan-only`): the demo scores 1.0 (the tagged leaf 90% wet, the
other leaves 0.28 cm²), IK errors are below 0.1 mm and the smallest joint margin is 8.6 degrees; `no_card` and `touch`
score 0.

## Fidelity and simplifications

- **The plant** is built in code: eight ovate leaves (8.5-5 cm long) on short stalks around a 21 cm stem, as a young
  pepper plant. It is drawn only, with no colliders, and its leaves don't bend.
- **The spray** is a droplet model, not a fluid. Its numbers are estimates until measured on a real 8 oz trigger
  bottle: 1 ml per stroke, a 40 degree cone, 12 m/s, and a volume median diameter of 200 µm (trigger sprays run about
  120-350 µm in the patents found). Drops don't splash, run off or evaporate. Spray passes through the hands and the
  bottle.
- **The bottle** is an 8 oz (237 ml) square PET bottle with a 28-400 trigger head, as boxes, full (286 g). The trigger
  is a sprung slider: 8 N preload, 20 N at full travel (12 mm). The preload is not yet measured. The liquid doesn't
  slosh.
- **The card** is rigid 2 mm corrugated board, 16 x 20 cm, 15 g.
- **The beads** are a sample of the parcels, drawn far larger than real droplets (3 mm across) so that the cameras
  show where spray landed, as a marker dye would.

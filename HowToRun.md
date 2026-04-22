# How to Run

## Setup

To run everything and recreate the figures, run
```bash
make setup         # create .venv and install requirements
make run-all       # full pipeline (reach → reach_hold → grasp → pick_place), 'all' curriculum
make figures         # regenerate all analysis figures from logs. Must update PPO attempt number in CONFIG blocks of each plot_*.py script before running.

```

To run the full pipeline for a specific curriculum, use one of the following:
```bash
make run-ordered   # same pipeline, 'ordered' curriculum
make run-random    # same pipeline, 'random' curriculum
make run           # all three curricula, sequentially
```

To run individual phases, visualize trained models, or resume training, see the instructions below.

## Sensor Curricula

Three curricula are supported, each producing its own model files tagged with the curriculum name (`_all`, `_ordered`, `_rand`).

| Curriculum | Flag | Description |
|---|---|---|
| **all** | *(default)* | All sensors active every episode. |
| **ordered** | `--curriculum ordered` | Fixed per-phase sensor subset (camera added in Phase 3). |
| **random** | `--curriculum random` | 30% per-sensor dropout each episode + SensorMaskSensor in obs. |

---

## Training Phases

Run each phase in sequence. Each phase loads the previous phase's model automatically.

### Phase 1A — Reach

```bash
python train_reach.py --timesteps 500000
python train_reach.py --timesteps 500000 --curriculum ordered
python train_reach.py --timesteps 500000 --curriculum random
```

### Phase 1B — Reach Hold

```bash
python train_reach.py --phase hold --load-model models/ppo_reach_all.zip --timesteps 1000000
python train_reach.py --phase hold --load-model models/ppo_reach_ordered.zip --timesteps 1000000 --curriculum ordered
python train_reach.py --phase hold --load-model models/ppo_reach_rand.zip --timesteps 1000000 --curriculum random
```

### Phase 2 — Grasp

Automatically loads the matching `ppo_reach_hold_<curriculum>.zip`.

```bash
python train_grasp.py --timesteps 5000000
python train_grasp.py --timesteps 5000000 --curriculum ordered
python train_grasp.py --timesteps 5000000 --curriculum random
```

### Phase 3 — Pick and Place

Automatically loads the matching `ppo_grasp_<curriculum>.zip`.

```bash
python train_pick_place.py --timesteps 7500000
python train_pick_place.py --timesteps 7500000 --curriculum ordered
python train_pick_place.py --timesteps 7500000 --curriculum random
```

---

## Resume Training

Pass `--no-reset` to skip value-function resets when resuming an interrupted run (not transferring).

```bash
python train_grasp.py --no-reset --load-model models/ppo_grasp_all.zip --timesteps 1500000
python train_pick_place.py --no-reset --load-model models/ppo_pick_place_all.zip --timesteps 2000000
```

### Pick-and-place with a wall-clock budget

Use `--max-hours H` on `train_pick_place.py` to stop when the cumulative wall-clock
hours across **all phases of this curriculum** (reach + reach_hold + grasp + every
pick_place run) reach `H`. The helper scans `logs/<phase>_<curriculum>/PPO_*/` event
files to compute prior hours, so resumes are counted correctly. Pair with a large
`--timesteps` so the budget is what actually trips the stop.

```bash
# Continue the ordered pick_place until total hours == the 'all' curriculum (~25 h)
python train_pick_place.py --resume --curriculum ordered --max-hours 25 --timesteps 100000000

# Or via the Makefile shortcut (same command):
make run-ordered-match-all HOURS=25
```

---

## Visualize a Trained Model

```bash
python see_arm_camera.py --model models/ppo_reach_all.zip --task reach
python see_arm_camera.py --model models/ppo_reach_hold_all.zip --task reach_hold
python see_arm_camera.py --model models/ppo_grasp_all.zip --task grasp
python see_arm_camera.py --model models/ppo_pick_place_all.zip --task pick_and_place
```

## Manual Control

```bash
python see_arm_camera.py # no args = manual control
```

---

## Analysis Plots

Figures are written to `analysis/figures/`. Edit the `CONFIG` block at the top
of each script to change curricula, PPO attempt number, smoothing, or metrics.

---

## Command-Line Flags

All training scripts:

| Flag | Default | Description |
|---|---|---|
| `--timesteps N` | script-specific | Total training steps |
| `--curriculum` | `all` | `all`, `ordered`, or `random` |
| `--load-model PATH` | phase-specific default | Weights to fine-tune from |
| `--from-scratch` | off | Ignore pretrained weights |
| `--render` | off | Watch training live in GUI |
| `--eval-episodes N` | 20 | Final evaluation episodes |

`train_reach.py` only:

| Flag | Default | Description |
|---|---|---|
| `--phase` | `reach` | `reach` = Phase 1A, `hold` = Phase 1B |

`train_grasp.py` only:

| Flag | Default | Description |
|---|---|---|
| `--keep-policy` | off | Keep policy weights, reset value head only |
| `--no-reset` | off | Skip all head resets (resuming, not transferring) |
| `--noise-scale` | 0.0 | Gaussian noise added to policy weights (0.1–0.3) |

`train_pick_place.py` only:

| Flag | Default | Description |
|---|---|---|
| `--no-reset` | off | Skip value function reset (resuming, not transferring) |
| `--resume` | off | Resume a pick_place checkpoint: implies `--no-reset`, preserves timestep counter, lowers `ent_coef` |
| `--max-hours H` | unset | Stop when cumulative wall-clock hours across ALL phases of this curriculum reach `H` (counts resumes) |

Transfer mode summary for `train_grasp.py`:

| Command | Value head | Policy head | Use when |
|---|---|---|---|
| *(default)* | reset | reset | Policy known to be stale |
| `--keep-policy` | reset | kept | Reach→Grasp transfer (recommended) |
| `--noise-scale 0.1` | reset | noise added | Break local minima gently |
| `--no-reset` | kept | kept | Resuming interrupted grasp training |

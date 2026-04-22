# Hierarchical RL for Robotic Arm Manipulation

> [!WARNING]  
> This project is a work in progress. The code is available, but training and evaluation are not yet complete, and the final thesis writeup is forthcoming. There is also vestigial code throughout the project, leftover from a LOT of experimentation and iteration. My primary focus at this time is not on this code, but rather the writeup. I will clean up the code and add more documentation once the thesis is done. In the meantime, please reach out if you have questions about the code or want to collaborate on further development. 

---

A BA Cognitive Science thesis project at Vassar College, advised by Ken Livingston (first reader) and Josh de Leeuw (second reader).

## Overview

This project trains a 4-DOF [Lynxmotion LSS arm](https://github.com/Lynxmotion/LSS-ROS2-Arms) in PyBullet to perform reach, grasp, and pick-and-place tasks using PPO. Training runs in three phases, with each phase fine-tuning from the policy learned in the previous one:

1. Reach (1A: approach a target, 1B: hold position at the target)
2. Grasp (close the gripper around a cube and lift)
3. Pick and Place (transport the cube to a target on a second table)

## Research Question

The thesis compares established learning curricula against a natural-development-inspired approach to the order in which sensors are introduced during training. Three sensor curricula are evaluated in parallel:

- `all`: every sensor (proprioception, camera, depth) is active in every episode from the start.
- `ordered`: a fixed per-phase sensor subset, with the camera withheld until Phase 3. This mimics a developmental trajectory in which richer sensing comes online only when the task requires it.
- `random`: each episode randomly drops 30% of the sensors and exposes a `SensorMaskSensor` in the observation, so the policy has to learn to handle missing modalities.

Running the same three-phase pipeline under each curriculum and comparing final task performance lets us ask whether developmental sensor introduction yields better policies than full-information training or robustness-style dropout.

## Results

Results, analysis, and the full thesis writeup will be added here once training and evaluation are complete.

## Project Layout

`envs/` holds the core Gymnasium environment. `arm_env.py` defines `ArmEnv`, the base PyBullet env running at 240 Hz; `tasks.py` defines `ReachTask`, `GraspTask`, and `PickAndPlaceTask` along with their reward functions; `sensors.py` contains the composable observation sensors (proprioceptive, camera, depth, mask); and `extractors.py` has the SB3 feature extractors.

Training scripts live at the project root: `train_reach.py`, `train_grasp.py`, and `train_pick_place.py`, one per phase. `see_arm_camera.py` runs a trained model in a GUI window or, with no arguments, lets you drive the arm manually. `diagnose_reward.py` shows a live breakdown of reward components for debugging, and `plot_training.py` plots TensorBoard logs across runs. `analysis/` contains the curricula-comparison plotting scripts (`plot_curricula.py`, `plot_cumulative.py`); see [HowToRun.md](HowToRun.md#analysis-plots) for usage.

Saved weights and `VecNormalize` stats live in `models/`, TensorBoard logs in `logs/`, and the robot/scene URDFs in `urdf_files/` and `lss_arm_description/`. The former was derived from the latter via a Docker-based extraction process documented in [HowToURDF.md](HowToURDF.md).

## Key Design Constants

The arm operates in a workspace bounded by `[-0.15, -0.15, 0.10]` to `[0.15, 0.15, 0.35]` metres. Joint velocity is capped at 1.5 rad/s, the realistic under-load spec for the LSS servos. The action space is 5D continuous in `[-1, 1]`: four arm joints plus one gripper.

A grasp counts as successful if the cube is lifted more than 5 cm with bilateral fingertip contact for 10 consecutive steps. A pick-and-place counts as successful if the cube ends up within 5 cm of the target on the destination table and is released. Each episode places a source table at a random angle and a destination table 120 to 240 degrees away from it.

## Setup

```bash
make setup
```

## Training

Each phase is run once per curriculum (`all`, `ordered`, `random`), and each phase automatically loads the matching model from the previous one. The minimum sequence:

```bash
make run-all 
```

This works, but will take a LONG time. On two EPYC 7601 CPUs, it takes ~3 days. On consumer hardware, it can take weeks. Thus, training each phase separately is recommended, both for faster iteration and to allow resuming if a run is interrupted. The commands below show how to run each phase separately, but the Makefile has shortcuts for the full pipeline under each curriculum.


```bash
# Phase 1A: Reach
python train_reach.py --timesteps 500000
python train_reach.py --timesteps 500000 --curriculum ordered
python train_reach.py --timesteps 500000 --curriculum random

# Phase 1B: Reach Hold (loads ppo_reach_<curriculum>.zip)
python train_reach.py --phase hold --load-model models/ppo_reach_all.zip --timesteps 1000000
python train_reach.py --phase hold --load-model models/ppo_reach_ordered.zip --timesteps 1000000 --curriculum ordered
python train_reach.py --phase hold --load-model models/ppo_reach_rand.zip --timesteps 1000000 --curriculum random

# Phase 2: Grasp (auto-loads ppo_reach_hold_<curriculum>.zip)
python train_grasp.py --timesteps 5000000
python train_grasp.py --timesteps 5000000 --curriculum ordered
python train_grasp.py --timesteps 5000000 --curriculum random

# Phase 3: Pick and Place (auto-loads ppo_grasp_<curriculum>.zip)
python train_pick_place.py --timesteps 7500000
python train_pick_place.py --timesteps 7500000 --curriculum ordered
python train_pick_place.py --timesteps 7500000 --curriculum random
```

Training uses 46 parallel envs and 10 eval envs via `SubprocVecEnv`. See [HowToRun.md](HowToRun.md) for the full set of run instructions, including how to resume interrupted runs, all command-line flags, transfer-mode options for `train_grasp.py`, and visualization commands.

## Visualizing a Trained Model

```bash
python see_arm_camera.py --model models/ppo_pick_place_all.zip --task pick_and_place
```

Run `see_arm_camera.py` with no arguments for manual control of the arm in the GUI.

## License

MIT, see [LICENSE](LICENSE).

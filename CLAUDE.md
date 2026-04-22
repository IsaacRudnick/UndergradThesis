# Thesis2 — Hierarchical RL for Robotic Arm Manipulation

Curriculum-based PPO training for a Lynxmotion LSS 4-DOF arm in PyBullet simulation.
Three progressive phases: **Reach → Grasp → Pick-and-Place**, each fine-tuning from the previous phase. That's run across three curriculums: **all**, **ordered**, and **random**. 

## Project Layout

- `envs/` — Core Gymnasium environment
  - `arm_env.py` — Base PyBullet env (ArmEnv), physics sim at 240 Hz
  - `tasks.py` — Task definitions with reward functions (ReachTask, GraspTask, PickAndPlaceTask)
  - `sensors.py` — Composable observation sensors (proprioceptive, camera, depth)
  - `extractors.py` — Custom SB3 feature extractors (FrozenCNNExtractor)
- `train_reach.py` / `train_grasp.py` / `train_pick_place.py` — Training scripts per phase
- `diagnose_reward.py` — Live reward breakdown visualization for debugging
- `see_arm_camera.py` — View trained models running + manual control
- `plot_training.py` — Single-run/per-metric TensorBoard plotting (older helper)
- `analysis/` — Curricula-comparison plotting
  - `_common.py` — Shared helpers: event-file loading, EMA smoothing, axis formatters, palette
  - `plot_curricula.py` — One 2×2 figure per phase (metric × {timesteps, per-phase wall time})
  - `plot_cumulative.py` — Final-phase view with x-axis offset by each curriculum's preceding-phase wall time
  - `figures/` — PNG outputs
- `models/` — Saved model weights (`.zip`) and normalization stats (`_vecnorm.pkl`)
- `logs/` — TensorBoard logs per phase
- `urdf_files/` — PyBullet URDFs for the arm and scene objects

## Critical Rules

1. **Reward function sync**: The reward function lives in `envs/tasks.py`. Any changes to reward logic MUST be mirrored in `diagnose_reward.py`.

2. **Transfer learning chain**: Phase 1 → `ppo_reach_hold.zip` → Phase 2 → `ppo_grasp.zip` → Phase 3 → `ppo_pick_place.zip`. Each phase loads the previous model. Breaking this chain breaks training.

3. **VecNormalize stats**: Grasp and pick-place models each have a paired `_vecnorm.pkl` file with observation/reward normalization stats. Always save/load these together with the model. Reach models do not use VecNormalize.

4. **Log directory naming**: TensorBoard logs live at `logs/<phase>_<curriculum>/PPO_<n>/` where `<phase>` is one of `reach`, `reach_hold`, `grasp`, `pick_place` and `<curriculum>` is one of `all`, `ordered`, `random`. The `analysis/` scripts rely on this exact layout.

5. **Makefile `figures` target**: `make figures` regenerates every analysis figure. When adding a new `analysis/plot_*.py` script, append a corresponding line to the `figures` target in the `Makefile` so it stays the single entry point for producing all figures.

## Training

- Algorithm: PPO (stable-baselines3)
- 46 parallel training envs, 10 eval envs (SubprocVecEnv)
- Run instructions in `HowToRun.md`
- Makefile shortcuts: `make setup` (venv + requirements), `make run-all` / `make run-ordered` / `make run-random` (full 4-phase pipeline per curriculum), `make run` (all three)

## Key Constants

- Workspace: `[-0.15, -0.15, 0.10]` to `[0.15, 0.15, 0.35]` (metres)
- Joint velocity limit: 1.5 rad/s (realistic under-load spec)
- Action space: 5D continuous `[-1, 1]` (4 arm joints + 1 gripper)
- Grasp point offset: `(0.11, 0.0, 0.0)` from joint 4
- Grasp success: lift >5cm + bilateral contact for 10 consecutive steps
- Two tables per episode: source table at random angle, destination table 60–120° away
- Pick-and-Place success: cube within 5cm of target on destination table AND released

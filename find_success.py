"""Find pick-and-place successes headless, then replay the EXACT recorded actions in GUI.

This proves whether headless successes are real by:
1. Recording every action + cube trajectory during headless successes
2. Replaying those exact actions in GUI (bypassing the model entirely)
3. Comparing observations between headless and GUI to diagnose differences

Usage:
    python find_success.py
    python find_success.py --model models/ppo_pick_place.zip --show 3
"""

import argparse
import os
import time

import numpy as np
import pybullet as p
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs import ArmEnv, PickAndPlaceTask
from envs.sensors import make_all_sensors
from envs.tasks import _spawn_table

_CUBE_Z = 0.10  # must match tasks.py


class ReplayPickAndPlaceTask(PickAndPlaceTask):
    """PickAndPlaceTask with forced positions for exact replay."""

    def __init__(self, cube_xy: tuple[float, float], destination_pos: np.ndarray,
                 tab1_xy: tuple[float, float], tab2_xy: tuple[float, float]):
        super().__init__()
        self._forced_cube_xy = cube_xy
        self._forced_destination_pos = destination_pos.copy()
        self._forced_tab1_xy = tab1_xy
        self._forced_tab2_xy = tab2_xy

    def setup_scene(self, env) -> None:
        # Spawn tables at the exact same positions as the recorded episode
        tab1_x, tab1_y = self._forced_tab1_xy
        env.table_id = _spawn_table(tab1_x, tab1_y)
        tab2_x, tab2_y = self._forced_tab2_xy
        env.table2_id = _spawn_table(tab2_x, tab2_y, color=[0.3, 0.4, 0.3, 1])

        # Cube at exact recorded position
        import os as _os
        cube_urdf = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)),
                                  "urdf_files", "urdf", "cube.urdf")
        cx, cy = self._forced_cube_xy
        env.object_id = p.loadURDF(cube_urdf, [cx, cy, _CUBE_Z])

        self._destination_pos = self._forced_destination_pos.copy()
        self._cube_spawn_pos = np.array([cx, cy, _CUBE_Z], dtype=np.float64)
        self._prev_dist_obj_target = float(
            np.linalg.norm(self._cube_spawn_pos - self._destination_pos))

        env.target_pos = self._cube_spawn_pos.copy()
        self._cube_initial_z = _CUBE_Z
        self._cube_dropped = False
        self._place_hold_counter = 0
        self._lift_bonus_earned = False
        self._last_lift_bonus = 0.0

        # Green sphere on destination
        if self._target_vis_id is not None:
            p.removeBody(self._target_vis_id)
        vs = p.createVisualShape(p.GEOM_SPHERE, radius=0.025,
                                 rgbaColor=[0, 1, 0, 0.6])
        self._target_vis_id = p.createMultiBody(
            baseVisualShapeIndex=vs,
            basePosition=self._destination_pos.tolist(),
        )

        p.resetJointState(env.arm_id, env.GRIPPER_JOINT, 1.5707)
        p.resetJointState(env.arm_id, env.MIMIC_JOINT, -1.5707)


def make_vec_env(task, render=False, vecnorm_path=None):
    """Create ArmEnv + DummyVecEnv + VecNormalize."""
    sensor_list, maskable = make_all_sensors()
    raw_env = ArmEnv(task=task, render_mode="human" if render else None,
                     sensors=sensor_list, maskable_sensors=maskable)
    vec_env = DummyVecEnv([lambda: raw_env])
    if vecnorm_path and os.path.exists(vecnorm_path):
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    else:
        if vecnorm_path:
            print(f"  Warning: {vecnorm_path} not found — running without normalization")
    return vec_env, raw_env


def _obs_vector(obs):
    """Extract the vector part from an observation (handles Dict or Box)."""
    if isinstance(obs, dict):
        return obs["vector"][0]  # remove batch dim
    return obs[0]


def main():
    parser = argparse.ArgumentParser(
        description="Find grasp successes headless, replay exact actions in GUI")
    parser.add_argument("--model", default="models/ppo_grasp.zip")
    parser.add_argument("--max-trials", type=int, default=50,
                        help="Max headless episodes to scan")
    parser.add_argument("--show", type=int, default=2,
                        help="Number of successes to replay visually")
    parser.add_argument("--slow", type=float, default=0.04,
                        help="Delay per step during visual replay (seconds)")
    args = parser.parse_args()

    vecnorm_path = args.model.replace(".zip", "_vecnorm.pkl")

    # ==================================================================
    # Phase 1: Headless scan — record actions + trajectories
    # ==================================================================
    print("=" * 70)
    print("  Phase 1: Headless scan (recording actions for exact replay)")
    print("=" * 70)

    task = PickAndPlaceTask()
    vec_env, raw_env = make_vec_env(task, render=False, vecnorm_path=vecnorm_path)
    model = PPO.load(args.model, env=vec_env, device="cpu")

    successes = []
    n_tried = 0

    for _ in range(args.max_trials):
        obs = vec_env.reset()
        cube_pos = raw_env.object_pos.copy()
        dest_pos = task._destination_pos.copy()
        tab1_pos = p.getBasePositionAndOrientation(raw_env.table_id)[0]
        tab2_pos = p.getBasePositionAndOrientation(raw_env.table2_id)[0]
        first_obs_vec = _obs_vector(obs).copy()

        actions = []
        cube_zs = []
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            actions.append(action.copy())
            obs, _, dones, infos = vec_env.step(action)
            cube_zs.append(float(raw_env.object_pos[2]))
            done = dones[0]
            info = infos[0]

        n_tried += 1
        success = info.get("is_success", False)
        dist = info.get("dist_obj_target", float("nan"))
        tag = "<<< SUCCESS" if success else ""
        print(f"  {n_tried:3d}  dist={dist:.4f}m  steps={len(actions):3d}  {tag}")

        if success:
            peak_z = max(cube_zs)
            print(f"        Cube Z: start={cube_zs[0]:.3f}  "
                  f"peak={peak_z:.3f}  end={cube_zs[-1]:.3f}")
            successes.append({
                "cube_xy": (float(cube_pos[0]), float(cube_pos[1])),
                "destination_pos": dest_pos,
                "tab1_xy": (tab1_pos[0], tab1_pos[1]),
                "tab2_xy": (tab2_pos[0], tab2_pos[1]),
                "dist": dist,
                "actions": actions,
                "cube_zs": cube_zs,
                "first_obs_vec": first_obs_vec,
            })
            if len(successes) >= args.show:
                break

    vec_env.close()

    rate = 100 * len(successes) / n_tried if n_tried else 0
    print(f"\n  {len(successes)} successes in {n_tried} trials ({rate:.1f}%)")

    if not successes:
        print("  No successes found. Try --max-trials 200")
        return

    # ==================================================================
    # Phase 2: Replay recorded actions in GUI (model NOT used)
    # ==================================================================
    print()
    print("=" * 70)
    print("  Phase 2: Replaying EXACT recorded actions in GUI")
    print("  (Model is NOT used — same actions as headless, proving")
    print("   whether the physics reproduces the same result)")
    print("=" * 70)

    for i, s in enumerate(successes):
        print(f"\n--- Replay {i + 1}/{len(successes)}: "
              f"headless dist={s['dist']:.4f}m ---")

        replay_task = ReplayPickAndPlaceTask(
            s["cube_xy"], s["destination_pos"], s["tab1_xy"], s["tab2_xy"])
        vec_env, raw_env = make_vec_env(
            replay_task, render=True, vecnorm_path=vecnorm_path)

        # --- 2a: Replay recorded actions ---
        obs = vec_env.reset()
        gui_first_obs_vec = _obs_vector(obs).copy()

        # Compare observations between headless and GUI
        vec_diff = np.abs(s["first_obs_vec"] - gui_first_obs_vec)
        print(f"  Obs vector diff (headless vs GUI):  "
              f"max={vec_diff.max():.4f}  mean={vec_diff.mean():.4f}")

        gui_cube_zs = []
        step_i = 0
        for step_i, action in enumerate(s["actions"]):
            obs, _, dones, infos = vec_env.step(action)
            gui_cube_zs.append(float(raw_env.object_pos[2]))
            time.sleep(args.slow)
            if dones[0]:
                break

        info = infos[0]
        gui_dist = info.get("dist_obj_target", float("nan"))
        gui_success = info.get("is_success", False)
        gui_peak = max(gui_cube_zs) if gui_cube_zs else 0
        headless_peak = max(s["cube_zs"])

        result = "SUCCESS" if gui_success else "FAIL"
        print(f"  Action replay:  {result}  "
              f"dist={gui_dist:.4f}m  steps={step_i + 1}")
        print(f"  Cube Z peak:  headless={headless_peak:.3f}  "
              f"gui={gui_peak:.3f}  "
              f"diff={abs(headless_peak - gui_peak):.3f}")

        # --- 2b: Run the MODEL in GUI at the same angle ---
        print(f"\n  Now running MODEL in GUI at same angle...")
        model = PPO.load(args.model, env=vec_env, device="cpu")
        obs = vec_env.reset()
        model_first_obs_vec = _obs_vector(obs).copy()

        # Compare model's GUI obs with headless obs
        model_diff = np.abs(s["first_obs_vec"] - model_first_obs_vec)
        print(f"  Obs vector diff (headless vs model-GUI):  "
              f"max={model_diff.max():.4f}  mean={model_diff.mean():.4f}")

        done = False
        step = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = vec_env.step(action)
            done = dones[0]
            info = infos[0]
            step += 1
            time.sleep(args.slow)

        model_success = info.get("is_success", False)
        model_lift = info.get("lift", 0.0)
        result = "SUCCESS" if model_success else "FAIL"
        print(f"  Model in GUI:   {result}  "
              f"lift={model_lift:+.4f}m  steps={step}")

        # Diagnosis
        if gui_success and not model_success:
            print("\n  DIAGNOSIS: Same actions work in GUI, but model produces")
            print("  different actions in GUI → observation difference (likely camera)")
        elif not gui_success:
            print("\n  DIAGNOSIS: Same actions fail in GUI → physics differs")
            print("  between p.DIRECT and p.GUI (rare but possible)")

        vec_env.close()

    print("\nDone.")


if __name__ == "__main__":
    main()

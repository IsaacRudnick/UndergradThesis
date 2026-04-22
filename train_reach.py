"""Phase 1 — Train on the Reach task.

Usage:
    python train_reach.py                                          # Phase 1A, all-sensors (default)
    python train_reach.py --curriculum ordered                     # Phase 1A, ordered subset
    python train_reach.py --curriculum random                      # Phase 1A, random dropout
    python train_reach.py --timesteps 1000000                      # longer run
    python train_reach.py --phase hold --load-model models/ppo_reach_all.zip  # Phase 1B
    python train_reach.py --render                                 # watch training live
"""

import argparse
import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, ProgressBarCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from envs import ArmEnv, ReachTask, ReachHoldTask
from envs.sensors import make_all_sensors, make_rand_sensors
from envs.extractors import FrozenCNNExtractor

N_ENVS = 96
N_EVAL_ENVS = 30

PHASE_MAP = {
    "reach": ReachTask,
    "hold": ReachHoldTask,
}

# ── Ordered curriculum sensor config ─────────────────────────────────────────
# Sensors active during this phase in 'ordered' mode.
# Set to None to activate all sensors (same as 'all' mode).
# Modify this set to change the ordered curriculum for Phase 1.
ORDERED_SENSORS = {
    "ProprioceptiveSensor",
    "EEPositionSensor",
    "TargetPositionSensor",
    "DistanceSensor",
}
_ALWAYS_ON = frozenset({"ProprioceptiveSensor", "TargetPositionSensor", "SensorMaskSensor"})


def make_env(task_cls, render: bool = False, curriculum: str = "all"):
    if curriculum == "random":
        sensor_list, maskable = make_rand_sensors()
    else:
        sensor_list, maskable = make_all_sensors()
        if curriculum == "ordered" and ORDERED_SENSORS is not None:
            for s in sensor_list:
                name = type(s).__name__
                if name not in _ALWAYS_ON:
                    s.is_active = name in ORDERED_SENSORS

    env = ArmEnv(
        task=task_cls(),
        render_mode="human" if render else None,
        sensors=sensor_list,
        maskable_sensors=maskable,
        curriculum="random" if curriculum == "random" else None,
    )
    return Monitor(env)


def make_vec_env(task_cls, n_envs: int = N_ENVS, curriculum: str = "all"):
    """Create a vectorised env with *n_envs* parallel workers."""
    def _make(rank: int):
        def _init():
            return make_env(task_cls, render=False, curriculum=curriculum)
        return _init
    return SubprocVecEnv([_make(i) for i in range(n_envs)], start_method="fork")


def evaluate(model, task_cls, n_episodes: int = 20, render: bool = False,
             curriculum: str = "all"):
    env = make_env(task_cls, render=render, curriculum=curriculum)
    successes = 0
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        if info.get("is_success", False):
            successes += 1
    env.close()
    print(f"Evaluation: {successes}/{n_episodes} successes "
          f"({100 * successes / n_episodes:.1f}%)")
    return successes / n_episodes


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Reach training")
    parser.add_argument("--phase", type=str, default="reach",
                        choices=["reach", "hold"],
                        help="'reach' = Phase 1A (learn to reach), "
                             "'hold' = Phase 1B (fine-tune for stillness)")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--load-model", type=str, default=None,
                        help="Continue training from saved weights (.zip)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume interrupted reach/hold training: auto-sets --load-model "
                             "and preserves the timestep counter (LR schedule continues)")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--curriculum", choices=["all", "ordered", "random"],
                        default="all",
                        help="Sensor curriculum: all=full sensors (default), "
                             "ordered=phase-specific subset, random=30%% per-episode dropout")
    parser.add_argument("--eval-episodes", type=int, default=20)
    args = parser.parse_args()

    # --resume: auto-set load path to the current phase's output model
    if args.resume and args.load_model is None:
        base_name = "ppo_reach" if args.phase == "reach" else "ppo_reach_hold"
        args.load_model = f"models/{args.curriculum}/{base_name}_{args.curriculum}.zip"

    task_cls = PHASE_MAP[args.phase]
    phase_label = "1A-Reach" if args.phase == "reach" else "1B-Hold"
    base_name = "ppo_reach" if args.phase == "reach" else "ppo_reach_hold"
    save_name = f"{base_name}_{args.curriculum}"
    log_dir = f"./logs/reach{'_hold' if args.phase == 'hold' else ''}_{args.curriculum}/"

    os.makedirs("models", exist_ok=True)
    os.makedirs(f"models/{args.curriculum}", exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    if args.render:
        env = make_env(task_cls, render=True, curriculum=args.curriculum)
    else:
        env = make_vec_env(task_cls, n_envs=N_ENVS, curriculum=args.curriculum)

    # Eval callback — saves best model during training
    eval_env = SubprocVecEnv(
        [lambda _i=i: make_env(task_cls, render=False, curriculum=args.curriculum)
         for i in range(N_EVAL_ENVS)],
        start_method="fork",
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{args.curriculum}/",
        log_path=log_dir,
        eval_freq=max(50_000 // N_ENVS, 1),
        n_eval_episodes=N_EVAL_ENVS,
        deterministic=True,
    )

    if args.load_model and os.path.exists(args.load_model):
        print(f"Resuming from {args.load_model}")
        model = PPO.load(args.load_model, env=env, device="cuda",
                         n_steps=512, batch_size=256, n_epochs=5,
                         learning_rate=1e-4, vf_coef=0.5,
                         max_grad_norm=0.5, target_kl=0.15)
        model.tensorboard_log = log_dir
    else:
        if args.phase == "hold" and not args.load_model:
            print("WARNING: Phase 1B (hold) works best when fine-tuning "
                  f"from a trained 1A model.  Use --load-model models/{args.curriculum}/ppo_reach_{args.curriculum}.zip")

        policy_kwargs = dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            features_extractor_class=FrozenCNNExtractor,
            features_extractor_kwargs=dict(
                vec_features=64,
                cnn_output_dim=32,
                freeze_cnn=True,
            ),
        )

        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            device="cuda",
            tensorboard_log=log_dir,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=256,
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            vf_coef=1.0,
            max_grad_norm=0.5,
            target_kl=0.05,
        )

    print(f"Training {phase_label} ({args.curriculum}) for {args.timesteps} steps ...")
    model.learn(total_timesteps=args.timesteps,
                callback=[eval_callback, ProgressBarCallback()],
                reset_num_timesteps=not args.resume)
    model.save(f"models/{args.curriculum}/{save_name}")
    print(f"Saved → models/{args.curriculum}/{save_name}.zip")

    env.close()
    eval_env.close()

    print("\nRunning final evaluation ...")
    evaluate(model, task_cls, n_episodes=args.eval_episodes,
             curriculum=args.curriculum)


if __name__ == "__main__":
    main()

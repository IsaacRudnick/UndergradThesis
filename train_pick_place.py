"""Phase 3 — Train on the Pick-and-Place task (fine-tuning from Grasp weights).

Usage:
    python train_pick_place.py                                         # all-sensors (default)
    python train_pick_place.py --curriculum ordered                    # ordered (all sensors active in Phase 3)
    python train_pick_place.py --curriculum random                     # random dropout
    python train_pick_place.py --load-model models/ppo_grasp_all.zip
    python train_pick_place.py --timesteps 2000000
    python train_pick_place.py --from-scratch
"""

import argparse
import os

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, ProgressBarCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from envs import ArmEnv, PickAndPlaceTask
from envs.sensors import make_all_sensors, make_rand_sensors
from envs.extractors import FrozenCNNExtractor

N_ENVS = 96
N_EVAL_ENVS = 30

# ── Ordered curriculum sensor config ─────────────────────────────────────────
# Phase 3 activates all sensors (camera included) in the ordered curriculum.
# Set to None to keep all sensors active (same as 'all' mode).
ORDERED_SENSORS = None  # all sensors active in Phase 3
_ALWAYS_ON = frozenset({"ProprioceptiveSensor", "TargetPositionSensor", "SensorMaskSensor"})

# Sensors that were active during Phase 2 ordered grasp training.
# Must match ORDERED_SENSORS in train_grasp.py.
_GRASP_ORDERED_SENSORS = {
    "ProprioceptiveSensor",
    "EEPositionSensor",
    "TargetPositionSensor",
    "DistanceSensor",
    "TouchSensor",
    "UltrasonicSensor",
    "ObjectPositionSensor",
}


def _zero_cnn_output_layer(model) -> None:
    """Zero the CNN's output linear layer at the grasp→pick-and-place transition.

    During ordered grasp, the CNN was frozen and always saw zero camera input,
    contributing a constant vector C to the policy features.  At this transition
    the camera activates for the first time.  Zeroing the output linear ensures
    the CNN starts at ~zero output on real images — matching the near-zero C it
    contributed during grasp — so the policy sees a smooth continuation rather
    than a sudden feature jump.  Once unfrozen, gradients from real images push
    the weights away from zero at a rate the rest of the network can adapt to.

    NatureCNN.linear is Sequential(Linear, ReLU); index 0 is the Linear layer.
    """
    extractors = model.policy.features_extractor.extractors
    cnn = extractors._modules.get("camera_rgbd")
    if cnn is None:
        return
    output_linear = cnn.linear[0]
    with torch.no_grad():
        output_linear.weight.zero_()
        output_linear.bias.zero_()
    print("  [transfer] Zeroed CNN output linear "
          "(camera activates from zero baseline)")


def _warmup_schedule(initial_lr: float, warmup_frac: float = 0.10):
    """Linear warmup from initial_lr/20 over the first warmup_frac of training."""
    def schedule(progress_remaining: float) -> float:
        elapsed = 1.0 - progress_remaining
        if elapsed < warmup_frac:
            return initial_lr * (0.05 + 0.95 * elapsed / warmup_frac)
        return initial_lr
    return schedule


def _reset_value_function(model):
    """Reset just the value function head for Phase 3.

    Grasp → Pick-and-Place is a smoother transition than Reach → Grasp,
    so we only reset the value function (different reward scale) but
    keep the policy weights intact.
    """
    policy = model.policy
    for module in policy.mlp_extractor.value_net:
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
    policy.value_net.reset_parameters()
    print("  [transfer] Reset value function MLP")


def make_env(render: bool = False, curriculum: str = "all"):
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
        task=PickAndPlaceTask(),
        render_mode="human" if render else None,
        sensors=sensor_list,
        maskable_sensors=maskable,
        curriculum="random" if curriculum == "random" else None,
    )
    return Monitor(env)


def make_vec_env(n_envs: int = N_ENVS, curriculum: str = "all"):
    """Create a vectorised env with *n_envs* parallel workers."""
    def _make(rank: int):
        def _init():
            return make_env(render=False, curriculum=curriculum)
        return _init
    return SubprocVecEnv([_make(i) for i in range(n_envs)])


def evaluate(model, n_episodes: int = 20, render: bool = False,
             curriculum: str = "all", vecnorm_path: str = None):
    from stable_baselines3.common.vec_env import DummyVecEnv
    raw_env = make_env(render=render, curriculum=curriculum)
    vec_env = DummyVecEnv([lambda: raw_env])
    if vecnorm_path and os.path.exists(vecnorm_path):
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    successes = 0
    for ep in range(n_episodes):
        obs = vec_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = vec_env.step(action)
            done = dones[0]
            info = infos[0]
        if info.get("is_success", False):
            successes += 1
    vec_env.close()
    print(f"Evaluation: {successes}/{n_episodes} successes "
          f"({100 * successes / n_episodes:.1f}%)")
    return successes / n_episodes


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: Pick-and-Place training")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--load-model", type=str, default=None,
                        help="Path to Phase 2 weights to fine-tune from "
                             "(default: ppo_grasp_<curriculum>.zip)")
    parser.add_argument("--from-scratch", action="store_true",
                        help="Train from scratch, ignoring pretrained weights")
    parser.add_argument("--no-reset", action="store_true",
                        help="Skip value function reset (use when resuming from a pick-and-place checkpoint)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume interrupted pick-and-place training: implies --no-reset, "
                             "preserves timestep counter (LR schedule continues), "
                             "and reduces ent_coef (policy is already converged)")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--curriculum", choices=["all", "ordered", "random"],
                        default="all",
                        help="Sensor curriculum: all=full sensors (default), "
                             "ordered=phase-specific subset, random=30%% per-episode dropout")
    parser.add_argument("--eval-episodes", type=int, default=20)
    args = parser.parse_args()

    # --resume implies --no-reset and auto-sets the load path to the pick-and-place model
    if args.resume:
        args.no_reset = True
        if args.load_model is None:
            args.load_model = f"models/{args.curriculum}/ppo_pick_place_{args.curriculum}.zip"

    save_name = f"ppo_pick_place_{args.curriculum}"
    vecnorm_save = f"models/{args.curriculum}/{save_name}_vecnorm.pkl"
    best_vecnorm_save = f"models/{args.curriculum}/best_model_vecnorm.pkl"

    if args.load_model is None:
        args.load_model = f"models/{args.curriculum}/ppo_grasp_{args.curriculum}.zip"

    os.makedirs("models", exist_ok=True)
    os.makedirs(f"models/{args.curriculum}", exist_ok=True)
    os.makedirs(f"logs/pick_place_{args.curriculum}", exist_ok=True)

    use_pretrained = (not args.from_scratch and
                      os.path.exists(args.load_model))

    if args.render:
        env = make_env(render=True, curriculum=args.curriculum)
    else:
        raw_env = make_vec_env(n_envs=N_ENVS, curriculum=args.curriculum)
        # Three cases for VecNormalize initialisation:
        #   1. Resuming a pick-and-place run: reload its own stats.
        #   2. Fresh start from grasp weights: seed from grasp stats.
        #      With the target_pos fix, the pre-grasp observation distribution
        #      is identical to GraspTask (target_pos = cube every step until
        #      has_grasp). Loading grasp stats means the transferred policy sees
        #      correctly-normalised observations from step 1 instead of burning
        #      ~200k steps on recalibration that risks a grip-shape local optimum.
        #      Stats continue to evolve to cover the transport/placement phases.
        #   3. No stats available: start fresh.
        resume_vecnorm = (args.no_reset and os.path.exists(vecnorm_save))
        grasp_vecnorm_path = args.load_model.replace(".zip", "_vecnorm.pkl")
        seed_from_grasp = (not resume_vecnorm and use_pretrained and
                           os.path.exists(grasp_vecnorm_path))
        if resume_vecnorm:
            env = VecNormalize.load(vecnorm_save, raw_env)
            env.training = True
            env.norm_reward = True
            print(f"  Loaded VecNormalize stats from {vecnorm_save}")
        elif seed_from_grasp:
            env = VecNormalize.load(grasp_vecnorm_path, raw_env)
            env.training = True
            env.norm_reward = True
            # Reset reward running stats: _LIFT_SCALE is 10x lower in
            # PickAndPlaceTask (50) than GraspTask (500).  GraspTask's ret_rms
            # variance would divide PickAndPlace rewards by a 10x-too-large
            # std, shrinking the gradient signal to near-zero.  obs_rms is
            # valid and kept; only reward calibration needs a fresh start.
            from stable_baselines3.common.running_mean_std import RunningMeanStd
            env.ret_rms = RunningMeanStd(shape=())
            print(f"  Seeded obs_rms from {grasp_vecnorm_path}, reset ret_rms (reward scale changed)")
        else:
            env = VecNormalize(raw_env, norm_obs=True, norm_reward=True,
                               clip_obs=10.0, clip_reward=10.0)

    eval_env = SubprocVecEnv(
        [lambda _i=i: make_env(render=False, curriculum=args.curriculum)
         for i in range(N_EVAL_ENVS)]
    )
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            clip_obs=10.0, training=False)
    if not args.render:
        eval_env.obs_rms = env.obs_rms

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{args.curriculum}/",
        log_path=f"./logs/pick_place_{args.curriculum}/",
        eval_freq=max(50_000 // N_ENVS, 1),
        n_eval_episodes=N_EVAL_ENVS,
        deterministic=True,
    )

    if use_pretrained:
        resuming = args.no_reset
        if resuming:
            print(f"Resuming training from {args.load_model} (no value function reset)")
        else:
            print(f"Loading pre-trained weights from {args.load_model}")
            print("  Transferring policy, resetting value function...")

        # Grasp → Pick-and-Place is a smoother transition.
        # Keep policy intact, just reset value function.
        base_lr = 1e-4
        if args.resume:
            # Resuming: policy is already converged.  Use flat LR (no warmup)
            # and lower entropy so we don't undo the learned behaviour.
            lr_schedule = 3e-5
            ent_coef = 0.01 if args.curriculum == "random" else 0.005
        else:
            lr_schedule = _warmup_schedule(base_lr, warmup_frac=0.10)
            # GraspTask terminates at lift success so the policy has zero
            # post-grasp experience; higher entropy forces exploration of
            # the carry/release/place sequence.
            ent_coef = 0.05 if args.curriculum == "random" else 0.03
        model = PPO.load(
            args.load_model,
            env=env,
            device="cuda",
            n_steps=512,
            batch_size=256,
            n_epochs=5,
            learning_rate=lr_schedule,
            vf_coef=0.5,
            ent_coef=ent_coef,
            max_grad_norm=0.5,
            target_kl=0.15,
        )

        if not resuming:
            _reset_value_function(model)

        if args.curriculum == "ordered" and not resuming:
            # Camera was disabled throughout ordered grasp (CNN frozen, zero
            # input).  Zero the output linear before unfreezing so the CNN
            # starts from a near-zero baseline on real images, matching what
            # it contributed during grasp rather than producing a sudden jump.
            _zero_cnn_output_layer(model)
        model.policy.features_extractor.unfreeze_cnn()

        model.tensorboard_log = f"./logs/pick_place_{args.curriculum}/"

    else:
        if not args.from_scratch and not os.path.exists(args.load_model):
            print(f"Warning: {args.load_model} not found — training from scratch")
        else:
            print("Training from scratch (no weight transfer)")

        policy_kwargs = dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            features_extractor_class=FrozenCNNExtractor,
            features_extractor_kwargs=dict(
                vec_features=64,
                cnn_output_dim=32,
                freeze_cnn=False,
            ),
        )

        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            device="cuda",
            tensorboard_log=f"./logs/pick_place_{args.curriculum}/",
            policy_kwargs=policy_kwargs,
            learning_rate=lambda p: max(3e-4 * p, 3e-5),
            n_steps=512,
            batch_size=256,
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            vf_coef=1.0,
            ent_coef=0.02,
            max_grad_norm=0.5,
            target_kl=0.10,
        )

    # Sync VecNormalize stats from training env to eval env each step
    from stable_baselines3.common.callbacks import BaseCallback
    class SyncNormCallback(BaseCallback):
        def __init__(self, eval_env, verbose=0):
            super().__init__(verbose)
            self.eval_env = eval_env
        def _on_step(self):
            if hasattr(self.training_env, 'obs_rms'):
                self.eval_env.obs_rms = self.training_env.obs_rms
            return True

    class SaveVecNormOnBestCallback(BaseCallback):
        """Save VecNormalize stats alongside best_model.zip when a new best is found."""
        def __init__(self, eval_callback, save_path, verbose=0):
            super().__init__(verbose)
            self.eval_callback = eval_callback
            self.save_path = save_path
            self._last_best = -np.inf
        def _on_step(self):
            if self.eval_callback.best_mean_reward > self._last_best:
                self._last_best = self.eval_callback.best_mean_reward
                if hasattr(self.training_env, 'save'):
                    self.training_env.save(self.save_path)
            return True

    print(f"Training Pick-and-Place ({args.curriculum}) for {args.timesteps} steps ...")
    callbacks = [eval_callback, ProgressBarCallback()]
    if not args.render:
        callbacks.append(SyncNormCallback(eval_env))
        callbacks.append(SaveVecNormOnBestCallback(eval_callback, best_vecnorm_save))
    model.learn(total_timesteps=args.timesteps, callback=callbacks,
                reset_num_timesteps=not args.resume)
    model.save(f"models/{args.curriculum}/{save_name}")
    if not args.render:
        env.save(vecnorm_save)
    print(f"Saved → models/{args.curriculum}/{save_name}.zip")

    env.close()
    eval_env.close()

    print("\nRunning final evaluation ...")
    final_vecnorm = vecnorm_save if not args.render else None
    evaluate(model, n_episodes=args.eval_episodes,
             curriculum=args.curriculum, vecnorm_path=final_vecnorm)


if __name__ == "__main__":
    main()

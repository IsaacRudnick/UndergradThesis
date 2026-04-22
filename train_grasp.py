"""Phase 2 — Train on the Grasp task (fine-tuning from Reach weights).

Usage:
    python train_grasp.py                                              # default (loads reach weights, all-sensors)
    python train_grasp.py --curriculum ordered                         # ordered sensor subset
    python train_grasp.py --curriculum random                          # random dropout
    python train_grasp.py --load-model models/ppo_reach_hold_all.zip   # explicit path
    python train_grasp.py --timesteps 1000000
    python train_grasp.py --from-scratch                               # ignore pretrained weights
"""

import argparse
import os

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, ProgressBarCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from envs import ArmEnv, GraspTask
from envs.sensors import make_all_sensors, make_rand_sensors
from envs.extractors import FrozenCNNExtractor

N_ENVS = 96
N_EVAL_ENVS = 30

# ── Ordered curriculum sensor config ─────────────────────────────────────────
# Sensors active during Phase 2 in 'ordered' mode.
# Set to None to activate all sensors (same as 'all' mode).
ORDERED_SENSORS = {
    "ProprioceptiveSensor",
    "EEPositionSensor",
    "TargetPositionSensor",
    "DistanceSensor",
    "TouchSensor",
    "UltrasonicSensor",
    "ObjectPositionSensor",
}
_ALWAYS_ON = frozenset({"ProprioceptiveSensor", "TargetPositionSensor", "SensorMaskSensor"})

# Sensors that were active during Phase 1 ordered reach training.
# Used to identify which weight columns in the feature extractor are
# untrained (always-zero input → zero gradient → random-init weights)
# so they can be zeroed at phase transition instead of corrupting features.
_REACH_ORDERED_SENSORS = {
    "ProprioceptiveSensor",
    "EEPositionSensor",
    "TargetPositionSensor",
    "DistanceSensor",
}


class SyncNormCallback(BaseCallback):
    """Sync VecNormalize stats from training env to eval env before evaluations."""

    def __init__(self, eval_env: VecNormalize, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env

    def _on_step(self) -> bool:
        # Sync observation normalization stats to eval env
        if hasattr(self.training_env, 'obs_rms'):
            self.eval_env.obs_rms = self.training_env.obs_rms
        return True


def _reset_mlp_heads(model, reset_policy: bool = True, reset_value: bool = True,
                     noise_scale: float = 0.0):
    """Reset the MLP heads of the policy.

    When transferring from Phase 1, the feature extractor (CNN or first layers)
    contains useful representations, but the MLP heads are tuned for a different
    task. Resetting them allows the network to learn new action mappings while
    preserving learned features.

    Args:
        model: The PPO model.
        reset_policy: If True, reset the policy MLP head.
        reset_value: If True, reset the value MLP head.
        noise_scale: If > 0, add Gaussian noise to policy weights instead of
                     full reset. This can help break local minima while
                     preserving some transfer benefit.
    """
    policy = model.policy

    if reset_value:
        # Reset value branch of MLP extractor
        for module in policy.mlp_extractor.value_net:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        # Reset final value output layer
        policy.value_net.reset_parameters()
        print("  [transfer] Reset value function MLP")

    if reset_policy:
        if noise_scale > 0:
            # Add noise instead of full reset (partial transfer)
            with torch.no_grad():
                for module in policy.mlp_extractor.policy_net:
                    if hasattr(module, "weight"):
                        noise = torch.randn_like(module.weight) * noise_scale
                        module.weight.add_(noise)
                    if hasattr(module, "bias") and module.bias is not None:
                        noise = torch.randn_like(module.bias) * noise_scale
                        module.bias.add_(noise)
                # Also add noise to action output layer
                if hasattr(policy.action_net, "weight"):
                    noise = torch.randn_like(policy.action_net.weight) * noise_scale
                    policy.action_net.weight.add_(noise)
            print(f"  [transfer] Added noise (scale={noise_scale}) to policy MLP")
        else:
            # Full reset of policy MLP
            for module in policy.mlp_extractor.policy_net:
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
            policy.action_net.reset_parameters()
            print("  [transfer] Reset policy MLP head")

        # Reset log_std to 0 (std=1) regardless of noise_scale.
        # Reach training runs with ent_coef=0, driving log_std very negative
        # (near-deterministic policy).  Without resetting it, the "reset"
        # policy head has random weights but tiny action variance, giving
        # almost no exploration despite the new random mean.  std=1 matches
        # the SB3 default initialisation at the start of reach training.
        if hasattr(policy, "log_std"):
            with torch.no_grad():
                policy.log_std.fill_(0.0)
            print("  [transfer] Reset log_std → 0.0 (std=1)")


def _new_sensor_indices(reach_sensors: set, grasp_sensors: set,
                        always_on: frozenset) -> list[int]:
    """Return the vector-obs indices that are newly active in grasp vs reach.

    Mirrors the sensor ordering in make_all_sensors() exactly so the indices
    match what the Linear(27, 64) layer actually receives.
    """
    from envs.sensors import (
        ProprioceptiveSensor, TargetPositionSensor, DistanceSensor,
        UltrasonicSensor, TouchSensor, EEPositionSensor, ObjectPositionSensor,
    )
    # Order must match make_all_sensors(): Proprio, Target, distance,
    # ultrasonic, touch, ee_pos, obj_pos  (camera is image, not in vector)
    ordered = [
        ProprioceptiveSensor(), TargetPositionSensor(),
        DistanceSensor(), UltrasonicSensor(), TouchSensor(),
        EEPositionSensor(), ObjectPositionSensor(),
    ]
    new_indices = []
    offset = 0
    for s in ordered:
        name = type(s).__name__
        size = s.get_obs_size()
        was_active = name in always_on or name in reach_sensors
        now_active = name in always_on or name in grasp_sensors
        if not was_active and now_active:
            new_indices.extend(range(offset, offset + size))
        offset += size
    return new_indices


def _zero_new_sensor_columns(model, new_indices: list[int]) -> None:
    """Zero weight columns for sensors newly activated at this phase transition.

    At the moment grasp begins, the feature extractor produces exactly the same
    output as at the end of reach: new sensor values are multiplied by zero and
    contribute nothing.  Gradients from those inputs then push the weights away
    from zero at a rate the rest of the network can adapt to — the same
    principle as zero-init in residual connections.
    """
    if not new_indices:
        return
    vec_linear = model.policy.features_extractor.extractors["vector"][0]
    with torch.no_grad():
        vec_linear.weight[:, new_indices] = 0.0
    print(f"  [transfer] Zeroed weight columns for new sensors "
          f"(indices {new_indices})")


def _warmup_schedule(initial_lr: float, warmup_frac: float = 0.10):
    """Return an LR schedule that linearly warms up from initial_lr/20
    over the first *warmup_frac* of training, then holds at initial_lr.

    Longer warmup (10%) and lower starting point (1/20) for stability
    when transferring to a very different task.
    """
    def schedule(progress_remaining: float) -> float:
        elapsed = 1.0 - progress_remaining
        if elapsed < warmup_frac:
            # Start at 5% of target LR, ramp to 100%
            return initial_lr * (0.05 + 0.95 * elapsed / warmup_frac)
        return initial_lr
    return schedule


def make_env(task_cls=GraspTask, render: bool = False, curriculum: str = "all"):
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


def make_vec_env(task_cls=GraspTask, n_envs: int = N_ENVS, curriculum: str = "all"):
    """Create a vectorised env with *n_envs* parallel workers."""
    def _make(rank: int):
        def _init():
            return make_env(task_cls=task_cls, render=False, curriculum=curriculum)
        return _init
    return SubprocVecEnv([_make(i) for i in range(n_envs)])


def evaluate(model, task_cls=GraspTask, n_episodes: int = 20, render: bool = False,
             curriculum: str = "all", vecnorm_path: str = None):
    from stable_baselines3.common.vec_env import DummyVecEnv
    raw_env = make_env(task_cls=task_cls, render=render, curriculum=curriculum)
    vec_env = DummyVecEnv([lambda: raw_env])

    # Load VecNormalize stats so observations match training
    if vecnorm_path and os.path.exists(vecnorm_path):
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    successes = 0
    total_lift = 0.0
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
        total_lift += info.get("lift", 0.0)
    vec_env.close()
    avg_lift = total_lift / n_episodes
    print(f"Evaluation: {successes}/{n_episodes} successes "
          f"({100 * successes / n_episodes:.1f}%), "
          f"avg lift: {avg_lift:.3f}")
    return successes / n_episodes


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Grasp training")
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--load-model", type=str, default=None,
                        help="Pre-trained weights to fine-tune from "
                             "(default: ppo_reach_hold_<curriculum>.zip)")
    parser.add_argument("--from-scratch", action="store_true",
                        help="Train from scratch, ignoring pretrained weights")
    parser.add_argument("--no-reset", action="store_true",
                        help="Skip MLP head reset (use when resuming grasp training)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume interrupted grasp training: implies --no-reset, "
                             "preserves timestep counter (LR schedule continues), "
                             "and reduces ent_coef (policy is already converged)")
    parser.add_argument("--keep-policy", action="store_true",
                        help="Reset only the value head, keep policy weights "
                             "(good for Reach→Grasp since reaching is a sub-task)")
    parser.add_argument("--noise-scale", type=float, default=0.0,
                        help="Add noise to policy weights instead of full reset "
                             "(0.1-0.3 recommended if using). 0 = full reset.")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--curriculum", choices=["all", "ordered", "random"],
                        default="all",
                        help="Sensor curriculum: all=full sensors (default), "
                             "ordered=phase-specific subset, random=30%% per-episode dropout")
    parser.add_argument("--eval-episodes", type=int, default=20)
    args = parser.parse_args()

    # --resume implies --no-reset and auto-sets the load path to the grasp model
    if args.resume:
        args.no_reset = True
        if args.load_model is None:
            args.load_model = f"models/{args.curriculum}/ppo_grasp_{args.curriculum}.zip"

    task_cls = GraspTask
    phase_label = "2-Grasp"
    save_name = f"ppo_grasp_{args.curriculum}"
    log_dir = f"./logs/grasp_{args.curriculum}/"

    if args.load_model is None:
        args.load_model = f"models/{args.curriculum}/ppo_reach_hold_{args.curriculum}.zip"

    os.makedirs("models", exist_ok=True)
    os.makedirs(f"models/{args.curriculum}", exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    if args.render:
        env = make_env(task_cls=task_cls, render=True, curriculum=args.curriculum)
    else:
        env = make_vec_env(task_cls=task_cls, n_envs=N_ENVS, curriculum=args.curriculum)
        # Wrap with VecNormalize for stable training (prevents NaN from large rewards)
        vecnorm_path = args.load_model.replace(".zip", "_vecnorm.pkl") if args.load_model else None
        if vecnorm_path and os.path.exists(vecnorm_path) and not args.from_scratch:
            print(f"Loading VecNormalize stats from {vecnorm_path}")
            env = VecNormalize.load(vecnorm_path, env)
        else:
            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    eval_env = SubprocVecEnv(
        [lambda _i=i: make_env(task_cls=task_cls, render=False, curriculum=args.curriculum)
         for i in range(N_EVAL_ENVS)]
    )
    # Share normalization stats with eval env (don't update during eval)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0,
                            training=False)
    # Sync normalization stats from training env to eval env
    if not args.render:
        eval_env.obs_rms = env.obs_rms
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{args.curriculum}/",
        log_path=log_dir,
        eval_freq=max(50_000 // N_ENVS, 1),
        n_eval_episodes=N_EVAL_ENVS,
        deterministic=True,
    )

    # Determine whether to transfer from pretrained weights
    use_pretrained = (not args.from_scratch and
                      os.path.exists(args.load_model))

    if use_pretrained:
        print(f"Loading pre-trained weights from {args.load_model}")
        if args.no_reset:
            print("  Fine-tuning (keeping all weights)")
        else:
            print("  Transferring feature extractor, resetting MLP heads...")

        if args.resume:
            # Resuming: policy is already converged.  Use flat LR (no warmup ramp)
            # and lower entropy so we don't undo the learned behaviour.
            base_lr = 3e-5
            lr_schedule = base_lr
            ent_coef = {"ordered": 0.01, "all": 0.002, "random": 0.005}[args.curriculum]
        else:
            base_lr = 5e-5  # Lower LR for transfer stability
            lr_schedule = _warmup_schedule(base_lr, warmup_frac=0.10)
            # Ordered: no camera → gripper closure must be discovered from the
            #   sensor signals alone.  Higher entropy keeps the reset policy
            #   exploring the gripper dimension long enough to find contact.
            # All: camera + unfrozen CNN provide rich gradient signal; lower
            #   entropy avoids over-exploration once reaching re-emerges.
            # Random: sensor dropout already injects stochasticity; 0.015 is
            #   enough to maintain robustness without destabilising.
            ent_coef = {"ordered": 0.05, "all": 0.005, "random": 0.015}[args.curriculum]

        model = PPO.load(
            args.load_model,
            env=env,
            device="cuda",
            n_steps=512,
            batch_size=256,
            n_epochs=3,  # Reduced from 5 to prevent overfitting per rollout
            learning_rate=lr_schedule,
            vf_coef=0.5,
            ent_coef=ent_coef,
            max_grad_norm=0.5,
            target_kl=0.15,  # Increased from 0.05 to allow adaptation
        )

        # Reset MLP heads (keep feature extractor)
        # Value function: always reset (reward scale changes at every transition).
        # Policy: reset so the agent explores with std≈1 (SB3 default init).
        #   The preserved reach-hold policy has near-zero action std (velocity
        #   penalty trained it to be still) and would barely move in GraspTask,
        #   giving ~-200 ep_rew.  A reset policy + ent_coef=0.05 explores the
        #   gripper dimension and eventually discovers contact.
        if not args.no_reset:
            _reset_mlp_heads(
                model,
                reset_policy=not args.keep_policy,
                reset_value=True,
                noise_scale=args.noise_scale,
            )

        if args.curriculum == "ordered":
            # Camera is never active in ordered mode — unfreezing the CNN
            # would only let biases drift from task gradients (noise, no signal).
            # Keep it frozen so its output remains a stable constant.
            # Zero out weight columns for sensors that were inactive during
            # reach so the feature extractor starts grasp producing the same
            # features it ended reach with.
            new_idx = _new_sensor_indices(
                _REACH_ORDERED_SENSORS, ORDERED_SENSORS, _ALWAYS_ON
            )
            _zero_new_sensor_columns(model, new_idx)
        else:
            model.policy.features_extractor.unfreeze_cnn()

        model.tensorboard_log = log_dir

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
            # I tried the below with .05 but it was too exploratory
            ent_coef=0.02,  # Balanced entropy for grasp exploration.
            max_grad_norm=0.5,
            target_kl=0.10,  # More permissive KL for new training
        )

    print(f"Training {phase_label} ({args.curriculum}) for {args.timesteps} steps ...")
    callbacks = [eval_callback, ProgressBarCallback()]
    if not args.render:
        callbacks.append(SyncNormCallback(eval_env))
    model.learn(total_timesteps=args.timesteps, callback=callbacks,
                reset_num_timesteps=not args.resume)
    model.save(f"models/{args.curriculum}/{save_name}")
    # Save normalization stats alongside model
    vecnorm_save = f"models/{args.curriculum}/{save_name}_vecnorm.pkl"
    if not args.render:
        env.save(vecnorm_save)
    print(f"Saved → models/{args.curriculum}/{save_name}.zip")

    env.close()
    eval_env.close()

    print("\nRunning final evaluation ...")
    vecnorm_path = vecnorm_save if not args.render else None
    evaluate(model, task_cls=task_cls, n_episodes=args.eval_episodes,
             curriculum=args.curriculum, vecnorm_path=vecnorm_path)


if __name__ == "__main__":
    main()

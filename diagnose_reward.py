"""Watch the trained grasp model run in GUI with live reward breakdown.

Loads the trained model, runs episodes visually, and prints a per-step
reward breakdown so you can see exactly what the agent is doing and
what it's earning (or missing).

Usage:
    python diagnose_reward.py                              # latest grasp model
    python diagnose_reward.py --model models/ppo_grasp.zip
    python diagnose_reward.py --episodes 5
    python diagnose_reward.py --manual                     # slider mode (no model)
"""

import argparse
import time
import numpy as np
import pybullet as p
from stable_baselines3 import PPO

from envs import ArmEnv, GraspTask
from envs.tasks import PickAndPlaceTask
from envs.arm_env import ARM_JOINT_INDICES, GRIPPER_JOINT, MIMIC_JOINT  # MIMIC/GRIPPER used in manual mode


def reward_breakdown(env, task):
    """Compute per-section reward breakdown for current state."""
    grasp_pos = env.grasp_point
    obj_pos = env.object_pos
    lift = obj_pos[2] - task._cube_initial_z

    # Gripper state (0 = closed, π/2 = fully open)
    gripper_angle = p.getJointState(env.arm_id, env.GRIPPER_JOINT)[0]

    # Mirror the drop-detection branch in GraspTask.compute_reward
    dropped = lift < -task._DROP_THRESH
    if dropped:
        grasp_dist = float(np.linalg.norm(grasp_pos - task._cube_spawn_pos))
    else:
        grasp_dist = float(np.linalg.norm(grasp_pos - obj_pos))

    left_c, right_c = task._finger_contacts(env)
    has_grasp = (left_c > 0 and right_c > 0
                 and gripper_angle < task._GRIPPER_CLOSE_THRESH)

    r_approach = -task._DIST_SCALE * grasp_dist

    # Proximity-gated gripper shaping
    proximity = max(0.0, 1.0 - grasp_dist / task._GRIP_SHAPE_RADIUS)
    grip_closed_frac = 1.0 - gripper_angle / 1.5707
    r_grip_shape = task._GRIP_SHAPE_SCALE * proximity * max(0.0, grip_closed_frac) if not dropped else 0.0

    r_grasp = task._GRASP_REWARD if (has_grasp and not dropped) else 0.0

    lift_min = getattr(task, '_LIFT_MIN', 0.005)
    r_lift = task._LIFT_SCALE * float(np.clip(lift, 0.0, 0.10)) if (has_grasp and not dropped and lift > lift_min) else 0.0

    # Mirror early-termination flag from compute_reward's drop detection
    if dropped:
        task._cube_dropped = True

    total = -task._TIME_PENALTY + r_approach + r_grip_shape + r_grasp + r_lift

    status = []
    if dropped:
        status.append("DROPPED")
    elif has_grasp:
        status.append("GRASP")
    elif left_c > 0 or right_c > 0:
        status.append("CONTACT(open)")
    grip_pct = int(100 * (1.0 - gripper_angle / 1.5707))
    status.append(f"grip:{grip_pct}%")
    if lift > lift_min:
        status.append(f"LIFT {lift:.3f}m")
    status_str = " | ".join(status) if status else "approaching"

    return grasp_dist, status_str, r_approach, r_grip_shape, r_grasp, r_lift, total


def reward_breakdown_pp(env, task):
    """Compute per-section reward breakdown for PickAndPlaceTask."""
    grasp_pos = env.grasp_point
    obj_pos = env.object_pos
    lift = obj_pos[2] - task._cube_initial_z
    grasp_dist = float(np.linalg.norm(grasp_pos - obj_pos))
    dist_obj_target = float(np.linalg.norm(obj_pos - task._destination_pos))
    gripper_angle = p.getJointState(env.arm_id, env.GRIPPER_JOINT)[0]

    left_c, right_c = task._finger_contacts(env)
    has_grasp = (left_c > 0 and right_c > 0
                 and gripper_angle < task._GRIPPER_CLOSE_THRESH)

    dropped = lift < -task._DROP_THRESH
    if dropped:
        task._cube_dropped = True
        spawn_dist = float(np.linalg.norm(grasp_pos - task._cube_spawn_pos))
        r_approach = -task._DIST_SCALE * spawn_dist
        total = -task._TIME_PENALTY + r_approach
        return grasp_dist, dist_obj_target, "DROPPED", r_approach, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, total

    r_approach = -task._DIST_SCALE * grasp_dist

    proximity = max(0.0, 1.0 - grasp_dist / task._GRIP_SHAPE_RADIUS)
    grip_closed_frac = 1.0 - gripper_angle / 1.5707
    r_grip_shape = task._GRIP_SHAPE_SCALE * proximity * max(0.0, grip_closed_frac)

    r_grasp = task._GRASP_REWARD if has_grasp else 0.0

    r_lift = 0.0
    if has_grasp and lift > 0.005:
        lift_taper = min(1.0, dist_obj_target / getattr(task, '_LIFT_TAPER_DIST', 0.15))
        r_lift = task._LIFT_SCALE * float(np.clip(lift, 0.0, 0.10)) * lift_taper

    # One-time lift bonus: reads what compute_reward set this step
    r_lift_bonus = task._last_lift_bonus

    r_transport = 0.0
    if has_grasp and lift > task._LIFT_MIN:
        progress = task._prev_dist_obj_target - dist_obj_target
        r_transport = task._TRANSPORT_SCALE * progress

    # Overheight penalty: grasped and above carry threshold
    r_overheight = 0.0
    if has_grasp and lift > task._OVERHEIGHT_THRESH:
        r_overheight = -task._OVERHEIGHT_PENALTY * (lift - task._OVERHEIGHT_THRESH)

    # Hold-distance cost: lift-scaled (mirrors tasks.py fix — no cliff at _LIFT_MIN)
    r_hold_cost = 0.0
    if has_grasp and lift > 0.005:
        lift_frac = min(1.0, lift / 0.10)
        r_hold_cost = -task._HOLD_DIST_PENALTY * (dist_obj_target / task._MAX_TRANSPORT_DIST) * lift_frac

    # Fling penalty: cube airborne above height threshold while not grasped
    r_fling = 0.0
    if not has_grasp:
        from envs.tasks import _CUBE_Z as _CZ
        excess_height = max(0.0, obj_pos[2] - (_CZ + task._RELEASE_HEIGHT_THRESH))
        r_fling = -task._FLING_PENALTY * excess_height

    # J1 alignment: reward for rotating base joint to face destination (scale=50)
    r_j1_align = 0.0
    if has_grasp and lift > task._LIFT_MIN:
        j1_current = p.getJointState(env.arm_id, 1)[0]
        j1_target = np.pi / 2.0 - np.arctan2(
            task._destination_pos[0], task._destination_pos[1])
        j1_error = j1_current - j1_target
        j1_error = (j1_error + np.pi) % (2 * np.pi) - np.pi
        align_frac = max(0.0, 1.0 - abs(j1_error) / np.pi)
        r_j1_align = task._J1_ALIGN_SCALE * align_frac * align_frac  # quadratic

    # Placement bonus: gated on cube speed
    cube_lin_vel, _ = p.getBaseVelocity(env.object_id)
    cube_speed = float(np.linalg.norm(cube_lin_vel))
    r_place = 0.0
    if (dist_obj_target < task.PLACE_THRESHOLD
            and not has_grasp
            and cube_speed < task._PLACE_MAX_SPEED):
        r_place = task._PLACE_BONUS

    # J4 (wrist) angle and tilt penalty (mirrors tasks.py)
    j4_angle = p.getJointState(env.arm_id, 4)[0]
    r_j4_tilt = 0.0
    if has_grasp and lift > task._LIFT_MIN:
        j4_tilt_thresh = getattr(task, '_J4_TILT_THRESH', 0.785)
        j4_tilt_penalty = getattr(task, '_J4_TILT_PENALTY', 0.0)
        excess_tilt = max(0.0, j4_angle - j4_tilt_thresh)
        r_j4_tilt = -j4_tilt_penalty * excess_tilt

    # Proximity bonus: stepped attractors toward placement zone (exclusive)
    r_prox = 0.0
    if has_grasp and lift > task._LIFT_MIN:
        prox_near  = getattr(task, '_PROX_THRESH_NEAR',  0.07)
        prox_mid   = getattr(task, '_PROX_THRESH_MID',   0.10)
        prox_far   = getattr(task, '_PROX_THRESH_FAR',   0.15)
        bonus_near = getattr(task, '_PROX_BONUS_NEAR',   0.0)
        bonus_mid  = getattr(task, '_PROX_BONUS_MID',    0.0)
        bonus_far  = getattr(task, '_PROX_BONUS_FAR',    0.0)
        if dist_obj_target < prox_near:
            r_prox = bonus_near
        elif dist_obj_target < prox_mid:
            r_prox = bonus_mid
        elif dist_obj_target < prox_far:
            r_prox = bonus_far

    # Release shaping: reward opening gripper when close + low (mirrors tasks.py)
    r_release = 0.0
    release_dist_thresh = getattr(task, '_RELEASE_DIST_THRESH', 0.10)
    release_lift_thresh = getattr(task, '_RELEASE_LIFT_THRESH', 0.06)
    release_scale       = getattr(task, '_RELEASE_SCALE',       0.0)
    if has_grasp and dist_obj_target < release_dist_thresh and lift < release_lift_thresh:
        grip_open_frac = gripper_angle / 1.5707
        r_release = release_scale * (1.0 - dist_obj_target / release_dist_thresh) * grip_open_frac

    task._prev_dist_obj_target = dist_obj_target
    total = (-task._TIME_PENALTY + r_approach + r_grip_shape + r_grasp + r_lift
             + r_lift_bonus + r_transport + r_overheight + r_hold_cost + r_fling
             + r_j1_align + r_j4_tilt + r_prox + r_release + r_place)

    status = []
    if has_grasp:
        status.append("GRASP")
    elif left_c > 0 or right_c > 0:
        status.append("CONTACT(open)")
    grip_pct = int(100 * (1.0 - gripper_angle / 1.5707))
    status.append(f"grip:{grip_pct}%")
    if lift > task._LIFT_MIN:
        status.append(f"LIFT {lift:.3f}m")
    if r_overheight < 0:
        status.append(f"TOOHIGH!")
    if r_fling < 0:
        status.append(f"FLING!")
    if r_place > 0:
        status.append("PLACED!")
    if r_lift_bonus > 0:
        status.append("LIFT_BONUS!")
    if r_j1_align > 9.0:
        status.append("J1OK!")
    if j4_angle > 0.785:  # > +45°: penalty zone
        status.append(f"J4TILT!")
    if r_prox == getattr(task, '_PROX_BONUS_NEAR', 0.0) and r_prox > 0:
        status.append("NEAR!")
    elif r_prox == getattr(task, '_PROX_BONUS_MID', 0.0) and r_prox > 0:
        status.append("MID!")
    elif r_prox == getattr(task, '_PROX_BONUS_FAR', 0.0) and r_prox > 0:
        status.append("FAR!")
    if r_release > 0:
        status.append("RLSE!")
    status_str = " | ".join(status) if status else "approaching"

    return grasp_dist, dist_obj_target, status_str, r_approach, r_grip_shape, r_grasp, r_lift, r_lift_bonus, r_transport, r_overheight, r_hold_cost, r_fling, r_j1_align, r_prox, r_release, j4_angle, r_j4_tilt, total


def run_model(args):
    """Load trained model and watch it run with reward overlay."""
    import os
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from envs.sensors import make_all_sensors

    task = PickAndPlaceTask() if args.task == "pick_place" else GraspTask()
    sensor_list, maskable = make_all_sensors()
    raw_env = ArmEnv(task=task, render_mode="human",
                     sensors=sensor_list, maskable_sensors=maskable)

    # Wrap in VecEnv + VecNormalize to match training setup
    vec_env = DummyVecEnv([lambda: raw_env])

    # Load saved normalization stats if available
    vecnorm_path = args.model.replace(".zip", "_vecnorm.pkl")
    if os.path.exists(vecnorm_path):
        print(f"Loading VecNormalize stats from {vecnorm_path}")
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False    # don't update stats during eval
        vec_env.norm_reward = False  # don't normalize reward during eval
    else:
        print(f"Warning: {vecnorm_path} not found — running without normalization")

    print(f"Loading model from {args.model} ...")
    model = PPO.load(args.model, env=vec_env, device="cpu")

    is_pp = isinstance(task, PickAndPlaceTask)

    for ep in range(args.episodes):
        obs = vec_env.reset()
        print(f"\n{'='*80}")
        print(f"  EPISODE {ep+1}")
        print(f"{'='*80}")
        print(f"  Cube at: {raw_env.object_pos}")
        if is_pp:
            print(f"  Dest at: {task._destination_pos}")
            print(f"  {'step':>4}  {'gdist':>6}  {'tdist':>6}  {'status':<34}  "
                  f"{'appr':>7}  {'grip':>7}  {'grasp':>7}  {'lift':>7}  {'lftb':>7}  {'trans':>7}  {'ovhgt':>7}  {'hold':>7}  {'fling':>7}  {'j1aln':>7}  {'prox':>7}  {'rlse':>7}  {'j4tlt':>7}  {'j4':>5}  {'TOTAL':>7}")
            print(f"  {'-'*4}  {'-'*6}  {'-'*6}  {'-'*34}  "
                  f"{'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*5}  {'-'*7}")
        else:
            print(f"  {'step':>4}  {'dist':>6}  {'status':<20}  "
                  f"{'appr':>7}  {'grip':>7}  {'grasp':>7}  {'lift':>7}  {'TOTAL':>7}")
            print(f"  {'-'*4}  {'-'*6}  {'-'*20}  "
                  f"{'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")

        done = False
        step = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = vec_env.step(action)
            info = infos[0]
            done = dones[0]
            step += 1

            if is_pp:
                grasp_dist, dist_obj_target, status_str, r_approach, r_grip_shape, r_grasp, r_lift, r_lift_bonus, r_transport, r_overheight, r_hold_cost, r_fling, r_j1_align, r_prox, r_release, j4_angle, r_j4_tilt, total = \
                    reward_breakdown_pp(raw_env, task)
                j4_deg = int(np.degrees(j4_angle))
                print(f"  {step:4d}  {grasp_dist:.4f}  {dist_obj_target:.4f}  "
                      f"{status_str:<34}  "
                      f"{r_approach:+.3f}  {r_grip_shape:+.3f}  {r_grasp:+.3f}  "
                      f"{r_lift:+.3f}  {r_lift_bonus:+.3f}  {r_transport:+.3f}  {r_overheight:+.3f}  {r_hold_cost:+.3f}  {r_fling:+.3f}  {r_j1_align:+.3f}  {r_prox:+.3f}  {r_release:+.3f}  {r_j4_tilt:+.3f}  {j4_deg:+4d}°  {total:+.3f}")
            else:
                grasp_dist, status_str, r_approach, r_grip_shape, r_grasp, r_lift, total = \
                    reward_breakdown(raw_env, task)
                print(f"  {step:4d}  {grasp_dist:.4f}  "
                      f"{status_str:<20}  "
                      f"{r_approach:+.3f}  {r_grip_shape:+.3f}  {r_grasp:+.3f}  "
                      f"{r_lift:+.3f}  {total:+.3f}")

            time.sleep(args.slow)

        print(f"\n  Episode ended: success={info.get('is_success', False)}  "
              f"lift={info.get('lift', 0):.4f}m  steps={step}")

    vec_env.close()


def run_manual(args):
    """Slider mode — drag the arm yourself and see live rewards."""
    if args.task == "pick_place":
        task = PickAndPlaceTask()
    else:
        task = GraspTask()
        if args.warm_start:
            task._WARM_START_FRAC = 1.0
    env = ArmEnv(task=task, render_mode="human")
    obs, _ = env.reset()

    print(f"Cube position: {env.object_pos}")
    print(f"\nUse the PyBullet joint sliders to move the arm.")
    print("Watch the terminal for live reward breakdown.\n")

    sliders = {}
    for idx in ARM_JOINT_INDICES:
        lo, hi = p.getJointInfo(env.arm_id, idx)[8], p.getJointInfo(env.arm_id, idx)[9]
        current = p.getJointState(env.arm_id, idx)[0]
        sliders[idx] = p.addUserDebugParameter(
            f"Joint {idx}", float(lo), float(hi), float(current))
    sliders["grip"] = p.addUserDebugParameter("Gripper", 0.0, 1.5707, 1.5707)

    last_print = 0.0
    try:
        while True:
            for i, idx in enumerate(ARM_JOINT_INDICES):
                val = p.readUserDebugParameter(sliders[idx])
                p.setJointMotorControl2(
                    env.arm_id, idx, p.POSITION_CONTROL,
                    targetPosition=val, force=6.5, maxVelocity=1.5)
            grip_val = p.readUserDebugParameter(sliders["grip"])
            p.setJointMotorControl2(
                env.arm_id, GRIPPER_JOINT, p.POSITION_CONTROL,
                targetPosition=grip_val, force=3.0, maxVelocity=1.5)
            p.setJointMotorControl2(
                env.arm_id, MIMIC_JOINT, p.POSITION_CONTROL,
                targetPosition=-grip_val, force=3.0, maxVelocity=1.5)

            for _ in range(10):
                p.stepSimulation()

            # Update grasp point sphere
            gp = env.grasp_point.tolist()
            if env._grasp_point_vis_id is None:
                vs = p.createVisualShape(
                    p.GEOM_SPHERE, radius=0.02, rgbaColor=[0, 1, 0, 0.7])
                env._grasp_point_vis_id = p.createMultiBody(
                    baseVisualShapeIndex=vs, basePosition=gp)
            else:
                p.resetBasePositionAndOrientation(
                    env._grasp_point_vis_id, gp, (0, 0, 0, 1))

            now = time.time()
            if now - last_print > 0.5:
                last_print = now
                if isinstance(task, PickAndPlaceTask):
                    grasp_dist, dist_obj_target, status_str, r_approach, r_grip_shape, r_grasp, r_lift, r_lift_bonus, r_transport, r_overheight, r_hold_cost, r_fling, r_j1_align, r_prox, r_release, j4_angle, r_j4_tilt, total = \
                        reward_breakdown_pp(env, task)
                    j4_deg = int(np.degrees(j4_angle))
                    print(f"gdist={grasp_dist:.3f}m tdist={dist_obj_target:.3f}m  "
                          f"[{status_str}]  j4={j4_deg:+d}°  "
                          f"R: appr={r_approach:+.2f} grip={r_grip_shape:+.2f} "
                          f"grasp={r_grasp:+.2f} lift={r_lift:+.2f} lftb={r_lift_bonus:+.2f} trans={r_transport:+.2f} "
                          f"ovhgt={r_overheight:+.2f} hold={r_hold_cost:+.2f} fling={r_fling:+.2f} "
                          f"j1aln={r_j1_align:+.2f} prox={r_prox:+.2f} rlse={r_release:+.2f} j4tlt={r_j4_tilt:+.2f}  TOTAL={total:+.2f}")
                else:
                    grasp_dist, status_str, r_approach, r_grip_shape, r_grasp, r_lift, total = \
                        reward_breakdown(env, task)
                    print(f"dist={grasp_dist:.3f}m  "
                          f"[{status_str}]  "
                          f"R: appr={r_approach:+.2f} grip={r_grip_shape:+.2f} "
                          f"grasp={r_grasp:+.2f} lift={r_lift:+.2f}  "
                          f"TOTAL={total:+.2f}")

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\nDone.")

    env.close()


def run_j4_test(_args):
    """Sweep J4 (wrist) through its range so you can confirm which angles look
    correct for carry and placement.

    J1-J3 are locked to a representative carry posture:
      J1=0.0  (base pointing forward)
      J2=0.5  (shoulder raised to carry height)
      J3=0.1  (elbow slightly bent, arm extended)

    J4 is swept in steps, pausing at each so you can see the physical pose.
    Key reference points printed in the terminal:
      -97°  = home / natural hang (URDF default)
        0°  = where the policy arrives after grasp
      +30°  = _J4_TILT_THRESH (penalty starts here)
      +120° = joint limit (where the cube drops)
    """
    from envs.arm_env import HOME_JOINTS, ARM_JOINT_INDICES, GRIPPER_JOINT, MIMIC_JOINT

    task = PickAndPlaceTask()
    env = ArmEnv(task=task, render_mode="human")
    env.reset()

    # Carry posture for J1-J3 — arm raised and extended forward
    CARRY_JOINTS = {1: 0.0, 2: 0.5, 3: 0.1}
    GRIPPER_CLOSED = 0.6  # partially closed, as during carry

    def _hold_pose(j4_rad):
        """Drive arm to carry posture with specific J4 angle for 60 sim steps."""
        for _ in range(60):
            for idx, pos in CARRY_JOINTS.items():
                p.setJointMotorControl2(env.arm_id, idx, p.POSITION_CONTROL,
                                        targetPosition=pos, force=6.5,
                                        maxVelocity=1.5)
            p.setJointMotorControl2(env.arm_id, 4, p.POSITION_CONTROL,
                                    targetPosition=j4_rad, force=6.5,
                                    maxVelocity=1.5)
            p.setJointMotorControl2(env.arm_id, GRIPPER_JOINT, p.POSITION_CONTROL,
                                    targetPosition=GRIPPER_CLOSED, force=5.0,
                                    maxVelocity=1.5)
            p.setJointMotorControl2(env.arm_id, MIMIC_JOINT, p.POSITION_CONTROL,
                                    targetPosition=-GRIPPER_CLOSED, force=2.5,
                                    maxVelocity=1.5)
            p.stepSimulation()

    # Angles to sweep through — include all diagnostically interesting points
    sweep_angles_deg = [-120, -97, -60, -30, 0, 20, 30, 45, 60, 90, 120]
    THRESH_DEG = 30   # _J4_TILT_THRESH in degrees (0.52 rad ≈ 30°)

    print("\n" + "="*60)
    print("  J4 WRIST SWEEP — visual reference")
    print("="*60)
    print(f"  Carry posture: J1={CARRY_JOINTS[1]:.1f}  J2={CARRY_JOINTS[2]:.1f}  J3={CARRY_JOINTS[3]:.1f}")
    print(f"  Penalty threshold: >{THRESH_DEG}°  (positive direction only)")
    print(f"  Joint limit: ±120°")
    print("="*60)
    print(f"  {'Angle':>8}   {'Note'}")
    print(f"  {'-'*8}   {'-'*40}")

    try:
        for deg in sweep_angles_deg:
            rad = np.radians(deg)
            if deg == -97:
                note = "<-- HOME / natural hang"
            elif deg == 0:
                note = "<-- where policy arrives after grasp"
            elif deg == THRESH_DEG:
                note = "<-- _J4_TILT_THRESH: penalty starts HERE"
            elif deg == 120:
                note = "<-- JOINT LIMIT: cube drops here"
            elif deg < 0:
                note = "(negative / downward tilt)"
            else:
                note = f"(+{deg - THRESH_DEG}° past threshold, cost={(deg-THRESH_DEG)*np.pi/180*15:.1f}/step)" if deg > THRESH_DEG else ""

            print(f"  {deg:>+7}°   {note}")

            # Transition smoothly to target
            current = p.getJointState(env.arm_id, 4)[0]
            steps = max(30, int(abs(np.degrees(rad - current)) * 0.8))
            for i in range(steps):
                interp = current + (rad - current) * (i + 1) / steps
                p.setJointMotorControl2(env.arm_id, 4, p.POSITION_CONTROL,
                                        targetPosition=interp, force=6.5,
                                        maxVelocity=1.5)
                for idx, pos in CARRY_JOINTS.items():
                    p.setJointMotorControl2(env.arm_id, idx, p.POSITION_CONTROL,
                                            targetPosition=pos, force=6.5,
                                            maxVelocity=1.5)
                p.stepSimulation()
                time.sleep(0.01)

            # Hold at this angle
            _hold_pose(rad)
            time.sleep(1.5)  # pause so you can inspect the pose

        print("\n  Sweep complete — window stays open. Ctrl-C to exit.\n")
        while True:
            p.stepSimulation()
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nDone.")

    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/ppo_grasp.zip")
    parser.add_argument("--task", type=str, default="grasp",
                        choices=["grasp", "pick_place"],
                        help="Task to diagnose (grasp or pick_place)")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--slow", type=float, default=0.04,
                        help="Delay per step in seconds (0.04 = ~25fps)")
    # Camera is always included (matches current training config).
    parser.add_argument("--manual", action="store_true",
                        help="Slider mode — no model, drag the arm yourself")
    parser.add_argument("--j4-test", action="store_true",
                        help="Sweep J4 through its full range to visually confirm carry angles")
    parser.add_argument("--warm-start", action="store_true",
                        help="Force warm-start (arm pre-positioned near cube)")
    args = parser.parse_args()

    if args.manual:
        run_manual(args)
    elif args.j4_test:
        run_j4_test(args)
    else:
        run_model(args)


if __name__ == "__main__":
    main()

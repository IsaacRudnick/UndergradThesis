# Visualize the arm with a target point in 3D space.
# Can run in manual mode (sliders) or load a trained model.
#
# Usage:
#   python see_arm_camera.py                     # Manual control
#   python see_arm_camera.py --model ./models/ppo_reach_final.zip  # Run trained model

import os
import sys
import time
import math
import argparse

import pybullet as p
import pybullet_data
import cv2
import numpy as np

from stable_baselines3 import PPO
from envs import ArmEnv, ReachTask, ReachHoldTask, GraspTask, PickAndPlaceTask
from envs.sensors import make_all_sensors
from envs.tasks import ReachTask as _ReachTask

# Task thresholds (metres) — pulled from tasks.py
SUCCESS_THRESHOLD = _ReachTask.SUCCESS_THRESHOLD  # 0.05 m
BONUS_THRESHOLD = _ReachTask.BONUS_THRESHOLD      # 0.10 m

# Workspace bounds (must match arm_env.py)
WORKSPACE_LOW = np.array([-0.15, -0.15, 0.10])
WORKSPACE_HIGH = np.array([0.15, 0.15, 0.35])

CAM_WIDTH = 256
CAM_HEIGHT = 256
CAM_FOV = 160.0
CAM_NEAR = 0.001
CAM_FAR = 2.0
CAM_LINK = 4
CAM_REL_POS = (0.05, 0.04, 0)
CAM_REL_ORN = p.getQuaternionFromEuler((math.pi / 2, 0.0, 0.0))
CAM_TARGET_REL = (0.01, 0.0, 0.0)

_prev_text_id = None
_target_visual_id = None


def show_camera_image(rgba, width, height):
    img = np.asarray(rgba, dtype=np.uint8).reshape((height, width, 4))[:, :, :3]
    cv2.imshow("EE Camera", cv2.resize(img[:, :, ::-1], (512, 512)))
    cv2.waitKey(1)


def gui_print(text, color=None):
    global _prev_text_id
    if color is None:
        color = [0, 0, 0]
    if _prev_text_id is not None:
        p.removeUserDebugItem(_prev_text_id)
    _prev_text_id = p.addUserDebugText(
        text, [0.2, 0, 0.1], textColorRGB=color, textSize=2
    )


def get_ee_position(body_id):
    """Get end effector position (link 4 + offset)."""
    link_state = p.getLinkState(body_id, 4, computeForwardKinematics=1)
    link_pos = np.array(link_state[4])
    link_orn = link_state[5]
    ee_pos, _ = p.multiplyTransforms(link_pos.tolist(), link_orn, (0.11, 0, 0), (0, 0, 0, 1))
    return np.array(ee_pos)


def spawn_target(position):
    """Create a red sphere at target position."""
    global _target_visual_id
    if _target_visual_id is not None:
        p.removeBody(_target_visual_id)
    visual_shape = p.createVisualShape(
        p.GEOM_SPHERE, radius=0.03, rgbaColor=[1, 0, 0, 0.7]
    )
    _target_visual_id = p.createMultiBody(
        baseVisualShapeIndex=visual_shape,
        basePosition=position.tolist(),
    )
    return _target_visual_id


def draw_workspace_bounds():
    """Draw the workspace bounding box."""
    corners = [
        (WORKSPACE_LOW[0], WORKSPACE_LOW[1], WORKSPACE_LOW[2]),
        (WORKSPACE_HIGH[0], WORKSPACE_LOW[1], WORKSPACE_LOW[2]),
        (WORKSPACE_HIGH[0], WORKSPACE_HIGH[1], WORKSPACE_LOW[2]),
        (WORKSPACE_LOW[0], WORKSPACE_HIGH[1], WORKSPACE_LOW[2]),
        (WORKSPACE_LOW[0], WORKSPACE_LOW[1], WORKSPACE_HIGH[2]),
        (WORKSPACE_HIGH[0], WORKSPACE_LOW[1], WORKSPACE_HIGH[2]),
        (WORKSPACE_HIGH[0], WORKSPACE_HIGH[1], WORKSPACE_HIGH[2]),
        (WORKSPACE_LOW[0], WORKSPACE_HIGH[1], WORKSPACE_HIGH[2]),
    ]
    for i in range(4):
        p.addUserDebugLine(corners[i], corners[(i+1) % 4], [0, 1, 0], lineWidth=2)
    for i in range(4):
        p.addUserDebugLine(corners[i+4], corners[(i+1) % 4 + 4], [0, 1, 0], lineWidth=2)
    for i in range(4):
        p.addUserDebugLine(corners[i], corners[i+4], [0, 1, 0], lineWidth=2)


def get_camera_pose(body_id, link_index=CAM_LINK):
    link_state = p.getLinkState(body_id, link_index, computeForwardKinematics=1)
    link_pos = link_state[4]
    link_orn = link_state[5]
    cam_pos_world, cam_orn_world = p.multiplyTransforms(
        link_pos, link_orn, CAM_REL_POS, CAM_REL_ORN
    )
    return cam_pos_world, cam_orn_world


def get_link_camera_image(body_id, link_index=CAM_LINK, width=CAM_WIDTH, height=CAM_HEIGHT, proj_matrix=None):
    cam_pos_world, cam_orn_world = get_camera_pose(body_id, link_index)
    cam_target_world, _ = p.multiplyTransforms(
        cam_pos_world, cam_orn_world, CAM_TARGET_REL, (0, 0, 0, 1)
    )
    up_world = p.rotateVector(cam_orn_world, (0, 0, -1))
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=cam_pos_world,
        cameraTargetPosition=cam_target_world,
        cameraUpVector=up_world
    )
    if proj_matrix is None:
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=CAM_FOV, aspect=float(width)/float(height),
            nearVal=CAM_NEAR, farVal=CAM_FAR
        )
    return p.getCameraImage(
        width=width, height=height,
        viewMatrix=view_matrix, projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
        flags=p.ER_NO_SEGMENTATION_MASK
    )


def run_manual(task_name="reach"):
    """Run in manual control mode with sliders."""
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    p.setRealTimeSimulation(0)

    p.loadURDF("plane.urdf")

    arm_path = os.path.join(os.path.dirname(__file__), "urdf_files", "urdf", "lss_arm_4dof.urdf")
    armId = p.loadURDF(
        arm_path, [0, 0, 0], p.getQuaternionFromEuler([0, 0, 0]),
        useFixedBase=True
    )

    draw_workspace_bounds()

    # Set up task scene
    task_cls = TASK_MAP.get(task_name)
    if task_cls and task_name != "reach":
        # Create a minimal env-like namespace so the task can spawn objects
        class _FakeEnv:
            arm_id = armId
            ARM_JOINT_INDICES = [1, 2, 3, 4]
            EE_LINK = 4
            FINGER_L_LINK = 5
            FINGER_R_LINK = 6
            GRIPPER_JOINT = 5
            MIMIC_JOINT = 6
            WORKSPACE_LOW = np.array([-0.15, -0.15, 0.10])
            WORKSPACE_HIGH = np.array([0.15, 0.15, 0.35])
            np_random = np.random.default_rng()
            object_id = None
            table_id = None
            table2_id = None
            target_pos = None
        fake_env = _FakeEnv()
        task = task_cls()
        task.setup_scene(fake_env)
        target_pos = fake_env.target_pos
    else:
        target_pos = np.random.uniform(WORKSPACE_LOW, WORKSPACE_HIGH)
        spawn_target(target_pos)

    print(f"\nManual Control Mode — task: {task_name}")
    print(f"Target position: {target_pos}")
    print(f"Workspace: {WORKSPACE_LOW} to {WORKSPACE_HIGH}")
    print(f"Success threshold: {SUCCESS_THRESHOLD} (scaled space)")
    print(f"Bonus threshold:   {BONUS_THRESHOLD} (scaled space)")
    print("\nControls:")
    print("  Sliders: Move joints")
    print("  'n': Reset environment")
    print("  'q': Quit")

    proj_matrix = p.computeProjectionMatrixFOV(
        fov=CAM_FOV, aspect=1.0, nearVal=CAM_NEAR, farVal=CAM_FAR
    )

    steps_per_second = 240
    camera_fps = 15
    min_distance_seen = float('inf')

    joint_limits = {
        1: (-3.14, 3.14),
        2: (-2.18, 2.18),
        3: (-1.92, 1.57),
        4: (-2.09, 2.09),
        5: (0, 1.57),
        6: (-1.57, 0),
    }

    def _make_sliders(old_params=None):
        """(Re)create sliders initialised to current joint positions."""
        if old_params:
            for _, pid in old_params:
                p.removeUserDebugItem(pid)
        params = []
        for jidx in [1, 2, 3, 4, 5, 6]:
            low, high = joint_limits[jidx]
            cur = p.getJointState(armId, jidx)[0]
            pid = p.addUserDebugParameter(f"Joint {jidx}", low, high, cur)
            params.append((jidx, pid))
        return params

    joint_params = _make_sliders()

    step = 0
    while True:
        for joint_idx, param_id in joint_params:
            target_angle = p.readUserDebugParameter(param_id)
            p.setJointMotorControl2(
                armId, joint_idx, p.POSITION_CONTROL,
                targetPosition=target_angle, force=10.0
            )

        p.stepSimulation()

        if step % (steps_per_second // camera_fps) == 0:
            width, height, rgba, _, _ = get_link_camera_image(
                armId, width=256, height=256, proj_matrix=proj_matrix
            )
            show_camera_image(rgba, width, height)

            ee_pos = get_ee_position(armId)
            distance = np.linalg.norm(ee_pos - target_pos)
            min_distance_seen = min(min_distance_seen, distance)

            # Compute reward (mirrors ReachTask.compute_reward)
            max_dist = 5.0
            reward = -distance / max_dist
            reward += max(0.0, 1.0 - distance / BONUS_THRESHOLD)
            if distance < SUCCESS_THRESHOLD:
                reward += 5.0

            # Color-coded status
            if distance < SUCCESS_THRESHOLD:
                status = "SUCCESS"
                color = [0, 1, 0]  # green
            elif distance < BONUS_THRESHOLD:
                status = "BONUS ZONE"
                color = [1, 0.6, 0]  # orange
            else:
                status = "---"
                color = [1, 0, 0]  # red

            gui_print(
                f"MANUAL DEBUG\n"
                f"Target: ({target_pos[0]:.1f}, {target_pos[1]:.1f}, {target_pos[2]:.1f})\n"
                f"EE:     ({ee_pos[0]:.1f}, {ee_pos[1]:.1f}, {ee_pos[2]:.1f})\n"
                f"Distance: {distance:.2f}  (min: {min_distance_seen:.2f})\n"
                f"Reward:   {reward:.3f}\n"
                f"Status:   {status}\n"
                f"Thresholds: success<{SUCCESS_THRESHOLD}  bonus<{BONUS_THRESHOLD}",
                color=color,
            )

        keys = p.getKeyboardEvents()
        if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
            break
        if ord('n') in keys and keys[ord('n')] & p.KEY_WAS_TRIGGERED:
            # Remove old scene objects
            if task_cls and task_name != "reach":
                if fake_env.object_id is not None:
                    p.removeBody(fake_env.object_id)
                if fake_env.table_id is not None:
                    p.removeBody(fake_env.table_id)
                if fake_env.table2_id is not None:
                    p.removeBody(fake_env.table2_id)
                # Reset arm joints to home pose
                home = {1: 0.0, 2: 1.0, 3: -0.7, 4: -1.7, 5: 0.0, 6: 0.0}
                for jidx, pos in home.items():
                    p.resetJointState(armId, jidx, pos)
                # Re-run task setup (spawns new table/cube, may do curriculum reset)
                task.setup_scene(fake_env)
                target_pos = fake_env.target_pos
            else:
                target_pos = np.random.uniform(WORKSPACE_LOW, WORKSPACE_HIGH)
                spawn_target(target_pos)
            # Recreate sliders from current joint positions
            joint_params = _make_sliders(joint_params)
            min_distance_seen = float('inf')
            print(f"\nReset — target: {target_pos}")

        step += 1
        time.sleep(1.0 / steps_per_second)

    cv2.destroyAllWindows()
    p.disconnect()


TASK_MAP = {
    "reach": ReachTask,
    "reach_hold": ReachHoldTask,
    "grasp": GraspTask,
    "pick_and_place": PickAndPlaceTask,
}


def run_with_model(model_path, task_name, episodes):
    """Load a trained model and run it in the GUI environment."""
    task_cls = TASK_MAP[task_name]

    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    sensor_list, maskable = make_all_sensors()
    raw_env = ArmEnv(task=task_cls(), render_mode="human",
                     sensors=sensor_list, maskable_sensors=maskable)

    # Wrap in VecEnv + VecNormalize to match training setup
    vec_env = DummyVecEnv([lambda: raw_env])
    vecnorm_path = model_path.replace(".zip", "_vecnorm.pkl")
    if os.path.exists(vecnorm_path):
        print(f"Loading VecNormalize stats from {vecnorm_path}")
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    else:
        print(f"Warning: {vecnorm_path} not found — running without normalization")

    model = PPO.load(model_path, env=vec_env)

    for ep in range(episodes):
        obs = vec_env.reset()
        done = False
        total_reward = 0.0
        step = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = vec_env.step(action)
            reward = rewards[0]
            info = infos[0]
            done = dones[0]
            total_reward += reward
            step += 1

            # Show camera feed with success overlay
            if raw_env.last_rgb is not None:
                display = cv2.resize(raw_env.last_rgb[:, :, ::-1], (512, 512))

                # Draw success/failure indicator on the camera feed
                is_success = info.get("is_success", False)
                distance = info.get("distance", info.get("dist_obj_target", None))
                lift = info.get("lift", None)

                if is_success:
                    label = "SUCCESS"
                    color = (0, 255, 0)  # green (BGR)
                else:
                    label = "NOT SUCCESS"
                    color = (0, 0, 255)  # red (BGR)

                # Status label with background
                cv2.rectangle(display, (0, 0), (512, 60), (0, 0, 0), -1)
                cv2.putText(display, label, (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

                # Extra metrics on a second line
                metrics = f"R={reward:+.2f}  Total={total_reward:.1f}"
                if distance is not None:
                    metrics += f"  Dist={distance:.3f}"
                if lift is not None:
                    metrics += f"  Lift={lift:.3f}"
                cv2.rectangle(display, (0, 60), (512, 95), (0, 0, 0), -1)
                cv2.putText(display, metrics, (10, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

                cv2.imshow("EE Camera", display)
                cv2.waitKey(1)

            # In-simulation HUD text (visible in the 3D PyBullet window)
            is_success = info.get("is_success", False)
            distance = info.get("distance", info.get("dist_obj_target", None))
            if is_success:
                hud_color = [0, 1, 0]
                hud_status = "SUCCESS"
            else:
                hud_color = [1, 0, 0]
                hud_status = "---"
            hud_lines = (
                f"Episode {ep+1}/{episodes}  Step {step}\n"
                f"Reward: {reward:+.3f}  Total: {total_reward:.1f}\n"
                f"Status: {hud_status}"
            )
            if distance is not None:
                hud_lines += f"\nDistance: {distance:.4f}"
            gui_print(hud_lines, color=hud_color)

            time.sleep(1.0 / 30)

        success = info.get("is_success", False)
        print(f"Episode {ep + 1}/{episodes}: "
              f"steps={step}, reward={total_reward:.2f}, "
              f"success={success}")

    cv2.destroyAllWindows()
    vec_env.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize arm with target reaching")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained model (.zip file)")
    parser.add_argument("--task", type=str, default="reach",
                        choices=["reach", "reach_hold", "grasp", "pick_and_place"],
                        help="Task to visualize")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to run (model mode)")
    # Camera sensors are always included (auto-detected from model is no longer needed).

    args = parser.parse_args()

    if args.model:
        if not os.path.exists(args.model):
            print(f"Error: Model file not found: {args.model}")
            sys.exit(1)
        run_with_model(args.model, args.task, args.episodes)
    else:
        run_manual(args.task)


if __name__ == "__main__":
    main()

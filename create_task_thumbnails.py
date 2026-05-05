"""Interactive thumbnail generator for each training task.

Usage:
    python create_task_thumbnails.py --task reach
    python create_task_thumbnails.py --task reach_hold
    python create_task_thumbnails.py --task grasp
    python create_task_thumbnails.py --task pick_and_place

Controls:
    Joint sliders  — position the arm
    'c'            — capture and save a 4K PNG from the current GUI viewpoint
    'n'            — reset / respawn scene with new random layout
    'q'            — quit

Saved files go to thumbnails/<task>_<NN>.png.

Overlays (visible in GUI, NOT baked into the saved image):
    Orange dashes  — grasp-point → reach target (or cube → destination for P&P)
    Cyan lines     — eye-in-hand camera FOV frustum (60° display angle)
"""

import argparse
import math
import os
import time

import cv2
import numpy as np
import pybullet as p
import pybullet_data

from envs import ReachTask, ReachHoldTask, GraspTask, PickAndPlaceTask

# ---------------------------------------------------------------------------
# Constants — must match arm_env.py / sensors.py
# ---------------------------------------------------------------------------
WORKSPACE_LOW  = np.array([-0.15, -0.15, 0.10])
WORKSPACE_HIGH = np.array([ 0.15,  0.15, 0.35])

ARM_JOINT_INDICES = [1, 2, 3, 4]
GRIPPER_JOINT = 5
MIMIC_JOINT   = 6
EE_LINK       = 4
FINGER_L_LINK = 5
FINGER_R_LINK = 6
EE_OFFSET     = (0.11, 0.0, 0.0)

CAM_REL_POS    = (0.05, 0.04, 0.0)
CAM_REL_ORN    = p.getQuaternionFromEuler((math.pi / 2, 0.0, 0.0))
CAM_TARGET_REL = (0.01, 0.0, 0.0)
CAM_FOV        = 160.0   # actual camera FOV (fisheye)
DISPLAY_FOV    = 60.0    # display angle for the frustum cone (60° looks clean)
CAM_NEAR       = 0.001
CAM_FAR        = 2.0

HOME_JOINTS = {1: 0.0, 2: 1.0, 3: -0.7, 4: -1.7, 5: 0.0, 6: 0.0}

JOINT_LIMITS = {
    1: (-3.14159, 3.14159),
    2: (-2.18166, 2.18166),
    3: (-1.91986, 1.57080),
    4: (-2.09440, 2.09440),
    5: (0.0, 1.5708),
}

CAPTURE_RES     = (3840, 2160)  # 4K UHD, scene view
CAM_CAPTURE_SIZE = 2160         # square (matches arm camera 1:1 aspect)

TASK_MAP = {
    "reach":          ReachTask,
    "reach_hold":     ReachHoldTask,
    "grasp":          GraspTask,
    "pick_and_place": PickAndPlaceTask,
}


# ---------------------------------------------------------------------------
# FakeEnv — minimal env-like namespace the task's setup_scene/post_settle need
# ---------------------------------------------------------------------------
class FakeEnv:
    def __init__(self, arm_id: int):
        self.arm_id       = arm_id
        self.object_id: int | None = None
        self.table_id:  int | None = None
        self.table2_id: int | None = None
        self.target_pos = None
        self.np_random  = np.random.default_rng()

        # Constants tasks read from the env
        self.ARM_JOINT_INDICES = ARM_JOINT_INDICES
        self.EE_LINK           = EE_LINK
        self.FINGER_L_LINK     = FINGER_L_LINK
        self.FINGER_R_LINK     = FINGER_R_LINK
        self.GRIPPER_JOINT     = GRIPPER_JOINT
        self.MIMIC_JOINT       = MIMIC_JOINT
        self.WORKSPACE_LOW     = WORKSPACE_LOW
        self.WORKSPACE_HIGH    = WORKSPACE_HIGH

    @property
    def grasp_point(self) -> np.ndarray:
        tip = (0.03, 0.0, 0.0)
        ls = p.getLinkState(self.arm_id, FINGER_L_LINK, computeForwardKinematics=1)
        rs = p.getLinkState(self.arm_id, FINGER_R_LINK, computeForwardKinematics=1)
        lt, _ = p.multiplyTransforms(list(ls[4]), ls[5], tip, (0, 0, 0, 1))
        rt, _ = p.multiplyTransforms(list(rs[4]), rs[5], tip, (0, 0, 0, 1))
        return (np.array(lt) + np.array(rt)) / 2.0

    @property
    def ee_pos(self) -> np.ndarray:
        state = p.getLinkState(self.arm_id, EE_LINK, computeForwardKinematics=1)
        ee, _ = p.multiplyTransforms(list(state[4]), state[5], EE_OFFSET, (0, 0, 0, 1))
        return np.array(ee)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def get_cam_world_frame(arm_id: int):
    """Return (cam_pos, forward, right, up) in world space for the EE camera."""
    state = p.getLinkState(arm_id, EE_LINK, computeForwardKinematics=1)
    cam_pos, cam_orn = p.multiplyTransforms(state[4], state[5], CAM_REL_POS, CAM_REL_ORN)
    cam_tgt, _       = p.multiplyTransforms(cam_pos, cam_orn, CAM_TARGET_REL, (0, 0, 0, 1))

    cam_pos = np.array(cam_pos)
    fwd     = np.array(cam_tgt) - cam_pos
    fwd    /= np.linalg.norm(fwd)
    up_raw  = np.array(p.rotateVector(cam_orn, (0, 0, -1)))
    right   = np.cross(fwd, up_raw)
    right  /= np.linalg.norm(right)
    up      = np.cross(right, fwd)
    return cam_pos, fwd, right, up


# ---------------------------------------------------------------------------
# Debug overlay drawing  (auto-expire via lifeTime so no ID tracking needed)
# ---------------------------------------------------------------------------
_OVERLAY_LIFE = 0.15  # seconds — just long enough to persist one frame at 60 Hz


def draw_dashed_line(a: np.ndarray, b: np.ndarray, color, n_dashes=14):
    for i in range(n_dashes):
        t0 = (2 * i)     / (2 * n_dashes)
        t1 = (2 * i + 1) / (2 * n_dashes)
        if t1 > 1.0:
            break
        p.addUserDebugLine(
            (a + t0 * (b - a)).tolist(),
            (a + t1 * (b - a)).tolist(),
            color, lineWidth=2, lifeTime=_OVERLAY_LIFE,
        )


def draw_fov_cone(arm_id: int, far: float = 0.18):
    """Draw eye-in-hand camera FOV cone using DISPLAY_FOV for visual clarity."""
    cam_pos, fwd, right, up = get_cam_world_frame(arm_id)
    half_tan = math.tan(math.radians(DISPLAY_FOV / 2.0))

    corners = [
        cam_pos + far * (fwd + half_tan * right  + half_tan * up),
        cam_pos + far * (fwd - half_tan * right  + half_tan * up),
        cam_pos + far * (fwd - half_tan * right  - half_tan * up),
        cam_pos + far * (fwd + half_tan * right  - half_tan * up),
    ]
    color = [0.2, 0.85, 1.0]
    for i in range(4):
        p.addUserDebugLine(cam_pos.tolist(), corners[i].tolist(),
                           color, lineWidth=1, lifeTime=_OVERLAY_LIFE)
        p.addUserDebugLine(corners[i].tolist(), corners[(i + 1) % 4].tolist(),
                           color, lineWidth=1, lifeTime=_OVERLAY_LIFE)

    # Label the camera position
    p.addUserDebugText("cam (160° FOV)", (cam_pos + up * 0.03).tolist(),
                       textColorRGB=color, textSize=0.8, lifeTime=_OVERLAY_LIFE)


def draw_workspace_box():
    """Draw the workspace bounding box (persistent — called once)."""
    lo, hi = WORKSPACE_LOW, WORKSPACE_HIGH
    verts = [
        (lo[0], lo[1], lo[2]), (hi[0], lo[1], lo[2]),
        (hi[0], hi[1], lo[2]), (lo[0], hi[1], lo[2]),
        (lo[0], lo[1], hi[2]), (hi[0], lo[1], hi[2]),
        (hi[0], hi[1], hi[2]), (lo[0], hi[1], hi[2]),
    ]
    c = [0.3, 0.9, 0.3]
    w = 1
    for i in range(4):
        p.addUserDebugLine(verts[i], verts[(i + 1) % 4], c, w)
        p.addUserDebugLine(verts[i + 4], verts[(i + 1) % 4 + 4], c, w)
        p.addUserDebugLine(verts[i], verts[i + 4], c, w)


# ---------------------------------------------------------------------------
# Scene management
# ---------------------------------------------------------------------------

def _safe_remove(body_id: int | None):
    if body_id is not None:
        try:
            p.removeBody(body_id)
        except Exception:
            pass


def spawn_scene(arm_id: int, env: FakeEnv, task) -> None:
    """Remove previous objects, reset arm, and re-run task.setup_scene."""
    _safe_remove(env.object_id)
    _safe_remove(env.table_id)
    _safe_remove(env.table2_id)
    env.object_id = env.table_id = env.table2_id = None
    env.target_pos = None

    for idx, pos in HOME_JOINTS.items():
        p.resetJointState(arm_id, idx, pos)

    task.setup_scene(env)

    # Mirror arm_env.py physics exactly so grasping is feasible
    for idx in ARM_JOINT_INDICES:
        p.changeDynamics(arm_id, idx, jointDamping=0.5)
    for link_idx in [FINGER_L_LINK, FINGER_R_LINK]:
        p.changeDynamics(arm_id, link_idx,
                         lateralFriction=2.0, spinningFriction=0.3, rollingFriction=0.05)
    if env.object_id is not None:
        p.changeDynamics(env.object_id, -1,
                         lateralFriction=2.0, spinningFriction=0.3, rollingFriction=0.05)
        for link_idx in range(p.getNumJoints(arm_id)):
            if link_idx not in (FINGER_L_LINK, FINGER_R_LINK):
                p.setCollisionFilterPair(arm_id, env.object_id, link_idx, -1,
                                         enableCollision=0)

    # Physics settle with rendering off so the GUI doesn't stall
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    for _ in range(50):
        p.stepSimulation()
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    task.post_settle(env)


# ---------------------------------------------------------------------------
# 4K capture
# ---------------------------------------------------------------------------

def _render_tiny(view_matrix, proj_matrix, width, height) -> np.ndarray:
    """Shared TinyRenderer call; returns H×W×3 uint8 RGB array."""
    _, _, rgba, _, _ = p.getCameraImage(
        width=width, height=height,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_TINY_RENDERER,
        shadow=1,
        lightDirection=[1.0, 1.5, 2.0],
        lightColor=[1.0, 0.98, 0.95],
        lightDistance=3.0,
        lightAmbientCoeff=0.45,
        lightDiffuseCoeff=0.45,
        lightSpecularCoeff=0.10,
    )
    return np.array(rgba, dtype=np.uint8).reshape((height, width, 4))[:, :, :3]


def capture_4k(arm_id: int, task_name: str, shot_index: int) -> tuple[str, str]:
    os.makedirs("thumbnails", exist_ok=True)

    # ---- Scene view (4K, 16:9) ----------------------------------------
    cam_info = p.getDebugVisualizerCamera()
    W, H = CAPTURE_RES
    scene_view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=cam_info[11],
        distance=cam_info[10],
        yaw=cam_info[8],
        pitch=cam_info[9],
        roll=0,
        upAxisIndex=2,
    )
    scene_proj = p.computeProjectionMatrixFOV(
        fov=50, aspect=W / H, nearVal=0.01, farVal=10.0,
    )
    print(f"Rendering scene {W}×{H}...")
    scene_img = _render_tiny(scene_view, scene_proj, W, H)
    scene_file = f"thumbnails/{task_name}_{shot_index:02d}.png"
    cv2.imwrite(scene_file, scene_img[:, :, ::-1])
    print(f"Saved: {scene_file}")

    # ---- Eye-in-hand camera view (square, 1:1 aspect) ------------------
    S = CAM_CAPTURE_SIZE
    ee_state   = p.getLinkState(arm_id, EE_LINK, computeForwardKinematics=1)
    cam_pos, cam_orn = p.multiplyTransforms(
        ee_state[4], ee_state[5], CAM_REL_POS, CAM_REL_ORN,
    )
    cam_tgt, _ = p.multiplyTransforms(cam_pos, cam_orn, CAM_TARGET_REL, (0, 0, 0, 1))
    cam_up     = p.rotateVector(cam_orn, (0, 0, -1))
    ee_view = p.computeViewMatrix(cam_pos, cam_tgt, cam_up)
    ee_proj = p.computeProjectionMatrixFOV(
        fov=CAM_FOV, aspect=1.0, nearVal=CAM_NEAR, farVal=CAM_FAR,
    )
    print(f"Rendering eye-in-hand {S}×{S}...")
    cam_img  = _render_tiny(ee_view, ee_proj, S, S)
    cam_file = f"thumbnails/{task_name}_{shot_index:02d}_cam.png"
    cv2.imwrite(cam_file, cam_img[:, :, ::-1])
    print(f"Saved: {cam_file}")

    return scene_file, cam_file


# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------
_hud_id = None


def update_hud(text, color=None):
    global _hud_id
    if color is None:
        color = [0.05, 0.05, 0.05]
    if _hud_id is not None:
        try:
            p.removeUserDebugItem(_hud_id)
        except Exception:
            pass
    _hud_id = p.addUserDebugText(
        text, [0.28, 0.0, 0.38],
        textColorRGB=color, textSize=1.4,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Interactive task thumbnail generator")
    parser.add_argument("--task", default="reach", choices=list(TASK_MAP),
                        help="Which task to visualize")
    args = parser.parse_args()

    task_name = args.task
    task_cls  = TASK_MAP[task_name]

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 240)
    p.setRealTimeSimulation(0)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)  # shadows only in TinyRenderer capture

    p.loadURDF("plane.urdf")

    arm_path = os.path.join(os.path.dirname(__file__), "urdf_files", "urdf", "lss_arm_4dof.urdf")
    arm_id = p.loadURDF(arm_path, [0, 0, 0], p.getQuaternionFromEuler([0, 0, 0]),
                        useFixedBase=True)

    for idx, pos in HOME_JOINTS.items():
        p.resetJointState(arm_id, idx, pos)

    task = task_cls()
    # Disable warm-starts so the thumbnail scene starts with a clean table layout
    if hasattr(task, '_WARM_START_FRAC'):
        task._WARM_START_FRAC = 0.0

    env = FakeEnv(arm_id)

    draw_workspace_box()
    spawn_scene(arm_id, env, task)

    # ---- Sliders (joints 1–5; joint 6 mimics joint 5 automatically) ----
    # Created once — never recreated, because p.removeUserDebugItem is
    # unreliable for parameters and leaves duplicates behind.
    sliders = []
    for jidx in [1, 2, 3, 4, 5]:
        lo, hi = JOINT_LIMITS[jidx]
        cur = p.getJointState(arm_id, jidx)[0]
        sid = p.addUserDebugParameter(f"Joint {jidx}", lo, hi, cur)
        sliders.append((jidx, sid))
    shot_index = 0

    update_hud(
        f"Task: {task_name}\n"
        " 'c'  capture 4K PNG\n"
        " 'n'  new random scene\n"
        " 'q'  quit"
    )

    print(f"\nTask: {task_name}")
    print("Controls: sliders → joints | 'c' → capture 4K | 'n' → reset | 'q' → quit\n")

    SIM_HZ  = 240
    step    = 0

    while True:
        for jidx, sid in sliders:
            val = p.readUserDebugParameter(sid)
            if jidx in ARM_JOINT_INDICES:
                # High force + velocity: fast enough to feel responsive,
                # physically simulated so the cube isn't hit by teleporting links.
                p.setJointMotorControl2(arm_id, jidx, p.POSITION_CONTROL,
                                        targetPosition=val, force=50.0, maxVelocity=5.0)
            else:
                # Gripper: training forces so it can actually squeeze the cube
                p.setJointMotorControl2(arm_id, GRIPPER_JOINT, p.POSITION_CONTROL,
                                        targetPosition=val, force=5.0, maxVelocity=1.5)
                p.setJointMotorControl2(arm_id, MIMIC_JOINT, p.POSITION_CONTROL,
                                        targetPosition=-val, force=2.5, maxVelocity=1.5)

        p.stepSimulation()

        # Overlays (drawn every 8 steps ≈ 30 Hz; auto-expire so no ID tracking needed)
        if step % 8 == 0:
            grasp_pt = env.grasp_point

            if env.target_pos is not None:
                # Line: grasp point → reach target (or cube for grasp tasks)
                draw_dashed_line(grasp_pt, env.target_pos, [1.0, 0.45, 0.0])

            # For pick-and-place also show cube → destination
            if (task_name == "pick_and_place"
                    and hasattr(task, '_destination_pos')
                    and task._destination_pos is not None
                    and env.object_id is not None):
                obj_pos, _ = p.getBasePositionAndOrientation(env.object_id)
                draw_dashed_line(np.array(obj_pos), task._destination_pos,
                                 [0.1, 1.0, 0.4])

            draw_fov_cone(arm_id)

        # Keyboard events
        keys = p.getKeyboardEvents()

        if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
            break

        if ord('n') in keys and keys[ord('n')] & p.KEY_WAS_TRIGGERED:
            spawn_scene(arm_id, env, task)
            update_hud(
                f"Task: {task_name}  (reset)\n"
                " 'c'  capture 4K PNG\n"
                " 'n'  new random scene\n"
                " 'q'  quit"
            )
            print("Scene reset.")

        if ord('c') in keys and keys[ord('c')] & p.KEY_WAS_TRIGGERED:
            scene_f, cam_f = capture_4k(arm_id, task_name, shot_index)
            shot_index += 1
            update_hud(f"Saved: {os.path.basename(scene_f)}\n"
                       f"       {os.path.basename(cam_f)}\n"
                       " 'c'  capture 4K PNG\n"
                       " 'n'  new random scene\n"
                       " 'q'  quit",
                       color=[0.0, 0.7, 0.1])

        step += 1
        time.sleep(1.0 / SIM_HZ)

    p.disconnect()


if __name__ == "__main__":
    main()

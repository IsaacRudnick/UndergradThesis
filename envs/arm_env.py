"""Base Gymnasium environment for the LSS 4-DOF arm.

Combines a composable sensor suite with pluggable Task objects so the
same environment class works for reach, grasp, *and* pick-and-place.
"""

import math
import os

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

from .sensors import make_all_sensors, JOINT_LIMITS, ImageSensor
from .tasks import Task, ReachTask

# ---------------------------------------------------------------------------
# Constants (all in real-world units — metres, radians)
# ---------------------------------------------------------------------------
WORKSPACE_LOW = np.array([-0.15, -0.15, 0.10])
WORKSPACE_HIGH = np.array([0.15, 0.15, 0.35])

SIM_FREQ = 240  # Hz
SUBSTEPS_PER_ACTION = 10

# Realistic velocity limit for Lynxmotion Smart Servos under load
# (URDF says 6.283 rad/s which is the no-load spec; real-world is ~1.5)
REALISTIC_MAX_VEL = 1.5  # rad/s

# Higher damping to prevent instant snapping to targets
REALISTIC_DAMPING = 0.5  # applied programmatically in PyBullet

ARM_JOINT_INDICES = [1, 2, 3, 4]
GRIPPER_JOINT = 5
MIMIC_JOINT = 6
EE_LINK = 4
FINGER_L_LINK = 5
FINGER_R_LINK = 6
EE_OFFSET = (0.11, 0.0, 0.0)  # 0.11 m

# Map for action → joint position conversion
_JOINT_LIMITS_LIST = [JOINT_LIMITS[i] for i in ARM_JOINT_INDICES]
_GRIPPER_LO, _GRIPPER_HI = JOINT_LIMITS[5]

# Camera settings (stored for visualisation / future CNN use)
CAM_WIDTH = 256
CAM_HEIGHT = 256
CAM_FOV = 160.0
CAM_NEAR = 0.001
CAM_FAR = 2.0
CAM_REL_POS = (0.05, 0.04, 0.0)
CAM_REL_ORN = p.getQuaternionFromEuler((math.pi / 2, 0.0, 0.0))
CAM_TARGET_REL = (0.01, 0.0, 0.0)

# Initial joint positions (home pose)
HOME_JOINTS = {1: 0.0, 2: 1.0, 3: -0.7, 4: -1.7, 5: 0.0, 6: 0.0}


class ArmEnv(gym.Env):
    """Gymnasium environment wrapping the PyBullet LSS-4DOF arm."""

    metadata = {"render_modes": ["human"], "render_fps": 24}

    # Expose constants so sensors / tasks can reference them via ``env.*``
    WORKSPACE_LOW = WORKSPACE_LOW
    WORKSPACE_HIGH = WORKSPACE_HIGH
    EE_LINK = EE_LINK
    FINGER_L_LINK = FINGER_L_LINK
    FINGER_R_LINK = FINGER_R_LINK
    GRIPPER_JOINT = GRIPPER_JOINT
    MIMIC_JOINT = MIMIC_JOINT

    def __init__(
        self,
        task: Task | None = None,
        render_mode: str | None = None,
        sensors=None,
        maskable_sensors=None,
        curriculum: str | None = None,
    ):
        super().__init__()
        self.task = task or ReachTask()
        self.render_mode = render_mode
        # Default to make_all_sensors when no explicit sensor list is given.
        if sensors is None:
            sensors, maskable_sensors = make_all_sensors()
        self.sensors = sensors
        self._maskable_sensors = maskable_sensors or []
        self._curriculum = curriculum  # None | 'random'

        # Separate vector and image sensors
        self._vector_sensors = [s for s in self.sensors
                                if not getattr(s, "is_image", False)]
        self._image_sensors = [s for s in self.sensors
                               if getattr(s, "is_image", False)]

        # Build observation space — always a Dict so the obs structure is
        # identical across all curriculum modes and camera-active states.
        vec_size = sum(s.get_obs_size() for s in self._vector_sensors)
        obs_spaces = {
            "vector": spaces.Box(
                low=-1.0, high=1.0, shape=(vec_size,), dtype=np.float32
            ),
        }
        for img_sensor in self._image_sensors:
            shape = img_sensor.get_obs_shape()
            if img_sensor.name == "camera_rgb":
                obs_spaces[img_sensor.name] = spaces.Box(
                    low=0, high=255, shape=shape, dtype=np.uint8
                )
            else:
                obs_spaces[img_sensor.name] = spaces.Box(
                    low=0.0, high=1.0, shape=shape, dtype=np.float32
                )
        self.observation_space = spaces.Dict(obs_spaces)

        # 4 arm joints + 1 gripper (all normalised to [-1, 1])
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
        )

        # State that gets set during reset
        self.arm_id: int | None = None
        self.object_id: int | None = None
        self.table_id: int | None = None
        self.table2_id: int | None = None
        self.target_pos = np.zeros(3)
        self._physics_client: int | None = None
        self._step_count = 0
        self._prev_action: np.ndarray = np.zeros(5, dtype=np.float32)
        self._prev_ee_pos: np.ndarray = np.zeros(3, dtype=np.float64)

        # Image buffers for visualisation / future CNN
        self.last_rgb: np.ndarray | None = None
        self.last_depth: np.ndarray | None = None
        self._proj_matrix = None

        # Debug visualisation (GUI only)
        self._grasp_point_vis_id: int | None = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @property
    def ee_pos(self) -> np.ndarray:
        link_state = p.getLinkState(self.arm_id, EE_LINK,
                                    computeForwardKinematics=1)
        link_pos = np.array(link_state[4])
        link_orn = link_state[5]
        ee, _ = p.multiplyTransforms(
            link_pos.tolist(), link_orn, EE_OFFSET, (0, 0, 0, 1)
        )
        return np.array(ee)

    @property
    def ee_velocity(self) -> np.ndarray:
        """EE displacement since the previous step (metres)."""
        return self.ee_pos - self._prev_ee_pos

    @property
    def grasp_point(self) -> np.ndarray:
        """Midpoint between the two finger contact surfaces.

        Computed by taking each finger's link frame (joint origin in world
        space) and moving ~3 cm along its local X axis to reach the contact
        zone, then averaging the two positions.
        """
        finger_tip = (0.03, 0.0, 0.0)  # offset along finger to contact zone
        l_state = p.getLinkState(self.arm_id, FINGER_L_LINK,
                                 computeForwardKinematics=1)
        r_state = p.getLinkState(self.arm_id, FINGER_R_LINK,
                                 computeForwardKinematics=1)
        l_pos, l_orn = l_state[4], l_state[5]
        r_pos, r_orn = r_state[4], r_state[5]
        l_tip, _ = p.multiplyTransforms(list(l_pos), l_orn,
                                        finger_tip, (0, 0, 0, 1))
        r_tip, _ = p.multiplyTransforms(list(r_pos), r_orn,
                                        finger_tip, (0, 0, 0, 1))
        return (np.array(l_tip) + np.array(r_tip)) / 2.0

    @property
    def object_pos(self) -> np.ndarray:
        if self.object_id is None:
            return np.zeros(3)
        pos, _ = p.getBasePositionAndOrientation(self.object_id)
        return np.array(pos)

    def _action_to_joints(self, action: np.ndarray):
        """Map [-1,1] actions to joint position targets."""
        arm_targets = []
        for i, idx in enumerate(ARM_JOINT_INDICES):
            lo, hi = JOINT_LIMITS[idx]
            mid = (hi + lo) / 2.0
            rng = (hi - lo) / 2.0
            arm_targets.append(mid + action[i] * rng)

        # Gripper: action[4] in [-1,1] → joint 5 in [lo, hi]
        grip_mid = (_GRIPPER_HI + _GRIPPER_LO) / 2.0
        grip_rng = (_GRIPPER_HI - _GRIPPER_LO) / 2.0
        grip_target = grip_mid + action[4] * grip_rng
        return arm_targets, grip_target

    def _get_obs(self):
        vec = np.concatenate([s.observe(self) for s in self._vector_sensors])
        # Sanitize before the observation reaches VecNormalize.
        # np.clip(nan, lo, hi) == nan, so clip_obs does NOT protect against
        # NaN/inf from a PyBullet physics explosion.  nan_to_num converts them
        # to finite values (0 for nan, ±1 for inf) so no NaN ever reaches the
        # network and corrupts weights permanently.
        vec = np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=-1.0)
        obs = {"vector": vec}
        for img_sensor in self._image_sensors:
            img = img_sensor.observe(self)
            if img.dtype == np.float32:
                img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
            obs[img_sensor.name] = img
        return obs

    def _capture_camera(self):
        """Render eye-in-hand camera images (RGB + depth)."""
        link_state = p.getLinkState(self.arm_id, EE_LINK,
                                    computeForwardKinematics=1)
        link_pos, link_orn = link_state[4], link_state[5]
        cam_pos, cam_orn = p.multiplyTransforms(
            link_pos, link_orn, CAM_REL_POS, CAM_REL_ORN
        )
        cam_target, _ = p.multiplyTransforms(
            cam_pos, cam_orn, CAM_TARGET_REL, (0, 0, 0, 1)
        )
        up = p.rotateVector(cam_orn, (0, 0, -1))
        view_matrix = p.computeViewMatrix(cam_pos, cam_target, up)

        if self._proj_matrix is None:
            self._proj_matrix = p.computeProjectionMatrixFOV(
                fov=CAM_FOV, aspect=1.0,
                nearVal=CAM_NEAR, farVal=CAM_FAR,
            )

        # Hide GUI-only debug sphere so it doesn't pollute camera images
        # (it doesn't exist in DIRECT mode where training happens).
        if self._grasp_point_vis_id is not None:
            p.changeVisualShape(self._grasp_point_vis_id, -1,
                                rgbaColor=[0, 0, 0, 0])

        # Explicit lighting parameters ensure identical TinyRenderer output
        # in both DIRECT and GUI modes (GUI shadow/lighting settings bleed
        # into TinyRenderer by default — known PyBullet bug).
        _, _, rgba, depth, _ = p.getCameraImage(
            width=CAM_WIDTH, height=CAM_HEIGHT,
            viewMatrix=view_matrix,
            projectionMatrix=self._proj_matrix,
            renderer=p.ER_TINY_RENDERER,
            shadow=0,
            lightDirection=[1, 1, 1],
            lightColor=[1.0, 1.0, 1.0],
            lightDistance=1.0,
            lightAmbientCoeff=0.6,
            lightDiffuseCoeff=0.35,
            lightSpecularCoeff=0.05,
        )

        # Restore debug sphere visibility
        if self._grasp_point_vis_id is not None:
            p.changeVisualShape(self._grasp_point_vis_id, -1,
                                rgbaColor=[0, 1, 0, 0.7])
        self.last_rgb = np.asarray(rgba, dtype=np.uint8).reshape(
            (CAM_HEIGHT, CAM_WIDTH, 4)
        )[:, :, :3]
        self.last_depth = np.asarray(depth, dtype=np.float32).reshape(
            (CAM_HEIGHT, CAM_WIDTH)
        )

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # (Re)create physics world
        if self._physics_client is not None:
            p.disconnect(self._physics_client)
            # Body IDs from the old client are now invalid
            if hasattr(self.task, '_target_vis_id'):
                self.task._target_vis_id = None
            if hasattr(self.task, '_hover_vis_id'):
                self.task._hover_vis_id = None
            self._grasp_point_vis_id = None

        if self.render_mode == "human":
            self._physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            # Render only on stepSimulation(), not continuously in background
            p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
            # Disable shadows — they bleed into TinyRenderer camera images
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        else:
            self._physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / SIM_FREQ)
        # Must come AFTER setTimeStep (which can reset the flag)
        p.setRealTimeSimulation(0)

        # Ground plane
        p.loadURDF("plane.urdf")

        # Arm
        arm_urdf = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "urdf_files", "urdf", "lss_arm_4dof.urdf",
        )
        self.arm_id = p.loadURDF(
            arm_urdf, [0, 0, 0],
            p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
        )

        # Set home pose
        for idx, pos in HOME_JOINTS.items():
            p.resetJointState(self.arm_id, idx, pos)

        # Override damping to realistic values (URDF has 0.01)
        for idx in ARM_JOINT_INDICES:
            p.changeDynamics(self.arm_id, idx, jointDamping=REALISTIC_DAMPING)

        # Clear previous objects
        self.object_id = None
        self.table_id = None
        self.table2_id = None

        # Random curriculum: independently drop each maskable sensor with p=0.2.
        # ProprioceptiveSensor, TargetPositionSensor, and SensorMaskSensor are
        # always on (not in _maskable_sensors) so they are never dropped.
        _RAND_DROP_PROB = 0.2
        if self._curriculum == "random" and self._maskable_sensors:
            for s in self._maskable_sensors:
                s.is_active = bool(self.np_random.random() > _RAND_DROP_PROB)

        # Task-specific scene setup
        self.task.setup_scene(self)

        # Explicit friction for fingers and object (PyBullet ignores Gazebo mu tags)
        # High lateral friction (rubber pads) + rolling/spinning friction to
        # prevent the cube sliding or rotating out of the grasp.
        # NOTE: avoid contactStiffness/contactDamping overrides — high
        # stiffness creates large repulsive impulses that eject the cube.
        # lateralFriction=2.0: doubled from 1.0 so the cube tolerates more
        # grasp-point drift during J1 rotation without slipping.
        # spinningFriction=0.3: cube can spin/pivot during sweep — higher value
        # prevents it pivoting out of the fingers.
        for link_idx in [FINGER_L_LINK, FINGER_R_LINK]:
            p.changeDynamics(self.arm_id, link_idx,
                             lateralFriction=2.0,
                             spinningFriction=0.3,
                             rollingFriction=0.05)
        if self.object_id is not None:
            p.changeDynamics(self.object_id, -1,
                             lateralFriction=2.0,
                             spinningFriction=0.3,
                             rollingFriction=0.05)
            # Disable collision between all non-finger arm links and the
            # object.  Only the finger links (5, 6) should interact with
            # the cube — the forearm (link 3) and wrist (link 4) meshes
            # extend into the grasp zone and knock the cube away before
            # the fingers can close around it.
            for link_idx in range(p.getNumJoints(self.arm_id)):
                if link_idx not in (FINGER_L_LINK, FINGER_R_LINK):
                    p.setCollisionFilterPair(
                        self.arm_id, self.object_id,
                        link_idx, -1, enableCollision=0,
                    )

        # Let physics settle (rendering off for determinism)
        if self.render_mode == "human":
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        for _ in range(50):
            p.stepSimulation()
        if self.render_mode == "human":
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        # Post-settle hook (e.g. reposition cube to actual grasp point)
        self.task.post_settle(self)

        # Reset sensors
        for s in self.sensors:
            s.reset(self)

        self._step_count = 0
        self._prev_action = np.zeros(5, dtype=np.float32)
        self._prev_ee_pos = self.ee_pos.copy()
        if any(s.is_active for s in self._image_sensors) or self.render_mode == "human":
            self._capture_camera()
        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        arm_targets, grip_target = self._action_to_joints(action)

        # Apply arm joint targets with realistic max velocity
        for target, idx in zip(arm_targets, ARM_JOINT_INDICES):
            p.setJointMotorControl2(
                self.arm_id, idx, p.POSITION_CONTROL,
                targetPosition=target, force=6.5,
                maxVelocity=REALISTIC_MAX_VEL,
            )

        # Gripper (with mimic enforcement)
        # force=5.0/2.5: raised from 3.0/1.45 so the gripper can resist the
        # lateral inertial forces on the cube during J1 rotation without the
        # finger angle creeping above _GRIPPER_CLOSE_THRESH and breaking has_grasp.
        p.setJointMotorControl2(
            self.arm_id, GRIPPER_JOINT, p.POSITION_CONTROL,
            targetPosition=grip_target, force=5.0,
            maxVelocity=REALISTIC_MAX_VEL,
        )
        p.setJointMotorControl2(
            self.arm_id, MIMIC_JOINT, p.POSITION_CONTROL,
            targetPosition=-grip_target, force=2.5,
            maxVelocity=REALISTIC_MAX_VEL,
        )

        # Physics sub-stepping (disable rendering during substeps so the
        # GUI thread cannot interfere — ensures DIRECT/GUI determinism)
        if self.render_mode == "human":
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        for _ in range(SUBSTEPS_PER_ACTION):
            p.stepSimulation()
        if self.render_mode == "human":
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        # Update grasp-point debug sphere (GUI only)
        if self.render_mode == "human":
            gp = self.grasp_point.tolist()
            if self._grasp_point_vis_id is None:
                vs = p.createVisualShape(
                    p.GEOM_SPHERE, radius=0.02,
                    rgbaColor=[0, 1, 0, 0.7],
                )
                self._grasp_point_vis_id = p.createMultiBody(
                    baseVisualShapeIndex=vs, basePosition=gp,
                )
            else:
                p.resetBasePositionAndOrientation(
                    self._grasp_point_vis_id, gp, (0, 0, 0, 1),
                )

        # Camera (only render when image sensors are needed)
        if any(s.is_active for s in self._image_sensors) or self.render_mode == "human":
            self._capture_camera()

        # Observations
        obs = self._get_obs()

        # Reward & termination
        reward = self.task.compute_reward(self, action)
        # Guard against a rare PyBullet physics explosion producing nan/inf
        # reward, which would corrupt VecNormalize's ret_rms permanently.
        if not np.isfinite(reward):
            reward = 0.0
        terminated, info = self.task.check_done(self)

        self._prev_ee_pos = self.ee_pos.copy()
        self._prev_action = action.copy()
        self._step_count += 1
        truncated = self._step_count >= self.task.max_episode_steps

        return obs, reward, terminated, truncated, info

    def close(self):
        if self._physics_client is not None:
            p.disconnect(self._physics_client)
            self._physics_client = None

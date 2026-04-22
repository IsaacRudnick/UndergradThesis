"""Composable sensor classes for the arm environment.

Each sensor implements a common interface so the environment can
auto-build the observation space from whichever sensors are active.
"""

import math
from abc import ABC, abstractmethod

import numpy as np
import pybullet as p


# ---------------------------------------------------------------------------
# Joint limits (un-scaled URDF values — positions are in radians)
# ---------------------------------------------------------------------------
JOINT_LIMITS = {
    1: (-math.pi, math.pi),
    2: (-2.1816615649929116, 2.1816615649929116),
    3: (-1.9198621771937625, 1.5707963267948966),
    4: (-2.0943951023931953, 2.0943951023931953),
    5: (0.0, 1.5707963267948966),
    6: (-1.5707963267948966, 0.0),
}

MAX_JOINT_VEL = 1.5  # rad/s — realistic max under load (URDF=6.283 is no-load)


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------
class Sensor(ABC):
    """Base interface for all vector sensors."""

    name: str = "base"
    is_image: bool = False
    is_active: bool = True

    @abstractmethod
    def get_obs_size(self) -> int:
        """Number of scalar values this sensor contributes."""

    def observe(self, env) -> np.ndarray:
        """Return a flat float32 array of length ``get_obs_size()``.

        When ``is_active`` is False returns zeros (zero-padding for curriculum
        modes where this sensor is temporarily disabled).
        """
        if not self.is_active:
            return np.zeros(self.get_obs_size(), dtype=np.float32)
        return self._observe(env)

    @abstractmethod
    def _observe(self, env) -> np.ndarray:
        """Sensor-specific observation logic.  Implement in subclasses."""

    def reset(self, env) -> None:
        """Called once at the start of each episode (optional override)."""


class ImageSensor(ABC):
    """Base interface for image-based sensors (CNN observations)."""

    name: str = "image_base"
    is_image: bool = True
    is_active: bool = True

    @abstractmethod
    def get_obs_shape(self) -> tuple[int, int, int]:
        """Return (H, W, C) shape of the image observation."""

    def observe(self, env) -> np.ndarray:
        """Return a uint8 or float32 array of shape ``get_obs_shape()``.

        When ``is_active`` is False returns zeros so the CNN sees a blank image.
        """
        if not self.is_active:
            return np.zeros(self.get_obs_shape(), dtype=np.float32)
        return self._observe(env)

    @abstractmethod
    def _observe(self, env) -> np.ndarray:
        """Sensor-specific observation logic.  Implement in subclasses."""

    def reset(self, env) -> None:
        """Called once at the start of each episode (optional override)."""


# ---------------------------------------------------------------------------
# Proprioceptive — joint positions + velocities  (12 values)
# ---------------------------------------------------------------------------
class ProprioceptiveSensor(Sensor):
    name = "proprioceptive"

    def __init__(self, joint_indices=None):
        self.joint_indices = joint_indices or [1, 2, 3, 4, 5, 6]

    def get_obs_size(self) -> int:
        return len(self.joint_indices) * 2  # pos + vel per joint

    def _observe(self, env) -> np.ndarray:
        positions = []
        velocities = []
        for idx in self.joint_indices:
            state = p.getJointState(env.arm_id, idx)
            pos, vel = state[0], state[1]
            lo, hi = JOINT_LIMITS[idx]
            # normalise position to [-1, 1]
            mid = (hi + lo) / 2.0
            rng = (hi - lo) / 2.0
            positions.append((pos - mid) / rng if rng > 0 else 0.0)
            # normalise velocity
            velocities.append(np.clip(vel / MAX_JOINT_VEL, -1.0, 1.0))
        return np.array(positions + velocities, dtype=np.float32)


# ---------------------------------------------------------------------------
# 1-D distance from EE to target  (1 value)
# ---------------------------------------------------------------------------
class DistanceSensor(Sensor):
    name = "distance"

    def __init__(self, max_dist: float = 0.5):
        self.max_dist = max_dist

    def get_obs_size(self) -> int:
        return 1

    def _observe(self, env) -> np.ndarray:
        # Use grasp point (between fingers) to match Phase 2 reward signal.
        # In Phase 1 this slightly changes the reference point but the MLP
        # heads are reset during transfer anyway.
        point = env.grasp_point
        dist = np.linalg.norm(point - env.target_pos)
        return np.array([np.clip(dist / self.max_dist, 0.0, 1.0)], dtype=np.float32)


# ---------------------------------------------------------------------------
# Ultrasonic — single forward ray from EE  (1 value)
# ---------------------------------------------------------------------------
class UltrasonicSensor(Sensor):
    name = "ultrasonic"

    def __init__(self, ray_length: float = 0.3):
        self.ray_length = ray_length

    def get_obs_size(self) -> int:
        return 1

    def _observe(self, env) -> np.ndarray:
        # Ray origin = EE position
        ray_from = env.ee_pos.tolist()
        # Forward direction in EE frame
        link_state = p.getLinkState(env.arm_id, env.EE_LINK,
                                    computeForwardKinematics=1)
        link_orn = link_state[5]
        fwd = np.array(p.rotateVector(link_orn, (1, 0, 0)))
        ray_to = (env.ee_pos + fwd * self.ray_length).tolist()
        result = p.rayTest(ray_from, ray_to)[0]
        hit_fraction = result[2]  # 1.0 when no hit
        return np.array([hit_fraction], dtype=np.float32)


# ---------------------------------------------------------------------------
# Touch / force on gripper fingers  (4 values)
# ---------------------------------------------------------------------------
class TouchSensor(Sensor):
    name = "touch"

    def __init__(self, max_force: float = 20.0):
        self.max_force = max_force

    def get_obs_size(self) -> int:
        return 4  # (contact_flag, force) x 2 fingers

    def _observe(self, env) -> np.ndarray:
        obs = np.zeros(4, dtype=np.float32)
        if env.object_id is None:
            return obs
        for i, link_idx in enumerate([env.FINGER_L_LINK, env.FINGER_R_LINK]):
            contacts = p.getContactPoints(bodyA=env.arm_id, bodyB=env.object_id,
                                          linkIndexA=link_idx)
            if contacts:
                obs[i * 2] = 1.0  # contact flag
                total_force = sum(c[9] for c in contacts)  # normal force
                obs[i * 2 + 1] = np.clip(total_force / self.max_force, 0.0, 1.0)
        return obs


# ---------------------------------------------------------------------------
# End-effector XYZ position  (3 values)
# ---------------------------------------------------------------------------
class EEPositionSensor(Sensor):
    name = "ee_position"

    def get_obs_size(self) -> int:
        return 3

    def _observe(self, env) -> np.ndarray:
        # Report grasp point position (between fingers) instead of virtual EE.
        # This aligns with the Phase 2 reward which uses env.grasp_point.
        low, high = env.WORKSPACE_LOW, env.WORKSPACE_HIGH
        mid = (high + low) / 2.0
        rng = (high - low) / 2.0
        return np.clip((env.grasp_point - mid) / rng, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Target position  (3 values)
# ---------------------------------------------------------------------------
class TargetPositionSensor(Sensor):
    name = "target_position"

    def get_obs_size(self) -> int:
        return 3

    def _observe(self, env) -> np.ndarray:
        low, high = env.WORKSPACE_LOW, env.WORKSPACE_HIGH
        mid = (high + low) / 2.0
        rng = (high - low) / 2.0
        return np.clip((env.target_pos - mid) / rng, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Object position  (3 values) — zeros when no object present
# ---------------------------------------------------------------------------
class ObjectPositionSensor(Sensor):
    name = "object_position"

    def get_obs_size(self) -> int:
        return 3

    def _observe(self, env) -> np.ndarray:
        if env.object_id is None:
            return np.zeros(3, dtype=np.float32)
        low, high = env.WORKSPACE_LOW, env.WORKSPACE_HIGH
        mid = (high + low) / 2.0
        rng = (high - low) / 2.0
        return np.clip((env.object_pos - mid) / rng, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Helper — create the default sensor suite
# ---------------------------------------------------------------------------
def make_default_sensors():
    """Return the full list of vector sensors used across all phases."""
    return [
        ProprioceptiveSensor(),
        DistanceSensor(),
        UltrasonicSensor(),
        TouchSensor(),
        EEPositionSensor(),
        TargetPositionSensor(),
        ObjectPositionSensor(),
    ]


# ---------------------------------------------------------------------------
# Image sensors  (CNN observations)
# ---------------------------------------------------------------------------
CNN_IMAGE_SIZE = 64  # downscale to 64x64 for CNN input


class CameraSensor(ImageSensor):
    """RGB camera mounted on the end effector.

    Returns a (H, W, 3) uint8 image downscaled to ``size x size``.
    """

    name = "camera_rgb"

    def __init__(self, size: int = CNN_IMAGE_SIZE):
        self.size = size

    def get_obs_shape(self) -> tuple[int, int, int]:
        return (self.size, self.size, 3)

    def _observe(self, env) -> np.ndarray:
        img = env.last_rgb  # (256, 256, 3) uint8
        if img is None:
            return np.zeros(self.get_obs_shape(), dtype=np.uint8)
        if img.shape[0] != self.size or img.shape[1] != self.size:
            import cv2
            img = cv2.resize(img, (self.size, self.size),
                             interpolation=cv2.INTER_AREA)
        return img.astype(np.uint8)


class DepthCameraSensor(ImageSensor):
    """Depth camera mounted on the end effector.

    Returns a (H, W, 1) float32 image (normalised 0-1), downscaled to
    ``size x size``.
    """

    name = "camera_depth"

    def __init__(self, size: int = CNN_IMAGE_SIZE):
        self.size = size

    def get_obs_shape(self) -> tuple[int, int, int]:
        return (self.size, self.size, 1)

    def _observe(self, env) -> np.ndarray:
        depth = env.last_depth  # (256, 256) float32
        if depth is None:
            return np.zeros(self.get_obs_shape(), dtype=np.float32)
        if depth.shape[0] != self.size or depth.shape[1] != self.size:
            import cv2
            depth = cv2.resize(depth, (self.size, self.size),
                               interpolation=cv2.INTER_AREA)
        # normalise to [0, 1]
        depth = np.clip(depth, 0.0, 1.0)
        return depth[:, :, np.newaxis].astype(np.float32)


class RGBDSensor(ImageSensor):
    """Combined RGB + Depth as a 4-channel image.

    Returns a (H, W, 4) float32 image: RGB channels scaled to [0,1]
    plus depth channel [0,1].
    """

    name = "camera_rgbd"

    def __init__(self, size: int = CNN_IMAGE_SIZE):
        self.size = size

    def get_obs_shape(self) -> tuple[int, int, int]:
        return (self.size, self.size, 4)

    def _observe(self, env) -> np.ndarray:
        import cv2

        shape = self.get_obs_shape()

        rgb = env.last_rgb
        depth = env.last_depth

        if rgb is None or depth is None:
            return np.zeros(shape, dtype=np.float32)

        if rgb.shape[0] != self.size or rgb.shape[1] != self.size:
            rgb = cv2.resize(rgb, (self.size, self.size),
                             interpolation=cv2.INTER_AREA)
        if depth.shape[0] != self.size or depth.shape[1] != self.size:
            depth = cv2.resize(depth, (self.size, self.size),
                               interpolation=cv2.INTER_AREA)

        rgb_f = rgb.astype(np.float32) / 255.0
        depth_f = np.clip(depth, 0.0, 1.0)[:, :, np.newaxis]
        return np.concatenate([rgb_f, depth_f], axis=-1).astype(np.float32)


def make_camera_sensors(mode: str = "rgbd", size: int = CNN_IMAGE_SIZE):
    """Return vector sensors + an image sensor for CNN training.

    Args:
        mode: ``"rgb"``, ``"depth"``, or ``"rgbd"`` (default).
        size: Image side length (default 64).
    """
    sensors = make_default_sensors()
    if mode == "rgb":
        sensors.append(CameraSensor(size=size))
    elif mode == "depth":
        sensors.append(DepthCameraSensor(size=size))
    else:
        sensors.append(RGBDSensor(size=size))
    return sensors


# ---------------------------------------------------------------------------
# Sensor mask — binary vector encoding which sensors are active
# (only included in the 'rand' curriculum where sensors vary per episode)
# ---------------------------------------------------------------------------
class SensorMaskSensor(Sensor):
    """Reports which sensors are currently active as a binary vector.

    One bit per maskable sensor: 1.0 = active, 0.0 = disabled.
    Always active itself; not eligible for random dropout.
    """

    name = "sensor_mask"

    def __init__(self, maskable_sensors: list):
        self._maskable = maskable_sensors

    def get_obs_size(self) -> int:
        return len(self._maskable)

    def _observe(self, env) -> np.ndarray:
        return np.array([float(s.is_active) for s in self._maskable],
                        dtype=np.float32)


# ---------------------------------------------------------------------------
# New curriculum-aware sensor factories
# ---------------------------------------------------------------------------
def make_all_sensors(size: int = CNN_IMAGE_SIZE):
    """All sensors active.  Used for 'all' and 'ordered' curricula.

    Returns:
        (sensor_list, maskable_sensors)
        sensor_list    — pass directly to ArmEnv(sensors=...).
        maskable_sensors — sensors eligible for ordered-curriculum deactivation;
                           pass to ArmEnv(maskable_sensors=...).
    """
    distance   = DistanceSensor()
    ultrasonic = UltrasonicSensor()
    touch      = TouchSensor()
    ee_pos     = EEPositionSensor()
    obj_pos    = ObjectPositionSensor()
    camera     = RGBDSensor(size=size)
    maskable   = [distance, ultrasonic, touch, ee_pos, obj_pos, camera]
    sensor_list = [
        ProprioceptiveSensor(), TargetPositionSensor(),
        distance, ultrasonic, touch, ee_pos, obj_pos, camera,
    ]
    return sensor_list, maskable


def make_rand_sensors(size: int = CNN_IMAGE_SIZE):
    """All sensors + SensorMaskSensor.  Used for the 'rand' curriculum only.

    The SensorMaskSensor reports which sensors are active each episode so the
    policy can condition on sensor availability.

    Returns:
        (sensor_list, maskable_sensors) — same convention as make_all_sensors.
    """
    distance   = DistanceSensor()
    ultrasonic = UltrasonicSensor()
    touch      = TouchSensor()
    ee_pos     = EEPositionSensor()
    obj_pos    = ObjectPositionSensor()
    camera     = RGBDSensor(size=size)
    maskable   = [distance, ultrasonic, touch, ee_pos, obj_pos, camera]
    mask_snsr  = SensorMaskSensor(maskable)
    sensor_list = [
        ProprioceptiveSensor(), TargetPositionSensor(),
        distance, ultrasonic, touch, ee_pos, obj_pos,
        mask_snsr, camera,
    ]
    return sensor_list, maskable

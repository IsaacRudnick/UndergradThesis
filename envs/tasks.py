"""Task configurations for the three training phases.

Each Task controls scene setup, reward computation, and termination logic
while sharing the same base environment and sensor suite.
"""

import os

import numpy as np
import pybullet as p
import pybullet_data

# ---------------------------------------------------------------------------
# Shared table geometry
# ---------------------------------------------------------------------------
_TABLE_RADIUS = 0.20   # metres: distance from arm base to table centre
_TABLE_Z = 0.08        # table centre Z  (top face ~0.085 m)
_CUBE_Z = 0.10         # cube resting height (table top + cube half-extent)
_TABLE_HALF = [0.06, 0.06, 0.005]  # table collision/visual half-extents

# Second table is placed 60–120° away from the first.
# Reduced from 120-240° so J1 only needs to rotate 60-120° instead of
# 120-240°.  The "go way up and over" workaround required 30 cm of lift;
# at 60-120° the geometry no longer allows that shortcut without extreme
# (penalised) heights, so the arm must actually rotate J1.
_TABLE2_ANGLE_MIN = 1.0 * np.pi / 3.0   # 60°
_TABLE2_ANGLE_MAX = 2.0 * np.pi / 3.0   # 120°


def _angle_to_xy(angle: float) -> tuple[float, float]:
    """Convert a polar angle to (x, y) at _TABLE_RADIUS."""
    return float(_TABLE_RADIUS * np.sin(angle)), float(_TABLE_RADIUS * np.cos(angle))


def _sample_table_xy(env) -> tuple[float, float]:
    """Return a random (x, y) for the table centre at a uniformly-sampled
    angle around the arm base, so the arm must generalise to all directions."""
    angle = env.np_random.uniform(0.0, 2.0 * np.pi)
    return _angle_to_xy(angle)


def _spawn_table(x: float, y: float, z: float = _TABLE_Z,
                 color=None) -> int:
    """Spawn a static table at (x, y, z). Returns body ID."""
    if color is None:
        color = [0.4, 0.3, 0.2, 1]
    table_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=_TABLE_HALF)
    table_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=_TABLE_HALF,
                                    rgbaColor=color)
    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=table_col,
        baseVisualShapeIndex=table_vis,
        basePosition=[x, y, z],
    )


def _spawn_second_table(env, angle1: float, color=None) -> int:
    """Spawn a second table 120–240° away from angle1. Returns body ID."""
    offset = env.np_random.uniform(_TABLE2_ANGLE_MIN, _TABLE2_ANGLE_MAX)
    x, y = _angle_to_xy(angle1 + offset)
    return _spawn_table(x, y, color=color)


class Task:
    """Base class for all tasks."""

    max_episode_steps: int = 200

    def setup_scene(self, env) -> None:
        """Spawn task-specific objects.  Called from ``env.reset()``."""

    def post_settle(self, env) -> None:
        """Called after env.reset()'s physics settling steps complete.

        Use this to finalise object placement that depends on the arm's
        settled position (e.g. warm-start cube placement).
        """

    def compute_reward(self, env, action: np.ndarray) -> float:
        """Return scalar reward for the current step."""
        return 0.0

    def check_done(self, env) -> tuple[bool, dict]:
        """Return (terminated, info_dict)."""
        return False, {}


# -----------------------------------------------------------------------
# Phase 1A — Reach (basic: learn to reach targets)
# -----------------------------------------------------------------------
class ReachTask(Task):
    """Learn to move EE to a random target.  Simple reward: distance
    penalty + flat success bonus.  This is the proven configuration that
    reaches ~90 % success rate."""

    max_episode_steps = 200
    _target_vis_id = None

    # thresholds in metres (real-world units)
    SUCCESS_THRESHOLD = 0.05
    BONUS_THRESHOLD = 0.10
    _MAX_DIST = 0.5  # approx max possible distance in workspace

    def setup_scene(self, env) -> None:
        # Spawn table and cube for visual scene consistency across all phases.
        angle = env.np_random.uniform(0.0, 2.0 * np.pi)
        tab_x, tab_y = _angle_to_xy(angle)
        env.table_id = _spawn_table(tab_x, tab_y)

        # Second table (cosmetic — visual consistency with Phase 3)
        env.table2_id = _spawn_second_table(env, angle)

        # Spawn cube on table
        cube_urdf = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                 "urdf_files", "urdf", "cube.urdf")
        cube_x = tab_x + env.np_random.uniform(-0.02, 0.02)
        cube_y = tab_y + env.np_random.uniform(-0.02, 0.02)
        env.object_id = p.loadURDF(cube_urdf, [cube_x, cube_y, _CUBE_Z])

        # Random target position in workspace (independent of cube)
        env.target_pos = env.np_random.uniform(env.WORKSPACE_LOW,
                                               env.WORKSPACE_HIGH)
        # visual marker
        if self._target_vis_id is not None:
            p.removeBody(self._target_vis_id)
        vs = p.createVisualShape(p.GEOM_SPHERE, radius=0.03,
                                 rgbaColor=[1, 0, 0, 0.7])
        self._target_vis_id = p.createMultiBody(
            baseVisualShapeIndex=vs,
            basePosition=env.target_pos.tolist(),
        )

    def compute_reward(self, env, action: np.ndarray) -> float:
        dist = np.linalg.norm(env.grasp_point - env.target_pos)
        # Normalised distance penalty in [-1, 0]
        reward = -dist / self._MAX_DIST
        # Shaped proximity bonus (grows as EE approaches target)
        reward += max(0.0, 1.0 - dist / self.BONUS_THRESHOLD)
        # Large bonus for reaching the target
        if dist < self.SUCCESS_THRESHOLD:
            reward += 5.0
        # Small action penalty
        reward -= 0.005 * float(np.sum(action ** 2))
        # Action smoothing: penalise rapid direction changes (jitter)
        action_delta = action - env._prev_action
        reward -= 0.05 * float(np.sum(action_delta ** 2))
        return reward

    def check_done(self, env) -> tuple[bool, dict]:
        dist = np.linalg.norm(env.grasp_point - env.target_pos)
        success = dist < self.SUCCESS_THRESHOLD
        return False, {"is_success": success, "distance": dist}


# -----------------------------------------------------------------------
# Phase 1B — Reach with precision hold (fine-tune from 1A)
# -----------------------------------------------------------------------
class ReachHoldTask(Task):
    """Fine-tune a model that already reaches targets so it learns to
    hold still at the exact position instead of oscillating.

    Simplifies Phase 1B: just add a strong penalty for arm movement velocity
    to discourage oscillation, while preserving the reaching signal from Phase 1A.
    """

    max_episode_steps = 200      # longer episodes allow more learning
    _target_vis_id = None

    SUCCESS_THRESHOLD = 0.05     # same as Phase 1A
    _BONUS_THRESHOLD = 0.10
    _MAX_DIST = 0.5

    # EE velocity (world-space, m/step) penalty weight
    # Strong penalty discourages movement; holding still is strongly rewarded
    _VEL_WEIGHT = 5.0

    def setup_scene(self, env) -> None:
        # Spawn table and cube for visual scene consistency with Phase 2 (GraspTask).
        # This helps CNN features transfer better to the grasp task.
        angle = env.np_random.uniform(0.0, 2.0 * np.pi)
        tab_x, tab_y = _angle_to_xy(angle)
        env.table_id = _spawn_table(tab_x, tab_y)

        # Second table (cosmetic — visual consistency with Phase 3)
        env.table2_id = _spawn_second_table(env, angle)

        # Spawn cube on table
        cube_urdf = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                 "urdf_files", "urdf", "cube.urdf")
        cube_x = tab_x + env.np_random.uniform(-0.02, 0.02)
        cube_y = tab_y + env.np_random.uniform(-0.02, 0.02)
        env.object_id = p.loadURDF(cube_urdf, [cube_x, cube_y, _CUBE_Z])

        # Target: random in workspace (independent of cube presence)
        env.target_pos = env.np_random.uniform(env.WORKSPACE_LOW,
                                               env.WORKSPACE_HIGH)
        if self._target_vis_id is not None:
            p.removeBody(self._target_vis_id)
        vs = p.createVisualShape(p.GEOM_SPHERE, radius=0.03,
                                 rgbaColor=[1, 0, 0, 0.7])
        self._target_vis_id = p.createMultiBody(
            baseVisualShapeIndex=vs,
            basePosition=env.target_pos.tolist(),
        )

    def compute_reward(self, env, action: np.ndarray) -> float:
        dist = np.linalg.norm(env.grasp_point - env.target_pos)

        # Keep the Phase 1A reaching signal: distance penalty + shaped bonus
        reward = -dist / self._MAX_DIST
        reward += max(0.0, 1.0 - dist / self._BONUS_THRESHOLD)
        if dist < self.SUCCESS_THRESHOLD:
            reward += 5.0  # same as Phase 1A

        # Add strong penalty for arm movement (EE velocity in world space)
        # This teaches the policy to hold still at the target
        ee_vel = float(np.linalg.norm(env.ee_velocity))
        reward -= self._VEL_WEIGHT * ee_vel

        # Action penalties (same as Phase 1A)
        reward -= 0.005 * float(np.sum(action ** 2))
        action_delta = action - env._prev_action
        reward -= 0.05 * float(np.sum(action_delta ** 2))

        return reward

    def check_done(self, env) -> tuple[bool, dict]:
        dist = np.linalg.norm(env.grasp_point - env.target_pos)
        success = dist < self.SUCCESS_THRESHOLD
        return False, {"is_success": success, "distance": dist}


# -----------------------------------------------------------------------
# Phase 2 — Grasp (restructured: contact-first, no progress terms)
# -----------------------------------------------------------------------
class GraspTask(Task):
    """Grasp task with simple three-stage reward: approach, grasp, lift.

    With proper gripper friction the agent doesn't need elaborate reward
    shaping — a clean signal is easier to learn from.
    """

    max_episode_steps = 200

    CUBE_HALF = 0.015       # metres
    LIFT_THRESHOLD = 0.03   # how far above initial pos counts as "lifted"
    SUCCESS_LIFT = 0.05
    SUCCESS_HOLD_STEPS = 10

    # Table angle randomization range (radians). Start narrow to make early
    # learning easier; widen once the baseline works.
    _ANGLE_RANGE = (0.0, np.pi / 2)

    # Fraction of episodes that start with arm pre-positioned near the cube.
    # Guarantees contact experiences early in training so the agent learns
    # what contact/grasp reward feels like before having to discover it.
    # Clipping fix: gripper is reset FULLY OPEN (π/2) and the cube is placed
    # at grasp-centre height in post_settle() so open fingers straddle the
    # cube with clearance — no geometric overlap on reset.
    _WARM_START_FRAC = 0.25

    # Reward magnitudes
    _TIME_PENALTY = 0.3     # mild per-step cost to discourage dawdling
    _DIST_SCALE = 2.0       # distance shaping weight
    _GRASP_REWARD = 5.0     # per-step reward for bilateral contact w/ closed gripper
    _LIFT_SCALE = 500.0     # per-metre lift reward (0.05m = +25/step, 5:1 vs grasp-only)

    # Gripper gating — prevents farming contact reward with open fingers
    _GRIPPER_CLOSE_THRESH = 1.0  # rad; gripper must be below this for grasp to count

    # Proximity-gated gripper shaping — bridges the gap between approach
    # and grasp by rewarding closing the gripper when near the cube.
    _GRIP_SHAPE_SCALE = 1.0   # max per-step bonus (5x less than GRASP_REWARD)
    _GRIP_SHAPE_RADIUS = 0.05 # metres; shaping falls to 0 beyond this

    _cube_initial_z: float = 0.0
    _cube_spawn_pos: np.ndarray = None
    _hold_counter: int = 0
    _cube_dropped: bool = False
    _target_vis_id = None
    _is_warm_start: bool = False

    # Pre-computed reaching joints (j2, j3, j4) that place the grasp
    # point at ~TABLE_RADIUS distance and ~CUBE_Z height when j1 is set
    # to point toward the table.  Found by grid search over the joint
    # space — IK consistently fails for this arm.
    _REACH_JOINTS = (-0.5, -0.7, 1.9)

    def setup_scene(self, env) -> None:
        # Spawn table with limited angle range (easier early learning)
        angle = env.np_random.uniform(self._ANGLE_RANGE[0], self._ANGLE_RANGE[1])

        self._is_warm_start = env.np_random.random() < self._WARM_START_FRAC

        cube_urdf = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                 "urdf_files", "urdf", "cube.urdf")

        if self._is_warm_start:
            # --- Warm-start: direct joint positioning (no IK) ---
            # Point arm toward the table angle; j1 = π/2 − angle maps
            # the arm's forward reach direction to the table direction.
            j1 = np.pi / 2.0 - angle
            j2, j3, j4 = self._REACH_JOINTS

            for idx, pos in [(1, j1), (2, j2), (3, j3), (4, j4)]:
                p.resetJointState(env.arm_id, idx, pos)
                # Hold position during the env's settling steps so the
                # arm doesn't sag under gravity before the agent acts.
                p.setJointMotorControl2(
                    env.arm_id, idx, p.POSITION_CONTROL,
                    targetPosition=pos, force=6.5, maxVelocity=1.5,
                )

            # Fully open gripper so fingers don't overlap the cube on reset.
            # post_settle() places the cube at grasp-centre height; open
            # fingers straddle it cleanly.  The agent closes the gripper
            # to grasp — this is the exact behaviour we want it to discover.
            p.resetJointState(env.arm_id, env.GRIPPER_JOINT, 1.5707)
            p.resetJointState(env.arm_id, env.MIMIC_JOINT, -1.5707)
            p.setJointMotorControl2(
                env.arm_id, env.GRIPPER_JOINT, p.POSITION_CONTROL,
                targetPosition=1.5707, force=3.0, maxVelocity=1.5,
            )
            p.setJointMotorControl2(
                env.arm_id, env.MIMIC_JOINT, p.POSITION_CONTROL,
                targetPosition=-1.5707, force=3.0, maxVelocity=1.5,
            )

            # Spawn table and cube at approximate position so that
            # collision filtering in env.reset() applies.  The cube
            # will be repositioned to the true post-settle grasp point
            # in post_settle().
            approx_tab_x, approx_tab_y = _angle_to_xy(angle)
            env.table_id = _spawn_table(approx_tab_x, approx_tab_y)
            env.object_id = p.loadURDF(cube_urdf,
                                       [approx_tab_x, approx_tab_y, _CUBE_Z])
            # Placeholder spawn pos — will be overwritten in post_settle()
            cube_x, cube_y = approx_tab_x, approx_tab_y
        else:
            # --- Normal episode: random placement ---
            tab_x, tab_y = _angle_to_xy(angle)
            cube_x = tab_x + env.np_random.uniform(-0.02, 0.02)
            cube_y = tab_y + env.np_random.uniform(-0.02, 0.02)

            env.table_id = _spawn_table(tab_x, tab_y)
            env.object_id = p.loadURDF(cube_urdf, [cube_x, cube_y, _CUBE_Z])
            # Normal episodes: gripper fully open for approach
            p.resetJointState(env.arm_id, env.GRIPPER_JOINT, 1.5707)
            p.resetJointState(env.arm_id, env.MIMIC_JOINT, -1.5707)

        # Second table (cosmetic — visual consistency with Phase 3)
        env.table2_id = _spawn_second_table(env, angle)

        self._cube_initial_z = _CUBE_Z
        self._cube_spawn_pos = np.array([cube_x, cube_y, _CUBE_Z], dtype=np.float64)
        self._hold_counter = 0
        self._cube_dropped = False

        # Target = object position (for observation consistency with Phase 1)
        env.target_pos = self._cube_spawn_pos.copy()

        # Visual marker on the cube (consistent red sphere)
        if self._target_vis_id is not None:
            p.removeBody(self._target_vis_id)
        vs = p.createVisualShape(p.GEOM_SPHERE, radius=0.02,
                                 rgbaColor=[1, 0, 0, 0.7])
        self._target_vis_id = p.createMultiBody(
            baseVisualShapeIndex=vs,
            basePosition=self._cube_spawn_pos.tolist(),
        )

    def post_settle(self, env) -> None:
        """Reposition the cube to the true post-settle grasp point.

        During warm-start episodes the cube is spawned at an approximate
        position in setup_scene() so collision filtering applies.  After
        the env's 50 settling steps complete, the arm has fully settled
        and we can read the *actual* grasp point and move the cube there.
        """
        if not self._is_warm_start:
            return

        grasp_pt = env.grasp_point
        cube_x, cube_y = float(grasp_pt[0]), float(grasp_pt[1])

        # Place the cube at the grasp XY but slightly below the grasp
        # centre Z so it sits between the open fingers without
        # overlapping the finger collision meshes.  The grasp height
        # varies with arm angle, so we use the actual grasp_pt Z.
        # Cube at grasp-centre height: open fingers straddle the cube cleanly
        # (no clipping), and closing the gripper presses fingers onto cube sides
        # at the correct height.  The -0.02 offset used previously caused finger
        # tips to overlap the top face of the cube with a partly-closed gripper.
        cube_z = float(grasp_pt[2])
        table_z = cube_z - self.CUBE_HALF - 0.005

        p.resetBasePositionAndOrientation(
            env.object_id,
            [cube_x, cube_y, cube_z],
            [0, 0, 0, 1],
        )
        p.resetBasePositionAndOrientation(
            env.table_id,
            [cube_x, cube_y, table_z],
            [0, 0, 0, 1],
        )

        # Zero out any residual velocity from collision resolution
        p.resetBaseVelocity(env.object_id, [0, 0, 0], [0, 0, 0])

        # Single physics step for contact detection
        p.stepSimulation()

        # Zero velocity again in case the step introduced any
        p.resetBaseVelocity(env.object_id, [0, 0, 0], [0, 0, 0])

        # Update spawn tracking to match the new position
        self._cube_initial_z = cube_z
        self._cube_spawn_pos = np.array([cube_x, cube_y, cube_z],
                                        dtype=np.float64)
        env.target_pos = self._cube_spawn_pos.copy()

        # Update visual marker
        if self._target_vis_id is not None:
            p.resetBasePositionAndOrientation(
                self._target_vis_id,
                self._cube_spawn_pos.tolist(),
                [0, 0, 0, 1],
            )

    def _finger_contacts(self, env):
        """Return (left_contact_count, right_contact_count)."""
        l_contacts = p.getContactPoints(bodyA=env.arm_id,
                                        bodyB=env.object_id,
                                        linkIndexA=env.FINGER_L_LINK)
        r_contacts = p.getContactPoints(bodyA=env.arm_id,
                                        bodyB=env.object_id,
                                        linkIndexA=env.FINGER_R_LINK)
        return len(l_contacts), len(r_contacts)

    # Drop detection: if cube falls this far below its spawn Z, it's
    # off the table and the episode is unrecoverable.
    _DROP_THRESH = 0.02       # metres below initial Z

    def compute_reward(self, env, action: np.ndarray) -> float:
        grasp_pos = env.grasp_point
        obj_pos = env.object_pos
        lift = obj_pos[2] - self._cube_initial_z

        # Gripper state (0 = closed, π/2 = fully open)
        gripper_angle = p.getJointState(env.arm_id, env.GRIPPER_JOINT)[0]

        # If the cube has fallen off the table, shape distance toward
        # the *spawn* position so the agent can't farm reward by chasing
        # the cube on the floor — but don't terminate: let it keep
        # trying so exploration near the cube isn't punished.
        if lift < -self._DROP_THRESH:
            self._cube_dropped = True
            spawn_dist = float(np.linalg.norm(grasp_pos - self._cube_spawn_pos))
            return -self._TIME_PENALTY - self._DIST_SCALE * spawn_dist

        grasp_dist = float(np.linalg.norm(grasp_pos - obj_pos))

        env.target_pos = obj_pos.copy()

        left_c, right_c = self._finger_contacts(env)
        # Require gripper partially closed — prevents farming contact
        # reward by touching cube with wide-open fingers.
        has_grasp = (left_c > 0 and right_c > 0
                     and gripper_angle < self._GRIPPER_CLOSE_THRESH)

        reward = -self._TIME_PENALTY

        # Approach: distance shaping pulls the arm toward the cube.
        reward -= self._DIST_SCALE * grasp_dist

        # Gripper shaping: when near the cube, reward closing the gripper.
        # Smoothly falls off with distance so it only activates close-up.
        # Max +1.0/step (vs +5.0 for actual grasp) — enough to guide
        # exploration without creating a local optimum.
        proximity = max(0.0, 1.0 - grasp_dist / self._GRIP_SHAPE_RADIUS)
        grip_closed_frac = 1.0 - gripper_angle / 1.5707  # 0=open, 1=closed
        reward += self._GRIP_SHAPE_SCALE * proximity * max(0.0, grip_closed_frac)

        # Grasp: per-step reward for holding the cube with closed gripper.
        # Gives the agent a stable rewarded state to learn from.
        if has_grasp:
            reward += self._GRASP_REWARD

        # Lift: 6:1 ratio vs grasp-only at success height, so lifting
        # clearly dominates holding at table level.
        if has_grasp and lift > 0.005:
            reward += self._LIFT_SCALE * float(np.clip(lift, 0.0, 0.10))

        # Action smoothness — mirrors ReachTask and PickAndPlaceTask so the
        # transferred policy doesn't need to unlearn jerky movements.
        reward -= 0.005 * float(np.sum(action ** 2))
        action_delta = action - env._prev_action
        reward -= 0.05 * float(np.sum(action_delta ** 2))

        return reward

    def check_done(self, env) -> tuple[bool, dict]:
        lift = env.object_pos[2] - self._cube_initial_z
        left_c, right_c = self._finger_contacts(env)
        gripper_angle = p.getJointState(env.arm_id, env.GRIPPER_JOINT)[0]
        has_grasp = (left_c > 0 and right_c > 0
                     and gripper_angle < self._GRIPPER_CLOSE_THRESH)

        # Terminate early if cube dropped — episode is unrecoverable,
        # continuing just teaches the agent to be passive.
        if self._cube_dropped:
            return True, {
                "is_success": False,
                "lift": lift,
                "contact_left": left_c > 0,
                "contact_right": right_c > 0,
            }

        # Require BOTH lift AND bilateral grasp — prevents false success
        # from bumping the cube upward without actually grasping it.
        if lift > self.SUCCESS_LIFT and has_grasp:
            self._hold_counter += 1
        else:
            self._hold_counter = 0

        success = self._hold_counter >= self.SUCCESS_HOLD_STEPS
        return success, {
            "is_success": success,
            "lift": lift,
            "contact_left": left_c > 0,
            "contact_right": right_c > 0,
        }



# -----------------------------------------------------------------------
# Phase 3 — Pick and Place (two-table: pick from table 1, place on table 2)
# -----------------------------------------------------------------------
class PickAndPlaceTask(Task):
    max_episode_steps = 300

    PLACE_THRESHOLD = 0.05  # cube within 5 cm of target = success

    # Start narrow like GraspTask — the transferred policy was only trained
    # on 0–90° and needs early successes before facing all orientations.
    _ANGLE_RANGE = (0.0, 2.0 * np.pi / 3.0)

    # Gripper gating (same as GraspTask — preserves transfer signal).
    _GRIPPER_CLOSE_THRESH = 1.0  # rad

    # Approach/grasp signals — mirror GraspTask so Phase 2 behaviour is
    # directly reinforced rather than trained away.
    _TIME_PENALTY = 0.3
    _DIST_SCALE = 2.0
    _GRASP_REWARD = 5.0
    _GRIP_SHAPE_SCALE = 1.0
    _GRIP_SHAPE_RADIUS = 0.05

    # Lift — 350x scale balances two competing failure modes:
    #   Too low (200x): table-level grasp is a stronger local optimum.
    #     At 200x, lifting breaks even with table-level (1410/ep) only if
    #     the arm maintains grip for 80+ steps.  But early in training the
    #     arm has no transport skills and drops the cube in ~50 steps,
    #     making the expected return from lifting LOWER than staying on the
    #     table.  Policy correctly learns "don't lift."
    #   Too high (500x): hover optimum dominates.  At 500x, hovering at max
    #     distance earns 47.7/step × 300 = 14310 vs transport earning 16410
    #     — the transport signal (14%) gets swamped by the hover density.
    #   At 350x: break-even T = 43 steps (achievable).  50 steps of lift
    #     earns 1635 > 1410 (table-level 300 steps), so the policy keeps
    #     lifting even when it drops frequently.  Hover at max distance =
    #     9810 vs destination = 11910 — transport is 21% better, a clear
    #     gradient without being swamped.
    _LIFT_SCALE = 350.0

    # Transport — only active once cube is lifted above the tilt threshold
    # (2 cm: tilting a 3 cm cube only raises its centre ~1.5 cm, so this
    # eliminates table-dragging and tilt shortcuts without blocking real lifts).
    _LIFT_MIN = 0.02          # metres
    _TRANSPORT_SCALE = 100.0  # per-step potential-based shaping when carrying toward target
    # Raised 30→100: at 30 the full 0.28 m journey earned only 8.4 total — cosmetic.
    # At 100 it earns 28 total and 1/step per cm of progress — a real gradient.
    _MAX_TRANSPORT_DIST = 0.5 # normalisation distance (approx max table gap)

    # J1 alignment: reward for rotating the base joint to face the destination
    # rather than reaching across in an awkward elbow/shoulder-only configuration.
    # Formula: scale × (1 − |err|/π)²  — quadratic, not linear.
    # At 0° error: +50/step; at 180° error: 0.
    # Scale=10 produced j1_error≈144° throughout carry — not strong enough.
    # Scale=50 linear overshot: at 86° error the arm still earns +26/step (82% of
    # max), subsidising the hover-at-source optimum.  The gradient from 86°→0° was
    # only +24/step — not worth the drop risk.
    # Squared at scale=50: at 86° error → +13.6/step (vs +26 linear); at 0° → +50
    # (unchanged).  The derivative at 86° is -16.6/rad vs linear's -15.9/rad — same
    # gradient strength for J1 rotation from the current hover position, just less
    # absolute reward for sitting there.  Combined with doubled _HOLD_DIST_PENALTY
    # this reduces hover reward from +31.7 → +2.4/step, while midpoint earns +33/step
    # and destination earns +115/step.
    _J1_ALIGN_SCALE = 50.0

    # J4 (wrist) tilt penalty: the policy freely rotates J4 during J1 rotation,
    # driving it to the ±120° joint limit.  At the limit the arm's reachable
    # workspace shifts, pushing the cube away from the destination, and the cube
    # eventually slides out of the tilted gripper.  Penalising excess tilt forces
    # the arm to keep the wrist roughly level during carry.
    # At J4=90°: -15*(1.571-0.785) = -11.8/step (tolerable with j1aln=+40)
    # At J4=120° (limit): -15*(2.094-0.785) = -19.6/step (strongly discourages limit)
    # Penalty is positive-direction only: -120° to +45° are all valid carry/placement
    # poses; only tilting the wrist further upward past +45° causes the cube to slip.
    _J4_TILT_THRESH  = 0.785  # ~45° — beyond this, positive tilt causes grip loss
    _J4_TILT_PENALTY = 15.0   # per-radian above threshold while carrying

    # Overheight penalty: penalise going above a reasonable carry height while
    # grasped.  The lift reward plateaus at 10 cm so there is no reward benefit
    # to going higher, but without a penalty the arm wanders to extreme joint
    # configurations (the "bends over backwards" behaviour) to maintain contact.
    # Threshold reduced 15→12 cm and penalty raised 200→500 to block the
    # "crane over" strategy: at lift=0.30m the arm was earning -90/step here
    # (barely noticeable) and arcing 30 cm high to reach the opposite side
    # without ever rotating J1.  At penalty=500 the same path costs -90/step,
    # making it far worse than rotating J1 (which costs nothing here).
    _OVERHEIGHT_THRESH = 0.12   # m above initial cube Z before penalty kicks in
    _OVERHEIGHT_PENALTY = 500.0 # per-metre-per-step while grasped above thresh

    # Hold-distance cost: breaks the hover optimum.  Scales with lift height so
    # the cost starts near-zero at table level and reaches full value at the
    # 10 cm lift cap.  This avoids the reward cliff that a binary threshold
    # creates: with a hard threshold at _LIFT_MIN the reward drops 6+/step the
    # instant lift crosses 2 cm, which (combined with the target_pos observation
    # switch at the same threshold) caused the policy to unlearn grasping entirely.
    # Smooth version: cost = PENALTY * (dist / MAX_DIST) * min(1, lift / 0.10)
    #   At 2 cm lift:  -120 * 0.56 * 0.2 = -13.4/step  (small, no cliff)
    #   At 5 cm lift:  -120 * 0.56 * 0.5 = -33.6/step  (source hover: nearly break-even)
    #   At 10 cm lift: -120 * 0.56 * 1.0 = -67.2/step  (strongly punishes high hover)
    #   At destination:  0/step regardless of lift   (dest = +115/step)
    # Raised from 30→60: broke hover at tdist≈0.20m.
    # Raised from 60→120: at 60, hovering at tdist≈0.28m with linear J1_ALIGN (scale=50)
    # still earned +31.7/step (J1 subsidy of +26 dominated).  With squared J1 the
    # subsidy drops to +13.6; doubling hold to 120 brings hover to +2.4/step — nearly
    # zero — while midpoint earns +33/step and destination earns +115/step.
    _HOLD_DIST_PENALTY = 120.0  # per-unit-normalised-distance while carrying

    # Stepped proximity bonuses while carrying (exclusive — only tightest band fires).
    # The placement bonus lives at tdist<0.05m which the arm has never reached, so
    # there is no gradient from it.  These create intermediate attractors at
    # reachable distances that progressively guide the arm into the placement zone.
    _PROX_THRESH_FAR  = 0.15   # m
    _PROX_THRESH_MID  = 0.10   # m
    _PROX_THRESH_NEAR = 0.07   # m
    _PROX_BONUS_FAR   = 15.0   # +15/step within 0.15 m while carrying
    _PROX_BONUS_MID   = 30.0   # +30/step within 0.10 m while carrying
    _PROX_BONUS_NEAR  = 60.0   # +60/step within 0.07 m while carrying

    # Lift-reward taper near destination: the +35/step lift reward at carry
    # height directly conflicts with lowering the cube for placement.  Scale
    # lift reward to zero linearly as the arm closes in, so there is no
    # incentive to stay high when the arm is already near the table.
    _LIFT_TAPER_DIST  = 0.15   # m — full lift reward at this distance; 0 at dest

    # Release shaping: reward for opening the gripper when the cube is close to
    # the destination AND already near table height.  Gives an immediate per-step
    # signal toward the release behaviour, which the arm has never been trained to
    # do.  Gated on lift < _RELEASE_LIFT_THRESH so the cube won't fling when released.
    _RELEASE_SCALE      = 30.0
    _RELEASE_DIST_THRESH = 0.10  # m — shaping active within this radius
    _RELEASE_LIFT_THRESH = 0.06  # m — cube must be close to table height

    # Placement bonus — per-step while cube is calmly near target.
    # Gated on velocity so a flung cube that happens to land nearby earns nothing.
    # Raised 50→150: the arm must overcome the loss of grasp+lift+j1aln rewards
    # (~90/step) when it releases.  At 150, releasing at the destination is worth
    # +60/step more than continuing to hover there while grasping.
    _PLACE_BONUS = 150.0

    # Calm-placement gates:
    #   _PLACE_MAX_SPEED   — cube must be below this speed (m/s) to earn
    #                        the placement bonus and count toward success.
    #   _RELEASE_HEIGHT_THRESH — free height margin above table (m).  Any step
    #                        the cube is higher than this while NOT grasped
    #                        incurs _FLING_PENALTY per metre of excess.
    #   _FLING_PENALTY     — per-metre-per-step penalty for the cube being
    #                        airborne above the threshold ungrasped.
    #   _PLACE_HOLD_STEPS  — consecutive calm steps needed to terminate as
    #                        success (mirrors GraspTask.SUCCESS_HOLD_STEPS).
    _PLACE_MAX_SPEED = 0.3        # m/s
    _RELEASE_HEIGHT_THRESH = 0.05 # m above _CUBE_Z before penalty kicks in
    _FLING_PENALTY = 200.0
    _PLACE_HOLD_STEPS = 5

    # Drop detection (mirrors GraspTask) — terminate early when cube falls
    # off the table so we don't waste steps on unrecoverable episodes.
    _DROP_THRESH = 0.02       # metres below initial Z

    # One-time lift bonus: fires once per episode the first time the arm
    # genuinely lifts the cube above _LIFT_BONUS_THRESH.  Prevents the
    # table-level grasp from being a stronger local optimum than lifting
    # by making the first successful lift worth attempting regardless of
    # what happens after (early termination from a drop, failed transport,
    # etc.).  Mirrors the implicit signal in GraspTask where a successful
    # lift terminates the episode with maximum accumulated reward.
    _LIFT_BONUS = 100.0
    _LIFT_BONUS_THRESH = 0.05  # m — same as GraspTask.SUCCESS_LIFT

    _destination_pos: np.ndarray = None
    _cube_spawn_pos: np.ndarray = None
    _cube_initial_z: float = 0.0
    _cube_dropped: bool = False
    _place_hold_counter: int = 0
    _prev_dist_obj_target: float = 0.0
    _lift_bonus_earned: bool = False  # reset each episode; gates one-time bonus
    _last_lift_bonus: float = 0.0    # set by compute_reward; read by diagnose_reward
    _target_vis_id = None

    def setup_scene(self, env) -> None:
        # Source table — angle constrained like GraspTask for early training
        angle1 = env.np_random.uniform(self._ANGLE_RANGE[0], self._ANGLE_RANGE[1])
        tab1_x, tab1_y = _angle_to_xy(angle1)
        env.table_id = _spawn_table(tab1_x, tab1_y)

        # Destination table (120–240° away)
        offset = env.np_random.uniform(_TABLE2_ANGLE_MIN, _TABLE2_ANGLE_MAX)
        angle2 = angle1 + offset
        tab2_x, tab2_y = _angle_to_xy(angle2)
        env.table2_id = _spawn_table(tab2_x, tab2_y,
                                     color=[0.3, 0.4, 0.3, 1])

        # Cube on source table
        cube_urdf = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                 "urdf_files", "urdf", "cube.urdf")
        cx = tab1_x + env.np_random.uniform(-0.02, 0.02)
        cy = tab1_y + env.np_random.uniform(-0.02, 0.02)
        env.object_id = p.loadURDF(cube_urdf, [cx, cy, _CUBE_Z])

        # Store the destination separately so check_done and the drop branch
        # can always reference it independently of env.target_pos.
        tx = tab2_x + env.np_random.uniform(-0.02, 0.02)
        ty = tab2_y + env.np_random.uniform(-0.02, 0.02)
        self._destination_pos = np.array([tx, ty, _CUBE_Z])

        self._cube_spawn_pos = np.array([cx, cy, _CUBE_Z], dtype=np.float64)
        self._prev_dist_obj_target = float(np.linalg.norm(self._cube_spawn_pos - self._destination_pos))

        # Start pointing at the cube (pre-grasp state) so the reset observation
        # matches GraspTask's distribution. compute_reward will update this
        # every step based on has_grasp.
        env.target_pos = self._cube_spawn_pos.copy()
        self._cube_initial_z = _CUBE_Z
        self._cube_dropped = False
        self._place_hold_counter = 0
        self._lift_bonus_earned = False
        self._last_lift_bonus = 0.0

        # Green sphere marker on destination table
        if self._target_vis_id is not None:
            p.removeBody(self._target_vis_id)
        vs = p.createVisualShape(p.GEOM_SPHERE, radius=0.025,
                                 rgbaColor=[0, 1, 0, 0.6])
        self._target_vis_id = p.createMultiBody(
            baseVisualShapeIndex=vs,
            basePosition=self._destination_pos.tolist(),
        )

        # Gripper fully open for approach
        p.resetJointState(env.arm_id, env.GRIPPER_JOINT, 1.5707)
        p.resetJointState(env.arm_id, env.MIMIC_JOINT, -1.5707)

    def _finger_contacts(self, env):
        """Return (left_contact_count, right_contact_count)."""
        l = p.getContactPoints(bodyA=env.arm_id, bodyB=env.object_id,
                               linkIndexA=env.FINGER_L_LINK)
        r = p.getContactPoints(bodyA=env.arm_id, bodyB=env.object_id,
                               linkIndexA=env.FINGER_R_LINK)
        return len(l), len(r)

    def _is_grasped(self, env):
        if env.object_id is None:
            return False
        left_c, right_c = self._finger_contacts(env)
        gripper_angle = p.getJointState(env.arm_id, env.GRIPPER_JOINT)[0]
        return (left_c > 0 and right_c > 0
                and gripper_angle < self._GRIPPER_CLOSE_THRESH)

    def compute_reward(self, env, action: np.ndarray) -> float:
        grasp_pos = env.grasp_point
        obj_pos = env.object_pos
        lift = obj_pos[2] - self._cube_initial_z
        grasp_dist = float(np.linalg.norm(grasp_pos - obj_pos))
        dist_obj_target = float(np.linalg.norm(obj_pos - self._destination_pos))
        gripper_angle = p.getJointState(env.arm_id, env.GRIPPER_JOINT)[0]

        left_c, right_c = self._finger_contacts(env)
        has_grasp = (left_c > 0 and right_c > 0
                     and gripper_angle < self._GRIPPER_CLOSE_THRESH)

        # Drop detection: cube fell off table — episode is unrecoverable.
        # Shape toward the spawn position so the agent can't farm reward
        # by chasing the cube on the floor.
        if lift < -self._DROP_THRESH:
            self._cube_dropped = True
            spawn_dist = float(np.linalg.norm(grasp_pos - self._cube_spawn_pos))
            return -self._TIME_PENALTY - self._DIST_SCALE * spawn_dist

        self._last_lift_bonus = 0.0  # cleared every live step; set below if bonus fires

        # Update target_pos so sensors gradually shift from cube → destination.
        # A hard switch at _LIFT_MIN created a sudden DistanceSensor jump
        # (0.02 → 0.70 normalised) that the GraspTask-transferred policy
        # interpreted as "I'm far from target → open gripper and approach",
        # causing it to drop the cube every time lift crossed 2 cm.  Instead,
        # linearly interpolate target_pos over the full lift range so
        # DistanceSensor and TargetPositionSensor change smoothly:
        #   t=0 at lift=_LIFT_MIN → target_pos == cube (GraspTask-identical obs)
        #   t=1 at lift=0.10 m   → target_pos == destination
        if has_grasp and lift > self._LIFT_MIN:
            t = float(np.clip((lift - self._LIFT_MIN) / (0.10 - self._LIFT_MIN),
                              0.0, 1.0))
            env.target_pos = (1.0 - t) * obj_pos + t * self._destination_pos
        else:
            env.target_pos = obj_pos.copy()

        reward = -self._TIME_PENALTY

        # Approach: pull EE toward cube (same gradient as GraspTask so Phase 2
        # approach behaviour is immediately reinforced, not retrained).
        reward -= self._DIST_SCALE * grasp_dist

        # Gripper shaping: reward closing the gripper when near the cube.
        # Mirrors GraspTask — bridges approach and grasp without a local optimum.
        proximity = max(0.0, 1.0 - grasp_dist / self._GRIP_SHAPE_RADIUS)
        grip_closed_frac = 1.0 - gripper_angle / 1.5707
        reward += self._GRIP_SHAPE_SCALE * proximity * max(0.0, grip_closed_frac)

        # Grasp: per-step reward for bilateral contact with closed gripper.
        # Kept active throughout transport so the agent is never incentivised
        # to drop the cube en route.
        if has_grasp:
            reward += self._GRASP_REWARD

        # Lift: mirrors GraspTask — strong incentive to pick the cube off the
        # table rather than just holding it at surface level.  Capped at 10 cm
        # so the reward plateaus once the arm clears the table.
        # Tapered to zero near the destination so the "stay high" incentive
        # doesn't conflict with lowering the cube for placement.
        if has_grasp and lift > 0.005:
            lift_taper = min(1.0, dist_obj_target / self._LIFT_TAPER_DIST)
            reward += self._LIFT_SCALE * float(np.clip(lift, 0.0, 0.10)) * lift_taper

        # One-time lift bonus: fires the first time the arm lifts the cube
        # above 5 cm per episode.  This definitively breaks the table-level
        # grasp optimum — at L=350 the break-even is 43 sustained steps, but
        # if the arm drops sooner (e.g. 30 steps), the bonus pushes it over.
        # Cannot be farmed: one bonus per episode, gated on genuine bilateral
        # grasp + real lift height (same threshold as GraspTask.SUCCESS_LIFT).
        if has_grasp and lift > self._LIFT_BONUS_THRESH and not self._lift_bonus_earned:
            self._lift_bonus_earned = True
            self._last_lift_bonus = self._LIFT_BONUS
            reward += self._LIFT_BONUS

        # Transport: potential-based shaping — reward proportional to progress
        # made toward destination this step.  Gated on lift threshold so the
        # agent can't score by dragging the cube along the table surface.
        if has_grasp and lift > self._LIFT_MIN:
            progress = self._prev_dist_obj_target - dist_obj_target
            reward += self._TRANSPORT_SCALE * progress

        # Proximity bonus: stepped attractors that give the arm experience in
        # regions it has never visited.  Exclusive — only the tightest band fires
        # to avoid stacking and to keep the gradient always pointing inward.
        if has_grasp and lift > self._LIFT_MIN:
            if dist_obj_target < self._PROX_THRESH_NEAR:
                reward += self._PROX_BONUS_NEAR
            elif dist_obj_target < self._PROX_THRESH_MID:
                reward += self._PROX_BONUS_MID
            elif dist_obj_target < self._PROX_THRESH_FAR:
                reward += self._PROX_BONUS_FAR

        # J1 alignment: while carrying, reward rotating the base joint to face
        # the destination table.  Without this the arm reaches across using
        # elbow/shoulder only and hits joint limits before it can lower the cube
        # for placement ("bending over backwards").
        # _angle_to_xy convention: x=sin(angle), y=cos(angle), so
        # j1_target = π/2 − atan2(dest_x, dest_y) faces the destination.
        if has_grasp and lift > self._LIFT_MIN:
            j1_current = p.getJointState(env.arm_id, 1)[0]
            j1_target = np.pi / 2.0 - np.arctan2(
                self._destination_pos[0], self._destination_pos[1])
            j1_error = j1_current - j1_target
            j1_error = (j1_error + np.pi) % (2 * np.pi) - np.pi  # wrap to [−π, π]
            align_frac = max(0.0, 1.0 - abs(j1_error) / np.pi)
            reward += self._J1_ALIGN_SCALE * align_frac * align_frac  # quadratic

            j4_current = p.getJointState(env.arm_id, 4)[0]
            excess_tilt = max(0.0, j4_current - self._J4_TILT_THRESH)
            reward -= self._J4_TILT_PENALTY * excess_tilt

        # Overheight penalty: costs per step the cube is above the carry
        # threshold while grasped.  The lift reward plateaus at 10 cm so there
        # is no reward for going higher, but without this cost the arm drifts to
        # extreme configurations to maintain contact ("bends over backwards").
        if has_grasp and lift > self._OVERHEIGHT_THRESH:
            reward -= self._OVERHEIGHT_PENALTY * (lift - self._OVERHEIGHT_THRESH)

        # Hold-distance cost: lift-scaled to avoid the reward cliff a binary
        # threshold would create.  Grows from ~0 at table level to the full
        # penalty at the 10 cm lift cap, creating a smooth gradient toward the
        # destination without discouraging lifting itself.
        if has_grasp and lift > 0.005:
            lift_frac = min(1.0, lift / 0.10)
            reward -= self._HOLD_DIST_PENALTY * (dist_obj_target / self._MAX_TRANSPORT_DIST) * lift_frac

        # Release shaping: immediate per-step reward for opening the gripper
        # when the cube is already close to the destination and near table height.
        # The arm was trained for millions of steps to keep the gripper closed;
        # without this signal it has no reason to release even when positioned
        # correctly.  Gated on lift < _RELEASE_LIFT_THRESH so releasing won't
        # cause a fling (cube stays below _RELEASE_HEIGHT_THRESH when dropped).
        if has_grasp and dist_obj_target < self._RELEASE_DIST_THRESH and lift < self._RELEASE_LIFT_THRESH:
            grip_open_frac = gripper_angle / 1.5707
            reward += self._RELEASE_SCALE * (1.0 - dist_obj_target / self._RELEASE_DIST_THRESH) * grip_open_frac

        # Fling penalty: every step the cube is above the free-height margin
        # while NOT held, penalise proportionally to the excess height.
        # This directly taxes the "bend arm back and throw" strategy — the cube
        # is in free flight above table level for several steps, each costing
        # _FLING_PENALTY * excess_metres.  A gentle low-level release (cube
        # stays within _RELEASE_HEIGHT_THRESH of table) incurs no penalty.
        if not has_grasp:
            excess_height = max(0.0, obj_pos[2] - (_CUBE_Z + self._RELEASE_HEIGHT_THRESH))
            reward -= self._FLING_PENALTY * excess_height

        # Placement bonus: only earned when the cube is near the target AND
        # nearly still.  A flung cube still moving fast after landing earns
        # nothing until it settles — which may never happen within the episode.
        cube_lin_vel, _ = p.getBaseVelocity(env.object_id)
        cube_speed = float(np.linalg.norm(cube_lin_vel))
        if (dist_obj_target < self.PLACE_THRESHOLD
                and not has_grasp
                and cube_speed < self._PLACE_MAX_SPEED):
            reward += self._PLACE_BONUS

        reward -= 0.005 * float(np.sum(action ** 2))
        action_delta = action - env._prev_action
        reward -= 0.05 * float(np.sum(action_delta ** 2))
        self._prev_dist_obj_target = dist_obj_target
        return reward

    def check_done(self, env) -> tuple[bool, dict]:
        # Always use _destination_pos — env.target_pos now tracks the cube
        # during approach so it can't be used for the success condition.
        dist_obj_target = float(np.linalg.norm(env.object_pos - self._destination_pos))
        grasped = self._is_grasped(env)

        # Terminate early when cube is dropped — continuing just wastes steps.
        if self._cube_dropped:
            return True, {"is_success": False,
                          "dist_obj_target": dist_obj_target}

        # Stable placement: cube must be close to target, not held, and nearly
        # still for _PLACE_HOLD_STEPS consecutive steps.  Mirrors GraspTask's
        # SUCCESS_HOLD_STEPS — a flung cube that bounces near the target and
        # keeps moving never accumulates enough hold steps to succeed.
        cube_lin_vel, _ = p.getBaseVelocity(env.object_id)
        cube_speed = float(np.linalg.norm(cube_lin_vel))
        calm = (dist_obj_target < self.PLACE_THRESHOLD
                and not grasped
                and cube_speed < self._PLACE_MAX_SPEED)
        if calm:
            self._place_hold_counter += 1
        else:
            self._place_hold_counter = 0

        success = self._place_hold_counter >= self._PLACE_HOLD_STEPS
        return success, {"is_success": success,
                         "dist_obj_target": dist_obj_target,
                         "cube_speed": cube_speed}

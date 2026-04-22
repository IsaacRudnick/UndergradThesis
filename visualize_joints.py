# Run through each joint of a robotic arm in PyBullet, oscillating it sinusoidally and returning it to zero.
# Display messages on-screen in the PyBullet GUI.
# To determine which joint is which through visual inspection.

import os
import time
import pybullet as p
import pybullet_data

# --- Helper for on-screen text in PyBullet ---
_prev_text_id = None


def gui_print(text):
    global _prev_text_id
    # Remove previous message
    if _prev_text_id is not None:
        p.removeUserDebugItem(_prev_text_id)

    # Add new message at a fixed location in front of the camera
    _prev_text_id = p.addUserDebugText(
        text,
        [0.2, 0, 0.1],            # Position in world coordinates
        textColorRGB=[0, 0, 0],
        textSize=3
    )


def main():
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # Hide annoying explorer.
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    planeId = p.loadURDF("plane.urdf")

    arm_path = os.path.join(os.path.dirname(__file__), "urdf_files", "urdf", "lss_arm_4dof.urdf")
    gui_print(f"Loading arm from:\n{arm_path}")
    arm_start_pos = [0, 0, 0]
    arm_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    armId = p.loadURDF(arm_path, arm_start_pos, arm_start_orientation,
                       useFixedBase=True)

    cube_start_pos = [0.03, 0, 0.005]
    cube_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    cubeId = p.loadURDF("cube_small.urdf", cube_start_pos, cube_start_orientation)

    num_joints = p.getNumJoints(armId)
    gui_print(f"Number of joints: {num_joints}")
    time.sleep(1.5)

    import math
    duration = 5
    steps_per_second = 240
    amplitude = 1.0
    frequency = 0.5

    for joint_idx in [1, 2, 3, 4, 5, 6]:
        gui_print(f"Oscillating joint {joint_idx}")
        for step in range(duration * steps_per_second):
            t = step / steps_per_second
            target = amplitude * math.sin(2 * math.pi * frequency * t)
            p.setJointMotorControl2(armId, joint_idx, p.POSITION_CONTROL,
                                    targetPosition=target)
            p.stepSimulation()
            time.sleep(1. / steps_per_second)

        gui_print(f"Returning joint {joint_idx} to zero")
        p.setJointMotorControl2(armId, joint_idx, p.POSITION_CONTROL, targetPosition=0.0)
        for _ in range(steps_per_second):
            p.stepSimulation()
            time.sleep(1. / steps_per_second)

    gui_print("Simulation complete.")
    time.sleep(2)

    p.disconnect()


if __name__ == "__main__":
    main()

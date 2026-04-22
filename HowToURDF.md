docker build -t lss-ros2-arms .

docker run --rm -it -v ${PWD}\urdf_files:/workspace/urdf_for_pybullet lss-ros2-arms /bin/bash

source /opt/ros/galactic/setup.bash
source /workspace/install/setup.bash
cd /workspace/src/LSS-ROS2-Arms/lss_arm_description/urdf

cp -r /workspace/src/LSS-ROS2-Arms/lss_arm_description/urdf/ /workspace/urdf_for_pybullet/

---

Then, download https://github.com/Lynxmotion/LSS-ROS2-Arms/tree/master/lss_arm_description
You can use https://download-directory.github.io/ to download only that part of the repo.

Unzip to a local folder named lss_arm_description

and move lss_arm_description/models/lss_arm_4dof/meshes to lss_arm_description/meshes

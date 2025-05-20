
from ikpy.chain import Chain
import h5py
import numpy as np

# Load the URDF
# urdf file link - https://github.com/agilexrobotics/mobile_aloha_sim/tree/2843ff11d2695c1563a1c9847f632aa9734f5bbc/aloha_description/arx5-urdf
chain = Chain.from_urdf_file("arx5p2/urdf/arx5p2.urdf")

# Load HDF5 Data
hdf5_file = h5py.File("../dataset_test/aloha_fork_pick_up_compressed/episode_43.hdf5", "r")
qpos = hdf5_file['observations/qpos'][:]  # Shape: (600, 14)

# Select correct joint positions for FK (adjust indices if needed)
for t, joint_state in enumerate(qpos):
    # Assuming first 8 positions match joint1 to joint8
    joint_angles = joint_state[:8]  # Insert 0 for fixed base joint

    fk_matrix = chain.forward_kinematics(joint_angles)
    ee_position = fk_matrix[:3, 3]

    print(f"Timestep {t}: End-Effector Position: {ee_position}")

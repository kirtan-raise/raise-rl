
import mujoco_py as mp
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R


# Load the MuJoCo model
model = mp.load_model_from_path("/home/ubuntu/workspace_rl/raise-rl/training/env/assets/rl.xml")
sim = mp.MjSim(model)


viewer = mp.MjViewer(sim) 

R_matrix = sim.data.get_geom_xmat('harmon_bracket')
rotation = R.from_matrix(R_matrix)
quaternion = rotation.as_quat()
print(quaternion)

# Define the target end-effector position and orientation
target_pos = np.array([0, 0.1, 0])  # Desired position
target_quat = np.array([0.7071068, 0, 0, 0.7071068])  # Desired orientation as quaternion

# Define the inverse kinematics error function
def ik_error(joint_positions):
    # Set the current joint positions in the MuJoCo simulation
    sim_state = sim.get_state()
    sim_state.qpos[:] = joint_positions 
    sim.set_state(sim_state)
    # print(sim.get_state().qpos)
    print(sim.data.qpos)
    sim.forward()

    
    # Forward kinematics: Compute the end-effector position and orientation
    current_end_effector_pos = sim.data.get_geom_xpos('harmon_bracket')
    print(current_end_effector_pos)
    R_matrix = sim.data.get_geom_xmat('harmon_bracket')
    rotation = R.from_matrix(R_matrix)
    current_end_effector_ori = rotation.as_quat()
    print(current_end_effector_ori)
    
    # Compute the error between the current and target end-effector pose
    error_pos = np.linalg.norm(current_end_effector_pos - target_pos)
    error_ori = np.linalg.norm(current_end_effector_ori - target_quat)
    total_error = error_pos + error_ori

    print(total_error)
    
    return total_error

# # Initial guess for joint positions
initial_joint_positions = np.zeros(model.nq)  # Assuming all joint angles start at 0
# initial_joint_positions[0] = 1  # Assuming all joint angles start at 0


# # Perform inverse kinematics optimization
result = minimize(ik_error, initial_joint_positions, method='slsqp', 
                  options={'ftol': 1e-5,
                           'disp': True,
                           'eps': 1e-2
                           },)

# # Extract the optimized joint positions
optimized_joint_positions = result.x
print(type(optimized_joint_positions))
print(optimized_joint_positions)

# Set the optimized joint positions to the MuJoCo simulation
sim_state = sim.get_state()
print(type(sim_state.qpos))
sim_state.qpos[:] = optimized_joint_positions
sim.set_state(sim_state)

sim.forward()

while True:
    viewer.render()

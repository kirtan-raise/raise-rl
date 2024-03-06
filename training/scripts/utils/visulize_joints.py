import mujoco_py as mp
import numpy as np


# Load the MuJoCo model
model = mp.load_model_from_path("/home/ubuntu/workspace_rl/raise-rl/training/env/assets/rl.xml")
sim = mp.MjSim(model)


viewer = mp.MjViewer(sim) 

joint_positions =   [  -0.0277612,   -0.517701,     1.48551,     -4.1094,    -1.54304,      1.5708]
sim_state = sim.get_state()
sim_state.qpos[:] = joint_positions 
sim.set_state(sim_state)
sim.forward()

while True:
    viewer.render()
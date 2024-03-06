import mujoco_py as mp


# Load the MuJoCo model
model = mp.load_model_from_path("/home/ubuntu/workspace_rl/raise-rl/training/env/assets/rl.xml")
sim = mp.MjSim(model)
viewer = mp.MjViewer(sim) 


# Current joint values
print(sim.data.qpos)
sim_state = sim.get_state()  
print(sim_state.qpos)

# Get body quat
print(sim.data.get_body_xquat('left_wrist_3_link'))

# Get geom_quat shape
print(model.geom_quat.shape)


print(model.geom_name2id('harmon_bracket'))

R_matrix = sim.data.get_geom_xmat('harmon_bracket')

# Set new joint values
sim_state.qpos[0] = 1.2  
sim.set_state(sim_state)

# Move simulatio by a step 
sim.step()

# Position of a geometry
sim.data.get_geom_xpos('harmon_bracket')

# Initial/ from model
print(model.geom_quat)


# Visulization
while True:
    sim.step()
    viewer.render()
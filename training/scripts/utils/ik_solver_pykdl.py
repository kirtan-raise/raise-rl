
import numpy as np
import PyKDL as kdl
import kdl_parser_py.urdf


# Load the MuJoCo model
urdf_file = "/home/ubuntu/workspace_rl/raise-rl/training/env/assets/ros_urdf/2_0/rl_2.urdf"
(ok, kdl_tree) =  kdl_parser_py.urdf.treeFromFile(urdf_file)

chain = kdl_tree.getChain("world", "left_harmon_bracket_tip")
end_effector_link_name = chain.getSegment(chain.getNrOfSegments() - 1).getName()
# print(end_effector_link_name)
# print(chain)
# print(chain.getNrOfSegments() )
initial_guess = [-0.1084817091571253, -0.9476657670787354, 0.9476657670787354, 
                 -3.6235062084593714, -1.4631193319903772, 1.5518755912780762]


limits = [6.283185307179586, 6.283185307179586, 3.141592653589793, 
                     6.283185307179586, 6.283185307179586, 6.283185307179586]

qmin = kdl.JntArray(chain.getNrOfJoints())
qmax = kdl.JntArray(chain.getNrOfJoints())

for i in range(len(limits)):
    qmin[i] = -1*limits[i]
    qmax[i] = limits[i]

# print(qmin)
# print(qmax)

fk_solver = kdl.ChainFkSolverPos_recursive(chain)
ik_solver = kdl.ChainIkSolverVel_pinv(chain)
ik_solver_pos = kdl.ChainIkSolverPos_NR_JL(chain, qmin, qmax, fk_solver, ik_solver, 500, 1e-4)

initial_joint_positions = kdl.JntArray(chain.getNrOfJoints())
# initial_joint_positions  = initial_guess

for i in range(chain.getNrOfJoints()):
    initial_joint_positions[i] = initial_guess[i]

desired_joint_positions = kdl.JntArray(chain.getNrOfJoints())

kdled_pose = kdl.Frame(kdl.Rotation.RPY(np.pi/2, 0.0, 0), kdl.Vector(0, 0.1, -0.15+0.0762 ))

res = ik_solver_pos.CartToJnt(initial_joint_positions, kdled_pose, desired_joint_positions)
F = kdl.Frame()
print(res)
print(desired_joint_positions)
fk_solver.JntToCart(desired_joint_positions, F, chain.getNrOfSegments())
print(F)
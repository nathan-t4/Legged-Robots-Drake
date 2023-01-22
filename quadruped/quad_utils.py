import numpy as np
from pydrake.multibody.tree import JointIndex
from utils import MakeNamedViewPositions

def set_quad_pose(plant, context, mode='default'):
    # Initialize joint positions for lying down config
    if mode == 'default':
        q0 = (0.1, -2.0, 2.6)
        z0 = 0.08
    elif mode == 'stand':
        q0 = (0.1,-0.8,1.2)
        z0 = 0.33
    elif mode == 'test':
        q0 = (0, -0.8, 1.6)
        z0 = 0.3
    else:
        raise RuntimeError('Invalid quad pose mode')
    
    PositionView = MakeNamedViewPositions(plant, "Positions")
    (hip_roll, hip_pitch, knee) = q0
    q0 = PositionView(np.zeros(plant.num_positions()))
    q0.torso_to_abduct_fl_j     = hip_roll
    q0.abduct_fl_to_thigh_fl_j  = hip_pitch
    q0.thigh_fl_to_knee_fl_j    = knee
    q0.torso_to_abduct_fr_j     = -hip_roll
    q0.abduct_fr_to_thigh_fr_j  = hip_pitch
    q0.thigh_fr_to_knee_fr_j    = knee
    q0.torso_to_abduct_hl_j     = hip_roll
    q0.abduct_hl_to_thigh_hl_j  = hip_pitch
    q0.thigh_hl_to_knee_hl_j    = knee
    q0.torso_to_abduct_hr_j     = -hip_roll
    q0.abduct_hr_to_thigh_hr_j  = hip_pitch
    q0.thigh_hr_to_knee_hr_j    = knee
    q0.body_qw                  = 1.0
    q0.body_z                   = z0
    plant.SetPositions(context, q0[:])

def GetQuadStateProjectionMatrix(plant):
    S = np.zeros((2*plant.num_actuated_dofs, plant.num_multibody_states()))
    num_q = plant.num_positions()
    j = 0
    for i in range(plant.num_joints()):
        joint = plant.get_joint(JointIndex(i))
        # skip floating body indices
        if joint.num_positions() != 1:
            continue
        S[j, joint.position_start()] = 1
        S[12+j, num_q + joint.velocity_start()] = 1
        j = j+1
    
    return S

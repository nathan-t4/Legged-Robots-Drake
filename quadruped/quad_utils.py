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

# Helper functions for ComDynamicsFullKinematicsPlanner
def setup_gait(gait='walking_trot'):
     # setup gait
    is_laterally_symmetric = False
    check_self_collision = False
    gait_params = {}
    if gait == 'running_trot':
        N = 21
        in_stance = np.zeros((4, N))
        in_stance[1, 3:17] = 1
        in_stance[2, 3:17] = 1
        speed = 0.9
        stride_length = .55
        is_laterally_symmetric = True
    elif gait == 'walking_trot':
        N = 21
        in_stance = np.zeros((4, N))
        in_stance[0, :11] = 1
        in_stance[1, 8:N] = 1
        in_stance[2, 8:N] = 1
        in_stance[3, :11] = 1
        speed = 0.4
        stride_length = .25
        is_laterally_symmetric = True
    elif gait == 'rotary_gallop':
        N = 41
        in_stance = np.zeros((4, N))
        in_stance[0, 7:19] = 1
        in_stance[1, 3:15] = 1
        in_stance[2, 24:35] = 1
        in_stance[3, 26:38] = 1
        speed = 1
        stride_length = .65
        check_self_collision = True
    elif gait == 'bound':
        N = 41
        in_stance = np.zeros((4, N))
        in_stance[0, 6:18] = 1
        in_stance[1, 6:18] = 1
        in_stance[2, 21:32] = 1
        in_stance[3, 21:32] = 1
        speed = 1.2
        stride_length = .55
        check_self_collision = True
    else:
        raise RuntimeError('Unknown gait.')

    T = stride_length / speed
    if is_laterally_symmetric:
        T = T / 2.0

    gait_params['N'] = N
    gait_params['in_stance'] = in_stance
    gait_params['speed'] = speed
    gait_params['stride_length'] = stride_length
    gait_params['check_self_collision'] = check_self_collision
    gait_params['is_laterally_symmetric'] = is_laterally_symmetric
    gait_params['T'] = T

    return gait_params

def HalfStrideToFullStride(PositionView, a):
    b = PositionView(np.copy(a))

    b.body_y = -a.body_y
    # Mirror quaternion so that roll=-roll, yaw=-yaw
    b.body_qx = -a.body_qx
    b.body_qz = -a.body_qz

    b.torso_to_abduct_fl_j = -a.torso_to_abduct_fr_j
    b.torso_to_abduct_fr_j = -a.torso_to_abduct_fl_j
    b.torso_to_abduct_hl_j = -a.torso_to_abduct_hr_j
    b.torso_to_abduct_hr_j = -a.torso_to_abduct_hl_j

    b.abduct_fl_to_thigh_fl_j = a.abduct_fr_to_thigh_fr_j
    b.abduct_fr_to_thigh_fr_j = a.abduct_fl_to_thigh_fl_j
    b.abduct_hl_to_thigh_hl_j = a.abduct_hr_to_thigh_hr_j
    b.abduct_hr_to_thigh_hr_j = a.abduct_hl_to_thigh_hl_j

    b.thigh_fl_to_knee_fl_j = a.thigh_fr_to_knee_fr_j
    b.thigh_fr_to_knee_fr_j = a.thigh_fl_to_knee_fl_j
    b.thigh_hl_to_knee_hl_j = a.thigh_hr_to_knee_hr_j
    b.thigh_hr_to_knee_hr_j = a.thigh_hl_to_knee_hl_j

    return b

# Helper functions for QuadPidController

def SetQuadActuationView(u_view, u):
    '''
        Set actuation view for quadrupeds
        TODO
        - make actuator names and order consistent for all quadruped models
        - try to do change u_view directly (so no need to return u_view)
    '''
    u_view.torso_to_abduct_fl_j_actuator     = u[0]
    u_view.abduct_fl_to_thigh_fl_j_actuator  = u[1]
    u_view.thigh_fl_to_knee_fl_j_actuator    = u[2]
    u_view.torso_to_abduct_fr_j_actuator     = u[3]
    u_view.abduct_fr_to_thigh_fr_j_actuator  = u[4]
    u_view.thigh_fr_to_knee_fr_j_actuator    = u[5]
    u_view.torso_to_abduct_hl_j_actuator     = u[6]
    u_view.abduct_hl_to_thigh_hl_j_actuator  = u[7]
    u_view.thigh_hl_to_knee_hl_j_actuator    = u[8]
    u_view.torso_to_abduct_hr_j_actuator     = u[9]
    u_view.abduct_hr_to_thigh_hr_j_actuator  = u[10]
    u_view.thigh_hr_to_knee_hr_j_actuator    = u[11]

    return u_view

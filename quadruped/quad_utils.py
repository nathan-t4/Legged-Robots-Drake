import numpy as np
from scipy.spatial.transform import Rotation as R

from pydrake.multibody.tree import JointIndex
from pydrake.common.containers import namedview
from pydrake.trajectories import PiecewisePolynomial
from utils import MakeNamedViewPositions, MakeNamedViewState, quat2eul

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
    S = np.zeros((2*plant.num_actuated_dofs(), plant.num_multibody_states()))
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

def HalfStrideToFullStride(StateView, a):
    b = StateView(np.copy(a))
    b.body_y = -a.body_y
    # Mirror quaternion so that roll=-roll, yaw=-yaw
    b.body_qx = -a.body_qx
    b.body_qz = -a.body_qz

    b.torso_to_abduct_fr_j_q = -a.torso_to_abduct_fl_j_q
    b.torso_to_abduct_fl_j_q = -a.torso_to_abduct_fr_j_q
    b.torso_to_abduct_hl_j_q = -a.torso_to_abduct_hr_j_q
    b.torso_to_abduct_hr_j_q = -a.torso_to_abduct_hl_j_q

    b.abduct_fl_to_thigh_fl_j_q = a.abduct_fr_to_thigh_fr_j_q
    b.abduct_fr_to_thigh_fr_j_q = a.abduct_fl_to_thigh_fl_j_q
    b.abduct_hl_to_thigh_hl_j_q = a.abduct_hr_to_thigh_hr_j_q
    b.abduct_hr_to_thigh_hr_j_q = a.abduct_hl_to_thigh_hl_j_q

    b.thigh_fl_to_knee_fl_j_q = a.thigh_fr_to_knee_fr_j_q
    b.thigh_fr_to_knee_fr_j_q = a.thigh_fl_to_knee_fl_j_q
    b.thigh_hl_to_knee_hl_j_q = a.thigh_hr_to_knee_hr_j_q
    b.thigh_hr_to_knee_hr_j_q = a.thigh_hl_to_knee_hl_j_q

    b.torso_to_abduct_fr_j_w = -a.torso_to_abduct_fl_j_w
    b.torso_to_abduct_fl_j_w = -a.torso_to_abduct_fr_j_w
    b.torso_to_abduct_hl_j_w = -a.torso_to_abduct_hr_j_w
    b.torso_to_abduct_hr_j_w = -a.torso_to_abduct_hl_j_w

    b.abduct_fl_to_thigh_fl_j_w = a.abduct_fr_to_thigh_fr_j_w
    b.abduct_fr_to_thigh_fr_j_w = a.abduct_fl_to_thigh_fl_j_w
    b.abduct_hl_to_thigh_hl_j_w = a.abduct_hr_to_thigh_hr_j_w
    b.abduct_hr_to_thigh_hr_j_w = a.abduct_hl_to_thigh_hl_j_w

    b.thigh_fl_to_knee_fl_j_w = a.thigh_fr_to_knee_fr_j_w
    b.thigh_fr_to_knee_fr_j_w = a.thigh_fl_to_knee_fl_j_w
    b.thigh_hl_to_knee_hl_j_w = a.thigh_hr_to_knee_hr_j_w
    b.thigh_hr_to_knee_hr_j_w = a.thigh_hl_to_knee_hl_j_w

    return b

def pHalfStrideToFullStride(PositionView, a):
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
        Sets actuation view for quadrupeds
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

def HalfStrideToTraj(**kwargs):
    '''
    Take half stride state trajectory to num_strides full stride trajectories
    - TODO: use full state ([q,v])
    '''
    # assert isinstance(kwargs['plant'], ) # TODO
    assert isinstance(kwargs['q_sol'], PiecewisePolynomial)
    assert isinstance(kwargs['v_sol'], PiecewisePolynomial)
    assert isinstance(kwargs['t_sol'], np.ndarray) # TODO
    assert isinstance(kwargs['stride_length'], float)
    assert isinstance(kwargs['num_strides'], float)

    plant = kwargs['plant']
    q_sol = kwargs['q_sol']
    v_sol = kwargs['v_sol']
    t_sol = kwargs['t_sol']
    stride_length = kwargs['stride_length']
    num_strides = kwargs['num_strides']
    is_laterally_symmetric = kwargs['is_laterally_symmetric']

    StateView = MakeNamedViewState(plant, 'x')
    t0, tf = t_sol[0], t_sol[-1]
    T = tf*num_strides*(2.0 if is_laterally_symmetric else 1.0)
    t_steps = np.hstack((np.arange(t0, T, 1.0/32.0), T)) # what is 1.0/32.0? dt from planner?
    
    # full_q_sol = np.empty(shape=(plant.num_positions(),len(t_steps)), dtype=np.float64) 
    full_x_sol = np.empty(shape=(plant.num_multibody_states(),len(t_steps)), dtype=np.float64)

    for i in range(len(t_steps)):
        t = t_steps[i]
        stride = (t - t0) // (tf - t0)
        ts = (t - t0) % (tf - t0)

        # qt_view = PositionView(q_sol.value(ts))
        xt_view = StateView(np.concatenate((q_sol.value(ts), v_sol.value(ts))))
        if is_laterally_symmetric:
            if stride % 2 == 1:
                # qt_view = HalfStrideToFullStride(PositionView, qt_view)
                # qt_view.body_x += stride_length/2.0
                xt_view = HalfStrideToFullStride(StateView, xt_view)
                xt_view.body_x += stride_length/2.0
            stride = stride // 2

        # qt_view.body_x += stride*stride_length
        xt_view.body_x += stride*stride_length

        # full_q_sol[:,i] = np.reshape(qt_view[:], -1)
        full_x_sol[:,i] = np.reshape(xt_view[:], -1)

    # return full_q_sol, t_steps
    return full_x_sol, t_steps

def ExcludeQuadFloatingBaseCoords(plant, A, axis=0):
    '''
        Exclude floating base coordinates from full state vector A
        np.vstack((qb, q, vb, v)) -> np.vstack((q,v))
    '''
    floating_body_idx = plant.GetFloatingBaseBodies() # this is a set
    has_quaternion_dofs = False

    try:
        floating_body = plant.get_body(floating_body_idx.pop()) 
        has_quaternion_dofs = floating_body.has_quaternion_dofs()
    except KeyError:
        pass

    # print('has quaternion dofs', has_quaternion_dofs)
    floating_base_end_idx = 7 if has_quaternion_dofs else 6
    if axis == 0:
        return np.vstack(
            (A[floating_base_end_idx:plant.num_positions(),:], A[plant.num_positions()+6:,:]),
            dtype=float
        )
    elif axis == 1:
        return np.hstack(
            (A[:,floating_base_end_idx:plant.num_positions()], A[:,plant.num_positions()+6:]),
            dtype=float
        )
    else:
        raise NotImplementedError()

def ExcludeQuadActuatedCoords(plant, A, axis=0, quat=False, quat2rpy=False):
    '''
        Exclude actuated coordinates from full state vector A
        np.vstack((qb, q, vb, v)) -> np.vstack((qb, vb))
    '''    
    num_positions = plant.num_positions()
    floating_base_end_idx = 7 if quat else 6 # (4+3) or (3+3)
    if axis == 0:
        if A.ndim == 1:
            return np.concatenate(
                (A[:floating_base_end_idx], A[num_positions:num_positions+6]), 
                dtype=float
            )
        elif A.ndim == 2:
            return np.vstack(
                (A[:floating_base_end_idx,:], A[num_positions:num_positions+6,:]),
                dtype=float
            )
    elif axis == 1:
        return np.hstack(
            (A[:,:floating_base_end_idx], A[:,num_positions:num_positions+6]),
            dtype=float
        )
    else:
        raise NotImplementedError()

def quat2rpy():
    pass

def CalcA(self, phi):
    '''
        Calculate discrete dynamics A_k for quadruped under assumptions of
        1. single-lumped mass body 
        2. negligible roll & pitch
        3. off-diagonal terms of inertia tensor are negligible

        x(k+1) = A_k @ x(k) + B_k @ f(k) + g
    '''
    zeros = np.zeros((3,3))
    ones = np.ones((3,3))
    Rz = R.from_euler('z',[phi],degrees=False)
    A = np.array(
        [[ones, zeros, Rz*self._dt, zeros],
            [zeros, ones, zeros, ones*self._dt],
            [zeros, zeros, ones, zeros],
            [zeros, zeros, zeros, ones]]
    )
    return A
    
def CalcB(phi, r, dt, M, Ib):
    '''
        Calculate discrete dynamics B_k under assumptions:
        1. single-lumped mass body 
        2. negligible roll & pitch
        3. off-diagonal terms of inertia tensor are negligible
        
        x(k+1) = A_k @ x(k) + B_k @ f(k) + g
    '''
    def to_skew_symmetric_matrix(lhs):
        assert len(lhs) == 3
        return np.array([[0,-lhs[2],lhs[1]],[lhs[2],0,-lhs[0]],[-lhs[1],lhs[0],0]])
    zeros = np.zeros((3,3))
    ones = np.ones((3,3))
    Rz = R.from_euler('z',[phi],degrees=False)
    Ig = Rz * Ib * Rz.T
    B = np.empty(shape=(12,3*np.shape(r)[1]), dtype=float)
    for i in range(np.shape(r)[1]):
        cross = np.linalg.inv(Ig)*to_skew_symmetric_matrix(r[:,i])
        B[:,i] = [zeros, zeros, cross*dt, ones*dt/M]
    return B

def CalcG(): 
    '''
        Calculate discrete dynamics g under assumptions:
        1. single-lumped mass body 
        2. negligible roll & pitch
        3. off-diagonal terms of inertia tensor are negligible
        
        x(k+1) = A_k @ x(k) + B_k @ f(k) + g
    '''
    g = np.vstack((np.zeros((9,1)), [[0],[0],[-9.81]]))
    assert np.shape(g) == (12,1)
    return g

def CalcFootPositions(plant, context):
    lf_foot_frame = plant.GetFrameByName('LF_FOOT')
    rf_foot_frame = plant.GetFrameByName('RF_FOOT')
    lh_foot_frame = plant.GetFrameByName('LH_FOOT')
    rh_foot_frame = plant.GetFrameByName('RH_FOOT')
    frames = [lf_foot_frame, rf_foot_frame, lh_foot_frame, rh_foot_frame]
    r = np.empty(shape=(3,len(frames)), dtype=float)
    for i in range(len(frames)):
        r[:,i] = plant.CalcPointsPositions(
            context=context,
            frame_B=frames[i],
            p_BQi=[0,0,0],
            frame_A=plant.world_frame()
        )
    return r
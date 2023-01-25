import numpy as np
from pydrake.all import (JointActuatorIndex, JointIndex, namedview, ExtractGradient, PiecewisePolynomial)
from typing import List, Tuple
'''
    Source for MakeNamedView*: https://github.com/RussTedrake/underactuated/blob/master/underactuated/utils.py
 '''
def MakeNamedViewPositions(mbp, view_name, add_suffix_if_single_position=False):
    names = [None] * mbp.num_positions()
    for ind in range(mbp.num_joints()):
        joint = mbp.get_joint(JointIndex(ind))
        if joint.num_positions() == 1 and not add_suffix_if_single_position:
            names[joint.position_start()] = joint.name()
        else:
            for i in range(joint.num_positions()):
                names[joint.position_start() + i] = \
                    f"{joint.name()}_{joint.position_suffix(i)}"
    for ind in mbp.GetFloatingBaseBodies():
        body = mbp.get_body(ind)
        start = body.floating_positions_start()
        for i in range(7 if body.has_quaternion_dofs() else 6):
            names[start
                  + i] = f"{body.name()}_{body.floating_position_suffix(i)}"
    return namedview(view_name, names)

def MakeNamedViewVelocities(mbp,
                            view_name,
                            add_suffix_if_single_velocity=False):
    names = [None] * mbp.num_velocities()
    for ind in range(mbp.num_joints()):
        joint = mbp.get_joint(JointIndex(ind))
        if joint.num_velocities() == 1 and not add_suffix_if_single_velocity:
            names[joint.velocity_start()] = joint.name()
        else:
            for i in range(joint.num_velocities()):
                names[joint.velocity_start() + i] = \
                    f"{joint.name()}_{joint.velocity_suffix(i)}"
    for ind in mbp.GetFloatingBaseBodies():
        body = mbp.get_body(ind)
        start = body.floating_velocities_start() - mbp.num_positions()
        for i in range(6):
            names[start
                  + i] = f"{body.name()}_{body.floating_velocity_suffix(i)}"
    return namedview(view_name, names)


def MakeNamedViewState(mbp, view_name):
    # TODO: this could become a nested named view, pending
    # https://github.com/RobotLocomotion/drake/pull/14973
    pview = MakeNamedViewPositions(mbp, f"{view_name}_pos", True)
    vview = MakeNamedViewVelocities(mbp, f"{view_name}_vel", True)
    return namedview(view_name, pview.get_fields() + vview.get_fields())


def MakeNamedViewActuation(mbp, view_name):
    names = [None] * mbp.get_actuation_input_port().size()
    for ind in range(mbp.num_actuators()):
        actuator = mbp.get_joint_actuator(JointActuatorIndex(ind))
        assert actuator.num_inputs() == 1
        names[actuator.input_start()] = actuator.name()
    return namedview(view_name, names)

# Need this because a==b returns True even if a = AutoDiffXd(1, [1, 2]), b= AutoDiffXd(2, [3, 4])
# That's the behavior of AutoDiffXd in C++, also.
def autoDiffArrayEqual(a,b):
    return np.array_equal(a, b) and np.array_equal(ExtractGradient(a), ExtractGradient(b))

def VerifyTrajectoryIsValidPolynomial(traj, **kwargs) -> PiecewisePolynomial:
    '''
        Verify traj is a valid (in the sense of MPC) PiecewisePolynomial 
        - Currently only used in ModelPredictiveController()
    '''    
    assert 'shape' in kwargs, 'Pass expected shape of traj as argument kwargs[\'shape\']'
    assert isinstance(kwargs['shape'], Tuple[float]) and len(kwargs['shape']) == 2, \
        'Argument kwargs[\'shape\'] should be a tuple of length 2'

    poly_traj = traj

    if isinstance(traj, List[float]) and 'total_time' in kwargs:
        # Create new PiecewisePolynomial with traj that is scaled to total_time
        breaks = np.arange(0, len(traj)) * (kwargs['total_time'] / len(traj))
        samples = traj
        poly_traj = PiecewisePolynomial().CubicWithContinuousSecondDerivatives(
            breaks=breaks,
            samples=samples,
            periodic_end_condition=True,
            zero_end_point_derivatives=False
        )
    elif traj is None:
        poly_traj = PiecewisePolynomial(np.zeros(kwargs['shape']))
    else:
        raise RuntimeError('Invalid trajectory input')
    
    assert (poly_traj.rows(), poly_traj.columns()) == kwargs['shape']
    
    return poly_traj

def GetStateProjectionMatrix(plant, robot_type):
    '''
        Returns state projection matrix for plant
        Source: https://github.com/RussTedrake/underactuated/blob/master/examples/littledog.ipynb 
    '''
    if robot_type == 'quad' or robot_type == 'biped':
        S = np.zeros((2*plant.num_actuated_dofs(), plant.num_multibody_states()))
        num_positions = plant.num_positions()
        num_actuated_dofs = plant.num_actuated_dofs()
        j = 0
        for i in range(plant.num_joints()):
            joint = plant.get_joint(JointIndex(i))
            # skip floating body indices
            if joint.num_positions() != 1:
                continue
            S[j, joint.position_start()] = 1
            S[num_actuated_dofs+j, num_positions + joint.velocity_start()] = 1 # TODO: verify (was 12 for quadruped)
            j = j+1        
    else:
        raise NotImplementedError('Unrecognized robot type')
    return S

def ModifyGains(plant, kp, kd, robot_type):
    '''
        Modifies gains (kp, kd) for robot_type
        - Assumes kp, kd are mutable
        Source: https://github.com/RussTedrake/underactuated/blob/master/examples/littledog.ipynb 
    '''
    if robot_type == 'quad':
        j = 0
        for i in range(plant.num_joints()):
            joint = plant.get_joint(JointIndex(i))
            # use lower gain for the knee joints
            if 'knee' in joint.name():
                kd[j] = 0.1
            j = j+1
    elif robot_type == 'biped':
        # TODO
        pass
    else:
        raise NotImplementedError('Unrecognized robot type')

def SetActuationView(u_view, u, robot_type):
    '''
        Set actuation view for robot_type
        TODO:
        - make actuator names and order consistent for all quadruped models
        - try to do change u_view directly (so no need to return u_view)
    '''
    if robot_type == 'quad':
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
    elif robot_type == 'biped':
        pass
    else:
        raise NotImplementedError('Unrecognized robot type')
    
    return u_view

def GetFootFrames(plant):
    '''[TODO] Returns foot frames of plant in a list'''
    pass

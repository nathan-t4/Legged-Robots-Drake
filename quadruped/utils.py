import numpy as np
from pydrake.all import (JointActuatorIndex, JointIndex, namedview, ExtractGradient, PiecewisePolynomial)
'''
    Source: https://github.com/RussTedrake/underactuated/blob/master/underactuated/utils.py
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

def VerifyTrajectoryIsValidPolynomial(traj, **kwargs):
    assert isinstance(traj, PiecewisePolynomial)
    if 'shape' in kwargs:
        # Verify shape of traj
        assert((traj.rows(), traj.columns()) == kwargs['shape']) 
    
    return True
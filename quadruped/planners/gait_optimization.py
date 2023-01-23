import numpy as np
import os
import sys

from functools import partial
from types import SimpleNamespace
from argparse import ArgumentParser

from pydrake.systems.framework import DiagramBuilder
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.geometry import StartMeshcat, MeshcatVisualizer
from pydrake.solvers import MathematicalProgram, Solve
from pydrake.math import RotationMatrix
from pydrake.autodiffutils import ExtractGradient, InitializeAutoDiff, AutoDiffXd, ExtractValue
from pydrake.multibody.inverse_kinematics import InverseKinematics, AddUnitQuaternionConstraintOnPlant, OrientationConstraint, PositionConstraint, MinimumDistanceConstraint
from pydrake.multibody.tree import JacobianWrtVariable
from pydrake.solvers import SnoptSolver
from pydrake.systems.analysis import Simulator
from pydrake.visualization import AddDefaultVisualization
from pydrake.trajectories import PiecewisePolynomial
from pydrake.systems.primitives import Saturation
from pydrake.all import eq

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from quad_utils import set_quad_pose
from utils import MakeNamedViewPositions, MakeNamedViewVelocities, autoDiffArrayEqual
from controllers.QuadPidController import QuadPidController

'''
    Credits to: https://github.com/RussTedrake/underactuated/blob/master/examples/littledog.ipynb

    COM dynamics + full kinematics planner

    TODO:
    - Add additional constraints
    - Make motion dynamically-feasible
    - Use meldis instead of meshcat (or use externally hosted meshcat)
    - Modify naming convention in mini cheetah urdf file
    - try with better q0 (walking_trot is fine with current q0)

    TODO (formatting):
    - Create class for GaitOptimization? Then make instance of GaitOptimization and visualize/simulate in simulate.py
'''
meshcat = StartMeshcat()

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

    
def gait_optimization_prog(plant, plant_context, gait='walking_trot'):
    q0 = plant.GetPositions(plant_context)
    body_frame = plant.GetFrameByName('body')
    quad = plant.GetModelInstanceByName('quad')

    PositionView = MakeNamedViewPositions(plant, 'Positions')
    VelocityView = MakeNamedViewVelocities(plant, 'Velocities')

    mu = 1 # rubber on rubber
    total_mass = plant.CalcTotalMass(plant_context, [quad])
    gravity = plant.gravity_field().gravity_vector()
    # TODO: inconsistent nq definition?
    nq = 12 
    foot_frame = [
        plant.GetFrameByName('LF_FOOT'),
        plant.GetFrameByName('RF_FOOT'),
        plant.GetFrameByName('LH_FOOT'),
        plant.GetFrameByName('RH_FOOT')]
    
    gait_params = SimpleNamespace(**setup_gait(gait=gait))

    prog = MathematicalProgram()

    # Time steps
    h = prog.NewContinuousVariables(gait_params.N-1, 'h')
    prog.AddBoundingBoxConstraint(0.5*gait_params.T/gait_params.N, 2.0*gait_params.T/gait_params.N, h)
    prog.AddLinearConstraint(sum(h) >= 0.9*gait_params.T)
    prog.AddLinearConstraint(sum(h) <= 1.1*gait_params.T)

    # Create one context per timestep (to maximize cache hits)
    context = [plant.CreateDefaultContext() for i in range(gait_params.N)]
    # We could get rid of this by implementing a few more Jacobians in MultibodyPlant
    ad_plant = plant.ToAutoDiffXd()

    # Joint positions and velocities
    nq = plant.num_positions()
    nv = plant.num_velocities()
    q = prog.NewContinuousVariables(nq, gait_params.N, 'q')
    v = prog.NewContinuousVariables(nv, gait_params.N, 'v')
    q_view = PositionView(q)
    v_view = VelocityView(v)
    q0_view = PositionView(q0)
    # Joint costs
    q_cost = PositionView([1]*nq)
    v_cost = VelocityView([1]*nv)
    q_cost.body_x = 0
    q_cost.body_y = 0
    q_cost.body_qx = 0
    q_cost.body_qy = 0
    q_cost.body_qz = 0
    q_cost.body_qw = 0
    q_cost.torso_to_abduct_fl_j  = 5
    q_cost.torso_to_abduct_fr_j  = 5
    q_cost.torso_to_abduct_hl_j  = 5
    q_cost.torso_to_abduct_hr_j  = 5
    v_cost.body_vx = 0
    v_cost.body_wx = 0
    v_cost.body_wy = 0
    v_cost.body_wz = 0
    for n in range(gait_params.N):
        # Joint limits
        prog.AddBoundingBoxConstraint(plant.GetPositionLowerLimits(),
                                      plant.GetPositionUpperLimits(), q[:, n])
        # Joint velocity limits
        prog.AddBoundingBoxConstraint(plant.GetVelocityLowerLimits(),
                                      plant.GetVelocityUpperLimits(), v[:, n])
        # Unit quaternions
        AddUnitQuaternionConstraintOnPlant(plant, q[:,n], prog)
        # Body orientation
        prog.AddConstraint(OrientationConstraint(plant,
                                                 body_frame, RotationMatrix(),
                                                 plant.world_frame(), RotationMatrix(),
                                                 0.1, context[n]), q[:,n])
        # Initial guess for all joint angles is the home position
        prog.SetInitialGuess(q[:,n], q0)  # Solvers get stuck if the quaternion is initialized with all zeros.

        # Running costs:
        prog.AddQuadraticErrorCost(np.diag(q_cost), q0, q[:,n])
        prog.AddQuadraticErrorCost(np.diag(v_cost), [0]*nv, v[:,n])

    # Make a new autodiff context for this constraint (to maximize cache hits)
    ad_velocity_dynamics_context = [
        ad_plant.CreateDefaultContext() for i in range(gait_params.N)
    ]

    def velocity_dynamics_constraint(vars, context_index):
        h, q, v, qn = np.split(vars, [1, 1+nq, 1+nq+nv])
        if isinstance(vars[0], AutoDiffXd):
            if not autoDiffArrayEqual(
                    q,
                    ad_plant.GetPositions(
                        ad_velocity_dynamics_context[context_index])):
                ad_plant.SetPositions(
                    ad_velocity_dynamics_context[context_index], q)
            v_from_qdot = ad_plant.MapQDotToVelocity(
                ad_velocity_dynamics_context[context_index], (qn - q) / h)
        else:
            if not np.array_equal(q, plant.GetPositions(
                    context[context_index])):
                plant.SetPositions(context[context_index], q)
            v_from_qdot = plant.MapQDotToVelocity(context[context_index],
                                                  (qn - q) / h)
        return v - v_from_qdot
    for n in range(gait_params.N-1):
        prog.AddConstraint(partial(velocity_dynamics_constraint,
                                   context_index=n),
                           lb=[0] * nv,
                           ub=[0] * nv,
                           vars=np.concatenate(
                               ([h[n]], q[:, n], v[:, n], q[:, n + 1])))

    # Contact forces
    contact_force = [
        prog.NewContinuousVariables(3, gait_params.N - 1, f'foot{foot}_contact_force')
        for foot in range(4)
    ]
    for n in range(gait_params.N-1):
        for foot in range(4):
            # Linear friction cone
            prog.AddLinearConstraint(
                contact_force[foot][0, n] <= mu * contact_force[foot][2, n])
            prog.AddLinearConstraint(
                -contact_force[foot][0, n] <= mu * contact_force[foot][2, n])
            prog.AddLinearConstraint(
                contact_force[foot][1, n] <= mu * contact_force[foot][2, n])
            prog.AddLinearConstraint(
                -contact_force[foot][1, n] <= mu * contact_force[foot][2, n])
            # normal force >=0, normal_force == 0 if not in_stance - Complimentary constraints on contact wrench & distance
            prog.AddBoundingBoxConstraint(
                0, gait_params.in_stance[foot, n] * 4 * 9.81 * total_mass,
                contact_force[foot][2, n])

    # Center of mass variables and constraints
    com = prog.NewContinuousVariables(3, gait_params.N, 'com')
    comdot = prog.NewContinuousVariables(3, gait_params.N, 'comdot')
    comddot = prog.NewContinuousVariables(3, gait_params.N-1, 'comddot')
    # Initial CoM x,y position == 0
    prog.AddBoundingBoxConstraint(0, 0, com[:2,0])
    # Initial CoM z vel == 0
    prog.AddBoundingBoxConstraint(0, 0, comdot[2,0])
    # CoM height
    prog.AddBoundingBoxConstraint(.125, np.inf, com[2,:])
    # CoM x velocity >= 0
    prog.AddBoundingBoxConstraint(0, np.inf, comdot[0,:])
    # CoM final x position
    if gait_params.is_laterally_symmetric:
        prog.AddBoundingBoxConstraint(gait_params.stride_length / 2.0, gait_params.stride_length / 2.0,
                                      com[0, -1])
    else:
        prog.AddBoundingBoxConstraint(gait_params.stride_length, gait_params.stride_length, com[0, -1])
    # CoM dynamics
    for n in range(gait_params.N-1):
        # Note: The original matlab implementation used backwards Euler (here and throughout),
        # which is a little more consistent with the LCP contact models.
        prog.AddConstraint(eq(com[:, n+1], com[:,n] + h[n]*comdot[:,n]))
        prog.AddConstraint(eq(comdot[:, n+1], comdot[:,n] + h[n]*comddot[:,n]))
        prog.AddConstraint(
            eq(
                total_mass * comddot[:, n],
                sum(contact_force[i][:, n] for i in range(4))
                + total_mass * gravity))

    # Angular momentum (about the center of mass)
    H = prog.NewContinuousVariables(3, gait_params.N, 'H')
    Hdot = prog.NewContinuousVariables(3, gait_params.N-1, 'Hdot')
    prog.SetInitialGuess(H, np.zeros((3, gait_params.N)))
    prog.SetInitialGuess(Hdot, np.zeros((3,gait_params.N-1)))
    # Hdot = sum_i cross(p_FootiW-com, contact_force_i)
    def angular_momentum_constraint(vars, context_index):
        q, com, Hdot, contact_force = np.split(vars, [nq, nq+3, nq+6])
        contact_force = contact_force.reshape(3, 4, order='F')
        if isinstance(vars[0], AutoDiffXd):
            q = ExtractValue(q)
            if not np.array_equal(q, plant.GetPositions(
                    context[context_index])):
                plant.SetPositions(context[context_index], q)
            torque = np.zeros(3)
            for i in range(4):
                p_WF = plant.CalcPointsPositions(context[context_index],
                                                 foot_frame[i], [0, 0, 0],
                                                 plant.world_frame())
                Jq_WF = plant.CalcJacobianTranslationalVelocity(
                    context[context_index], JacobianWrtVariable.kQDot,
                    foot_frame[i], [0, 0, 0], plant.world_frame(),
                    plant.world_frame())
                ad_p_WF = InitializeAutoDiff(
                    p_WF, np.hstack((Jq_WF, np.zeros((3, 18)))))
                torque = torque + np.cross(
                    ad_p_WF.reshape(3) - com, contact_force[:, i])
        else:
            if not np.array_equal(q, plant.GetPositions(
                    context[context_index])):
                plant.SetPositions(context[context_index], q)
            torque = np.zeros(3)
            for i in range(4):
                p_WF = plant.CalcPointsPositions(context[context_index],
                                                 foot_frame[i], [0, 0, 0],
                                                 plant.world_frame())
                torque += np.cross(p_WF.reshape(3) - com, contact_force[:,i])
        return Hdot - torque
    for n in range(gait_params.N-1):
        prog.AddConstraint(eq(H[:,n+1], H[:,n] + h[n]*Hdot[:,n]))
        Fn = np.concatenate([contact_force[i][:,n] for i in range(4)])
        prog.AddConstraint(partial(angular_momentum_constraint,
                                   context_index=n),
                           lb=np.zeros(3),
                           ub=np.zeros(3),
                           vars=np.concatenate(
                               (q[:, n], com[:, n], Hdot[:, n], Fn)))

    # com == CenterOfMass(q), H = SpatialMomentumInWorldAboutPoint(q, v, com)
    # Make a new autodiff context for this constraint (to maximize cache hits)
    com_constraint_context = [ad_plant.CreateDefaultContext() for i in range(gait_params.N)]
    def com_constraint(vars, context_index):
        qv, com, H = np.split(vars, [nq+nv, nq+nv+3])
        if isinstance(vars[0], AutoDiffXd):
            if not autoDiffArrayEqual(
                    qv,
                    ad_plant.GetPositionsAndVelocities(
                        com_constraint_context[context_index])):
                ad_plant.SetPositionsAndVelocities(com_constraint_context[context_index], qv)
            com_q = ad_plant.CalcCenterOfMassPositionInWorld(
                com_constraint_context[context_index], [quad])
            H_qv = ad_plant.CalcSpatialMomentumInWorldAboutPoint(
                com_constraint_context[context_index], [quad],
                com).rotational()
        else:
            if not np.array_equal(qv, plant.GetPositionsAndVelocities(context[context_index])):
                plant.SetPositionsAndVelocities(context[context_index], qv)
            com_q = plant.CalcCenterOfMassPositionInWorld(
                context[context_index], [quad])
            H_qv = plant.CalcSpatialMomentumInWorldAboutPoint(
                context[context_index], [quad], com).rotational()
        return np.concatenate((com_q - com, H_qv - H))
    for n in range(gait_params.N):
        prog.AddConstraint(partial(com_constraint, context_index=n),
                           lb=np.zeros(6),
                           ub=np.zeros(6),
                           vars=np.concatenate(
                               (q[:, n], v[:, n], com[:, n], H[:, n])))

    # TODO: Add collision constraints
    if gait_params.check_self_collision:
        '''
            TODO
            - Not sure how to implement. Does drake support collision avoidance?
            - Only consider certain geoms (ex. calfs and not quads)

        Check here: https://github.com/RobotLocomotion/drake/blob/master/bindings/pydrake/multibody/test/inverse_kinematics_test.py 
        '''
        # # maybe only pass context on relevent bodies?
        # ik_context = [ad_plant.CreateDefaultContext() for i in range(N)] 
        # ik = InverseKinematics(plant, ik_context, with_joint_limits=False)

        # See: https://drake.mit.edu/doxygen_cxx/namespacedrake_1_1solvers.html#adc2a57e87ae4e4088e12967a25a7c229
        
        # MinimumDistanceConstraint checks distance btw all candidate pairs
        # of geometries in SceneGraphInspector.GetCollisionCandidates
        
        # If can't get MinimumDistanceConstraint working, can implement manually
        # prog.AddConstraint(MinimumDistanceConstraint(plant, min_distance, ))
    
    # Kinematic constraints
    def fixed_position_constraint(vars, context_index, frame):
        q, qn = np.split(vars, [nq])
        if not np.array_equal(q, plant.GetPositions(context[context_index])):
            plant.SetPositions(context[context_index], q)
        if not np.array_equal(qn, plant.GetPositions(context[context_index+1])):
            plant.SetPositions(context[context_index+1], qn)
        p_WF = plant.CalcPointsPositions(context[context_index], frame,
                                         [0, 0, 0], plant.world_frame())
        p_WF_n = plant.CalcPointsPositions(context[context_index + 1], frame,
                                           [0, 0, 0], plant.world_frame())
        if isinstance(vars[0], AutoDiffXd):
            J_WF = plant.CalcJacobianTranslationalVelocity(
                context[context_index], JacobianWrtVariable.kQDot, frame,
                [0, 0, 0], plant.world_frame(), plant.world_frame())
            J_WF_n = plant.CalcJacobianTranslationalVelocity(
                context[context_index + 1], JacobianWrtVariable.kQDot, frame,
                [0, 0, 0], plant.world_frame(), plant.world_frame())
            return InitializeAutoDiff(
                p_WF_n - p_WF,
                J_WF_n @ ExtractGradient(qn) - J_WF @ ExtractGradient(q))
        else:
            return p_WF_n - p_WF
    for i in range(4):
        for n in range(gait_params.N):
            if gait_params.in_stance[i, n]:
                # foot should be on the ground (world position z=0)
                prog.AddConstraint(
                    PositionConstraint(plant, plant.world_frame(),
                                       [-np.inf, -np.inf, 0],
                                       [np.inf, np.inf, 0], foot_frame[i],
                                       [0, 0, 0], context[n]), q[:, n])
                if n > 0 and gait_params.in_stance[i, n-1]:
                    # feet should not move during stance.
                    prog.AddConstraint(partial(fixed_position_constraint,
                                               context_index=n - 1,
                                               frame=foot_frame[i]),
                                       lb=np.zeros(3),
                                       ub=np.zeros(3),
                                       vars=np.concatenate((q[:, n - 1], q[:,
                                                                           n])))
            else:
                min_clearance = 0.01
                prog.AddConstraint(
                    PositionConstraint(plant, plant.world_frame(),
                                       [-np.inf, -np.inf, min_clearance],
                                       [np.inf, np.inf, np.inf], foot_frame[i],
                                       [0, 0, 0], context[n]), q[:, n])

    # Periodicity constraints
    if gait_params.is_laterally_symmetric:
        # Joints
        def AddAntiSymmetricPair(a, b):
            prog.AddLinearEqualityConstraint(a[0] == -b[-1])
            prog.AddLinearEqualityConstraint(a[-1] == -b[0])
        def AddSymmetricPair(a, b):
            prog.AddLinearEqualityConstraint(a[0] == b[-1])
            prog.AddLinearEqualityConstraint(a[-1] == b[0])
        # TODO: Should generalize joint names for all quadruped models
        AddAntiSymmetricPair(q_view.torso_to_abduct_fl_j,
                             q_view.torso_to_abduct_fr_j)
        AddAntiSymmetricPair(q_view.torso_to_abduct_hl_j,
                             q_view.torso_to_abduct_hr_j)
        AddSymmetricPair(q_view.abduct_fl_to_thigh_fl_j,
                         q_view.abduct_fr_to_thigh_fr_j)
        AddSymmetricPair(q_view.abduct_hl_to_thigh_hl_j,
                         q_view.abduct_hr_to_thigh_hr_j)    
        AddSymmetricPair(q_view.thigh_fl_to_knee_fl_j,
                         q_view.thigh_fr_to_knee_fr_j)
        AddSymmetricPair(q_view.thigh_hl_to_knee_hl_j,
                         q_view.thigh_hr_to_knee_hr_j)
        prog.AddLinearEqualityConstraint(q_view.body_y[0] == -q_view.body_y[-1])
        prog.AddLinearEqualityConstraint(q_view.body_z[0] == q_view.body_z[-1])
        # Body orientation must be in the xz plane:
        prog.AddBoundingBoxConstraint(0, 0, q_view.body_qx[[0,-1]])
        prog.AddBoundingBoxConstraint(0, 0, q_view.body_qz[[0,-1]])

        # Floating base velocity
        prog.AddLinearEqualityConstraint(
            v_view.body_vx[0] == v_view.body_vx[-1])
        prog.AddLinearEqualityConstraint(
            v_view.body_vy[0] == -v_view.body_vy[-1])
        prog.AddLinearEqualityConstraint(
            v_view.body_vz[0] == v_view.body_vz[-1])

        # CoM velocity periodicity constraints
        prog.AddLinearEqualityConstraint(comdot[0,0] == comdot[0,-1])
        prog.AddLinearEqualityConstraint(comdot[1,0] == -comdot[1,-1])
        prog.AddLinearEqualityConstraint(comdot[2,0] == comdot[2,-1])

    else:
        # Everything except body_x is periodic
        q_selector = PositionView([True]*nq)
        q_selector.body_x = False
        prog.AddLinearConstraint(eq(q[q_selector,0], q[q_selector,-1]))
        prog.AddLinearConstraint(eq(v[:,0], v[:,-1]))
    
    # TODO: Set solver parameters (mostly to make the worst case solve times less bad)
    snopt = SnoptSolver().solver_id()
    prog.SetSolverOption(snopt, 'Iterations Limits', 1e5)
    prog.SetSolverOption(snopt, 'Major Iterations Limit', 200)
    prog.SetSolverOption(snopt, 'Major Feasibility Tolerance', 5e-6)
    prog.SetSolverOption(snopt, 'Major Optimality Tolerance', 1e-4)
    prog.SetSolverOption(snopt, 'Superbasics limit', 2000)
    prog.SetSolverOption(snopt, 'Linesearch tolerance', 0.9)
    # prog.SetSolverOption(snopt, 'Print file', 'snopt.out')

    # TODO a few more costs/constraints from
    # from https://github.com/RobotLocomotion/LittleDog/blob/master/gaitOptimization.m

    result = Solve(prog)
    print(result.get_solver_id().name())
    # print(result.is_success())  # We expect this to be false if iterations are limited.
    
    vars = {}
    vars['result'] = result
    vars['h'] = h
    vars['q'] = q
    vars['PositionView'] = PositionView
    vars = SimpleNamespace(**vars)

    return result, vars, gait_params

def visualize_gait_optimization(sim_time_step, gait):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, sim_time_step)
    parser = Parser(plant)
    quadruped_model_file = os.path.relpath('./models/mini_cheetah/mini_cheetah_mesh.urdf')
    quad = parser.AddModelFromFile(quadruped_model_file, model_name='quad')
    plant.Finalize()

    visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    set_quad_pose(plant, plant_context, mode='stand')
    diagram.ForcedPublish(context)

    result, vars, gait_params = gait_optimization_prog(
        plant=plant, plant_context=plant_context, gait=gait)

    is_laterally_symmetric, stride_length = gait_params.is_laterally_symmetric, gait_params.stride_length

    result = vars.result
    PositionView, h, q = vars.PositionView, vars.h, vars.q

    # Animate trajectory
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    t_sol = np.cumsum(np.concatenate(([0],result.GetSolution(h))))
    q_sol = PiecewisePolynomial.FirstOrderHold(t_sol, result.GetSolution(q))
    visualizer.StartRecording()
    num_strides = 4
    t0 = t_sol[0]
    tf = t_sol[-1]
    T = tf*num_strides*(2.0 if is_laterally_symmetric else 1.0)
    for t in np.hstack((np.arange(t0, T, 1.0/32.0), T)):
        context.SetTime(t)
        stride = (t - t0) // (tf - t0)
        ts = (t - t0) % (tf - t0) 
        qt = PositionView(q_sol.value(ts))
        if is_laterally_symmetric:
            if stride % 2 == 1:
                qt = HalfStrideToFullStride(PositionView, qt)
                qt.body_x += stride_length/2.0
            stride = stride // 2
        qt.body_x += stride*stride_length
        plant.SetPositions(plant_context, qt[:])
        diagram.ForcedPublish(context)

    visualizer.StopRecording()
    visualizer.PublishRecording()

def simulate_gait_optimization(sim_time_step, gait):
    '''
        TODO
        - simulate instead of animate
        - need to make optimal trajectory into continuous piecewise polynomials to feed as desired trajectory to MPC
    '''
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, sim_time_step)
    parser = Parser(plant)
    quadruped_model_file = os.path.relpath('./models/mini_cheetah/mini_cheetah_mesh.urdf')
    quad = parser.AddModelFromFile(quadruped_model_file, model_name='quad')
    plant.Finalize()

    # Start visualization
    AddDefaultVisualization(builder)

    # TODO: Add ModelPredictiveController

    # Add PidController with saturation
    pid_controller = builder.AddSystem(QuadPidController(plant=plant))
    torque_limiter = builder.AddSystem(Saturation(
        min_value = plant.GetLowerEffortLimit(),
        max_value = plant.GetUpperEffortLimit()
    ))

    builder.Connect(pid_controller.get_output_port(), torque_limiter.get_input_port())
    builder.Connect(torque_limiter.get_output_port(), plant.get_actuation_input_port())
    builder.Connect(plant.get_state_output_port(), pid_controller.get_input_port_estimated_state())

    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    plant_context = plant.GetMyContextFromRoot(context)
    set_quad_pose(plant, plant_context, mode='stand')
    diagram.ForcedPublish(context)

    result, vars, gait_params = gait_optimization_prog(
        plant=plant, plant_context=plant_context, gait=gait)

    is_laterally_symmetric, stride_length = gait_params.is_laterally_symmetric, gait_params.stride_length

    result = vars.result
    PositionView, h, q = vars.PositionView, vars.h, vars.q
    # breaks = breaks, samples = q_sol(breaks), knot[i] = (breaks[i], samples[i])
    t_sol = np.cumsum(np.concatenate(([0],result.GetSolution(h))))
    # q_sol = PiecewisePolynomial.FirstOrderHold(t_sol, result.GetSolution(q))
    num_strides = 4
    t0 = t_sol[0]
    tf = t_sol[-1]
    T = tf*num_strides*(2.0 if is_laterally_symmetric else 1.0)
    breaks = np.hstack((np.arange(t0, T, 1.0/32.0), T))
    samples = result.GetSolution(q)
    
    q_sol_cubic = PiecewisePolynomial().CubicWithContinuousSecondDerivative(
        breaks=breaks, samples=samples, 
        periodic_end_condition=True,
        zero_end_point_derivatives=False)
    
    
    '''
        context = diagram.CreateDefaultContext()
        plant_context = plant.GetMyContextFromRoot(simulator.get_mutable_context())
        t_sol = np.cumsum(np.concatenate(([0],result.GetSolution(h))))
        q_sol = PiecewisePolynomial.FirstOrderHold(t_sol, result.GetSolution(q)) # result.GetSolution(q) is shape (nq, N)
        
        num_strides = 4
        t0, tf = t_sol[0], t_sol[-1]
        T = tf*num_strides*(2.0 if is_laterally_symmetric else 1.0)
        controller, S = AddPidControllerForQuad(builder=builder, plant=plant)
        xd = S @ plant.get_state_output_port().Eval(plant_context) # TODO: change to q_sol.value(ts)
        controller.get_input_port_desired_state().SetFromVector(xd) 
    '''

def main():
    parser = ArgumentParser()
    parser.add_argument('--gait', dest='gait', type=str, default='walking_trot')
    args = parser.parse_args()

    visualize_gait_optimization(sim_time_step=1e-3, gait=args.gait)
    
if __name__ == '__main__':
    main()
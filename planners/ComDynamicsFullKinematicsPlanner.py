import numpy as np
import os
import sys

from functools import partial
from types import SimpleNamespace

from pydrake.systems.framework import LeafSystem, BasicVector
from pydrake.solvers import MathematicalProgram, Solve
from pydrake.math import RotationMatrix
from pydrake.autodiffutils import ExtractGradient, InitializeAutoDiff, AutoDiffXd, ExtractValue
from pydrake.multibody.inverse_kinematics import InverseKinematics, AddUnitQuaternionConstraintOnPlant, OrientationConstraint, PositionConstraint, MinimumDistanceConstraint
from pydrake.multibody.tree import JacobianWrtVariable
from pydrake.solvers import SnoptSolver
from pydrake.all import eq

from quadruped.quad_utils import setup_gait
from utils import MakeNamedViewPositions, MakeNamedViewVelocities, autoDiffArrayEqual

'''
    COM dynamics + full kinematics planner
    Credits to: https://github.com/RussTedrake/underactuated/blob/master/examples/littledog.ipynb    

    TODO:
    - Add additional constraints
    - Make motion dynamically-feasible
    - Generalize for quadrupeds and bipeds

    TODO (formatting):
    - Make into an actual LeafSystem.
'''

class ComDynamicsFullKinematicsPlanner(LeafSystem):
    def __init__(self, plant, plant_context, gait='walking_trot'):
        super().__init__()
        self.__plant = plant
        self.__plant_context = plant_context
        self._gait = gait
        
        self.__plant.DeclareVectorOutputPort(
            name='state_traj',
            model_value=BasicVector(2*self.__plant.num_actuated_dofs()),
            calc=self.SetStateTraj
        )
    
    def SetStateTraj(self, context, output):
        pass

    def RunPlanner(self):
        q0 = self.__plant.GetPositions(self.__plant_context)
        body_frame = self.__plant.GetFrameByName('body')
        quad = self.__plant.GetModelInstanceByName('quad')

        PositionView = MakeNamedViewPositions(self.__plant, 'Positions')
        VelocityView = MakeNamedViewVelocities(self.__plant, 'Velocities')

        mu = 1 # rubber on rubber
        total_mass = self.__plant.CalcTotalMass(self.__plant_context, [quad])
        gravity = self.__plant.gravity_field().gravity_vector()
        # TODO: inconsistent nq definition?
        nq = 12 
        # TODO: change to function GetFootFrames(plant=plant)
        foot_frame = [
            self.__plant.GetFrameByName('LF_FOOT'),
            self.__plant.GetFrameByName('RF_FOOT'),
            self.__plant.GetFrameByName('LH_FOOT'),
            self.__plant.GetFrameByName('RH_FOOT')]
        
        gait_params = SimpleNamespace(**setup_gait(gait=self._gait))

        prog = MathematicalProgram()

        # Time steps
        h = prog.NewContinuousVariables(gait_params.N-1, 'h')
        prog.AddBoundingBoxConstraint(0.5*gait_params.T/gait_params.N, 2.0*gait_params.T/gait_params.N, h)
        prog.AddLinearConstraint(sum(h) >= 0.9*gait_params.T)
        prog.AddLinearConstraint(sum(h) <= 1.1*gait_params.T)

        # Create one context per timestep (to maximize cache hits)
        context = [self.__plant.CreateDefaultContext() for i in range(gait_params.N)]
        # We could get rid of this by implementing a few more Jacobians in MultibodyPlant
        ad_plant = self.__plant.ToAutoDiffXd()

        # Joint positions and velocities
        nq = self.__plant.num_positions()
        nv = self.__plant.num_velocities()
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
            prog.AddBoundingBoxConstraint(self.__plant.GetPositionLowerLimits(),
                                        self.__plant.GetPositionUpperLimits(), q[:, n])
            # Joint velocity limits
            prog.AddBoundingBoxConstraint(self.__plant.GetVelocityLowerLimits(),
                                        self.__plant.GetVelocityUpperLimits(), v[:, n])
            # Unit quaternions
            AddUnitQuaternionConstraintOnPlant(self.__plant, q[:,n], prog)
            # Body orientation
            prog.AddConstraint(OrientationConstraint(self.__plant,
                                                    body_frame, RotationMatrix(),
                                                    self.__plant.world_frame(), RotationMatrix(),
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
                if not np.array_equal(q, self.__plant.GetPositions(
                        context[context_index])):
                    self.__plant.SetPositions(context[context_index], q)
                v_from_qdot = self.__plant.MapQDotToVelocity(context[context_index],
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
                if not np.array_equal(q, self.__plant.GetPositions(
                        context[context_index])):
                    self.__plant.SetPositions(context[context_index], q)
                torque = np.zeros(3)
                for i in range(4):
                    p_WF = self.__plant.CalcPointsPositions(context[context_index],
                                                    foot_frame[i], [0, 0, 0],
                                                    self.__plant.world_frame())
                    Jq_WF = self.__plant.CalcJacobianTranslationalVelocity(
                        context[context_index], JacobianWrtVariable.kQDot,
                        foot_frame[i], [0, 0, 0], self.__plant.world_frame(),
                        self.__plant.world_frame())
                    ad_p_WF = InitializeAutoDiff(
                        p_WF, np.hstack((Jq_WF, np.zeros((3, 18)))))
                    torque = torque + np.cross(
                        ad_p_WF.reshape(3) - com, contact_force[:, i])
            else:
                if not np.array_equal(q, self.__plant.GetPositions(
                        context[context_index])):
                    self.__plant.SetPositions(context[context_index], q)
                torque = np.zeros(3)
                for i in range(4):
                    p_WF = self.__plant.CalcPointsPositions(context[context_index],
                                                    foot_frame[i], [0, 0, 0],
                                                    self.__plant.world_frame())
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
                if not np.array_equal(qv, self.__plant.GetPositionsAndVelocities(context[context_index])):
                    self.__plant.SetPositionsAndVelocities(context[context_index], qv)
                com_q = self.__plant.CalcCenterOfMassPositionInWorld(
                    context[context_index], [quad])
                H_qv = self.__plant.CalcSpatialMomentumInWorldAboutPoint(
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
                [TODO]
                - Not sure how to implement. Does drake support collision avoidance?

            Check here: https://github.com/RobotLocomotion/drake/blob/master/bindings/pydrake/multibody/test/inverse_kinematics_test.py 
            '''
            # ik_context = [ad_plant.CreateDefaultContext() for i in range(N)] 
            # ik = InverseKinematics(self.__plant, ik_context, with_joint_limits=False)

            # See: https://drake.mit.edu/doxygen_cxx/namespacedrake_1_1solvers.html#adc2a57e87ae4e4088e12967a25a7c229
            
            # MinimumDistanceConstraint checks distance btw all candidate pairs
            # of geometries in SceneGraphInspector.GetCollisionCandidates
            
            # If can't get MinimumDistanceConstraint working, can implement manually
            # prog.AddConstraint(MinimumDistanceConstraint(self.__plant, min_distance, ))
            pass
        
        # Kinematic constraints
        def fixed_position_constraint(vars, context_index, frame):
            q, qn = np.split(vars, [nq])
            if not np.array_equal(q, self.__plant.GetPositions(context[context_index])):
                self.__plant.SetPositions(context[context_index], q)
            if not np.array_equal(qn, self.__plant.GetPositions(context[context_index+1])):
                self.__plant.SetPositions(context[context_index+1], qn)
            p_WF = self.__plant.CalcPointsPositions(context[context_index], frame,
                                            [0, 0, 0], self.__plant.world_frame())
            p_WF_n = self.__plant.CalcPointsPositions(context[context_index + 1], frame,
                                            [0, 0, 0], self.__plant.world_frame())
            if isinstance(vars[0], AutoDiffXd):
                J_WF = self.__plant.CalcJacobianTranslationalVelocity(
                    context[context_index], JacobianWrtVariable.kQDot, frame,
                    [0, 0, 0], self.__plant.world_frame(), self.__plant.world_frame())
                J_WF_n = self.__plant.CalcJacobianTranslationalVelocity(
                    context[context_index + 1], JacobianWrtVariable.kQDot, frame,
                    [0, 0, 0], self.__plant.world_frame(), self.__plant.world_frame())
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
                        PositionConstraint(self.__plant, self.__plant.world_frame(),
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
                        PositionConstraint(self.__plant, self.__plant.world_frame(),
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
        vars['v'] = v
        vars['PositionView'] = PositionView
        vars['VelocityView'] = VelocityView
        vars = SimpleNamespace(**vars)

        return result, vars, gait_params
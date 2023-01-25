import numpy as np

from pydrake.systems.framework import LeafSystem, BasicVector
from pydrake.systems.primitives import FirstOrderTaylorApproximation
from pydrake.solvers import MathematicalProgram, Solve, LinearEqualityConstraint, Binding, BoundingBoxConstraint, QuadraticCost
from pydrake.trajectories import PiecewisePolynomial
from pydrake.symbolic import Variable
from pydrake.common.containers import namedview
from pydrake.math import IsPositiveDefinite

from utils import VerifyTrajectoryIsValidPolynomial, MakeNamedViewPositions, MakeNamedViewVelocities, MakeNamedViewActuation
from quadruped.quad_utils import GetQuadStateProjectionMatrix
# Refer to: https://stackoverflow.com/questions/72146203/linear-model-predictive-control-optimization-running-slowly


class ModelPredictiveController(LeafSystem):
    """Discrete Whole Body (Linear) Model Predictive Controller for Quadrupeds
    - Input: Reference (optimized) state traj (traj_init_x) & control traj (traj_init_u)
    - Output: desired state [q,v] & actuator torques [u]

    Alternatively, simplified model MPC 
    - Input: Contact patches
    - Output: CoM trajectory
    
    TODO:
    - Finish implementing + testing
    - 

    TODO (secondary):
    - Generalize to legged robots (instead of quadrupeds)
    - what constraints to add?
        - friction cone -> need GRF (estimate or as decision var)
        - no-slip -> need feet pos and feet jacobians
    - what exactly to pass in for dt?
        - 1/time_horizon is MPC frequency
        - In practice, a robot is a discrete system, so dt will be min sensor update frequency?
    - Set access modifiers for all read-only variables
    """

    def __init__(self, plant, system, Q, R, dt, time_period, time_horizon, traj_init_x=None, traj_init_u=None):
        super().__init__()
        self.__plant = plant
        self.__context = self.__plant.CreateDefaultContext() # Only to get current state. Can maybe remove?

        # System parameters
        # TODO: Does (sub)system (e.g. quad, biped) need to be an argument
        self.__system = system
        self._nq = self.__system.num_actuated_dofs()
        self._S = GetQuadStateProjectionMatrix(system)
        self._P_y = self.__system.MakeActuationMatrix()[6:,:].T

        self.PositionView = MakeNamedViewPositions(self.__plant, 'Positions')
        self.VelocityView = MakeNamedViewVelocities(self.__plant, 'Velocities')
        self.ActuationView = MakeNamedViewActuation(self.__plant, 'Actuations')

        self._Q = Q
        self._R = R

        if isinstance(Q, np.ndarray(self.nq, dtype=np.float64)):
            self._Q = np.diag(self._Q)
            self._R = np.diag(self._R)
        else:
            assert isinstance(Q, np.ndarray((self._nq,self._nq), dtype=np.float64)), \
                'Q should be have shape=(num_actuated_dofs,num_actuated_dofs) and dtype=float'
            assert isinstance(R, np.ndarray((self._nq,self._nq), dtype=np.float64)), \
                'R should be have shape=(num_actuated_dofs,num_actuated_dofs) and dtype=float'
        
        assert IsPositiveDefinite(self._Q), 'Q is not positive definite'
        assert IsPositiveDefinite(self._R), 'R is not positive definite'

        # MPC parameters
        self._k = 0 # MPC iteration 
        self._timesteps_btw_MPC = 0
        self._dt = dt
        self._time_period = time_period 
        self._time_horizon = time_horizon

        # Add assertions for dt, time_period, and time_horizon
        assert isinstance(time_horizon, float) and time_horizon > 0
        assert isinstance(time_period, float) and time_period >= time_horizon
        assert isinstance(dt, float) and dt > 0 and dt < time_horizon

        self._N = self._time_horizon // self._dt # TODO: floor division should be ok?
        self._Nt = self._time_period // self._dt

        traj_x_constraints = {
            'shape': (self.__plant.get_num_multibody_states(), self._Nt),
            'total_time': self._time_period
        }
        traj_u_constraints = {
            'shape': (self._nq, self._Nt-1),
            'total_time': self._time_period
        }

        # Transform inputs traj_init_* to PiecewisePolynomial with constraints
        self._traj_init_x = VerifyTrajectoryIsValidPolynomial(traj_init_x, traj_x_constraints)
        self._traj_init_u = VerifyTrajectoryIsValidPolynomial(traj_init_u, traj_u_constraints)

        self._estimated_state_input_port = self.DeclareVectorInputPort(
            model_vector=BasicVector(size=self.__plant.num_multibody_states()), 
            name='estimated_state'
        )
        
        self._control_output_port = self.DeclareVectorOutputPort(
            model_vector=BasicVector(size=self._nq),
            name='control',
            calc=self.SetTorque
        )
        self._desired_state_output_port = self.DeclareVectorOutputPort(
            model_vector=BasicVector(size=2*self._nq),
            name='desired_state',
            calc=self.SetDesiredState
        )
        
        # TODO: what to do at first timestep? First run one MPC and use optimal_u from there?
        #       or just use traj_init_x and traj_init_u until first iteration of RunModelPredictiveController
        self.DeclarePeriodicPublishEvent(
            period_sec=self._N,
            offset_sec=0.0,
            publish=self.RunModelPredictiveController
        )

        self.prog = MathematicalProgram()

        # Set initial guess for solver using traj_init_* from planner 
        self.prog.SetInitialGuess(self._x, self._traj_init_x)
        self.prog.SetInitialGuess(self._u, self._traj_init_u)

        # These are the results from RunModelPredictiveController
        self._optimal_x = np.empty((2*self._nq,), dtype=float) 
        self._optimal_u = np.empty((self._nq,), dtype=float)
        self._previous_u = np.empty((self._nq,), dtype=float)

        # Initialize traj_init_* slices used for each finite horizon (RunModelPredictiveController)
        self._traj_init_x_slice = np.empty((2*self._nq,self._N), dtype=np.float64)
        self._traj_init_u_slice = np.empty((self._nq,self._N), dtype=np.float64)

        # Initialize variables
        self._x = np.empty((2*self._nq, self._N), dtype=Variable) 
        self._u = np.empty((self._nq, self._N-1), dtype=Variable)
        self._x_view = np.empty((self._N,), dtype=namedview)
        self._u_view = np.empty((self._N,), dtype=namedview)

        # Initialize linearized dynamics matrices
        A = np.empty((self.__plant.num_multibody_states(),self.__plant.num_multibody_states()), dtype=np.float64)
        B = np.empty((self.__plant.num_multibody_states(),self._nq), dtype=np.float64)
        # Exclude floating base coords [TODO: verify shape]
        self.__floating_base_state_index_end = self.__plant.get_multibody_states() - (2 * self._nq) # should == 13 for quad
        A = A[self.__floating_base_state_index_end:,self.__floating_base_state_index_end:]
        B = B[self.__floating_base_state_index_end:,:]

        # Initialize linearized dynamics and torque limit constraints
        self.linearized_dynamics_constraints = np.empty(self._N-1, dtype=Binding[LinearEqualityConstraint])
        self.torque_limit_constraints = np.empty(self._N-1, dtype=Binding[BoundingBoxConstraint])

        # Initialize state error and input error quadratic costs
        self.state_error_quadratic_cost = np.empty(self._N-1, dtype=Binding[QuadraticCost])
        self.input_error_quadratic_cost = np.empty(self._N-1, dtype=Binding[QuadraticCost])

        # TODO: Initialize other constraints?

        for n in range(self._N-1):
            self._u[:,n] = self.prog.NewContinuousVariables(self._nq, 'u' + str(n))
            self._x[:,n] = self.prog.NewContinuousVariables(2*self._nq, 'x' + str(n))
            self._u_view[n] = self.ActuationView(self._u[:,n])
            self._x_view[n] = self.PositionView(self._x[:,n])
            # Add linearized dynamics constraints to MathematicalProgram
            self.linearized_dynamics_constraints[n] = self.prog.AddLinearEqualityConstraint(
                Aeq=A,
                beq=self._x[:n+1] + A@self._traj_init_x_slice[:,n] - B@(self._u[:,n]-self._traj_init_u_slice[:,n]),
                vars=self._x[:,n]
            ) 
            # Add torque limits constraints to MathematicalProgram
            self.torque_limit_constraints[n] = self.prog.AddBoundingBoxConstraint(
                self.__system.GetEffortLowerLimit(), 
                self.__system.GetEffortUpperLimit(), 
                self._u[:,n]
            )
            # Add state error cost to MathematicalProgram
            self.prog.AddQuadraticErrorCost(
                Q=2*self._Q,
                x_desired=self._traj_init_x_slice[:,n],
                vars=self._x[:,n]
            )
            # Add input error cost to MathematicalProgram
            self.prog.AddQuadraticErrorCost(
                Q=2*self._R,
                x_desired=self._traj_init_u_slice[:,n],
                vars=self._u[:,n]
            )
        self._x[:,self._N] = self.prog.NewContinuousVariables(2*self._nq, 'x' + str(self._N))
        self._x_view[self._N] = self.PositionView(self._x[:,self._N])

        # Initialize position constraints
        x0 = self.__context.get_state()
        x0_proj = self._S @ x0
        x0_view = self.PositionView(x0_proj)
        self.initial_value_constraint = self.prog.AddBoundingBoxConstraint(
            lb=x0_proj, 
            ub=x0_proj, 
            vars=self._x[:,0]
        )
        self.final_value_constraint = self.prog.AddBoundingBoxConstraint(
            lb=self._traj_init_x_slice[:,-1], 
            ub=self._traj_init_x_slice[:,-1], 
            vars=self._x[:,self._N]    
        )        
    
    def SetTorque(self, context, output):
        '''
            TODO: 
            - check if u needs to be negative or multiplied by self._P_y. See QuadPidController.GetPdTorque
            - how to align time (self._timesteps_btw_MPC) btw SetTorque and SetDesiredState
        '''
        self._previous_u = self._optimal_u[:,self._timesteps_btw_MPC]
        output.SetFromVector(self._previous_u)
        self._timesteps_btw_MPC += 1

    def SetDesiredState(self, context, output):
        self._previous_x = self._optimal_x[:,self._timesteps_btw_MPC]
        output.SetFromVector(self._previous_x)

    def RunModelPredictiveController(self, context): 
        '''
            MPC at one time step
            - Trajectory stabilization with receding horizon LQR and updated constraints using UpdateCoefficient
            - Runs with frequency 1/self._N
        '''
        current_context = context
        
        # Enforce periodicity when self._k+self._N > self._Nt
        assert len(self._traj_init_x) == self._N
        assert len(self._traj_init_u) == self._N

        start_time = self._k
        end_time = (self._k+self._N) if (self._k+self._N) <= self._Nt else ((self._k+self._N)%self._Nt)
        # end_time = (self._k+self._N) if (self._k+self._N) <= self._Nt else self._Nt # no periodicity version

        if end_time > self._Nt:
            self._traj_init_x_slice = self._S @ (self._traj_init_x[:,start_time:] + self._traj_init_x[:,:(end_time%self._N)])
            self._traj_init_u_slice = self._P_y @ (self._traj_init_u[:,start_time:] + self._traj_init_u[:,:(end_time%self._N)])
        else:
            self._traj_init_x_slice = self._S @ self._traj_init_x[:,start_time:end_time]
            self._traj_init_u_slice = self._P_y @ self._traj_init_u[:,start_time:end_time]

        # Verify length of initial trajectory slices are correct
        assert len(self._traj_init_x_slice) == self.Nt
        assert len(self._traj_init_u_slice) == self.Nt

        # TODO: verify S @ x0 == self._optimal_x[:,0] (or at least close to a tolerance)

        # Update initial value constraint
        x0 = self._S @ current_context.get_state() 
        u0 = self._optimal_u[:,0] # previous action
        x0_view = self.PositionView(x0)
        self.initial_value_constraint.evaluator().UpdateCoefficients(new_lb=x0, new_ub=x0)

        system_input_port_index = self.__system.get_state_output_port(model_instance=self.__system) 
        system_output_port_index = self.__plant.get_actuation_input_port(model_instance=self.__system)

        # Update linearized dynamics constraints
        # A[k] @ (x[k]-x0[k]) + B[k] @ (u[k]-u0[k]) = x[k+1]
        for n in range(self._N-1):
            # Get linearized system about (x0, u0) from current_context (returns an AffineSystem)
            linearized_system = FirstOrderTaylorApproximation(
                system=self.__plant,
                context=current_context,
                input_port_index=system_input_port_index,
                output_port_inex=system_output_port_index,
            )

            A, B = linearized_system.A(), linearized_system.B()

            # Skip floating body indices 
            A = A[self.__floating_base_state_index_end:, self.__floating_base_state_index_end:]
            B = B[self.__floating_base_state_index_end,:]

            self.linearized_dynamics_constraints[n].evaluator().UpdateCoefficients(
                Aeq=A, 
                beq=self._x[:,n+1] + A@self._traj_init_x_slice[:,n] - B@(self._u[:,n]-self._traj_init_u_slice[:,n])
            )
        
        # Update final value constraint
        xf = self._traj_init_x_slice[:,-1]
        xf_view = self.PositionView(xf)
        self.final_value_constraint.evaluator().UpdateCoefficients(new_lb=xf, new_ub=xf)

        # self.prog.AddLinearConstraint(dirtran.initial_state() == current_state - state_ref) # initial condition

        # Update quadratic cost in error coordinates
        # TODO: technically, we only need to remove quadratic_cost[:,0], 
        #       change vars for quadratic[:,1:-2], and add quadratic_cost[:,-1]
        for n in range(self._N-1):
            self.prog.RemoveCost(self.state_error_quadratic_cost[n])
            self.prog.RemoveCost(self.input_error_quadratic_cost[n])
            self.state_error_quadratic_cost[n] = self.prog.AddQuadraticErrorCost(
                Q=2*self._Q,
                b=np.zeros((self._nq,1)),
                desired_x=self._traj_init_x[:,n],
                vars=self._S@self._x[:,n]
            )
            self.input_error_quadratic_cost[n] = self.prog.AddQuadraticErrorCost(
                Q=2*self._R,
                b=np.zeros((self._nq,1)),
                desired_x=self._traj_init_u[:,n],
                vars=self._P_y@self._u[:,n]
            )

        # TODO: verify size of self.state_error_quadratic_cost[self.N]
        # self.prog.RemoveCost(self.state_error_quadratic_cost[self._N])
        # self.state_error_quadratic_cost[self._N] = self.prog.AddQuadraticErrorCost(
        #     Q=2*self._Q,
        #     b=np.zeros((self._nq,1)),
        #     desired_x=self._traj_init_u[:,-1],
        #     vars=self._P_y@self._u[:,n]
        # )

        result = Solve(self.prog)

        # Save optimal x and optimal u
        self._optimal_x = result.GetSolution(self._x)
        self._optimal_u = result.GetSolution(self._u)

        self._k = self._k + 1
        self._timesteps_btw_MPC = 0
    
    def get_state_port(self):
        return self._estimated_state_input_port

    def get_control_port(self):
        return self._control_output_port
    
    def get_traj_init_x(self):
        return self._traj_init_x
    
    def get_traj_init_u(self):
        return self._traj_init_u
import numpy as np

from pydrake.systems.framework import LeafSystem, BasicVector
from pydrake.systems.primitives import Linearize
from pydrake.solvers import MathematicalProgram, Solve, LinearEqualityConstraint, QuadraticConstraint, BoundingBoxConstraint
from pydrake.trajectories import PiecewisePolynomial
from pydrake.symbolic import Variable
from pydrake.common.containers import namedview

# TODO: Use absolute path for imports
from ..utils import VerifyTrajectoryIsValidPolynomial, MakeNamedViewPositions, MakeNamedViewVelocities, MakeNamedViewActuation
from ..quad_utils import GetQuadStateProjectionMatrix
# Refer to: https://stackoverflow.com/questions/72146203/linear-model-predictive-control-optimization-running-slowly


class ModelPredictiveController(LeafSystem):
    """(Linear) Model Predictive Controller for Quadrupeds
    - Input: quadruped (actuated) state [n=12]
    - Output: control inputs tau [n=12]

    TODO:
    - Finish implementing + testing

    TODO (secondary):
    - Generalize to legged robots (instead of quadrupeds)
    - For quad, still need to add friction cone constraints 
    - figure out contexts
    - Make __init__ parameters into keyword arguments?
    - what constraints to add
        - friction cone -> need GRF (estimate or as decision var)
        - no-slip -> need feet pos -> need feet jacobians...
    - what exactly to pass in for dt?
        - 1/time_horizon is MPC frequency
        - In practice, a robot is a discrete system, so dt will be min sensor update frequency?
    - Set access modifiers for all read-only variables
    """

    def __init__(self, plant, system, base_context, Q, R, dt, time_period, time_horizon, traj_init_x=None, traj_init_u=None):
        LeafSystem.__init__(self)
        self.__plant = plant
        self.base_context = base_context
        
        self.context = self.__plant.GetMyMutableContextFromRoot(self.base_context)

        # System parameters
        self.system = system
        self.nq = self.system.num_actuated_dofs()
        self.S = GetQuadStateProjectionMatrix(system)
        self.P_y = self.system.MakeActuationMatrix()[6:,:].T

        self.PositionView = MakeNamedViewPositions(self.__plant, 'Positions')
        self.VelocityView = MakeNamedViewVelocities(self.__plant, 'Velocities')
        self.ActuationView = MakeNamedViewActuation(self.__plant, 'Actuations')
    
        # MPC parameters
        self.k = 0 # MPC iteration 
        self.Q = Q
        self.R = R
        self.dt = dt
        self.time_period = time_period 
        self.time_horizon = time_horizon # MPC frequency?
        self.optimal_x = np.empty((self.nq,), dtype=float) 
        self.optimal_u = np.empty((self.nq,), dtype=float)
        self.N = self.time_horizon // self.dt # TODO: floor division should be ok?
        self.Nt = self.time_period // self.dt

        # should also make sure traj_init_x time matches up with time_period 
        # (aka shape(traj_init_x) == (self.nq,M) and shape(traj_init_u) == (self.nq,M-1))
        # where M = self.time_period / self.dt
        # Then, for each iteration, we take traj_init_x[:,k:k+T] and traj_init_u[:,k:k+T]
        # where T = self.time_horizon / self.dt

        traj_constraints = {
            'shape': (self.nq,self.Nt)
        }

        # Initialize trajectory to zeros if traj_init_* == None        
        self.traj_init_x = traj_init_x if not None else PiecewisePolynomial(np.zeros((self.nq,self.Nt)))
        self.traj_init_u = traj_init_u if not None else PiecewisePolynomial(np.zeros((self.nq,self.Nt)))

        assert VerifyTrajectoryIsValidPolynomial(self.traj_init_x, traj_constraints)
        assert VerifyTrajectoryIsValidPolynomial(self.traj_init_u, traj_constraints) 

        self.DeclareVectorInputPort(
            model_vector=BasicVector(size=self.__plant.num_multibody_states()), 
            name='estimated_state')
        
        self.DeclareVectorOutputPort(
            model_vector=BasicVector(size=self.nq),
            name='control',
            calc=self.optimal_u[:,0]) # TODO: check to use index 0 or 1

        # self.quad = self.__plant.GetSubsystemByName('quad')
        
        self.prog = MathematicalProgram()

        # Run MPC at frequency 1/self.N
        self.DeclarePeriodicDiscreteUpdateEvent(
          period_sec=self.N,
          update=self.RunModelPredictiveController())

        # Initialize variables
        self.x = np.empty((2*self.nq, self.N), dtype=Variable) # self.x shape is (2*self.nq, self.N) since np.vstack(q, qdot)?
        self.u = np.empty((self.nq, self.N-1), dtype=Variable)
        self.x_view = np.empty((self.N,), dtype=namedview)
        self.u_view = np.empty((self.N,), dtype=namedview)
        for n in range(self.N-1):
            self.u[:,n] = self.prog.NewContinuousVariables(self.nq, 'u' + str(n))
            self.x[:,n] = self.prog.NewContinuousVariables(2*self.nq, 'x' + str(n))
            self.u_view[n] = self.ActuationView(self.u[:,n])
            self.x_view[n] = self.PositionView(self.x[:,n])
        self.x[:,self.N] = self.prog.NewContinuousVariables(2*self.nq, 'x' + str(self.N))
        self.x_view[self.N] = self.PositionView(self.x[:,self.N])

        # Initialize position constraints (they will be updated in self.RunModelPredictiveController())
        x0 = self.context.get_state()
        x0_view = self.PositionView(x0)
        self.initial_value_constraint = self.prog.AddBoundingBoxConstraint(self.S@x0, self.S@x0, self.x[:,0])
        # Let self.S @ x0 act as placeholder for self.S @ xf for now.
        self.final_value_constraint = self.prog.AddBoundingBoxConstraint(self.S@x0, self.S@x0, self.x[:,self.N])

        # Initialize torque limits and linearized dynamics constraints
        A = np.empty((self.__plant.num_multibody_states(),self.__plant.num_multibody_states()), dtype=np.float64)
        B = np.empty((self.__plant.num_multibody_states(),self.nq), dtype=np.float64)
        # Exclude floating base coords # TODO: VERIFY
        A = A[13:,13:]
        B = B[13:,:]
        self.linearized_dynamics_constraints = np.empty((self.nq,self.N-1), dtype=LinearEqualityConstraint)
        self.torque_limit_constraints = np.empty((self.nq,self.N-1), dtype=BoundingBoxConstraint)

        for n in range(self.N-1):
            self.linearized_dynamics_constraints[:,n] = self.prog.AddLinearEqualityConstraint(
                Aeq=A,
                Beq=self.x[:n+1] + A@self.x[:,n] - B@(self.u[:,n]-self.traj_init_u[:,0])) 
            self.torque_limits_constraints[:,n] = self.prog.AddBoundingBoxConstraint(
                self.system.GetEffortLowerLimit(), 
                self.system.GetEffortUpperLimit(), 
                self.u[:,n]) 

        # Initialize quadratic cost (for all timesteps...)
        # For each MPC iteration, reset quadratic costs (remove and add) 
        self.state_error_quadratic_cost = np.empty(self.N, dtype=QuadraticConstraint)
        self.input_error_quadratic_cost = np.empty(self.N-1, dtype=QuadraticConstraint)

        # TODO: Initialize friction cone constraints AND/OR non-slip constraints

    def RunModelPredictiveController(self): 
        # cpp implementation uses DirectTranscription, where linearized discrete dynamics are a constraint by construction
        '''
            MPC optimization at one time step
            - Inputs: traj_init_u, traj_init_x, base_context, current_context, Q, R
            - Returns: optimal_u, optimal_x
            Pass optimal u to output port? (technically to PD input port)
        '''
        system_input_port_index = self.system.get_state_output_port(model_instance=self.system) # this is system_state
        system_output_port_index = self.__plant.get_actuation_input_port(model_instance=self.system)

        # context for linear system should be system state at current timestep (get from simulator?)
        context_at_timestep = self.__plant.GetMutableSubsystemContext()
        
        # TODO: enforce periodicity when self.k+self.N > self.Nt?
        # TODO: fix start/end time to respect python slicing
        start_time = self.k
        end_time = (self.k+self.N) if (self.k+self.N) <= self.Nt else ((self.k+self.N)%self.Nt)
        # end_time = (self.k+self.N) if (self.k+self.N) <= self.Nt else self.Nt # no periodicity version
        traj_init_x_slice = self.S @ self.traj_init_x[:,start_time:end_time]
        traj_init_u_slice = self.P_y @ self.traj_init_u[:,start_time:end_time]

        # Set initial guess using traj_init_* from planner 
        self.prog.SetInitialGuess(self.x, traj_init_x_slice)
        self.prog.SetInitialGuess(self.u, traj_init_u_slice)

        # Update initial value constraint
        # TODO: 
        #   - is initial value x0 OR (x0-traj_init_x[:,0])?
        #   - fix inconsistency self.S@x0 vs xf        
        x0 = context_at_timestep.get_state() # TODO: verify S @ x0 == self.optimal_x[:,0] (or at least close to a tolerance)
        u0 = self.optimal_u[:,0]
        x0_view = self.PositionView(x0)
        self.initial_value_constraint.UpdateCoefficients(new_lb=self.S@x0, new_ub=self.S@x0)

        linearized_system = Linearize(
            system=self.system,
            context=context_at_timestep, 
            input_port_index=system_input_port_index, 
            output_port_index=system_output_port_index,
            equilibrium_check_tolerance=1e-6)

        A, B = linearized_system.A(), linearized_system.B()
        # Update linearized dynamics constraints
        # linearized_system is w.r.t. error coordinates (x-x0), (u-u0)
        #       A(t) @ (x[k]-x0) + B(t) @ (u[k]-u0) = x[k+1]
        #       S @ x[k+1] = S @ A(t) @ (x[k] - x0) + S @ B(t) @ (u[k]-u0)
        # Take A[13:,13:] and B[13:,:] # TODO: VERIFY 

        A = A[13:,13:]
        B = B[13:,:]

        for n in range(self.N-1):
            self.linearized_dynamics_constraints[:,n].UpdateCoefficients(
                Aeq=A, 
                beq=self.x[:,n+1] + A@self.x[:,n] - B@(self.u[:,n]-u0))
        
        # Update final value constraint
        xf = traj_init_x_slice[:,-1]
        xf_view = self.PositionView(xf)
        self.final_value_constraint.UpdateCoefficients(new_lb=xf, new_ub=xf)

        # self.prog.AddLinearConstraint(dirtran.initial_state() == current_state - state_ref) # initial condition

        # Update quadratic cost in error coordinates
        # TODO: technically, we only need to remove quadratic_cost[:,0], 
        #       change vars for quadratic[:,1:-2], and add quadratic_cost[:,-1]
        # TODO: index self.traj_*[:,n] may not work with periodicity...consider counting from back -n...?
        #       we know len(self.traj_*) == self.N
        for n in range(self.N-1):
            self.prog.RemoveCost(self.state_error_quadratic_cost[:,n])
            self.prog.RemoveCost(self.input_error_quadratic_cost[:,n])
            self.state_error_quadratic_cost[:,n] = self.prog.AddQuadraticErrorCost(
                Q=2*self.Q,
                b=np.zeros((self.nq,1)),
                desired_x=self.traj_init_x[:,n],
                vars=self.S@self.x[:,n]
            )
            self.input_error_quadratic_cost[:,n] = self.prog.AddQuadraticErrorCost(
                Q=2*self.R,
                b=np.zeros((self.nq,1)),
                desired_x=self.traj_init_u[:,n],
                vars=self.P_y@self.u[:,n]
            )
        self.prog.RemoveCost(self.state_error_quadratic_cost[:,self.N])
        self.state_error_quadratic_cost[:,self.N] = self.prog.AddQuadraticErrorCost(
            Q=2*self.Q,
            b=np.zeros((self.nq,1)),
            desired_x=self.traj_init_u[:,-1],
            vars=self.P_y@self.u[:,n]
        )

        result = Solve(self.prog)

        # Return optimal x and optimal u
        self.optimal_x = result.GetSolution(self.x)
        self.optimal_u = result.GetSolution(self.u)

        self.k = self.k + 1
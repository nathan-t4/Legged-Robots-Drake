import numpy as np

from pydrake.systems.framework import LeafSystem, BasicVector
from pydrake.solvers import MathematicalProgram, Solve, Binding, LinearEqualityConstraint, BoundingBoxConstraint, QuadraticCost, LinearConstraint
from pydrake.symbolic import Variable
from pydrake.math import IsPositiveDefinite

from utils import VerifyTrajectoryIsValidPolynomial, MakeNamedViewPositions, MakeNamedViewVelocities, MakeNamedViewActuation
from quadruped.quad_utils import GetQuadStateProjectionMatrix, ExcludeQuadActuatedCoords, CalcA, CalcB, CalcG, CalcFootPositions


class QuadModelPredictiveController(LeafSystem):
    """
        [TODO]: Still working on this. There are probably a lot of bugs...

        Lumped-Mass Rigid Body MPC for Quadrupeds
        - as detailed in D. Kim, 'Highly Dynamic Quadruped Locomotion via Whole-Body Impulse Control'
        - Input: gait type, speed and direction commands (x = [q_body; v_body], gait type)
        - Output: contact states, reaction force, foot and body position commands

        x(k+1) = A(k) @ x(k) + B(k) @ f(k) + g
        
        x = np.vstack((q_body, v_body))
        f = np.vstack((f1, f2, f3, f4))
    """

    def __init__(self, plant, Q, R, dt, time_period, time_horizon, traj_init_x=None, traj_init_f=None, model_name=None):
        super().__init__()
        self.__plant = plant
        self.__context = self.__plant.CreateDefaultContext()

        # System parameters
        self._nc = 4 # num of contact points
        self._nf = 12 # num of multibody states for the floating body (size of x)
        self._nq = self.__plant.num_multibody_states()
        
        model_instance = 0 if model_name is None else self.__plant.GetModelInstanceByName(model_name)
        # Total mass
        self._M = self.__plant.CalcTotalMass(self.__context)
        # Trunk body frame
        self.__body_frame = self.__plant.GetFrameByName('body')
        # Spatial inertia in trunk body frame
        self._Ib = self.__plant.CalcSpatialInertia(
          context=self.__context,
          frame_F=self.__body_frame,
          body_indexes=self.__plant.GetBodyIndices(model_instance)
        )
        # Static friction coefficient
        self._mu = 1
        # State projection matrix
        self._S = GetQuadStateProjectionMatrix(plant)
        # Actuation matrix
        self._P_y = self.__plant.MakeActuationMatrix()[6:,:].T

        # print('Total mass: ', self._M)
        # print('Spatial inertia: ', self._Ib.CopyToFullMatrix6())

        # Positive-definite QR gains
        self._Q = Q
        self._R = R

        # print(np.shape(Q), np.shape(R))
        # These should be 12x12 matrices
        if np.shape(Q) == (self._nf,) and np.shape(R) == (self._nf,):
            self._Q = np.diag(self._Q)
            self._R = np.diag(self._R)
        else:
            assert np.shape(Q) == (self._nf, self._nf), 'Q should be have shape=(12,12)'
            assert np.shape(R) == (self._nf, 3*self._nc), 'R should be have shape=(12,3*nc)'
        
        assert IsPositiveDefinite(self._Q), 'Q is not positive definite'
        assert IsPositiveDefinite(self._R), 'R is not positive definite'

        # MPC parameters
        self._k = 0 # MPC iteration 
        self._timesteps_btw_MPC = 0
        self._dt = dt
        self._time_period = time_period 
        self._time_horizon = time_horizon

        # Add assertions for dt, time_period, and time_horizon
        assert isinstance(dt, float) and 0 < dt
        assert isinstance(time_horizon, float) and dt < time_horizon
        assert isinstance(time_period, float) and time_horizon <= time_period

        self._N = int(self._time_horizon // self._dt) # TODO: floor division should be ok?
        self._Nt = int(self._time_period // self._dt)

        # print(self._N, self._Nt)

        traj_x_constraints = {
            'shape': (self._nf, self._Nt),
            'total_time': self._time_period
        }
        # TODO: shape[1] is not accounted for in VerifyTrajectoryIsValidPolynomial. Is total time for u also self._time_period
        traj_f_constraints = {
            'shape': (3*self._nc, self._Nt),
            'total_time': self._time_period
        }

        # Transform inputs traj_init_* to PiecewisePolynomial with constraints
        self._traj_init_x = VerifyTrajectoryIsValidPolynomial(traj_init_x, **traj_x_constraints)
        self._traj_init_f = VerifyTrajectoryIsValidPolynomial(traj_init_f, **traj_f_constraints)

        # Discretize PiecewisePolynomial
        self._traj_init_x_discrete = np.empty(shape=(self._nf,self._Nt), dtype=np.float64)
        self._traj_init_f_discrete = np.empty(shape=(3*self._nc,self._Nt), dtype=np.float64)

        t_steps = np.linspace(start=0, stop=self._time_period, num=self._Nt)

        for i in range(len(t_steps)):
            self._traj_init_x_discrete[:,i] = self._traj_init_x.value(t=t_steps[i]).flatten()
            self._traj_init_f_discrete[:,i] = self._traj_init_f.value(t=t_steps[i]).flatten()
        
        # For plant context
        self._estimated_state_input_port = self.DeclareVectorInputPort(
            model_vector=BasicVector(size=self._nq), 
            name='estimated_body_state'
        )
        # self._contact_force_input_port = self.DeclareVectorInputPort(
        #     model_vector=BasicVector(size=3*self._nc), # TODO
        #     name='contact_force_input_port'
        # )
        self._desired_foot_pos_output_port = self.DeclareVectorOutputPort(
            model_value=BasicVector(size=3*self._nc),
            name='control',
            calc=self.SetForce
        )
        self._desired_body_state_output_port = self.DeclareVectorOutputPort(
            model_value=BasicVector(size=self._nf),
            name='desired_body_state',
            calc=self.SetDesiredState
        )
        
        self.DeclarePeriodicPublishEvent(
            period_sec=self._N,
            offset_sec=0.0,
            publish=self.RunModelPredictiveController
        )

        self.prog = MathematicalProgram()

        # Initialize the outputs from MPC (locally optimal state and control)
        self._optimal_x = self._traj_init_x_discrete[:,0:self._N]
        self._optimal_f = self._traj_init_f_discrete[:,0:self._N]
        self._previous_f = self._traj_init_f_discrete[:,0:self._N]

        # Initialize traj_init_* slices used for each finite horizon (RunModelPredictiveController)
        self._traj_init_x_slice = np.empty((self._nf,self._N), dtype=np.float64)
        self._traj_init_f_slice = np.empty((3*self._nc,self._N), dtype=np.float64)

        self._traj_init_x_slice = self._traj_init_x_discrete[:,0:self._N]
        self._traj_init_f_slice = self._traj_init_f_discrete[:,0:self._N-1]

        # Initialize variables
        self._x = np.empty((self._nf, self._N), dtype=Variable) 
        self._f = np.empty((3*self._nc, self._N-1), dtype=Variable)

        # Initialize linearized dynamics matrices
        # A is (12x12) == (self._nf, self._nf). B is (12,(3*4)) == (self.nf, 3*self.nc) where self.nc is the number of contacts
        A = np.zeros((self._nf,self._nf))
        B = np.zeros((self._nf,3*self._nc))

        # Initialize linearized dynamics and torque limit constraints
        self.linearized_dynamics_constraints = np.empty(self._N-1, dtype=Binding[LinearEqualityConstraint])
        self.torque_limit_constraints = np.empty(self._N-1, dtype=Binding[BoundingBoxConstraint])

        # Initialize state error and input error quadratic costs
        self.state_error_quadratic_cost = np.empty(self._N, dtype=Binding[QuadraticCost])
        self.force_magnitude_quadratic_cost = np.empty(self._N-1, dtype=Binding[QuadraticCost])

        # Initialize friction cone constraints [TODO: replace with Lorentz cone constraint]
        self.friction_cone_x_constraint = np.empty(shape=(self._N-1,self._nc), dtype=Binding[LinearConstraint])
        self.friction_cone_y_constraint = np.empty(shape=(self._N-1,self._nc), dtype=Binding[LinearConstraint])
        self.friction_cone_z_constraint = np.empty(shape=(self._N-1,self._nc), dtype=Binding[LinearConstraint])
        
        # Initialize variables [x: floating body state] and [f: 3D contact forces]
        self._x = self.prog.NewContinuousVariables(self._nf, self._N, 'x')
        self._f = self.prog.NewContinuousVariables(3*self._nc, self._N-1, 'f') 

        for n in range(self._N-1):
            # Initialize linearized dynamics constraints to MathematicalProgram
            self.linearized_dynamics_constraints[n] = self.prog.AddLinearEqualityConstraint(
                Aeq = A,
                beq = np.zeros(shape=(3*self._nc,1)),
                vars = self._x[:,n]
            )

            # Initialize torque limits constraints to MathematicalProgram
            self.torque_limit_constraints[n] = self.prog.AddBoundingBoxConstraint(
                self.__plant.GetEffortLowerLimits(), 
                self.__plant.GetEffortUpperLimits(), 
                self._f[:,n]
            )

            # Initialize linearized friction cone constraints 
            # TODO: Replace with Lorentz cone constraint?
            for j in range(self._nc):
                Fx = self._f[3*j,n]
                Fy = self._f[3*j+1,n]
                Fz = self._f[3*j+2,n]
                self.friction_cone_x_constraint = self.prog.AddLinearConstraint(Fx <= self._mu * Fz)
                self.friction_cone_y_constraint = self.prog.AddLinearConstraint(Fy <= self._mu * Fz)
                self.friction_cone_z_constraint = self.prog.AddLinearConstraint(Fz >= 0)

            # Initialize state error cost to MathematicalProgram
            self.prog.AddQuadraticErrorCost(
                Q=2*self._Q,
                x_desired=self._traj_init_x_slice[:,n],
                vars=self._x[:,n]
            )

            # Add input error cost to MathematicalProgram
            self.prog.AddQuadraticErrorCost(
                Q=2*self._R,
                x_desired=self._traj_init_f_slice[:,n],
                vars=self._f[:,n]
            )
        # Initialize quadratic error cost at last timestep
        self.prog.AddQuadraticErrorCost(
                Q=2*self._Q,
                x_desired=self._traj_init_x_slice[:,self._N-1],
                vars=self._x[:,self._N-1]
        )

        # Set initial guess for solver using traj_init_* from planner
        self.prog.SetInitialGuess(self._x, self._traj_init_x_slice)
        self.prog.SetInitialGuess(self._f, self._traj_init_f_slice)

        # Initialize position constraints
        x0 = self.__context.get_state().get_discrete_state().value()
        # TODO: Implement quat2rpy (quat=False only to match sizes)
        x0_body = ExcludeQuadActuatedCoords(self.__plant, x0, quat=False, quat2rpy=True) 

        # TODO: Errors on boundary condition constraints 
        # self.initial_value_constraint = self.prog.AddBoundingBoxConstraint(
        #     lb=x0_body, 
        #     ub=x0_body, 
        #     vars=self._x[:,0].transpose()
        # )
        # self.final_value_constraint = self.prog.AddBoundingBoxConstraint(
        #     lb=self._optimal_x[:,-1], 
        #     ub=self._optimal_x[:,-1], 
        #     vars=self._x[:,-1].transpose()  
        # )
    
    def GetVectorInput(self, context):
        estimated_state = self.EvalVectorInput(context, self._estimated_state_input_port.get_index()).get_value()
        return estimated_state
    
    def SetForce(self, context, output):
        '''
            TODO: 
            - how to align time (self._timesteps_btw_MPC) btw SetTorque and SetDesiredState
        '''
        self._previous_f = self._optimal_f[:,self._timesteps_btw_MPC]
        output.SetFromVector(self._previous_f)
        self._timesteps_btw_MPC += 1

    def SetDesiredState(self, context, output):
        self._previous_x = self._optimal_x[:,self._timesteps_btw_MPC]
        output.SetFromVector(self._previous_x)

    def RunModelPredictiveController(self, context): 
        '''
            MPC at one time step
            - Trajectory stabilization with receding horizon LQR and updated constraints using UpdateCoefficient
            - Runs with frequency 1/self._N

            min x[.] f[.]: l_f(x[N]) + sum(l(x[n],f[n]), 0, N-1)
            s.t.:
                x[n+1] = A@x[n] + B@f[n] + g, n in [0,N-1]
                x[0] = x0
                friction cone constraints
        '''
        # [TODO]: Use GetVectorInput() here
        current_context = context 
        
        # Enforce periodicity when self._k+self._N > self._Nt
        assert len(self._traj_init_x_discrete) == self._Nt
        assert len(self._traj_init_f_discrete) == self._Nt

        start_idx = self._k
        end_idx = self._k + self._N
        # end_idx = (self._k+self._N) if (self._k+self._N) <= self._Nt else ((self._k+self._N)%self._Nt) # periodic verison
        # end_idx = (self._k+self._N) if (self._k+self._N) <= self._Nt else self._Nt # non periodic version

        if end_idx > self._Nt:
            self._traj_init_x_slice = np.hstack((
                    self._traj_init_x_discrete[:,start_idx:],
                    self._traj_init_x_discrete[:,:(end_idx%self._Nt)]
            ))
            self._traj_init_f_slice = np.hstack((
                self._traj_init_f_discrete[:,start_idx:],
                self._traj_init_f_discrete[:,:(end_idx%self._Nt)]
            ))
        else:
            self._traj_init_x_slice = self._traj_init_x_discrete[:,start_idx:end_idx]
            self._traj_init_f_slice = self._traj_init_f_discrete[:,start_idx:end_idx]

        # Verify length of initial trajectory slices are correct
        assert len(self._traj_init_x_slice) == self._N
        assert len(self._traj_init_f_slice) == self._N

        # Set initial guess for MathematicalProgram
        self.prog.SetInitialGuess(self._x, self._traj_init_x_slice)
        self.prog.SetInitialGuess(self._f, self._traj_init_f_slice)

        # Update initial value constraint
        x0 = ExcludeQuadActuatedCoords(
            self.__plant, 
            current_context.get_state().get_discrete_state().value(), 
            quat=True
        )
        f0 = self._optimal_f[:,0] # previous action
        self.initial_value_constraint.evaluator().UpdateCoefficients(new_lb=x0, new_ub=x0)

        # Update linearized dynamics constraints
        # A[k] @ x[k] + B[k] @ u[k] = x[k+1]
        for n in range(self._N):
            # TODO: r = []
            r = CalcFootPositions(plant=self.__plant, context=self.__context) 
            yaw = self._x[-1,n]
            A, B = CalcA(phi=yaw), CalcB(phi=yaw,r=r,dt=self._dt,M=self._M,Ib=self._Ib)
            g = CalcG()
            self.linearized_dynamics_constraints[n].evaluator().UpdateCoefficients(
                Aeq=A, 
                beq=self._x[:,n+1] - B@self._f[:,n] - g
            )
        
        # Update final value constraint to final x from previous MPC
        self.final_value_constraint.evaluator().UpdateCoefficients(
            new_lb=self._optimal_x[:,-1], 
            new_ub=self._optimal_x[:,-1])

        # Update quadratic cost in error coordinates
        for n in range(self._N):
            self.prog.RemoveCost(self.state_error_quadratic_cost[n])
            self.prog.RemoveCost(self.force_magnitude_quadratic_cost[n])
            self.state_error_quadratic_cost[n] = self.prog.AddQuadraticErrorCost(
                Q=2*self._Q,
                b=np.zeros((self._nf,1)),
                desired_x=self._traj_init_x[:,n],
                vars=self._S@self._x[:,n]
            )
            self.force_magnitude_quadratic_cost[n] = self.prog.AddQuadraticErrorCost(
                Q=2*self._R,
                b=np.zeros((3*self._nc,1)),
                desired_x=self._traj_init_f[:,n],
                vars=self._P_y@self._f[:,n]
            )

        result = Solve(self.prog)

        # Save optimal x and optimal u
        self._optimal_x = result.GetSolution(self._x)
        self._optimal_f = result.GetSolution(self._f)

        self._k = self._k + 1
        self._timesteps_btw_MPC = 0
    
    def get_estimated_state_input_port(self):
        return self._estimated_state_input_port
    
    # def get_contact_force_input_port(self):
        # return self._contact_force_input_port
    
    def get_desired_body_state_output_port(self):
        return self._desired_body_state_output_port

    def get_desired_foot_pos_output_port(self):
        return self._desired_foot_pos_output_port
    
    def get_traj_init_x(self):
        return self._traj_init_x
    
    def get_traj_init_f(self):
        return self._traj_init_f
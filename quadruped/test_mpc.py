import numpy as np
import os
import sys

from pydrake.systems.primitives import Saturation
from pydrake.systems.analysis import Simulator
from pydrake.trajectories import PiecewisePolynomial
from pydrake.visualization import AddDefaultVisualization

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils import MakeNamedViewPositions, MakeNamedViewVelocities, MakeNamedViewActuation, quat2eul
from controllers.LeggedRobotPidController import LeggedRobotPidController
from controllers.QuadModelPredictiveController import QuadModelPredictiveController
from planners.ComDynamicsFullKinematicsPlanner import ComDynamicsFullKinematicsPlanner
from quadruped.simulate import setup_env
from quadruped.quad_utils import set_quad_pose, HalfStrideToTraj, ExcludeQuadActuatedCoords

def testMPC(sim_time_step=1e-3, sim_time=10.0, sim_rate=1.0, sim_type='stand'):
    builder, plant = setup_env(sim_time_step)

    # Start visualization
    AddDefaultVisualization(builder)

    # Initialize ComDynamicsFullKinematicsPlanner. Returns [q,v] for one half-stride
    planner = ComDynamicsFullKinematicsPlanner(plant=plant, plant_context=plant.CreateDefaultContext())
    result, vars, gait_params = planner.RunPlanner()
    is_laterally_symmetric, stride_length = gait_params.is_laterally_symmetric, gait_params.stride_length
    result = vars.result
    h = vars.h
    q, v = vars.q, vars.v
    PositionView, VelocityView = vars.PositionView, vars.VelocityView
    t_sol = np.cumsum(np.concatenate(([0],result.GetSolution(h))))
    q_sol = PiecewisePolynomial.FirstOrderHold(t_sol, result.GetSolution(q))
    v_sol = PiecewisePolynomial.FirstOrderHold(t_sol, result.GetSolution(v))

    # Extend the half-stride to (num_strides) full-length strides
    x_full_sol, t_full_sol = HalfStrideToTraj(
        plant=plant, 
        q_sol=q_sol,
        v_sol=v_sol,
        t_sol=t_sol,
        stride_length=stride_length, 
        num_strides=4.0,
        is_laterally_symmetric=is_laterally_symmetric
    )

    assert np.shape(x_full_sol)[0] == plant.num_multibody_states()

    # Filter out actuated DOFs (all leg actuators)
    x_body_sol = ExcludeQuadActuatedCoords(plant, x_full_sol, quat=True)
    assert np.shape(x_body_sol)[0] == 13 # (qw,qx,qy,qz,x,y,z,wx,wy,wz,vx,vy,vz)

    # Convert from quaternion to euler angles [TODO: check velocities]
    x_body_sol_eul = np.empty(shape=(np.size(x_body_sol,0)-1,np.size(x_body_sol,1))) # quaternion to euler angles
    for i in range(np.shape(x_body_sol)[1]):
        quat = x_body_sol[:4,i]
        x_body_sol_eul[:3,i] = quat2eul(quat)
        x_body_sol_eul[3:,i] = x_body_sol[4:,i]

    # Initialize full_f_sol to zeros [TODO: Initialize with contact forces from ComDynamicsFullKinematicsPlanner]
    full_f_sol = np.zeros(shape=(plant.num_actuated_dofs(), len(t_full_sol)-1), dtype=np.float64)

    # Initialize PID controller and torque saturation   
    pid_controller = builder.AddSystem(LeggedRobotPidController(plant=plant))
    torque_limiter = builder.AddSystem(Saturation(
        min_value=plant.GetEffortLowerLimits(),
        max_value=plant.GetEffortUpperLimits()
    ))

    # Initialize namedviews for gains
    PositionCostView = MakeNamedViewPositions(plant, 'PositionCost')
    VelocityCostView = MakeNamedViewVelocities(plant, 'VelocityCost')
    # print(VelocityCostView(np.zeros(plant.num_velocities())))
    ActuationView = MakeNamedViewActuation(plant, 'Actuation')

    # Initialize state error (Q) and input error (R) costs 
    q_cost = [1]*np.shape(x_body_sol_eul)[0] # == 12
    r_cost = ActuationView([1]*plant.num_actuated_dofs())

    mpc_controller = builder.AddSystem(QuadModelPredictiveController(
        plant=plant,
        model_name='quad',
        Q=q_cost,
        R=r_cost[:],
        dt=float(sim_time_step),
        time_horizon=1e-2,
        time_period=float(sim_time), 
        traj_init_x=x_body_sol_eul,
        traj_init_f=full_f_sol,
    ))

    # Connect plant, planner, MPC, PID, and torque_limiter
    # TODO: Make planner into an actual LeafSystem
    # builder.Connect(planner.get_contact_force_output_port(), mpc_controller.get_estimated_contact_force_input_port())
    builder.Connect(plant.get_state_output_port(), pid_controller.get_estimated_state_input_port())  
    builder.Connect(plant.get_state_output_port(), mpc_controller.get_estimated_state_input_port())  
    # TODO: Add feedforward torque to PID controller? Also MPC doesn't have control output anymore.
    # builder.Connect(mpc_controller.get_desired_body_state_output_port(), pid_controller.get_desired_state_input_port())
    # builder.Connect(mpc_controller.get_control_output_port(), pid_controller.get_feedforward_input_port()) 
    builder.Connect(pid_controller.get_output_port(), torque_limiter.get_input_port())
    builder.Connect(torque_limiter.get_output_port(), plant.get_actuation_input_port())

    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    plant_context = plant.GetMyContextFromRoot(context)

    # Set quadruped initial condition -> standing
    set_quad_pose(plant=plant, context=plant_context, mode='stand')

    # Start simulation
    simulator.set_target_realtime_rate(sim_rate)
    simulator.Initialize()
    simulator.AdvanceTo(sim_time) 

def main():
    testMPC()

if __name__ == '__main__':
    main()
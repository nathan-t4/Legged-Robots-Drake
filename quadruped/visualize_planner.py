import numpy as np
import os
import sys

from argparse import ArgumentParser

from pydrake.systems.framework import DiagramBuilder
from pydrake.multibody.parsing import Parser
from pydrake.geometry import StartMeshcat, MeshcatVisualizer
from pydrake.systems.primitives import Saturation
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.analysis import Simulator
from pydrake.trajectories import PiecewisePolynomial
from pydrake.visualization import AddDefaultVisualization

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from planners.ComDynamicsFullKinematicsPlanner import ComDynamicsFullKinematicsPlanner
from controllers.LeggedRobotPidController import LeggedRobotPidController
from quadruped.quad_utils import set_quad_pose, pHalfStrideToFullStride # TODO: replace with HalfStrideToFullStride

'''
    TODO:
    - Use meldis instead of meshcat (or use externally hosted meshcat)
    - Finish simulate_gait_optimization
    - Modify naming convention in mini cheetah urdf file
    - try with better q0 (walking_trot is fine with current q0)
'''

meshcat = StartMeshcat()

def visualize_gait_optimization(sim_time_step, gait):
    '''
        Visualization of the optimized gait (not a simulation)
    '''
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, sim_time_step)
    parser = Parser(plant)
    quadruped_model_file = os.path.relpath('./quadruped/models/mini_cheetah/mini_cheetah_mesh.urdf')
    quad = parser.AddModelFromFile(quadruped_model_file, model_name='quad')
    plant.Finalize()

    # Connect MeshcatVisualizer
    visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    set_quad_pose(plant, plant_context, mode='stand')
    diagram.ForcedPublish(context)

    planner = ComDynamicsFullKinematicsPlanner(plant=plant, plant_context=plant_context, gait=gait)
    result, vars, gait_params = planner.RunPlanner()
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
                qt = pHalfStrideToFullStride(PositionView, qt) # TODO: Remove function. use StateView instead
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
    quadruped_model_file = os.path.relpath('./quadruped/models/mini_cheetah/mini_cheetah_mesh.urdf')
    quad = parser.AddModelFromFile(quadruped_model_file, model_name='quad')
    plant.Finalize()

    # Start visualization
    AddDefaultVisualization(builder)

    # TODO: Add ModelPredictiveController

    # Add PidController with saturation
    pid_controller = builder.AddSystem(LeggedRobotPidController(plant=plant))
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

    planner = ComDynamicsFullKinematicsPlanner(plant=plant, plant_context=plant_context, gait=gait)
    result, vars, gait_params = planner.RunPlanner()

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
    
    q_sol_cubic = PiecewisePolynomial.CubicWithContinuousSecondDerivative(
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
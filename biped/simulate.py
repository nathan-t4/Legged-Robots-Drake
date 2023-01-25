import numpy as np
import os
import sys

from pydrake.common import FindResourceOrThrow
from pydrake.systems.framework import DiagramBuilder
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, CoulombFriction
from pydrake.multibody.tree import JointIndex
from pydrake.systems.analysis import Simulator
from pydrake.visualization import AddDefaultVisualization
from pydrake.geometry import HalfSpace
from pydrake.math import RigidTransform
from pydrake.systems.primitives import Saturation

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
print(SCRIPT_DIR)

from argparse import ArgumentParser
from controllers.LeggedRobotPidController import LeggedRobotPidController
from biped_utils import set_atlas_pose
# from graphviz import Source

def setup_env(sim_time_step=1e-3):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, sim_time_step)
    parser = Parser(plant)
    # Atlas model is from the Drake repo
    biped_model_file = os.path.join(SCRIPT_DIR, './models/atlas/urdf/atlas_convex_hull.urdf') 
    biped = parser.AddModelFromFile(biped_model_file, model_name='biped')
    
    # Add ground plane (Note: Meshcat does not display HalfSpace geometry (yet))
    ground_friction = CoulombFriction(static_friction=1.0, dynamic_friction=1.0)
    plant.RegisterCollisionGeometry(plant.world_body(), RigidTransform(), 
                                    HalfSpace(), 'GroundCollisionGeometry',
                                    ground_friction)
    # ground_color = np.array([0.5, 0.5, 0.5, 0.2])
    # plant.RegisterVisualGeometry(plant.world_body(), RigidTransform(),
    #                              HalfSpace(), 'GroundVisualGeometry', 
    #                              ground_color)

    plant.Finalize()

    # Verify model is the atlas biped 
    assert plant.num_velocities() == 36
    assert plant.num_positions() == 37

    return builder, plant
    
def passive_sim(sim_time_step=1e-2, sim_time=3.0, sim_rate=0.5):
    builder, plant = setup_env(sim_time_step)
    
    # Start visualization
    AddDefaultVisualization(builder)

    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    plant_context = diagram.GetMutableSubsystemContext(plant, context)

    # Set actuation to zero - passive simulation
    tau = np.zeros(plant.num_actuated_dofs())
    plant.get_actuation_input_port().FixValue(plant_context, tau)

    # View block diagram
    # s = Source(diagram.GetGraphvizString(), filename="test.gv", format="png")
    # s.view()

    # set_atlas_pose(plant, plant_context, mode='default')

    # Set pelvis initial position to [x0,y0,z0] = [0,0,0.95] m
    atlas = plant.GetBodyByName(name='pelvis')
    plant.SetFreeBodyPose(plant_context, atlas, RigidTransform(p=[0,0,0.95]))

    # Simulator settings
    simulator.set_target_realtime_rate(sim_rate)
    simulator.Initialize()
    simulator.AdvanceTo(sim_time) 

def active_sim(sim_time_step=1e-2, sim_time=10, sim_rate=1, sim_type='stand'):
    '''
        (TODO) simulation of atlas standing via joint-level PD control
    '''
    builder, plant = setup_env(sim_time_step)

    pid_controller = builder.AddSystem(LeggedRobotPidController(plant=plant, robot_type='biped'))
    torque_limiter = builder.AddSystem(Saturation(
        min_value=plant.GetEffortLowerLimits(),
        max_value=plant.GetEffortUpperLimits()
    ))

    builder.Connect(pid_controller.get_output_port(), torque_limiter.get_input_port())
    builder.Connect(torque_limiter.get_output_port(), plant.get_actuation_input_port())
    builder.Connect(plant.get_state_output_port(), pid_controller.get_input_port_estimated_state())  

    # Start visualization
    AddDefaultVisualization(builder)

    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    plant_context = plant.GetMyContextFromRoot(context)

    # TODO
    set_atlas_pose(plant, plant_context, sim_type)

    # Set desired state for pid_controller
    x0 = plant.get_state_output_port().Eval(plant_context)
    pid_controller.get_input_port_desired_state().FixValue(pid_controller.GetMyContextFromRoot(context),x0)

    # Start simulation
    simulator.set_target_realtime_rate(sim_rate)
    simulator.Initialize()
    simulator.AdvanceTo(sim_time) 

def main():
    parser = ArgumentParser()
    parser.add_argument('--type', dest='type', type=str, default='stand')
    parser.add_argument('--dt', dest='dt', type=float, default=1e-2)
    parser.add_argument('--sim_time',dest='sim_time', type=float, default=3)
    parser.add_argument('--sim_rate',dest='sim_rate', type=float, default=.5)
    args = parser.parse_args()

    passive_sim(sim_time_step=args.dt)
    # active_sim(sim_time_step=args.dt, sim_time=args.sim_time, sim_rate=args.sim_rate, sim_type=args.type)

if __name__ == '__main__':
    main()
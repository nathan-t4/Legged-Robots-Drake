# Quadruped in Drake

Quadruped simulation with Drake. Work in progress.

<img src=https://user-images.githubusercontent.com/19811248/214022195-8c3ff128-9016-434c-b6ab-b2096fd5b782.mp4 width=500/>

Video shows a visualization of the optimized 'walking trot' gait on Mini Cheetah ([Run example](#gait-optimization-com-dynamis-with-full-kinematics-planner---visualization))

Credits to the [LittleDog example](https://github.com/RussTedrake/underactuated/blob/master/examples/littledog.ipynb) in the underactuated textbook

## Run
Start meldis:
```
bazel run //tools:meldis -- --open-window
```

Start a new terminal and run...

### Standing with PID control
```
python3 ./path_to_repo/quadruped/simulate.py
```

- ```--type```: Type of simulation (default='stand')
- ```--dt```: Simulation time step (default=1e-3)
- ```--sim_time```: Maximum time simulation will advance to (default=10)
- ```--sim_rate```: Simulation rate (default=1.0)

### Gait optimization (CoM dynamis with full kinematics planner) - visualization
```
python3 ./path_to_repo/quadruped/visualize_planner.py
```
- ```--gait```: Pre-defined contact sequence (default='walking_trot')

### Gait optimization (CoM dynamis with full kinematics planner) - simulation
```
TODO
```

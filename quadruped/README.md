# Quadruped in Drake

Quadruped simulation with Drake. Work in progress.

<img src=https://user-images.githubusercontent.com/19811248/214022195-8c3ff128-9016-434c-b6ab-b2096fd5b782.mp4 width=500/>

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

### Gait optimization - visualization
```
python3 ./path_to_repo/quadruped/planners/gait_optimization.py
```
- ```--gait```: Pre-defined contact sequence (default='walking_trot')
### Gait optimization - simulation
```
TODO
```

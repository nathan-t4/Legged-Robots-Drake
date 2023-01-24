import numpy as np

from utils import MakeNamedViewPositions

def set_atlas_pose(plant, context, mode='default'):
    '''Set atlas pose [TODO]'''
    if mode == 'default':
        z0 = 0.95
    else:
        raise RuntimeError('Invalid biped pose mode')
    
    PositionView = MakeNamedViewPositions(plant, "Positions")
    q0 = PositionView(np.zeros(plant.num_positions()))

    plant.SetPositions(context, q0[:])
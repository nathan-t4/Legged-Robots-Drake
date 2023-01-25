import numpy as np

from utils import MakeNamedViewPositions

def set_atlas_pose(plant, context, mode='default'):
    '''Set atlas pose [TODO]'''
    pelvis_z = 0.95
    arm_shx = 0.0

    if mode == 'stand':
        pelvis_z = 0.95
        arm_shx = 1.57
    elif mode =='default':
        pass
    else:
        raise RuntimeError('Invalid biped pose mode')
    
    PositionView = MakeNamedViewPositions(plant, "Positions")
    q0 = PositionView(np.zeros(plant.num_positions()))
    # print(q0) # To get all the joint names
    q0.pelvis_qw = 1.0
    q0.pelvis_qx = 0.0
    q0.pelvis_qy = 0.0
    q0.pelvis_qz = 0.0
    q0.pelvis_x = 0.0
    q0.pelvis_y = 0.0
    q0.pelvis_z = pelvis_z
    q0.back_bkz = 0.0
    q0.back_bky = 0.0
    q0.back_bkx = 0.0
    q0.l_arm_shz = 0.0
    q0.l_arm_shx = -arm_shx
    q0.l_arm_ely = 0.0
    q0.l_arm_elx = 0.0
    q0.l_arm_uwy = 0.0
    q0.l_arm_mwx = 0.0
    q0.l_arm_lwy = 0.0
    q0.neck_ay = 0.0
    q0.r_arm_shz = 0.0
    q0.r_arm_shx = arm_shx
    q0.r_arm_ely = 0.0
    q0.r_arm_elx = 0.0
    q0.r_arm_uwy = 0.0
    q0.r_arm_mwx = 0.0
    q0.r_arm_lwy = 0.0
    q0.l_leg_hpz = 0.0
    q0.l_leg_hpx = 0.0
    q0.l_leg_hpy = 0.0
    q0.l_leg_kny = 0.0
    q0.l_leg_aky = 0.0
    q0.l_leg_akx = 0.0
    q0.r_leg_hpz = 0.0
    q0.r_leg_hpx = 0.0
    q0.r_leg_hpy = 0.0
    q0.r_leg_akx = 0.0
    q0.r_leg_aky = 0.0
    q0.r_leg_kny = 0.0

    plant.SetPositions(context, q0[:])

def setup_atlas_gait():
    '''[TODO]'''
    pass
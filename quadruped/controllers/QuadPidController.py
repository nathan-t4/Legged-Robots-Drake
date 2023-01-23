import numpy as np
import sys
import os

from pydrake.systems.framework import LeafSystem, BasicVector
from pydrake.multibody.tree import JointIndex

from utils import MakeNamedViewActuation

# Note: RuntimeError when extending PidController: 
#       Python-extended C++ class does not inherit from pybind11::wrapper<>, and the instance will be sliced. 
#       Either avoid this situation, or the type extends pybind11::wrapper<>.

class QuadPidController(LeafSystem):
    """
        PID Controller for quadrupeds
        - Input: estimated state (num_multibody_states,) AND desired state (num_multibody_states,)
        - Output: controllable torques (num_actuated_dofs,)
        
        Note: Saturation is not incorporated at the moment to explicitly make user connect saturation block
        
        TODO:
        - better default gains based on mass matrix?
        - Add integral term (as continuous state?) 
    """
    def __init__(self, plant, gains=(10.0,1.0)):
        LeafSystem.__init__(self)

        self.__plant = plant
        self.__context = self.__plant.CreateDefaultContext()
        self.__num_actuated_dofs = self.__plant.num_actuated_dofs() 
        self.__num_multibody_states = self.__plant.num_multibody_states()

        self._gains = gains
        self._kp = np.ones(self.__num_actuated_dofs) * self._gains[0]
        self._kd = np.ones(self.__num_actuated_dofs) * self._gains[1]

        self._S = np.zeros((2*self.__num_actuated_dofs,self.__num_multibody_states))
        
        # (L39:51) Source: https://github.com/RussTedrake/underactuated/blob/master/examples/littledog.ipynb 
        # TODO: move somewhere else?
        num_positions = self.__plant.num_positions()
        j = 0
        for i in range(self.__plant.num_joints()):
            joint = self.__plant.get_joint(JointIndex(i))
            # skip floating body indices
            if joint.num_positions() != 1:
                continue
            self._S[j, joint.position_start()] = 1
            self._S[12+j, num_positions + joint.velocity_start()] = 1
            # use lower gain for the knee joints
            if 'knee' in joint.name():
                # self._kd[j] = 0.1
                pass
            j = j+1
        
        # Make gains into matrices
        self._kp = np.diag(self._kp)
        self._kd = np.diag(self._kd)

        # self._P_y = self.__plant.MakeActuationMatrix()[6:,:].T

        # Define system ports
        self._estimated_state_input_port = self.DeclareVectorInputPort(
            model_vector=BasicVector(size=self.__num_multibody_states), 
            name='estimated_state'
        )
        self._desired_state_input_port = self.DeclareVectorInputPort(
            model_vector=BasicVector(size=self.__num_multibody_states),
            name='desired_state'
        )
        self._control_output_port = self.DeclareVectorOutputPort(
            model_value=BasicVector(size=self.__num_actuated_dofs),
            name='control',
            calc=self.GetPdTorque
        )
    
    def GetVectorInput(self, context):
        estimated_state = self.EvalVectorInput(context, self._estimated_state_input_port.get_index()).get_value()
        desired_state = self.EvalVectorInput(context, self._desired_state_input_port.get_index()).get_value()
        return estimated_state, desired_state

    def GetPdTorque(self, context, output):
        x, xd = self.GetVectorInput(context)
        
        x = self._S @ x
        q = x[:self.__num_actuated_dofs]
        v = x[-self.__num_actuated_dofs:]

        xd = self._S @ xd
        qd = xd[:self.__num_actuated_dofs]
        vd = xd[-self.__num_actuated_dofs:]

        # TODO: why do I need negative sign here? (check actuation direction in mini-cheetah urdf)
        u = -(self._kp@(q-qd) + self._kd@(v-vd))
        output.SetFromVector(u)

        # ActuationView = MakeNamedViewActuation(self.__plant, 'u')
        # u_view = ActuationView(np.zeros(self.__num_actuated_dofs))
        # u_view.torso_to_abduct_fl_j_actuator     = u[0]
        # u_view.abduct_fl_to_thigh_fl_j_actuator  = u[1]
        # u_view.thigh_fl_to_knee_fl_j_actuator    = u[2]
        # u_view.torso_to_abduct_fr_j_actuator     = u[3]
        # u_view.abduct_fr_to_thigh_fr_j_actuator  = u[4]
        # u_view.thigh_fr_to_knee_fr_j_actuator    = u[5]
        # u_view.torso_to_abduct_hl_j_actuator     = u[6]
        # u_view.abduct_hl_to_thigh_hl_j_actuator  = u[7]
        # u_view.thigh_hl_to_knee_hl_j_actuator    = u[8]
        # u_view.torso_to_abduct_hr_j_actuator     = u[9]
        # u_view.abduct_hr_to_thigh_hr_j_actuator  = u[10]
        # u_view.thigh_hr_to_knee_hr_j_actuator    = u[11]

        # output.SetFromVector(u_view[:])
    
    def get_input_port_estimated_state(self):
        '''Get estimated state input port'''
        return self._estimated_state_input_port

    def get_input_port_desired_state(self):
        '''Get desired state input port'''
        return self._desired_state_input_port

    def get_output_port_control(self):
        '''Get control output port'''
        return self._control_output_port

    def get_Kp_vector(self):
        '''Get proportional gains'''
        return self._kp
    
    def get_Ki_vector(self):
        '''Get integral gains'''
        pass

    def get_Kd_vector(self):
        '''Get derivative gains'''
        return self._kd
    
    # Read-only attributes
    @property
    def S(self):
        '''Get state projection matrix'''
        return self._S
    # @property
    # def P_y(self):
    #     '''Get output projection matrix'''
    #     return self._P_y 
import numpy as np
import sys
import os

from pydrake.systems.framework import LeafSystem, BasicVector
from pydrake.multibody.tree import JointIndex

from utils import (MakeNamedViewActuation, GetStateProjectionMatrix, ModifyGains, SetActuationView)

# Note: RuntimeError when extending PidController: 
#       Python-extended C++ class does not inherit from pybind11::wrapper<>, and the instance will be sliced. 
#       Either avoid this situation, or the type extends pybind11::wrapper<>.

class LeggedRobotPidController(LeafSystem):
    """
        PID Controller for legged robots
        - Input: estimated state (num_multibody_states,) AND desired state (num_multibody_states,)
        - Output: controllable torques (num_actuated_dofs,)
        
        Note: Saturation is not incorporated at the moment to explicitly make user connect saturation block
        
        TODO:
        - Make LeggedRobotPidController
        - better default gains based on mass matrix?
        - Add integral term (as continuous state?) 
    """
    def __init__(self, plant, robot_type='quad', gains=(10,1)):
        super().__init__()

        self.__plant = plant
        self.__context = self.__plant.CreateDefaultContext()
        self.__num_actuated_dofs = self.__plant.num_actuated_dofs() 
        self.__num_multibody_states = self.__plant.num_multibody_states()
        self.__robot_type = robot_type

        assert len(gains) == 2

        self._gains = gains
        self._kp = np.ones(self.__num_actuated_dofs) * self._gains[0]
        self._kd = np.ones(self.__num_actuated_dofs) * self._gains[1]

        self._S = np.zeros((2*self.__num_actuated_dofs,self.__num_multibody_states))
        
        # TODO: test if self.kd was changed
        self._S = GetStateProjectionMatrix(plant=self.__plant)
        ModifyGains(plant=self.__plant,kp=self._kp, kd=self._kd, robot_type=self.__robot_type)
        print(self._kd)
        
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
        # u_view = SetActuationView(u_view, u, self.__robot_type)
        # output.SetFromVector(u[:])
    
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
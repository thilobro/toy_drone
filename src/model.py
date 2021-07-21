# casadi model of toy drone

import casadi as ca
import numpy as np
from utils import rk4


class ToyDroneModel():
    # TODO: docstrings

    def __init__(self, parameters):
        self._parameters = parameters
        self._state = np.zeros(6)
        self._controls =  np.zeros(2)
        self._build_model_functions()
        # state consists of 2 positions, 2 velocities, 1 angle, one angular velocity
        # controls consists of 2 forces

    def _build_model_functions(self):
        state = ca.SX.sym("state", 6)
        controls = ca.SX.sym("controls", 2)
        # TODO: how to manage disturbance states?
        disturbancce_state = ca.SX.sym("disturbance_state", 9)

        self._ode = self._build_ode(state, controls)
        self._disturbed_ode = self._build_disturbed_ode(state, controls)
        self._sensor_model = self._build_sensor_model(state, controls)
        self._state_jacobian = self._build_state_jacobian(state, controls)
        self._control_jacobian = self._build_control_jacobian(state, controls)
        self._sensor_jacobian = self._build_sensor_jacobian(state, controls)

    def _build_disturbed_ode(self, state, controls):
        # TODO: build ode with disturbance states, acceleration and torque disturbance
        disturbances = ca.SX.sym("disturbances", 3)
        disturbed_state = ca.vertcat(state, disturbances)
        disturbed_ode = ca.Function("disturbed_ode",
                                          [disturbed_state, controls],
                                          [ca.vertcat(self._ode(disturbed_state[:-3],
                                                                controls), 0, 0, 0)])
        return disturbed_ode

    def _build_ode(self, state, controls):
        velocity = state[2:4]
        angle = state[4]
        angular_velocity = state[5]

        dposition = velocity
        dvelocity = 1.0/self._parameters["mass"] * ((controls[0] + controls[1])
                                                    * ca.vertcat(ca.sin(angle), -ca.cos(angle))
                                                    + ca.vertcat(0, self._parameters["gravity"]))
        dangle = angular_velocity
        dangular_velocity = 1.0/self._parameters["moment_of_inertia"]\
            * self._parameters["arm_length"] * (controls[0] - controls[1])
        dstate = ca.vertcat(dposition, dvelocity, dangle, dangular_velocity)
        # TODO: add noise
        return ca.Function("ode", [state, controls], [dstate])

    def _build_sensor_model(self, state, controls):
        # GPS + gyro + accelerometer
        # angle = state[4]
        # acceleration = 1.0/self._parameters["mass"] * ((controls[0] + controls[1])
        #                                                * ca.vertcat(ca.sin(angle), -ca.cos(angle))
        #                                                + ca.vertcat(0, self._parameters["gravity"]))
        # sensor_values = ca.vertcat(state[:2], state[5], acceleration)
        # return ca.Function("sensor_model", [state, controls], [sensor_values])
        return ca.Function("sensor_model", [state, controls], [state])

    def _build_state_jacobian(self, state, controls):
        return ca.Function("state_jacobian", [state, controls],
                           [ca.jacobian(self._ode(state, controls), state)])

    def _build_control_jacobian(self, state, controls):
        return ca.Function("control_jacobian", [state, controls],
                           [ca.jacobian(self._ode(state, controls), controls)])

    def _build_sensor_jacobian(self, state, controls):
        return ca.Function("sensor_jacobian", [state, controls],
                           [ca.jacobian(self._sensor_model(state, controls), state)])

    def get_sensor_model(self):
        return self._sensor_model

    def make_step(self, controls, dt):
        self._controls = np.clip(controls, 0, self._parameters["max_force_input"])
        self._state = rk4(self._ode, self._state, self._controls, dt).full().squeeze()
        return self._state

    def get_ode(self):
        return self._ode

    def get_sensor_values(self):
        return self._sensor_model(self._state, self._controls).full().squeeze()

    def get_state_jacobian(self):
        return self._state_jacobian

    def get_control_jacobian(self):
        return self._control_jacobian

    def get_sensor_jacobian(self):
        return self._sensor_jacobian

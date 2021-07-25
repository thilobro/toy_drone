# casadi model of toy drone

import casadi as ca
import numpy as np
from toy_drone.utils import rk4


class ToyDroneModel():
    """
    ToyDroneModel Model of a 2d drone with two propellers. States are position, velocity,
    orientation and angular velocity. Controls are two forces of the props.
    """

    def __init__(self, parameters):
        """
        __init__ Constructor

        Args:
            parameters: dictionary of model paramters
        """
        self._state_dim = 6
        self._controls_dim = 2
        self._parameters = parameters
        self._state = np.zeros(6)
        self._controls = np.zeros(2)
        self._build_model_functions()

    def _build_model_functions(self):
        """
        _build_model_functions Build the casadi functions for ODE, sensor model and jacobians.
        """

        state = ca.SX.sym("state", self._state_dim)
        controls = ca.SX.sym("controls", self._controls_dim)

        self._ode = self._build_ode(state, controls)
        self._sensor_model = self._build_sensor_model(state, controls)
        self._state_jacobian = self._build_state_jacobian(state, controls)
        self._control_jacobian = self._build_control_jacobian(state, controls)
        self._sensor_jacobian = self._build_sensor_jacobian(state, controls)

    def _build_ode(self, state, controls):
        """
        _build_ode Build the ode of the model in the form of xdot = f(x, u) as a casadi function.

        Args:
            state: symbolic state
            controls: symbolic controls

        Returns:
            casadi ode function
        """
        velocity = state[2:4]
        angle = state[4]
        angular_velocity = state[5]

        dposition = velocity
        dvelocity = 1.0/self._parameters["mass"] * ((controls[0] + controls[1])
                                                    * ca.vertcat(ca.sin(angle), -ca.cos(angle))
                                                    + ca.vertcat(0, self._parameters["mass"]
                                                                 * self._parameters["gravity"]))
        dangle = angular_velocity
        dangular_velocity = 1.0/self._parameters["moment_of_inertia"]\
            * self._parameters["arm_length"] * (controls[0] - controls[1])
        dstate = ca.vertcat(dposition, dvelocity, dangle, dangular_velocity)
        # TODO: add noise
        return ca.Function("ode", [state, controls], [dstate])

    def _build_sensor_model(self, state, controls):
        """
        _build_sensor_model Build the sensor model as a casadi function. All sensors (GPS, gyro,
        accelerometer) are in one sensor equation of the form y = h(x, u).

        Args:
            state: symbolic state
            controls: symbolic controls

        Returns:
            casadi sensor model function
        """
        angle = state[4]
        acceleration = 1.0/self._parameters["mass"] * ((controls[0] + controls[1])
                                                       * ca.vertcat(ca.sin(angle), -ca.cos(angle))
                                                       + ca.vertcat(0, self._parameters["gravity"]))
        gps = state[:2]
        gyro = state[5]
        sensor_values = ca.vertcat(gps, gyro, acceleration)
        return ca.Function("sensor_model", [state, controls], [sensor_values])

    def _build_state_jacobian(self, state, controls):
        """
        _build_state_jacobian Build casadi function of the state jacobian df/dx(x, u) at state x
        and control u.

        Args:
            state: symbolic state
            controls: symbolic controls

        Returns:
            casadi state jacobian function
        """
        return ca.Function("state_jacobian", [state, controls],
                           [ca.jacobian(self._ode(state, controls), state)])

    def _build_control_jacobian(self, state, controls):
        """
        _build_control_jacobian Build casadi function of the control jacobian df/du(x, u) at state x
        and contol u.

        Args:
            state: symbolic state
            controls: symbolic controls

        Returns:
            casadi control jacobian function
        """
        return ca.Function("control_jacobian", [state, controls],
                           [ca.jacobian(self._ode(state, controls), controls)])

    def _build_sensor_jacobian(self, state, controls):
        """
        _build_sensor_jacobian Build casadi function of the sensor jacobian dh/dx(x, u) at state x
        and control u.

        Args:
            state: symbolic state
            controls: symbolic controls

        Returns:
            casadi sensor jacobian function
        """
        return ca.Function("sensor_jacobian", [state, controls],
                           [ca.jacobian(self._sensor_model(state, controls), state)])

    def make_step(self, controls, dt):
        """
        make_step Integrate the model forward for time step dt with rk4. Clip inputs to simulate
        physical constraints of the model.

        Args:
            controls: control inputs (zero-order-hold)
            dt: integration time step

        Returns:
            new state
        """
        self._controls = np.clip(controls, 0, self._parameters["max_force_input"])
        self._state = rk4(self._ode, self._state, self._controls, dt).full().squeeze()
        return self._state

    def get_sensor_model(self):
        return self._sensor_model

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

    def get_state_dim(self):
        return self._state_dim

    def get_controls_dim(self):
        return self._controls_dim

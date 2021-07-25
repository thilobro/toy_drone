import numpy as np
from toy_drone.utils import rk4


class ExtendedKalmanFilter():
    """
    ExtendedKalmanFilter Extended Kalman Filter implementation with correction and prediction step
    that are performed when a new sensor value is input.
    """

    def __init__(self, initial_state, initial_controls, state_covariance,
                 sensor_covariance, state_model, sensor_model, state_jacobian,
                 sensor_jacobian, dt):
        """
        __init__ Constructor.

        Args:
            initial_state: Initial state at which estimation is started
            initial_controls: Initial controls at which estimation is started
            state_covariance: Tuning matrix for the state covariance
            sensor_covariance: Tuning matrix for the sensor covariance
            state_model: Model ode of the system in the form of a casadi function
            sensor_model: Sensor model in the form of a casadi function
            state_jacobian: State jacobian as a casadi function of state and controls
            sensor_jacobian: Sensor jacobian as a casadi function of state and controls
            dt: Discretization time step
        """
        self._state_covariance = state_covariance
        self._sensor_covariance = sensor_covariance
        self._state_model = state_model
        self._sensor_model = sensor_model
        self._sensor_jacobian = sensor_jacobian
        self._state_jacobian = state_jacobian
        self._controls = initial_controls
        self._estimated_state = initial_state
        self._estimated_covariance = state_covariance
        self._dt = dt

    def _correction_step(self, measurement):
        """
        _correction_step Performs a correction step for a sensor measurement. Corrects the estimated
        state and the estimated covariance. Before the correction step, we always perform a
        prediction step first.

        Args:
            measurement: Sensor measurement of all sensor measurements y for the sensor
            equation y = h(x, u)
        """
        self._prediction_step()
        measurement_residual = measurement - self._sensor_model(self._estimated_state,
                                                                self._controls).full().squeeze()
        H = self._sensor_jacobian(self._estimated_state, self._controls).full().squeeze()
        covariance_residual = H @ self._estimated_covariance @ H.T\
            + self._sensor_covariance
        kalman_gain = self._estimated_covariance @ H.T @ covariance_residual
        self._estimated_state += kalman_gain @ measurement_residual
        self._estimated_covariance = (np.eye(self._estimated_covariance.shape[0])
                                      - kalman_gain @ H) @ self._estimated_covariance

    def _prediction_step(self):
        """
        _prediction_step Performs a prediction step with the model. Predicts estimated state and
        estimated covariance.
        """
        self._estimated_state = rk4(self._state_model, self._estimated_state,
                                    self._controls, self._dt).full().squeeze()

        F = self._state_jacobian(self._estimated_state, self._controls).full().squeeze()
        self._estimated_covariance = F @ self._estimated_covariance @ F.T\
            + self._state_covariance

    def input_measurement(self, measurement):
        """
        input_measurement Input a sensor measurement and perform a correction step with it.

        Args:
            measurement: Sensor measurement of all sensor measurements y for the sensor
            equation y = h(x, u)
        """
        self._correction_step(measurement)

    def input_controls(self, controls):
        self._controls = controls

    def get_estimate(self):
        return self._estimated_state

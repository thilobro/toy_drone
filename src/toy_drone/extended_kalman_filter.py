import numpy as np
from toy_drone.utils import rk4


class ExtendedKalmanFilter():
    # TODO: docstrings
    def __init__(self, initial_state, initial_controls, state_covariance,
                 sensor_covariance, state_model, sensor_model, state_jacobian,
                 sensor_jacobian, dt):
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
        # TODO: state and sensor jacobians must be functions
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
        # TODO: state and sensor jacobians must be functions
        self._estimated_state = rk4(self._state_model, self._estimated_state,
                                    self._controls, self._dt).full().squeeze()

        F = self._state_jacobian(self._estimated_state, self._controls).full().squeeze()
        self._estimated_covariance = F @ self._estimated_covariance @ F.T\
            + self._state_covariance

    def input_measurement(self, measurement):
        self._correction_step(measurement)

    def input_controls(self, controls):
        self._controls = controls

    def get_estimate(self):
        return self._estimated_state

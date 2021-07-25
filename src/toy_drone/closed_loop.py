import numpy as np


class ClosedLoop():
    def __init__(self, model, controller, estimator):
        self._model = model
        self._controller = controller
        self._estimator = estimator

    def run_simulation(self, N, dt, reference_data):
        reference_size = self._controller.get_reference_size()
        state_data = np.zeros([N, 6])
        estimated_state_data = np.zeros([N, 6])
        control_data = np.zeros([N - 1, 2])
        for i in range(N - 1):
            controls = self._controller.compute_controls(estimated_state_data[i],
                                                         reference_data[i:i+reference_size]
                                                         .flatten())
            control_data[i] = controls
            state_data[i + 1] = self._model.make_step(controls, dt)

            sensor_values = self._model.get_sensor_values()
            self._estimator.input_controls(controls)
            self._estimator.input_measurement(sensor_values)
            estimated_state_data[i + 1] = self._estimator.get_estimate()

        return state_data, estimated_state_data, control_data

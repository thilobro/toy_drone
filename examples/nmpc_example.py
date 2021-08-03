import numpy as np
import matplotlib.pyplot as plt
from toy_drone.model import ToyDroneModel
from toy_drone.controller import Controller
from toy_drone.extended_kalman_filter import ExtendedKalmanFilter
from toy_drone.plotting import plot_drone_trajectory
from toy_drone.plotting import plot_drone_states_and_controls
from toy_drone.nmpc import Nmpc
from toy_drone.closed_loop import ClosedLoop

# define discretization parameters
simulation_time = 10
dt = 5 * 1e-1
N = int(simulation_time/dt)

# set up model
parameters = {"mass": 1, "moment_of_inertia": 1, "arm_length": 1,
              "gravity": 9.81, "max_force_input": None}
hover_force = (parameters["mass"] * parameters["gravity"])/2.0
parameters["max_force_input"] = hover_force * 1.5
drone = ToyDroneModel(parameters)

# set up estimator
initial_state = np.zeros(6)
initial_controls = np.zeros(2)
state_covariance = 1e-3 * np.eye(6)
sensor_covariance = 1e-6 * np.eye(5)
state_model = drone.get_ode()
sensor_model = drone.get_sensor_model()
state_jacobian = drone.get_state_jacobian()
sensor_jacobian = drone.get_sensor_jacobian()
kalman_filter = ExtendedKalmanFilter(initial_state, initial_controls, state_covariance,
                                     sensor_covariance, state_model, sensor_model,
                                     state_jacobian, sensor_jacobian, dt)

# set up nmpc
N_nmpc = 40
Q = np.diag([1, 1, 0, 0, 0, 0])
Q_term = np.diag([1, 1, 1, 1, 1, 1])
R = np.diag([1, 1]) * 1e-6
dR = np.diag([1, 1]) * 1e-3
control_bounds = [0, parameters["max_force_input"]]
nmpc = Nmpc(drone, Q, Q_term, R, dR, control_bounds, dt, N_nmpc)
nmpc_controller = Controller(nmpc.compute_control, lambda: 0, N_nmpc)

# set up simulation
closed_loop = ClosedLoop(drone, nmpc_controller, kalman_filter)

# set up reference
reference_data = np.zeros([N + N_nmpc, 6])
reference_data[:, 0] = 1
# reference_data[:, 0] = np.sin(np.linspace(0, 2 * np.pi, N + N_nmpc))
# reference_data[:, 1] = -1 + np.cos(-np.linspace(0, 2 * np.pi, N + N_nmpc))

# run simulation
state_data, estimated_state_data, control_data = closed_loop.run_simulation(N, dt, reference_data)

# plot results
plot_drone_trajectory(state_data, reference_data)
plot_drone_states_and_controls(state_data, reference_data, control_data, dt)
plt.show()

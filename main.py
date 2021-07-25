import numpy as np
import matplotlib.pyplot as plt
from toy_drone.model import ToyDroneModel
from toy_drone.lqr import Lqr
from toy_drone.controller import Controller
from toy_drone.extended_kalman_filter import ExtendedKalmanFilter
from toy_drone.plotting import plot_drone_trajectory
from toy_drone.plotting import plot_drone_states
from toy_drone.nmpc import Nmpc
from toy_drone.closed_loop import ClosedLoop


parameters = {"mass": 1, "moment_of_inertia": 1, "arm_length": 1,
              "gravity": 9.81, "max_force_input": None}
hover_force = (parameters["mass"] * parameters["gravity"])/2.0
parameters["max_force_input"] = hover_force * 1.5
N = 1000
state_data = np.zeros([N, 6])
estimated_state_data = np.zeros([N, 6])
dt = 1e-1
drone = ToyDroneModel(parameters)

state_jacobian = drone.get_state_jacobian()(np.zeros(6), hover_force * np.ones(2)).full().squeeze()
control_jacobian = drone.get_control_jacobian()(np.zeros(6),
                                                hover_force * np.ones(2)).full().squeeze()
state_cost = np.diag([1, 1, 0, 0, 0, 0])
control_cost = np.eye(2) * 1e-3
lqr = Lqr(state_jacobian, control_jacobian, state_cost, control_cost, dt)

lqr_controller = Controller(lqr.compute_control, lambda: hover_force, 1)

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

N_nmpc = 40
Q = np.diag([1, 1, 0, 0, 0, 1e-3])
Q_term = np.diag([1, 1, 1, 1, 1, 1])
R = np.diag([1, 1]) * 1e-6
dR = np.diag([1, 1]) * 1e-2
control_bounds = [0, parameters["max_force_input"]]
nmpc = Nmpc(drone, Q, Q_term, R, dR, control_bounds, dt, N_nmpc)

reference_data = np.zeros([N + N_nmpc, 6])
control_data = np.zeros([N - 1, 2])
reference_data[:, 0] = np.sin(np.linspace(0, 2 * np.pi, N + N_nmpc))
reference_data[:, 1] = -1 + np.cos(-np.linspace(0, 2 * np.pi, N + N_nmpc))

nmpc_controller = Controller(nmpc.compute_control, lambda: 0, N_nmpc)

# TODO: write unit tests

closed_loop = ClosedLoop(drone, lqr_controller, kalman_filter)
state_data, estimated_state_data, control_data = closed_loop.run_simulation(N, dt, reference_data)

plot_drone_trajectory(state_data, reference_data)
plot_drone_states(state_data, control_data)
plt.show()

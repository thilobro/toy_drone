import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from model import ToyDroneModel
from lqr import Lqr
from controller import Controller
from extended_kalman_filter import ExtendedKalmanFilter


parameters = {"mass": 10, "moment_of_inertia": 0.1, "arm_length": 0.1,
              "gravity": 9.81, "max_force_input": 1000}
hover_force = (parameters["mass"] * parameters["gravity"])/2.0
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


def feedforward_law():
    return hover_force


controller = Controller(lqr.compute_control, feedforward_law)

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

reference_data = np.zeros([N, 6])
reference_data[:, 0] = np.sin(np.linspace(0, 2 * np.pi, N))
reference_data[:, 1] = -1 + np.cos(-np.linspace(0, 2 * np.pi, N))

# TODO: write unit tests

for i in range(N - 1):
    error = estimated_state_data[i] - reference_data[i]
    controls = controller.compute_controls(error)
    state_data[i + 1] = drone.make_step(controls, dt)

    sensor_values = drone.get_sensor_values()
    kalman_filter.input_controls(controls)
    kalman_filter.input_measurement(sensor_values)
    estimated_state_data[i + 1] = kalman_filter.get_estimate()


fig, ax = plt.subplots(1)
ax.plot(state_data[:, 0], -state_data[:, 1], alpha=0.2)
ts = ax.transData
for i in range(0, N, 20):
    pos_x = state_data[i, 0]
    pos_z = -state_data[i, 1]
    vel_x = state_data[i, 2] * 1e-1
    vel_z = -state_data[i, 3] * 1e-1
    angle = state_data[i, 4]
    att_x = 0.1 * np.sin(angle)
    att_z = 0.1 * np.cos(angle)
    circle_position = mpatches.Circle((pos_x, pos_z), radius=0.01, color='g')
    arrow_speed = mpatches.FancyArrowPatch((pos_x, pos_z), (pos_x + vel_x, pos_z + vel_z), arrowstyle='->', color='r', mutation_scale=2)
    arrow_attitude = mpatches.FancyArrowPatch((pos_x, pos_z), (pos_x + att_x, pos_z + att_z), arrowstyle='->', color='k', mutation_scale=2)
    ax.add_patch(circle_position)
    ax.add_patch(arrow_speed)
    ax.add_patch(arrow_attitude)
ax.plot(reference_data[:, 0], -reference_data[:, 1], '--')
ax.set_aspect('equal')
plt.show()

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def plot_drone_trajectory(trajectory_data, reference_data):
    """
    plot_drone_trajectory Plots the drone's trajectory on a x-z plot with negative z axis drawn up
    for intuitive visualization. Drone position is drawn blue with a green dot every few samples,
    velocity a red vector, orientation a black vector facing up.
    Reference position is shown dashed.

    Args:
        trajectory_data: array with the trajectory data
        reference_data: array with the reference_data
    """
    fig, ax = plt.subplots(1)
    fig.suptitle("Drone position, orientation and velocity in a x-z plot")
    ax.set_xlabel("x position in [m]")
    ax.set_ylabel("z position in [m]")
    ax.plot(trajectory_data[:, 0], -trajectory_data[:, 1], alpha=0.3)
    for i in range(0, trajectory_data.shape[0], 20):
        pos_x = trajectory_data[i, 0]
        pos_z = -trajectory_data[i, 1]
        vel_x = trajectory_data[i, 2] * 1e-1
        vel_z = -trajectory_data[i, 3] * 1e-1
        angle = trajectory_data[i, 4]
        att_x = 0.07 * np.sin(angle)
        att_z = 0.07 * np.cos(angle)
        circle_position = mpatches.Circle((pos_x, pos_z), radius=0.01, color='g')
        arrow_speed = mpatches.FancyArrowPatch((pos_x, pos_z), (pos_x + vel_x, pos_z + vel_z),
                                               arrowstyle='->', color='r', mutation_scale=2)
        arrow_attitude = mpatches.FancyArrowPatch((pos_x, pos_z), (pos_x + att_x, pos_z + att_z),
                                                  arrowstyle='->', color='k', mutation_scale=2)
        ax.add_patch(circle_position)
        ax.add_patch(arrow_speed)
        ax.add_patch(arrow_attitude)
    ax.plot(reference_data[:, 0], -reference_data[:, 1], '--')
    ax.set_aspect('equal')


def plot_drone_states_and_controls(trajectory_data, reference_data, controls_data, dt):
    """
    plot_drone_states_and_controls Subplots for every drone state and control. Reference is drawn
    dashed.

    Args:
        trajectory_data: array with the trajectory data
        reference_data: array with the reference_data
        controls_data: array with the controls_data
        dt: discretization timestep
    """
    N = trajectory_data.shape[0]
    t = np.linspace(0, N*dt, N)
    label_list = ["pos x [m]", "pos z [m]", "vel x [m/s]", "vel z [m/s]",
                  "orientation [rad]", "ang vel [rad/s]",
                  "left prop [N]", "right prop [N]"]
    fig, ax = plt.subplots(8, sharex="all")
    fig.suptitle("Drone states and controls over time")
    for i in range(6):
        ax[i].set_ylabel(label_list[i])
        ax[i].plot(t, trajectory_data[:, i])
        ax[i].plot(t, reference_data[:N, i], "--")
    ax[6].plot(t[:-1], controls_data[:, 0])
    ax[6].set_ylabel(label_list[6])
    ax[7].plot(t[:-1], controls_data[:, 1])
    ax[7].set_ylabel(label_list[7])
    ax[7].set_xlabel("time in [s]")

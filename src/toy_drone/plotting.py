import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def plot_drone_trajectory(trajectory_data, reference_data):
    fig, ax = plt.subplots(1)
    ax.plot(trajectory_data[:, 0], -trajectory_data[:, 1], alpha=0.2)
    for i in range(0, trajectory_data.shape[0], 20):
        pos_x = trajectory_data[i, 0]
        pos_z = -trajectory_data[i, 1]
        vel_x = trajectory_data[i, 2] * 1e-1
        vel_z = -trajectory_data[i, 3] * 1e-1
        angle = trajectory_data[i, 4]
        att_x = 0.1 * np.sin(angle)
        att_z = 0.1 * np.cos(angle)
        circle_position = mpatches.Circle((pos_x, pos_z), radius=0.01, color='g')
        arrow_speed = mpatches.FancyArrowPatch((pos_x, pos_z), (pos_x + vel_x, pos_z + vel_z),
                                               arrowstyle='->', color='r', mutation_scale=2)
        arrow_attitude = mpatches.FancyArrowPatch((pos_x, pos_z), (pos_x + att_x, pos_z + att_z),
                                                  arrowstyle='->', color='k', mutation_scale=2)
        ax.add_patch(circle_position)
        ax.add_patch(arrow_speed)
        ax.add_patch(arrow_attitude)
    ax.plot(reference_data[:, 0], -reference_data[:, 1], '--')
    # ax.set_aspect('equal')
    plt.show()


def plot_drone_states(trajectory_data, controls_data):
    fig, ax = plt.subplots(8)
    for i in range(6):
        ax[i].plot(trajectory_data[:, i])
    ax[6].plot(controls_data[:, 0])
    ax[7].plot(controls_data[:, 1])
    plt.show()

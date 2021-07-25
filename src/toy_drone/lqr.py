import numpy as np


class Lqr():
    # TODO: docstrings
    def __init__(self, state_jacobian, control_jacobian, state_cost, control_cost, dt):
        A, B = self.discretize_jacobians(state_jacobian, control_jacobian, dt)
        self.compute_and_set_optimal_feedback(A, B, state_cost, control_cost)

    @staticmethod
    def solve_ricatti_recursion(A, B, Q, R, max_iterations=1000, epsilon=1e-8):
        P = Q
        for i in range(max_iterations):
            P_new = A.transpose() @ P @ A - (A.transpose() @ P @ B)\
                @ np.linalg.inv(R + B.transpose() @ P @ B) @ (B.transpose() @ P @ A) + Q
            if np.linalg.norm(P - P_new) < epsilon:
                break
            P = P_new
        return P

    def compute_and_set_optimal_feedback(self, A, B, Q, R):
        P = self.solve_ricatti_recursion(A, B, Q, R)
        self._K = np.linalg.inv(R + B.transpose() @ P @ B) @ B.transpose() @ P @ A

    def compute_control(self, state, reference):
        return -self._K @ (state - reference)

    @staticmethod
    def discretize_jacobians(state_jacobian, control_jacobian, dt):
        A = np.eye(state_jacobian.shape[0]) + dt * state_jacobian
        B = dt * control_jacobian
        return A, B

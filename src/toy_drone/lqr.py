import numpy as np


class Lqr():
    """
    Lqr Linear Quadratic Regulator.
    """

    def __init__(self, state_jacobian, control_jacobian, state_cost, control_cost, dt):
        """
        __init__ Constructor.

        Args:
            state_jacobian: State jacobian df/dx(x, u) evaluated at a operating point
            control_jacobian: control jacobian df/du(x, u) evaluated at a operating point
            state_cost: Quadratic state cost tuning matrix
            control_cost: Quadratic control cost tuning matrix
            dt: Discretization time step
        """
        A, B = self.discretize_jacobians(state_jacobian, control_jacobian, dt)
        self.compute_and_set_optimal_feedback_matrix(A, B, state_cost, control_cost)

    def compute_and_set_optimal_feedback_matrix(self, A, B, Q, R):
        """
        compute_and_set_optimal_feedback_matrix Computes the optimal feedbakc gain by solving
        the DARE.

        Args:
            A: Discrete state jacobian at operating point
            B: Discrete control jacobian at operating point
            Q: Quadratic state cost tuning matrix
            R: Quadratic control cost tuning matrix
        """
        P = self.solve_ricatti_recursion(A, B, Q, R)
        self._K = np.linalg.inv(R + B.transpose() @ P @ B) @ B.transpose() @ P @ A

    def compute_control(self, state, reference):
        """
        compute_control Compute a control with the optimal feedback gain K.

        Args:
            state: state at which control should be computed
            reference: Reference used for error computation
        """
        return -self._K @ (state - reference)

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

    @staticmethod
    def discretize_jacobians(state_jacobian, control_jacobian, dt):
        A = np.eye(state_jacobian.shape[0]) + dt * state_jacobian
        B = dt * control_jacobian
        return A, B

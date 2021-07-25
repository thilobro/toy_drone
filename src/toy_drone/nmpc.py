import numpy as np
import casadi as ca
from toy_drone.utils import rk4


class Nmpc():
    """
    Nmpc Nonlinear Model Predictive Controller with quadratic running and terminal cost.
    Additional quadratic cost on changes in the controls to penalize spikes.
    Ode discretized with rk4 and multiple shooting constraints.
    """

    def __init__(self, model, state_running_cost, state_terminal_cost,
                 control_running_cost, dcontrol_running_cost, control_bounds, dt, N):
        """
        __init__ Constructor

        Args:
            model: model for the system that nmpc is controlling
            state_running_cost: tuning matrix for quadratic state cost
            state_terminal_cost: tuning matrix for quadratic terminal state cost
            control_running_cost: tuning matrix for quadratic control cost
            dcontrol_running_cost: tuning matrix for quadratic cost on control changes
            control_bounds: upper and lower bounds on the controls
            dt: discretization time step
            N: nmpc horizon length
        """
        self._ode = model.get_ode()
        self._state_dim = model.get_state_dim()
        self._controls_dim = model.get_controls_dim()
        self._control_bounds = control_bounds
        self._dt = dt
        self._horizon = N
        self._Q = state_running_cost
        self._R = control_running_cost
        self._Q_term = state_terminal_cost
        self._dR = dcontrol_running_cost
        self._ocp = self._build_ocp()
        self._V_initial = np.zeros((self._state_dim + self._controls_dim)
                                   * self._horizon + self._state_dim)

    def _model_step(self, state, controls):
        """
        _model_step Integrate the model forward.

        Args:
            state: symbolic state
            controls: symbolic controls
        """
        return rk4(self._ode, state, controls, self._dt)

    @staticmethod
    def _quadratic_norm(vector, matrix):
        """
        _quadratic_norm Compute the quadratic norm x.T Q x for a vector x and a matrix Q.
        """
        return ca.mtimes(ca.mtimes(vector.T, matrix), vector)

    def _running_cost(self, state, controls, dcontrols, reference):
        """
        _running_cost Compute running cost for a single time step.

        Args:
            state: symbolic state at the time step
            controls: symbolic controls at the time step
            dcontrols: symbolic control difference at the time step
            reference: symbolic referenc at the time step

        Returns:
            running cost term for single time step
        """
        cost = self._quadratic_norm(state - reference, self._Q)\
            + self._quadratic_norm(controls, self._R) + self._quadratic_norm(dcontrols, self._dR)
        return cost

    def _terminal_cost(self, state, reference):
        """
        _terminal_cost Compute terminal cost.

        Args:
            state: symbolic state at last time step
            reference: symbolic reference at last time step

        Returns:
            Terminal cost term
        """
        cost = self._quadratic_norm(state - reference, self._Q_term)

        return cost

    def _build_ocp(self):
        """
        _build_ocp Build up OCP out of casadi expressions and generate numerical solver.
        """
        ocp_variables = []
        ocp_variables += [ca.SX.sym('V0', self._state_dim + self._controls_dim)]
        params = []
        params += [ca.SX.sym('x0', self._state_dim)]
        equality_constraints = []
        # initial state constraint
        equality_constraints += [params[0] - ocp_variables[0][:self._state_dim]]
        cost_function = 0
        for i in range(1, self._horizon):
            # build variable struct
            ocp_variables += [ca.SX.sym(f'V{i}', self._state_dim + self._controls_dim)]

            # build cost function
            params += [ca.SX.sym(f'ref{i}', self._state_dim)]
            cost_function += self._running_cost(ocp_variables[i][:self._state_dim],
                                                ocp_variables[i][self._state_dim:],
                                                ocp_variables[i][self._state_dim:]
                                                - ocp_variables[i-1][self._state_dim:],
                                                params[i])

            # build MS constraints
            equality_constraints += [self._model_step(ocp_variables[i - 1][:self._state_dim],
                                                      ocp_variables[i - 1][self._state_dim:])
                                     - ocp_variables[i][:self._state_dim]]

        # add terminal cost and constraints
        ocp_variables += [ca.SX.sym(f'V{self._horizon}', self._state_dim)]
        equality_constraints += [self._model_step(ocp_variables[self._horizon - 1]
                                                  [:self._state_dim],
                                                  ocp_variables[self._horizon - 1]
                                                  [self._state_dim:])
                                 - ocp_variables[self._horizon][:self._state_dim]]
        params += [ca.SX.sym(f'ref{self._horizon}', self._state_dim)]
        cost_function += self._terminal_cost(ocp_variables[self._horizon][:self._state_dim],
                                             params[self._horizon])

        # build nlp
        nlp = {'x': ca.vertcat(*ocp_variables), 'f': cost_function,
               'g': ca.vertcat(*equality_constraints), 'p': ca.vertcat(*params)}

        # build solver
        solver_options = {'ipopt': {'print_level': 0}, 'print_time': False}
        self._solver = ca.nlpsol('solver', 'ipopt', nlp, solver_options)

    def _solve_ocp(self, state, reference):
        """
        _solve_ocp Solve the OCP for a set of parameters.

        Args:
            state: current state that is used as initial constraint for the OCP
            reference: reference of length of the horizon that should be tracked

        Returns:
            Optimal trajectory in state and controls
        """
        lbx = [-ca.inf] * self._state_dim + [self._control_bounds[0]] * self._controls_dim
        lbx = lbx * (self._horizon) + [-ca.inf] * self._state_dim
        ubx = [ca.inf] * self._state_dim + [self._control_bounds[1]] * self._controls_dim
        ubx = ubx * (self._horizon) + [ca.inf] * self._state_dim
        solution = self._solver(x0=self._V_initial, p=np.hstack((state, reference)), ubg=0, lbg=0,
                                ubx=ubx, lbx=lbx)
        return solution['x']

    def _extract_initial_guess(self, optimal_trajectory):
        """
        _extract_initial_guess Extract the initial guess for the next iteration by shifting the
        optimal trajectory to the left by one time step and duplicating the last time step.
        Since we don't have controls for the last time step, we fill the up with zeros.

        Args:
            optimal_trajectory: Optimal state and controls trajectory

        Returns:
            Initial guess for the next iteration
        """
        initial_guess = optimal_trajectory[self._state_dim + self._controls_dim:]
        initial_guess = ca.vertcat(initial_guess, [0] * self._controls_dim)
        initial_guess = ca.vertcat(initial_guess, optimal_trajectory[-(self._state_dim):])
        return initial_guess

    def compute_control(self, state, reference):
        """
        compute_control Compute an nmpc control output. Solve the OCP and returns the first control
        of the solution. The rest is used to compute the new initial guess for the next iteration.

        Args:
            state: state at which the control should be computed
            reference: reference for the horizon

        Returns:
            Optimal nmpc control
        """
        # solve OCP
        optimal_trajectory = self._solve_ocp(state, reference)
        # extract V_initial for next iteration
        self._V_initial = self._extract_initial_guess(optimal_trajectory)
        # extract first controls and apply
        optimal_controls = optimal_trajectory[self._state_dim:
                                              self._state_dim + self._controls_dim]
        return optimal_controls.full().squeeze()

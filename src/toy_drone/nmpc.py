import numpy as np
import casadi as ca
from toy_drone.utils import rk4


class Nmpc():
    # TODO: docstrings
    def __init__(self, model, state_running_cost, state_terminal_cost,
                 control_running_cost, dcontrol_running_cost, control_bounds, dt, N):
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
        self._ocp = self.build_ocp()
        self._V_initial = np.zeros((self._state_dim + self._controls_dim)
                                   * self._horizon + self._state_dim)

    def model_step(self, state, controls):
        return rk4(self._ode, state, controls, self._dt)

    @staticmethod
    def quadratic_norm(vector, matrix):
        return ca.mtimes(ca.mtimes(vector.T, matrix), vector)

    def running_cost(self, state, controls, dcontrols, reference):
        cost = self.quadratic_norm(state - reference, self._Q)\
            + self.quadratic_norm(controls, self._R) + self.quadratic_norm(dcontrols, self._dR)
        return cost

    def terminal_cost(self, state, reference):
        cost = self.quadratic_norm(state - reference, self._Q_term)

        return cost

    def build_ocp(self):
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
            cost_function += self.running_cost(ocp_variables[i][:self._state_dim],
                                               ocp_variables[i][self._state_dim:],
                                               ocp_variables[i][self._state_dim:]
                                               - ocp_variables[i-1][self._state_dim:],
                                               params[i])

            # build MS constraints
            equality_constraints += [self.model_step(ocp_variables[i - 1][:self._state_dim],
                                                     ocp_variables[i - 1][self._state_dim:])
                                     - ocp_variables[i][:self._state_dim]]

        # add terminal cost and constraints
        ocp_variables += [ca.SX.sym(f'V{self._horizon}', self._state_dim)]
        equality_constraints += [self.model_step(ocp_variables[self._horizon - 1][:self._state_dim],
                                                 ocp_variables[self._horizon - 1][self._state_dim:])
                                 - ocp_variables[self._horizon][:self._state_dim]]
        params += [ca.SX.sym(f'ref{self._horizon}', self._state_dim)]
        cost_function += self.terminal_cost(ocp_variables[self._horizon][:self._state_dim],
                                            params[self._horizon])

        # build nlp
        nlp = {'x': ca.vertcat(*ocp_variables), 'f': cost_function,
               'g': ca.vertcat(*equality_constraints), 'p': ca.vertcat(*params)}

        # build solver
        solver_options = {'ipopt': {'print_level': 0}, 'print_time': False}
        self._solver = ca.nlpsol('solver', 'ipopt', nlp, solver_options)

    def compute_control(self, state, reference):
        # solve OCP
        lbx = [-ca.inf] * self._state_dim + [self._control_bounds[0]] * self._controls_dim
        lbx = lbx * (self._horizon) + [-ca.inf] * self._state_dim
        ubx = [ca.inf] * self._state_dim + [self._control_bounds[1]] * self._controls_dim
        ubx = ubx * (self._horizon) + [ca.inf] * self._state_dim
        optimal_trajectory = self._solver(x0=self._V_initial,
                                          p=np.hstack((state, reference)), ubg=0, lbg=0,
                                          ubx=ubx, lbx=lbx)
        # extract V_initial for next iteration
        new_initial_guess = optimal_trajectory['x'][self._state_dim + self._controls_dim:]
        new_initial_guess = ca.vertcat(new_initial_guess, [0] * self._controls_dim)
        new_initial_guess = ca.vertcat(new_initial_guess,
                                       optimal_trajectory['x'][-(self._state_dim):])
        self._V_initial = new_initial_guess
        # extract first controls and apply
        optimal_controls = optimal_trajectory['x'][self._state_dim:
                                                   self._state_dim + self._controls_dim]
        return optimal_controls.full().squeeze()

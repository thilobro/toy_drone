import numpy as np
import casadi as ca
from toy_drone.utils import rk4


class Nmpc():
    # TODO: docstrings
    def __init__(self, model, dt, N):
        self._ode = model.get_ode()
        self._state_dim = model.get_state_dim()
        self._controls_dim = model.get_controls_dim()
        self._dt = dt
        self._horizon = N
        self._ocp = self.build_ocp()
        self._V_initial = np.zeros((self._state_dim + self._controls_dim) * self._horizon + self._state_dim)

    def model_step(self, state, controls):
        return rk4(self._ode, state, controls, self._dt)

    @staticmethod
    def running_cost(state, controls, dcontrols, reference):
        # TODO: tuning should come frome outside
        cost = ca.dot(state[:2] - reference[:2], state[:2] - reference[:2]) * 1\
            + ca.dot(state[2:4] - reference[2:4], state[2:4] - reference[2:4]) * 0\
            + ca.dot(state[4] - reference[4], state[4] - reference[4]) * 0\
            + ca.dot(state[5] - reference[5], state[5] - reference[5]) * 1e-3\
            + ca.dot(controls, controls) * 1e-6\
            + ca.dot(dcontrols, dcontrols) * 1e-2
        return cost

    @staticmethod
    def terminal_cost(state, reference):
        # TODO: tuning should come frome outside
        cost = ca.dot(state[:2] - reference[:2], state[:2] - reference[:2]) * 1\
            + ca.dot(state[2:4] - reference[2:4], state[2:4] - reference[2:4]) * 1\
            + ca.dot(state[4] - reference[4], state[4] - reference[4]) * 1\
            + ca.dot(state[5] - reference[5], state[5] - reference[5]) * 1\

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
                                               ocp_variables[i][self._state_dim:] - ocp_variables[i-1][self._state_dim:],
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
        self._solver = ca.nlpsol('solver', 'ipopt', nlp)

    def compute_control(self, state, reference):
        # solve OCP
        lbx = [-ca.inf] * 6 + [0, 0]
        lbx = lbx * (self._horizon) + [-ca.inf] * 6
        ubx = [ca.inf] * 8 * (self._horizon) + [ca.inf] * 6
        optimal_trajectory = self._solver(x0=self._V_initial,
                                          p=np.hstack((state, reference)), ubg=0, lbg=0,
                                          ubx=ubx, lbx=lbx)
        # extract V_initial for next iteration
        new_initial_guess = optimal_trajectory['x'][self._state_dim + self._controls_dim:]
        new_initial_guess = ca.vertcat(new_initial_guess, [0, 0])
        new_initial_guess = ca.vertcat(new_initial_guess, optimal_trajectory['x'][-(self._state_dim):])
        self._V_initial = new_initial_guess
        # extract first controls and apply
        optimal_controls = optimal_trajectory['x'][self._state_dim:self._state_dim + self._controls_dim]
        return optimal_controls

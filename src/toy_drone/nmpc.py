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
        self._V_initial = np.zeros((self._state_dim + self._controls_dim) * (self._horizon - 1))

    def model_step(self, state, controls):
        return rk4(self._ode, state, controls, self._dt)

    @staticmethod
    def running_cost(state, controls, dcontrols, reference):
        controls = ca.dot(state[:2] - reference[:2], state[:2] - reference[:2]) * 1\
            + ca.dot(state[2:4] - reference[2:4], state[2:4] - reference[2:4]) * 1e-1\
            + ca.dot(state[4] - reference[4], state[4] - reference[4]) * 0\
            + ca.dot(state[5] - reference[5], state[5] - reference[5]) * 1e-4\
            + ca.dot(controls, controls) * 1e-6\
            + ca.dot(dcontrols, dcontrols) * 1e-2
        return controls

    def build_ocp(self):
        ocp_variables = []
        ocp_variables += [ca.SX.sym('V0', self._state_dim + self._controls_dim)]
        params = []
        params += [ca.SX.sym('x0', self._state_dim)]
        equality_constraints = []
        # initial state constraint
        equality_constraints += [params[0] - ocp_variables[0][:self._state_dim]]
        cost_function = 0
        for i in range(1, self._horizon - 1):
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

            # build path constraints if we have some
            # TODO: add constraints on inputs

        # build nlp
        nlp = {'x': ca.vertcat(*ocp_variables), 'f': cost_function,
               'g': ca.vertcat(*equality_constraints), 'p': ca.vertcat(*params)}

        # build solver
        self._solver = ca.nlpsol('solver', 'ipopt', nlp)

    def compute_control(self, state, reference):
        # solve OCP
        lbx = [-ca.inf] * 6 + [0, 0]
        lbx = lbx * (self._horizon - 1)
        ubx = [ca.inf] * 8 * (self._horizon - 1)
        optimal_trajectory = self._solver(x0=self._V_initial,
                                          p=np.hstack((state, reference)), ubg=0, lbg=0,
                                          ubx=ubx, lbx=lbx)
        # extract V_initial for next iteration
        new_initial_guess = optimal_trajectory['x'][self._state_dim + self._controls_dim:((self._state_dim + self._controls_dim)*(self._horizon - 1))]
        new_initial_guess = ca.vertcat(new_initial_guess, optimal_trajectory['x'][(self._state_dim + self._controls_dim) * (self._horizon - 2):(self._state_dim + self._controls_dim) * (self._horizon - 1)])
        self._V_initial = new_initial_guess
        optimal_controls = optimal_trajectory['x'][self._state_dim:self._state_dim + self._controls_dim]
        # print(optimal_trajectory['x'])
        # extract first controls
        return optimal_controls

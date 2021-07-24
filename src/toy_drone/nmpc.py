import numpy as np
import casadi as ca
from toy_drone.utils import rk4


class Nmpc():
    # TODO: docstrings
    def __init__(self, model):
        self._ode = model.get_ode()
        self._state_dim = model.get_state_dim()
        self._controls_dim = model.get_controls_dim()
        self._dt = 1e-3
        self._horizon = 5
        self._ocp = self.build_ocp()

    def model_step(self, state, controls):
        return rk4(self._ode, state, controls, 1e-3)

    @staticmethod
    def running_cost(state, reference):
        return ca.dot(state - reference, state - reference)

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
            cost_function += self.running_cost(ocp_variables[i][:self._state_dim], params[i])

            # build MS constraints
            equality_constraints += [self.model_step(ocp_variables[i - 1][:self._state_dim],
                                                     ocp_variables[i - 1][self._state_dim:])
                                     - ocp_variables[i][:self._state_dim]]

            # build path constraints if we have some

        # build nlp
        nlp = {'x': ca.vertcat(*ocp_variables), 'f': cost_function,
               'g': ca.vertcat(*equality_constraints), 'p': ca.vertcat(*params)}

        # build solver
        self._solver = ca.nlpsol('solver', 'ipopt', nlp)

    def compute_control(self, state, reference):
        # solve OCP
        optimal_trajectory = self._solver(
            x0=np.zeros((self._state_dim + self._controls_dim) * (self._horizon - 1)),
            p=np.hstack((state, reference)))
        # extract first controls
        return optimal_trajectory['x'][self._state_dim:self._controls_dim]

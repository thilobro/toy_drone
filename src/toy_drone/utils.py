def rk4(ode, x0, u, dt, N=10):
    for i in range(N):
        k1 = dt * ode(x0, u)
        k2 = dt * ode(x0 + k1/2.0, u)
        k3 = dt * ode(x0 + k2/2.0, u)
        k4 = dt * ode(x0 + k3, u)
        x_0 = x0 + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return x_0

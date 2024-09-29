import numpy as np
import matplotlib.pyplot as plt


# Derivative function for Lotka-Volterra competition equations
def dNdt_comp(t, N, a=1, b=2, c=1, d=3):
    dN1dt = a * N[0] * (1 - N[0]) - b * N[0] * N[1]
    dN2dt = c * N[1] * (1 - N[1]) - d * N[1] * N[0]
    return np.array([dN1dt, dN2dt])

# Predator-prey model
def dNdt_predator_prey(t, N, a=1, b=2, c=1, d=3):
    dN1dt = a * N[0] - b * N[0] * N[1]
    dN2dt = -c * N[1] + d * N[0] * N[1]
    return np.array([dN1dt, dN2dt])

# Euler method solver
def euler_solve(func, N1_init=0.5, N2_init=0.5, dT=0.1, t_final=100, a=1, b=2, c=1, d=3):
    time = np.arange(0, t_final, dT)
    N1 = np.zeros(len(time))
    N2 = np.zeros(len(time))
    N1[0], N2[0] = N1_init, N2_init
    
    for i in range(1, len(time)):
        dN = func(time[i-1], [N1[i-1], N2[i-1]], a, b, c, d)
        N1[i] = N1[i-1] + dN[0] * dT
        N2[i] = N2[i-1] + dN[1] * dT
    
    return time, N1, N2

# Runge-Kutta method using DOP853
def solve_rk8(func, N1_init=0.5, N2_init=0.5, t_final=100, a=1, b=2, c=1, d=3):
    from scipy.integrate import solve_ivp
    # Wrapper for scipy solve_ivp
    result = solve_ivp(func, [0, t_final], [N1_init, N2_init], args=(a, b, c, d), method='DOP853', max_step=10)
    return result.t, result.y[0, :], result.y[1, :]

# Solving the competition model using Euler method
time, N1, N2 = euler_solve(dNdt_comp, N1_init=0.3, N2_init=0.6, dT=1.0, t_final=100, a=1, b=2, c=1, d=3)

# Solving the competition model using RK8
time_rk8, N1_rk8, N2_rk8 = solve_rk8(dNdt_comp, N1_init=0.3, N2_init=0.6, t_final=100, a=1, b=2, c=1, d=3)

# Plot results for competition model
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.plot(time, N1, label="Species 1 (Euler - Competition)")
plt.plot(time, N2, label="Species 2 (Euler - Competition)")
plt.plot(time_rk8, N1_rk8, linestyle='--', label="Species 1 (RK8 - Competition)")
plt.plot(time_rk8, N2_rk8, linestyle='--', label="Species 2 (RK8 - Competition)")
plt.xlabel('Time (years)')
plt.ylabel('Population')
plt.title("Competition Model")
plt.legend()

# Solving the predator-prey model using Euler method
time, N1, N2 = euler_solve(dNdt_predator_prey, N1_init=0.3, N2_init=0.6, dT=0.05, t_final=100, a=1, b=2, c=1, d=3)

# Solving the predator-prey model using RK8
time_rk8, N1_rk8, N2_rk8 = solve_rk8(dNdt_predator_prey, N1_init=0.3, N2_init=0.6, t_final=100, a=1, b=2, c=1, d=3)


# Plot results for predator-prey model
plt.subplot(1, 2, 2)
plt.plot(time, N1, label="Prey (Euler - Predator-Prey)")
plt.plot(time, N2, label="Predator (Euler - Predator-Prey)")
plt.plot(time_rk8, N1_rk8, linestyle='--', label="Prey (RK8 - Predator-Prey)")
plt.plot(time_rk8, N2_rk8, linestyle='--', label="Predator (RK8 - Predator-Prey)")
plt.xlabel('Time (years)')
plt.ylabel('Population')
plt.title("Predator-Prey Model")
plt.legend()

plt.tight_layout()
plt.show()

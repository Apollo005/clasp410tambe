#!/usr/bin/env python3

'''
This file contains tools and scripts for completing Lab 2 for CLaSP 410. 
We explore the questions found below:

-How does the performance of the Euler method solver compare to the 8th-order DOP853 method for both sets of equations? 
-How do the initial conditions and coefficient values affect the final result and general behavior of the two species? 
-How do the initial conditions and coefficient values affect the final result and general behavior
of the two species?

In this lab, I will create two ODE solvers. One will use the first-order-accurate
Euler method with a constant forward time step. The second will use the Scipy implementation
of the Dormand-Prince embedded 8th-order Runge-Kutta method with adaptive stepsize, called
DOP853. This method is an ”industry standard” ODE solver that is efficient, high-order, and
accurate. These solvers will be used to solve the Lotka-Volterra competition and predator-prey models
and analyze the growth/decline of the species (in broad terms)

To reproduce the plots shown in the lab report, simply run the code and you will get the same figures displayed
in the report...

Note for the phase diagrams and population vs time plots for the predator-prey equation modeling, I produce more
output than in the report so you can "x" out of them or look at them if preferred!
'''

import numpy as np
import matplotlib.pyplot as plt


# Derivative function for Lotka-Volterra competition equations
def dNdt_comp(t, N, a=1, b=2, c=1, d=3):
    '''
    This function calculates the Lotka-Volterra competition equations for
    two species. Given normalized populations, `N1` and `N2`, as well as the
    four coefficients representing population growth and decline,
    calculate the time derivatives dN_1/dt and dN_2/dt and return to the
    caller.
    This function accepts `t`, or time, as an input parameter to be
    compliant with Scipy's ODE solver. However, it is not used in this
    function.
    Parameters
    ----------
    t : float
    The current time (not used here).
    N : two-element list
    The current value of N1 and N2 as a list (e.g., [N1, N2]).
    a, b, c, d : float, defaults=1, 2, 1, 3
    The value of the Lotka-Volterra coefficients.
    Returns
    -------
    dN1dt, dN2dt : floats
    The time derivatives of `N1` and `N2`.
    '''
    dN1dt = a * N[0] * (1 - N[0]) - b * N[0] * N[1]
    dN2dt = c * N[1] * (1 - N[1]) - d * N[1] * N[0]
    return np.array([dN1dt, dN2dt])

# Predator-prey model
def dNdt_predator_prey(t, N, a=1, b=2, c=1, d=3):
    '''
    This function calculates the Lotka-Volterra predator-prey equations for
    two species. Given normalized populations `N1` (prey) and `N2` (predator),
    as well as four coefficients representing the interaction between predator
    and prey populations, calculate the time derivatives dN_1/dt (prey) and
    dN_2/dt (predator) and return them to the caller.

    This function accepts `t`, or time, as an input parameter to be compliant 
    with Scipy's ODE solver. However, it is not explicitly used in this function.

    Parameters
    ----------
    t : float
        The current time (not used here).
    N : two-element list
        The current values of N1 (prey population) and N2 (predator population), 
        passed as a list (e.g., [N1, N2]).
    a, b, c, d : float, defaults=1, 2, 1, 3
    Returns
    -------
    dN1dt, dN2dt : numpy.ndarray
        The time derivatives of `N1` (prey) and `N2` (predator).
    '''
    dN1dt = a * N[0] - b * N[0] * N[1]
    dN2dt = -c * N[1] + d * N[0] * N[1]
    return np.array([dN1dt, dN2dt])

# Euler method solver
def euler_solve(func, N1_init=0.5, N2_init=0.5, dT=0.1, t_final=100, a=1, b=2, c=1, d=3):
    '''
    Solves a system of differential equations using the Euler method.
    This function numerically solves a system of ordinary differential equations (ODEs) for two variables N1 and N2 using the Euler method. 
    The system is defined by a function `func` which computes the time derivatives of N1 and N2. 
    Initial values and system parameters (a, b, c, d) are provided as inputs.

    Parameters
    ----------
    func : function
        A Python function that takes `time`, a list [`N1`, `N2`], and parameters `a`, `b`, `c`, and `d`, 
        and returns a list of the time derivatives [dN1/dt, dN2/dt].
    N1_init : float, optional
        The initial value for the population (or quantity) of N1. Default is 0.5.
    N2_init : float, optional
        The initial value for the population (or quantity) of N2. Default is 0.5.
    dT : float, optional
        The time step for the Euler method. Default is 0.1.
    t_final : float, optional
        The final time until which the system is solved. Default is 100.
    a, b, c, d : float, defaults=1, 2, 1, 3
    Returns
    -------
    time : numpy.ndarray
        An array of time values from 0 to `t_final` with steps of `dT`.
    N1 : numpy.ndarray
        The computed values of N1 at each time step.
    N2 : numpy.ndarray
        The computed values of N2 at each time step.
    '''
    # Configure the initial value problem solver
    time = np.arange(0, t_final, dT)
    N1 = np.zeros(len(time))
    N2 = np.zeros(len(time))
    N1[0], N2[0] = N1_init, N2_init
    # Perform the integration
    for i in range(1, len(time)):
        dN = func(time[i-1], [N1[i-1], N2[i-1]], a, b, c, d)
        N1[i] = N1[i-1] + dN[0] * dT
        N2[i] = N2[i-1] + dN[1] * dT
    # Return values to caller.
    return time, N1, N2

# Runge-Kutta method using DOP853
def solve_rk8(func, N1_init=0.5, N2_init=0.5, t_final=100, a=1, b=2, c=1, d=3):
    '''
    Solve the Lotka-Volterra competition and predator/prey equations using
    Scipy's ODE class and the adaptive step 8th order solver.
    Parameters
    ----------
    func : function
    A python function that takes `time`, [`N1`, `N2`] as inputs and
    returns the time derivative of N1 and N2.
    N1_init, N2_init : float
    Initial conditions for `N1` and `N2`, ranging from (0,1]
    dT : float, default=10
    Largest timestep allowed in years.
    t_final : float, default=100
    Integrate until this value is reached, in years.
    a, b, c, d : float, default=1, 2, 1, 3
    Lotka-Volterra coefficient values
    Returns
    -------
    time : Numpy array
    Time elapsed in years.
    N1, N2 : Numpy arrays
    Normalized population density solutions.
    '''
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

# Function to compare the Euler method with different time steps for competition model
def compare_euler_rk8_comp(func=dNdt_comp, N1_init=0.3, N2_init=0.6, t_final=100, a=1, b=2, c=1, d=3):
    '''
    Compares the Euler method with various time steps against the RK8 (DOP853) method 
    for solving the competition model of Lotka-Volterra equations.

    Parameters:
    func: Callable, the derivative function to solve (default: dNdt_comp).
    N1_init: float, initial population of species 1.
    N2_init: float, initial population of species 2.
    t_final: float, the final time value for the simulation.
    a, b, c, d: floats, model coefficients.

    Returns:
    None
    '''
    time_steps = [1.0, 0.5, 0.1]  # Different Euler time steps to try
    plt.figure(figsize=(12, 8))

    # Loop over each time step
    for dT in time_steps:
        time, N1, N2 = euler_solve(func, N1_init, N2_init, dT=dT, t_final=t_final, a=a, b=b, c=c, d=d)
        plt.plot(time, N1, label=f"Species 1 (Euler dT={dT})")
        plt.plot(time, N2, label=f"Species 2 (Euler dT={dT})")

    # RK8 Method (DOP853)
    time_rk8, N1_rk8, N2_rk8 = solve_rk8(func, N1_init, N2_init, t_final=t_final, a=a, b=b, c=c, d=d)
    plt.plot(time_rk8, N1_rk8, linestyle='--', label="Species 1 (RK8)", color='black')
    plt.plot(time_rk8, N2_rk8, linestyle='--', label="Species 2 (RK8)", color='gray')

    plt.xlabel('Time (years)')
    plt.ylabel('Normalized Population')
    plt.title("Comparison of Euler and RK8 Methods for Competition Model")
    plt.legend()
    plt.show()

# Function to compare the Euler method with different time steps for predator-prey model
def compare_euler_rk8_preprey(func = dNdt_predator_prey, N1_init=0.3, N2_init=0.6, t_final=100, a=1, b=2, c=1, d=3):
    '''
    Compares the Euler method with various time steps against the RK8 (DOP853) method 
    for solving the predator-prey model of Lotka-Volterra equations.

    Parameters:
    func: Callable, the derivative function to solve (default: dNdt_predator_prey).
    N1_init: float, initial population of prey (species 1).
    N2_init: float, initial population of predator (species 2).
    t_final: float, the final time value for the simulation.
    a, b, c, d: floats, model coefficients.

    Returns:
    None
    '''
    time_steps = [0.065, 0.05, 0.01]  # Different Euler time steps to try
    plt.figure(figsize=(12, 8))

    # Loop over each time step
    for dT in time_steps:
        time, N1, N2 = euler_solve(func, N1_init, N2_init, dT=dT, t_final=t_final, a=a, b=b, c=c, d=d)
        plt.plot(time, N1, label=f"Species 1 (Euler dT={dT})")
        plt.plot(time, N2, label=f"Species 2 (Euler dT={dT})")

        # RK8 Method (DOP853)
        time_rk8, N1_rk8, N2_rk8 = solve_rk8(func, N1_init, N2_init, t_final=t_final, a=a, b=b, c=c, d=d)
        plt.plot(time_rk8, N1_rk8, linestyle='--', label="Species 1 (RK8)", color='black')
        plt.plot(time_rk8, N2_rk8, linestyle='--', label="Species 2 (RK8)", color='gray')

        plt.xlabel('Time (years)')
        plt.ylabel('Normalized Population')
        plt.title(f"Comparison of Euler and RK8 Methods for Predator-Prey Model (dT = {dT})")
        plt.legend()
        plt.show()

# Run comparison for the Competition model
compare_euler_rk8_comp()

# Run comparison for the Predator-Prey model
compare_euler_rk8_preprey()

#---------------------------------------------------------

time, N1, N2 = euler_solve(dNdt_comp, N1_init=0.4, N2_init=0.4, dT=0.1, t_final=100, a=4, b=3, c=2, d=1)
time_rk8, N1_rk8, N2_rk8 = solve_rk8(dNdt_comp, N1_init=0.4, N2_init=0.4, t_final=100, a=4, b=3, c=2, d=1)

# Plot results for competition model
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.plot(time, N1, label="Species 1 (Euler - Competition)")
plt.plot(time, N2, label="Species 2 (Euler - Competition)")
plt.plot(time_rk8, N1_rk8, linestyle='--', label="Species 1 (RK8 - Competition)")
plt.plot(time_rk8, N2_rk8, linestyle='--', label="Species 2 (RK8 - Competition)")
plt.xlabel('Time (years)')
plt.ylabel('Population')
plt.title("Competition Model Equilibrium")
plt.legend()
plt.show()

#---------------------------------------------------------

# Plot populations vs time
def plot_population_vs_time(time, N1, N2, title):
    '''
    Plots the populations of two species (prey and predator) versus time.

    Parameters:
    time: array, time values for the simulation.
    N1: array, population of prey (species 1).
    N2: array, population of predator (species 2).
    title: str, title of the plot.

    Returns:
    None
    '''
    plt.figure(figsize=(10, 5))
    plt.plot(time, N1, label="Prey", color='blue')
    plt.plot(time, N2, label="Predator", color='red')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title(f'{title}: Population vs Time')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot phase diagram (prey vs predator)
def plot_phase_diagram(N1, N2, title):
    '''
    Plots the phase diagram of prey population versus predator population.

    Parameters:
    N1: array, population of prey (species 1).
    N2: array, population of predator (species 2).
    title: str, title of the plot.

    Returns:
    None
    '''
    plt.figure(figsize=(5, 5))
    plt.plot(N1, N2, color='green')
    plt.xlabel('Prey Population')
    plt.ylabel('Predator Population')
    plt.title(f'{title}: Phase Diagram')
    plt.grid(True)
    plt.show()

# Function to simulate and plot for varying parameters
def simulate_predator_prey(initial_conditions, a, b, c, d, t_final=100):
    '''
    Simulates the predator-prey model using the RK8 (DOP853) method and plots the results 
    (population vs time and phase diagram).

    Parameters:
    initial_conditions: tuple, initial populations for prey and predator (N1_init, N2_init).
    a, b, c, d: floats, model coefficients for prey and predator interaction.
    t_final: float, the final time value for the simulation.

    Returns:
    None
    '''
    N1_init, N2_init = initial_conditions
    time, N1, N2 = solve_rk8(dNdt_predator_prey, N1_init=N1_init, N2_init=N2_init, t_final=t_final, a=a, b=b, c=c, d=d)

    # Plot population vs time
    plot_population_vs_time(time, N1, N2, title=f'Initial Conditions: Prey={N1_init}, Predator={N2_init}, Coefficients a={a}, b={b}, c={c}, d={d}')

    # Plot phase diagram (Prey vs Predator)
    plot_phase_diagram(N1, N2, title=f'Initial Conditions: Prey={N1_init}, Predator={N2_init}, Coefficients a={a}, b={b}, c={c}, d={d}')

# Test the predator-prey model with varying initial conditions and coefficients
initial_conditions_list = [
    (0.3, 0.6),  # Example 1: Prey starts lower than predator
    (0.5, 0.5),  # Example 2: Equal initial populations
    (0.7, 0.2),  # Example 3: Prey population higher
]

coefficients_list = [
    (1.0, 0.5, 0.5, 0.3),  # Example 1: Moderate interaction
    (1.5, 0.8, 0.4, 0.2),  # Example 2: Stronger predator impact
    (0.8, 0.3, 0.7, 0.1),  # Example 3: Higher predator mortality
]

# Run the simulations from the lists defined above with varying intial conditions and coefficients
# This code will provide 
for initial_conditions in initial_conditions_list:
    for coefficients in coefficients_list:
        a, b, c, d = coefficients
        simulate_predator_prey(initial_conditions, a, b, c, d, t_final=100)

-Learn how to implement solvers for Ordinary Differential Equations (ODEs) both from
scratch and using external libraries.
-Compare the performance of low-order and high-order ODE solvers.
-Explore the Lotka-Volterra equations for population growth for two competing species and for a predator-prey pair of species.

In this lab, I will create two ODE solvers. One will use the first-order-accurate
Euler method with a constant forward time step. The second will use the Scipy implementation
of the Dormand-Prince embedded 8th-order Runge-Kutta method with adaptive stepsize, called
DOP853. This method is an ”industry standard” ODE solver that is efficient, high-order, and
accurate
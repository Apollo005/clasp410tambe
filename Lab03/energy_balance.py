#!/usr/bin/env python3
'''
A set of tools and routines for solving the N-layer atmosphere energy
balance problem and perform some useful analysis.

'''

import numpy as np
import matplotlib.pyplot as plt

#  Define some useful constants here.
sigma = 5.67E-8  # Steffan-Boltzman constant.

#####################################

def n_layer_atmos(N, epsilon, S0=1350, albedo=0.33, debug=False):
    '''
    Solve the n-layer atmosphere problem and return temperature at each layer.

    Parameters
    ----------
    N : int
        Set the number of layers.
    epsilon : float, default=1.0
        Set the emissivity of the atmospheric layers.
    albedo : float, default=0.33
        Set the planetary albedo from 0 to 1.
    S0 : float, default=1350
        Set the incoming solar shortwave flux in Watts/m^2.
    debug : boolean, default=False
        Turn on debug output.

    Returns
    -------
    temps : Numpy array of size N+1
        Array of temperatures from the Earth's surface (element 0) through
        each of the N atmospheric layers, from lowest to highest.
    '''

    # Create matrices:
    A = np.zeros([N+1, N+1])
    b = np.zeros(N+1)
    b[0] = -S0/4 * (1-albedo)

    if debug:
        print(f"Populating N+1 x N+1 matrix (N = {N})")

    # Populate our A matrix piece-by-piece.
    for i in range(N+1):
        for j in range(N+1):
            if debug:
                print(f"Calculating point i={i}, j={j}")
            # Diagonal elements are always -2 ('cept at Earth's surface.)
            if i == j:
                A[i, j] = -1*(i > 0) - 1
                # print(f"Result: A[i,j]={A[i,j]}")
            else:
                # This is the pattern we solved for in class!
                m = np.abs(j-i) - 1
                A[i, j] = epsilon * (1-epsilon)**m
    # At Earth's surface, epsilon =1, breaking our pattern.
    # Divide by epsilon along surface to get correct results.
    A[0, 1:] /= epsilon

    # Verify our A matrix.
    if debug:
        print(A)

    # Get the inverse of our A matrix.
    Ainv = np.linalg.inv(A)

    # Multiply Ainv by b to get Fluxes.
    fluxes = np.matmul(Ainv, b)

    # Convert fluxes to temperatures.
    # Fluxes for all atmospheric layers
    temps = (fluxes/epsilon/sigma)**0.25
    temps[0] = (fluxes[0]/sigma)**0.25  # Flux at ground: epsilon=1.

    return temps

print(n_layer_atmos(3, 0.5))

#####################################

# First experiment: Vary emissivity and measure surface temperature with 1 atmospheric layer
N = 1  # Single-layer atmosphere
epsilon_values = np.linspace(0.1, 1.0, 50)  # Vary emissivity from 0.1 to 1.0
surface_temps_1_layer = [n_layer_atmos(N, epsilon)[0] for epsilon in epsilon_values]

# Plot surface temperature vs. emissivity for 1-layer atmosphere
plt.plot(epsilon_values, surface_temps_1_layer, label="1-layer atmosphere")
plt.xlabel("Emissivity")
plt.ylabel("Surface Temperature (K)")
plt.title("Surface Temperature vs Emissivity (1-Layer Atmosphere)")
plt.grid(True)
plt.show()

# Estimate emissivity for Earth when surface temperature is 288K
epsilon_earth_guess = epsilon_values[np.argmin(np.abs(np.array(surface_temps_1_layer) - 288))]
print(f"Estimated Earth emissivity for 288K: {epsilon_earth_guess}")

# Second experiment: Vary number of layers with emissivity of 0.255
epsilon = 0.255  # Effective emissivity of Earth's atmosphere
layer_counts = np.arange(1, 10)  # Vary from 1 to 10 layers
surface_temps_vary_layers = [n_layer_atmos(N, epsilon)[0] for N in layer_counts]

# Plot altitude (number of layers) vs surface temperature
plt.plot(layer_counts, surface_temps_vary_layers, label="Varying Layers", color = 'red')
plt.xlabel("Number of Atmospheric Layers")
plt.ylabel("Surface Temperature (K)")
plt.title("Surface Temperature vs Number of Layers (Emissivity = 0.255)")
plt.grid(True)
plt.show()

# Find number of layers to reach ~288K
layers_needed = layer_counts[np.argmin(np.abs(np.array(surface_temps_vary_layers) - 288))]
print(f"Number of layers needed for 288K surface temperature: {layers_needed}")


# Define Venus parameters
T_venus = 700  # K
S0_venus = 2600  # W/m²
epsilon_venus = 1.0  # Perfectly absorbing layers
albedo_venus = 0.33  # Venus has a high albedo due to cloud cover

# Iterate over possible N values
N_vals = range(1, 50)  # We'll try up to 50 layers initially
temps_surface = []

# Compute surface temperature for each N
for N in N_vals:
    temps = n_layer_atmos(N, epsilon_venus, S0=S0_venus, albedo=albedo_venus)
    temps_surface.append(temps[0])  # Record surface temperature

# Find the N closest to Venus surface temperature of 700 K
N_best = N_vals[np.argmin(np.abs(np.array(temps_surface) - T_venus))]

print(f"The best estimate for the number of atmospheric layers on Venus is N = {N_best}")
print(f"With a corresponding surface temperature of {temps_surface[N_best - 1]:.2f} K")

# Plot N-layers vs Surface Temperature
plt.figure(figsize=(8, 6))
plt.plot(N_vals, temps_surface, marker='o', linestyle='-', color='b', label='Surface Temperature')
plt.axhline(y=T_venus, color='r', linestyle='--', label=f'Observed Temp: {T_venus}K')
plt.axvline(x=N_best, color='g', linestyle='--', label=f'N = {N_best}')
plt.title('N-layers vs Surface Temperature on Venus')
plt.xlabel('Number of Atmospheric Layers (N)')
plt.ylabel('Surface Temperature (K)')
plt.legend()
plt.grid(True)
plt.show()

#################################

def n_layer_atmos_nuclear_winter(N, epsilon, S0=1350, albedo=0.33, debug=False):
    '''
    Solve the n-layer atmosphere problem under a nuclear winter scenario and 
    return temperature at each layer.

    Parameters
    ----------
    N : int
        Set the number of layers.
    epsilon : float, default=1.0
        Set the emissivity of the atmospheric layers.
    albedo : float, default=0.33
        Set the planetary albedo from 0 to 1.
    S0 : float, default=1350
        Set the incoming solar shortwave flux in Watts/m^2.
    debug : boolean, default=False
        Turn on debug output.

    Returns
    -------
    temps : Numpy array of size N+1
        Array of temperatures from the Earth's surface (element 0) through
        each of the N atmospheric layers, from lowest to highest.
    '''

    # Create matrices:
    A = np.zeros([N+1, N+1])
    b = np.zeros(N+1)

    # In a nuclear winter scenario, all solar radiation is absorbed by the top layer.
    b[N] = -S0 / 4 * (1 - albedo)

    if debug:
        print(f"Populating N+1 x N+1 matrix (N = {N})")

    # Populate our A matrix piece-by-piece.
    for i in range(N+1):
        for j in range(N+1):
            if debug:
                print(f"Calculating point i={i}, j={j}")
            # Diagonal elements are always -2 ('cept at Earth's surface.)
            if i == j:
                A[i, j] = -1*(i > 0) - 1
            else:
                # This is the pattern we solved for in class!
                m = np.abs(j-i) - 1
                A[i, j] = epsilon * (1-epsilon)**m
    # At Earth's surface, epsilon =1, breaking our pattern.
    # Divide by epsilon along surface to get correct results.
    A[0, 1:] /= epsilon

    # Verify our A matrix.
    if debug:
        print(A)

    # Get the inverse of our A matrix.
    Ainv = np.linalg.inv(A)

    # Multiply Ainv by b to get Fluxes.
    fluxes = np.matmul(Ainv, b)

    # Convert fluxes to temperatures.
    # Fluxes for all atmospheric layers
    temps = (fluxes / epsilon / sigma)**0.25
    temps[0] = (fluxes[0] / sigma)**0.25  # Flux at ground: epsilon=1.

    return temps

# Simulation parameters
N = 5
epsilon = 0.5
S0 = 1350

# Calculate temperatures
temps = n_layer_atmos_nuclear_winter(N, epsilon, S0)

# Generate altitude profile
altitudes = np.arange(N+1)

# Plot altitude vs. temperature
plt.figure(figsize=(8,6))
plt.plot(temps, altitudes, marker='o', linestyle='-', color='b')
plt.xlabel('Temperature (K)')
plt.ylabel('Altitude (Layer)')
plt.title('Altitude vs. Temperature under Nuclear Winter Scenario')
plt.grid(True)
plt.show()
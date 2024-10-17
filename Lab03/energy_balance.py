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

# Second experiment: Vary emissivity and measure surface temperature with 1 atmospheric layer
N = 1  # Single-layer atmosphere for this experiment
epsilon_values = np.linspace(0.1, 1.0, 50)  # Vary emissivity from 0.1 to 1.0 in 50 steps
# For each emissivity, calculate the surface temperature and store the result
surface_temps_1_layer = [n_layer_atmos(N, epsilon)[0] for epsilon in epsilon_values]

# Plot surface temperature vs. emissivity for the 1-layer atmosphere
plt.plot(epsilon_values, surface_temps_1_layer, label="1-layer atmosphere")
plt.xlabel("Emissivity")
plt.ylabel("Surface Temperature (K)")
plt.title("Surface Temperature vs Emissivity (1-Layer Atmosphere)")
plt.grid(True)
plt.show()

# Estimate emissivity for Earth when surface temperature is 288K
epsilon_earth_guess = epsilon_values[np.argmin(np.abs(np.array(surface_temps_1_layer) - 288))]
# Print the estimated emissivity where the surface temperature is closest to 288K (Earth-like conditions)
print(f"Estimated Earth emissivity for 288K: {epsilon_earth_guess}")

# Third experiment: Vary number of layers with emissivity fixed at 0.255
epsilon = 0.255  # Effective emissivity of Earth's atmosphere
layer_counts = np.arange(1, 10)  # Vary number of layers from 1 to 10
# For each number of layers, calculate the surface temperature
surface_temps_vary_layers = [n_layer_atmos(N, epsilon)[0] for N in layer_counts]

# Plot the number of atmospheric layers vs surface temperature
plt.plot(layer_counts, surface_temps_vary_layers, label="Varying Layers", color='red')
plt.xlabel("Number of Atmospheric Layers")
plt.ylabel("Surface Temperature (K)")
plt.title("Surface Temperature vs Number of Layers (Emissivity = 0.255)")
plt.grid(True)
plt.show()

# Find the number of layers needed to reach a surface temperature close to 288K
layers_needed = layer_counts[np.argmin(np.abs(np.array(surface_temps_vary_layers) - 288))]
# Print the number of layers required for a surface temperature near 288K
print(f"Number of layers needed for 288K surface temperature: {layers_needed}")

#####################################

# Define Venus parameters
T_venus = 700  # K (Venus surface temperature)
S0_venus = 2600  # W/m² (Solar constant for Venus)
epsilon_venus = 1.0  # Perfect emissivity (Venus's thick atmosphere absorbs all radiation)
albedo_venus = 0.70  # Venus's albedo (reflectivity), due to its thick cloud cover

# Try various numbers of layers (up to 50) to estimate Venus's atmosphere properties
N_vals = range(1, 80)  # Number of atmospheric layers from 1 to 50
temps_surface = []  # List to store the surface temperatures

# Compute surface temperature for each N value
for N in N_vals:
    temps = n_layer_atmos(N, epsilon_venus, S0=S0_venus, albedo=albedo_venus)  # Calculate temperatures
    temps_surface.append(temps[0])  # Record the surface temperature (first element in temps array)

# Find the N value where the surface temperature is closest to Venus's observed temperature (700K)
N_best = N_vals[np.argmin(np.abs(np.array(temps_surface) - T_venus))]

# Print the number of layers (N_best) and the corresponding surface temperature
print(f"The best estimate for the number of atmospheric layers on Venus is N = {N_best}")
print(f"With a corresponding surface temperature of {temps_surface[N_best - 1]:.2f} K")

# Plot the number of layers vs surface temperature for Venus
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

# Fourth experiment: Define the nuclear_winter_atmos function
def nuclear_winter_atmos(N, epsilon, S0=1350, albedo=0.33, debug=False):
    '''
    Solve the n-layer atmosphere problem for a nuclear winter scenario.

    Parameters
    ----------
    N : int
        Number of atmospheric layers.
    epsilon : float
        Emissivity of the atmosphere.
    S0 : float, optional
        Incoming solar shortwave flux (default is 1350 W/m²).
    albedo : float, optional
        Planetary albedo (default is 0.33).
    debug : bool, optional
        Whether to print debug information (default is False).

    Returns
    -------
    temps : numpy array
        Array of temperatures for each layer including the surface.
    '''

    # Initialize matrices A and b
    A = np.zeros([N+1, N+1])  # Coefficient matrix A (N+1 x N+1 for N layers + surface)
    b = np.zeros(N+1)  # Right-hand side vector b
    # In nuclear winter, all solar flux is absorbed by the top layer, so set the flux in the top layer.
    b[-1] = -S0/4 * (1-albedo)

    # Fill the matrix A based on the physical relationships between fluxes and temperatures
    for i in range(N+1):
        for j in range(N+1):
            if i == j:
                A[i, j] = -1*(i > 0) - 1  # Diagonal elements: surface and atmospheric layers
            else:
                m = np.abs(j-i) - 1  # Non-diagonal elements: interaction between layers
                A[i, j] = epsilon * (1-epsilon)**m  # Off-diagonal elements

    # Adjust the surface layer (index 0) to account for emissivity effects
    A[0, 1:] /= epsilon

    # Invert the matrix A to solve for fluxes
    Ainv = np.linalg.inv(A)
    fluxes = np.matmul(Ainv, b)  # Solve Ainv * b = fluxes

    # Convert fluxes to temperatures using the Stefan-Boltzmann law
    temps = (fluxes/epsilon/sigma)**0.25  # Temperatures in the atmosphere
    temps[0] = (fluxes[0]/sigma)**0.25  # Surface temperature, no division by epsilon

    return temps

# Define the plot_altitude_profile function to visualize temperature distribution by altitude
def plot_altitude_profile(temps):
    '''
    Plot the altitude profile of temperatures.

    Parameters
    ----------
    temps : numpy array
        Temperatures for each layer, including the surface.
    '''
    altitudes = np.arange(len(temps))  # Altitude represented by layer index
    plt.figure(figsize=(10, 6))  # Set figure size
    plt.plot(temps, altitudes, 'ro-')  # Plot temperature vs altitude
    plt.xlabel('Temperature (K)')  # X-axis label
    plt.ylabel('Altitude (layer)')  # Y-axis label (layers as altitude)
    plt.title('Altitude vs Temperature in Nuclear Winter Scenario')  # Title of the plot
    plt.grid(True)  # Display grid
    plt.show()

# Run the nuclear winter scenario simulation
N = 5  # Number of layers
epsilon = 0.5  # Emissivity
S0 = 1350  # Solar constant (W/m²)

# Calculate temperatures for the nuclear winter scenario
temps = nuclear_winter_atmos(N, epsilon, S0)

# Print the surface temperature in the nuclear winter scenario
print(f"Surface temperature in nuclear winter scenario: {temps[0]:.2f} K")

# Plot the altitude-temperature profile
plot_altitude_profile(temps)
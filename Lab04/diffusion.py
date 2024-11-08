#!/usr/bin/env python3
"""
This lab investigates heat diffusion within Arctic permafrost layers using forward-difference numerical methods 
for solving one-dimensional diffusion equations. Through this study, we aim to model and understand the thermal 
profiles of permafrost in response to seasonal temperature fluctuations and evaluate how global warming impacts 
these profiles.

We start by implementing a forward-difference solver for the heat equation, assessing solver stability with respect 
to time step, spatial resolution, and diffusion rate. The model is then applied to simulate the permafrost depth 
and active layer characteristics in Kangerlussuaq, Greenland, a region experiencing pronounced seasonal temperature 
variations.

Using historical data and simulated warming scenarios, we examine changes in permafrost stability and the depth 
of the active layer, projecting long-term changes due to rising atmospheric temperatures. This lab illustrates the 
impact of climate change on Arctic biomes, focusing on the thermal evolution of permafrost and its implications 
for carbon release and biome stability.

To run this code simply run the file and check the output figures and their usage/implications within the lab document
Read the comments to ensure Python does not crash (basically comment and uncomment the loops below line 154 to ensure code runtime efficiency)
"""

import numpy as np
import matplotlib.pyplot as plt

# Average monthly temperatures in Kangerlussuaq, Greenland (°C)
T_KANGER = np.array([-19.7, -21.0, -17.0, -8.4, 2.3, 8.4, 10.7, 8.5, 3.1, -6.0, -12.0, -16.9])

def temp_kanger(t, shift=0):
    """
    Calculates an annual sinusoidal approximation of temperature at Kangerlussuaq over time with an optional temperature shift.
    
    Parameters:
    - t: array-like, time in days (can be an array representing multiple time points).
    - shift: float, a temperature shift to simulate warming effects.
    
    Returns:
    - Temperature values (°C) at each time point in t with the given temperature shift.
    """
    amplitude = (T_KANGER - T_KANGER.mean()).max()  # Maximum deviation from the mean temperature
    return amplitude * np.sin(np.radians(t) - np.pi/2) + T_KANGER.mean() + shift

def heatdiff(xmax, tmax, dx, dt, shift, c2=0.0216, validate=False, visual=False):
    """
    Simulates heat diffusion through soil for permafrost analysis in response to warming.
    
    Parameters:
    - xmax: float, maximum depth (m).
    - tmax: float, maximum time duration (days).
    - dx: float, spatial step size (m).
    - dt: float, time step size (days).
    - shift: float, temperature shift for simulating warming.
    - c2: float, thermal diffusivity of the ground (default: 0.0216).
    - validate: bool, if True, print U array for debugging.
    - visual: bool, if True, generate visual output of temperature profile.

    Returns:
    - xgrid: array, spatial grid.
    - tgrid: array, time grid.
    - U: 2D array, temperature distribution over depth and time.
    - active_layer_depth: float, depth of active layer where seasonal temperature changes.
    - permafrost_depth: float, depth where permafrost starts.
    """
    if dt >= (dx ** 2) / (2 * c2):
        raise ValueError(f"Unstable configuration: Reduce dt to less than {(dx ** 2) / (2 * c2)}")

    # Create spatial (depth) and temporal grids
    M = int(xmax / dx) + 1  # Number of spatial points
    N = int(tmax / dt) + 1  # Number of time points
    xgrid = np.linspace(0, xmax, M)  # Spatial grid from 0 to xmax
    tgrid = np.linspace(0, tmax, N)  # Time grid from 0 to tmax

    # Initialize temperature distribution matrix
    U = np.zeros((M, N))

    # Initial and boundary conditions
    if validate:
        # Validation condition: set initial profile to a specific function for testing
        U[:, 0] = 4 * xgrid - 4 * xgrid**2
        U[0, :], U[-1, :] = 0, 0  # Boundary conditions at surface and bottom depth
    else:
        # Actual scenario: apply temperature model at the surface with specified shift
        U[0, :] = temp_kanger(tgrid, shift)
        U[-1, :] = 5  # Constant temperature at the bottom boundary

    # Stability factor for the diffusion process
    r = c2 * dt / dx**2

    # Heat diffusion loop over time
    for j in range(N - 1):
        # Update temperature at each depth except boundaries using finite difference
        U[1:-1, j+1] = U[1:-1, j] * (1 - 2 * r) + r * (U[2:, j] + U[:-2, j])

    if validate:
        print(np.array2string(U))
        return xgrid, tgrid, U

    if not validate:
        if visual:
            # Display temperature distribution over time and depth
            fig, ax1 = plt.subplots()
            temp_map = ax1.pcolor(tgrid / 365, xgrid, U, cmap='seismic', vmin=-25, vmax=25)
            ax1.set(title='Ground Temperature: Kangerlussuaq, Greenland', xlabel='Time (Years)', ylabel='Depth (m)')
            ax1.invert_yaxis()
            plt.colorbar(temp_map, ax=ax1, label='Temperature (°C)')
            plt.figtext(0.5, -0.04, f"Temperature Offset: {shift}°C", ha="center", fontsize=12)
            plt.show()

        # Identify active layer depth and permafrost depth using summer and winter temperature profiles
        steps_per_year = int(-365 / dt)
        summer_temps = U[:, steps_per_year:].max(axis=1)  # Max temperature at each depth over summer
        winter_temps = U[:, steps_per_year:].min(axis=1)  # Min temperature at each depth over winter

        # Determine active layer depth (first depth where summer temperature crosses from positive to negative)
        active_layer_depth = None
        for i in range(1, len(summer_temps)):
            if summer_temps[i] < 0 and summer_temps[i-1] > 0:
                depth1, depth2 = xgrid[i-1], xgrid[i]
                temp1, temp2 = summer_temps[i-1], summer_temps[i]
                active_layer_depth = depth1 + (0 - temp1) * (depth2 - depth1) / (temp2 - temp1)
                break

        # Determine permafrost depth (first depth from bottom where summer temperature reaches 0°C)
        permafrost_depth = None
        for i in range(len(summer_temps) - 1, -1, -1):
            if abs(summer_temps[i]) < 1e-2:
                permafrost_depth = xgrid[i]
                break

        # Visualize seasonal temperature profiles with marked active layer and permafrost depths
        if visual:
            fig, ax2 = plt.subplots(figsize=(10, 8))
            ax2.plot(winter_temps, xgrid, label='Winter', color='blue')
            ax2.plot(summer_temps, xgrid, label='Summer', linestyle='--', color='red')
            ax2.invert_yaxis()
            ax2.set(xlabel='Temperature (°C)', ylabel='Depth (m)', title=f"Seasonal Profiles with Temperature Offset: {shift}°C")
            if active_layer_depth is not None:
                ax2.axhline(y=active_layer_depth, color='green', linestyle=':', label=f'Active Layer Depth ({active_layer_depth:.2f} m)')
            if permafrost_depth is not None:
                ax2.axhline(y=permafrost_depth, color='purple', linestyle='--', label=f'Permafrost Depth ({permafrost_depth:.2f} m)')
            ax2.legend()
            ax2.grid(True)
            plt.figtext(0.5, 0.01, f"Time Elapsed: {tmax/365:.2f} Years", ha="center", fontsize=12)
            plt.show()

        return xgrid, tgrid, U, active_layer_depth, permafrost_depth


# Validation run with heat diffusion
# heatdiff(1, .2, .2, .02, 0, c2=1, validate=True)

# Arrays for time points in years (converted to days) for different visualizations
yr_arr1 = [3650, 7300, 10950, 21900, 36500]

# Uncomment these lines 155-156 and run and then comment when running other simulations below:
# Simulations for different time points without temperature shifts
# for tmax in yr_arr1:
#     heatdiff(100, tmax, 0.5, 0.2, 0, visual=True)

# Uncomment these lines 160-161 and run and then comment when running other simulations below:
# Simulations for different time points with temperature shifts of 0.5°C, 1°C, and 3°C
# for tmax in yr_arr1:
#     heatdiff(100, tmax, 0.5, 0.2, 0.5, visual=True)

# Uncomment these lines 164-165 and run and then comment when running other simulations below:
# for tmax in yr_arr1:
#     heatdiff(100, tmax, 0.5, 0.2, 1, visual=True)

# Uncomment these lines 168-169 and run and then comment when running other simulations below:
# for tmax in yr_arr1:
#     heatdiff(100, tmax, 0.5, 0.2, 3, visual=True)

# Arrays for temperature shifts and time points
temp_shifts = [0.5, 1, 3]
yr_arr = [3650, 7300, 10950]

# Store results
active_layer_depths = {shift: [] for shift in temp_shifts}
permafrost_depths = {shift: [] for shift in temp_shifts}

# Run simulations for each temperature shift and each time period
for shift in temp_shifts:
    for tmax in yr_arr:
        _, _, _, active_layer_depth, permafrost_depth = heatdiff(100, tmax, 0.5, 0.2, shift, visual=False)
        active_layer_depths[shift].append(active_layer_depth)
        permafrost_depths[shift].append(permafrost_depth)

# Plot the results of active layer depth and permafrost depth over time under temperature shifts
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

# Plot active layer depth over time for different temperature shifts
for shift in temp_shifts:
    ax1.plot([t / 365 for t in yr_arr], active_layer_depths[shift], marker='o', label=f'Temp Shift: {shift}°C')
ax1.set_title('Active Layer Depth over Time under Temperature Shifts')
ax1.set_xlabel('Time (Years)')
ax1.set_ylabel('Active Layer Depth (m)')
ax1.invert_yaxis()
ax1.legend()
ax1.grid(True)

# Plot permafrost depth over time for different temperature shifts
for shift in temp_shifts:
    ax2.plot([t / 365 for t in yr_arr], permafrost_depths[shift], marker='o', label=f'Temp Shift: {shift}°C')
ax2.set_title('Permafrost Depth over Time under Temperature Shifts')
ax2.set_xlabel('Time (Years)')
ax2.set_ylabel('Permafrost Depth (m)')
ax2.set_ylim(-5, 100)  # Adjusted y-axis range for better visibility
ax2.invert_yaxis()
ax2.legend()
ax2.grid(True)

plt.suptitle('Impact of Global Warming on Active Layer and Permafrost Depths in Kangerlussuaq')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

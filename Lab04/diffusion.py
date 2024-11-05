import numpy as np
import matplotlib.pyplot as plt

def heatdiff(xmax=1, tmax=.2, dx=.2, dt=.02, c2=1, neumann=False, debug=False):
    '''
    Parameters:
    -----------
    xmax : float, defaults to 1
        Set the domain upper boundary location. In meters.
    tmax : float, defaults to .2s
        Set the domain time limit in seconds.
    neumann : bool, defaults to False
        Switch to Neumann boundary conditions if true where dU/dx = 0
        Default behavior is Dirichlet where U=0 at boundaries.

    Returns:
    --------

    '''

    # Start by calculating size of array: MxN
    M = int(np.round(xmax / dx + 1))
    N = int(np.round(tmax / dt + 1))

    xgrid, tgrid = np.arange(0, xmax+dx, dx), np.arange(0, tmax+dt, dt)

    if debug:
        print(f'Our grid goes from 0 to {xmax}m and 0 to {tmax}s')
        print(f'Our spatial step is {dx} and time step is {dt}')
        print(f'There are {M} points in space and {N} points in time.')
        print('Here is our spatial grid:')
        print(xgrid)
        print('Here is our time grid:')
        print(tgrid)

    # Initialize our data array:
    U = np.zeros([M, N])

    # Set initial conditions:
    U[:, 0] = 4*xgrid - 4*xgrid**2

    # Set boundary conditions:
    U[0, :] = 0
    U[-1, :] = 0

    # Set our "r" constant.
    r = c2 * dt / dx**2

    # Solve! Forward differnce ahoy.
    for j in range(N-1):
        U[1:-1, j+1] = (1-2*r) * U[1:-1, j] + \
            r*(U[2:, j] + U[:-2, j])
        # Set Neumann-type boundary conditions:
        if neumann:
            U[0, j+1] = U[1, j+1]
            U[-1, j+1] = U[-2, j+1]

    # Return grid and result:
    return xgrid, tgrid, U

x, t, heat = heatdiff()

# Validation Bounds
fig, ax = plt.subplots(figsize=(8, 6))
c = ax.pcolor(t, x, heat, cmap='plasma', vmin=0, vmax=1)
plt.colorbar(c, ax=ax, label='Temperature (°C)')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.title('Heat Diffusion Solution')
plt.show()

# Validation of the solver with specific conditions
def heatdiff_validate(xmax=1, tmax=0.2, dx=0.2, dt=0.02, c2=1):
    M = int(np.round(xmax / dx + 1))
    N = int(np.round(tmax / dt + 1))
    xgrid, tgrid = np.linspace(0, xmax, M), np.linspace(0, tmax, N)

    # Initialize U and set initial and boundary conditions
    U = np.zeros([M, N])
    U[:, 0] = 4 * xgrid - 4 * xgrid**2  # U(x, 0) = 4x - 4x^2
    U[0, :] = U[-1, :] = 0  # U(0, t) = U(1, t) = 0

    r = c2 * dt / dx**2  # Compute r for stability
    if r > 0.5:  # Stability condition
        raise ValueError("The configuration is not numerically stable.")

    for j in range(N-1):
        U[1:-1, j+1] = (1 - 2 * r) * U[1:-1, j] + r * (U[2:, j] + U[:-2, j])

    return xgrid, tgrid, U

# Run validation
x, t, U = heatdiff_validate()

# Plot the result for validation
plt.figure()
for i in range(0, len(t), 5):  # Plot at intervals of 5 time steps
    plt.plot(x, U[:, i], label=f't={t[i]:.2f}s')
plt.xlabel('Distance (m)')
plt.ylabel('Temperature (°C)')
plt.title('Heat Diffusion: Validation Case')
plt.legend()
plt.show()

heatdiff_validate()

def temp_kanger(t, temp_shift=0):
    '''
    For an array of times in days, return timeseries of temperature for
    Kangerlussuaq, Greenland with optional temperature shift for climate change scenarios.
    '''
    t_kanger = np.array([-19.7, -21.0, -17.0, -8.4, 2.3, 8.4,
                         10.7, 8.5, 3.1, -6.0, -12.0, -16.9]) + temp_shift
    t_amp = (t_kanger - t_kanger.mean()).max()
    return t_amp * np.sin(np.pi / 180 * t - np.pi / 2) + t_kanger.mean()

def permafrost_model(depth_max=100, time_max=365*10, dx=1, dt=1, 
                    c2=0.25e-6, temp_shift=0):
    '''
    Solve heat equation for permafrost with Kangerlussuaq surface temperatures.
    '''
    
    # Convert c2 from m²/s to m²/day for daily timesteps
    c2 = c2 * 86400  # seconds per day
    
    # Calculate grid dimensions
    M = int(np.round(depth_max / dx + 1))
    N = int(np.round(time_max / dt + 1))
    
    # Create spatial and temporal grids
    depth = np.linspace(0, depth_max, M)
    time = np.linspace(0, time_max, N)
    
    # Check stability
    r = c2 * dt / (dx**2)
    if r > 0.5:
        raise ValueError(f"Solution unstable! r = {r:.3f} > 0.5")
    
    # Initialize temperature array with a slight gradient
    T = np.zeros((M, N))
    surface_temp = -5     # surface starts cooler
    geothermal_gradient = 0.1  # degrees Celsius per meter
    T[:, 0] = surface_temp + geothermal_gradient * depth  # Set initial gradient
    
    # Set boundary conditions
    T[0, :] = temp_kanger(time, temp_shift)             # Upper boundary: surface temperature
    T[-1, :] = T[-2, :] + geothermal_gradient * dx      # Lower boundary: geothermal warming
    
    # Solve using forward difference
    for j in range(N-1):
        T[1:-1, j+1] = (1-2*r)*T[1:-1, j] + r*(T[2:, j] + T[:-2, j])
        # Update lower boundary dynamically to simulate geothermal heat influx
        T[-1, j+1] = T[-2, j+1] + geothermal_gradient * dx

    return depth, time, T, dt

def plot_results(depth, time, T, dt, temp_shift=0):
    '''
    Create visualization of results.
    '''
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    
    # Plot 1: Heat map of temperature over time and depth
    pcm = ax1.pcolormesh(time / 365, depth, T, cmap='seismic', vmin=-25, vmax=25)
    ax1.invert_yaxis()
    ax1.set_xlabel('Time (years)')
    ax1.set_ylabel('Depth (m)')
    ax1.set_title(f'Ground Temperature Evolution (+{temp_shift}°C scenario)')
    plt.colorbar(pcm, ax=ax1, label='Temperature (°C)')
    
    # Plot 2: Temperature profiles for summer and winter
    year_steps = int(365 / dt)
    last_year = T[:, -year_steps:]
    winter = last_year.min(axis=1)
    summer = last_year.max(axis=1)
    
    ax2.plot(winter, depth, 'b-', label='Winter')
    ax2.plot(summer, depth, 'r-', label='Summer')
    ax2.axvline(0, color='k', linestyle='--', alpha=0.5)
    ax2.invert_yaxis()
    ax2.set_xlabel('Temperature (°C)')
    ax2.set_ylabel('Depth (m)')
    ax2.set_title(f'Seasonal Temperature Profiles (+{temp_shift}°C scenario)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def analyze_layers(depth, T, dt):
    '''
    Analyze active and permafrost layer depths.
    '''
    year_steps = int(365 / dt)
    last_year = T[:, -year_steps:]
    winter = last_year.min(axis=1)
    summer = last_year.max(axis=1)
    
    # Find active layer depth (where summer temp goes above 0°C)
    active_layer = depth[np.where(summer > 0)[0][-1]] if np.any(summer > 0) else 0
    
    # Find permafrost bottom (where winter temp goes above 0°C)
    permafrost_bottom = depth[np.where(winter < 0)[0][-1]] if np.any(winter < 0) else None
    
    return active_layer, permafrost_bottom

# Run baseline scenario
depth, time, T, dt = permafrost_model(temp_shift=0)
plot_results(depth, time, T, dt, temp_shift=0)

# Analyze and print results for each scenario
scenarios = [0, 0.5, 1.0, 3.0]
for temp_shift in scenarios:
    depth, time, T, dt = permafrost_model(temp_shift=temp_shift)
    active_layer, permafrost_bottom = analyze_layers(depth, T, dt)
    print(f"\nScenario: +{temp_shift}°C")
    print(f"Active layer depth: {active_layer:.1f}m")
    if permafrost_bottom is not None:
        print(f"Permafrost bottom depth: {permafrost_bottom:.1f}m")
        print(f"Permafrost thickness: {permafrost_bottom - active_layer:.1f}m")
    else:
        print("No permafrost layer present")
    
    # Display the updated results
    plot_results(depth, time, T, dt, temp_shift=temp_shift)
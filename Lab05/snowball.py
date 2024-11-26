#!/usr/bin/env python3

'''
The "Snowball Earth" hypothesis proposes a radical climatic scenario where the entire Earth was 
completely glaciated approximately 650-700 million years ago. During this hypothetical period, 
the planet would have been covered entirely in ice, with no exposed liquid water or land masses. 
This extraordinary concept is supported by geological evidence such as worldwide glacial striations and deposit formations. 
The hypothesis challenges our understanding of planetary climate systems by suggesting a state of near-total planetary freezing. 
While some scientists support this theory, others argue that achieving and escaping such a global glaciation would be 
physically extremely challenging. An alternative "slushball Earth" hypothesis suggests that while massive glaciation occurred, 
a narrow band of open ocean might have remained around the equator.
'''

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

# Some constants:
radearth = 6357000.  # Earth radius in meters.
mxdlyr = 50.         # depth of mixed layer (m)
sigma = 5.67e-8      # Steffan-Boltzman constant
C = 4.2e6            # Heat capacity of water
rho = 1020           # Density of sea-water (kg/m^3)

# Experiment 1 code

def gen_grid(nbins=18):
    '''
    Generate a grid from 0 to 180 lat (where 0 is south pole, 180 is north)
    where each returned point represents the cell center.

    Parameters
    ----------
    nbins : int, defaults to 18
        Set the number of latitude bins.

    Returns
    -------
    dlat : float
        Grid spacing in degrees.
    lats : Numpy array
        Array of cell center latitudes.
    '''

    dlat = 180 / nbins  # Latitude spacing.
    lats = np.arange(0, 180, dlat) + dlat/2.

    # Alternative way to obtain grid:
    # lats = np.linspace(dlat/2., 180-dlat/2, nbins)

    return dlat, lats


def temp_warm(lats_in):
    '''
    Create a temperature profile for modern day "warm" earth.

    Parameters
    ----------
    lats_in : Numpy array
        Array of latitudes in degrees where temperature is required

    Returns
    -------
    temp : Numpy array
        Temperature in Celcius.
    '''

    # Get base grid:
    dlat, lats = gen_grid()

    # Set initial temperature curve:
    T_warm = np.array([-47, -19, -11, 1, 9, 14, 19, 23, 25, 25,
                       23, 19, 14, 9, 1, -11, -19, -47])
    coeffs = np.polyfit(lats, T_warm, 2)

    # Now, return fitting:
    temp = coeffs[2] + coeffs[1]*lats_in + coeffs[0] * lats_in**2

    return temp


def insolation(S0, lats):
    '''
    Given a solar constant (`S0`), calculate average annual, longitude-averaged
    insolation values as a function of latitude.
    Insolation is returned at position `lats` in units of W/m^2.

    Parameters
    ----------
    S0 : float
        Solar constant (1370 for typical Earth conditions.)
    lats : Numpy array
        Latitudes to output insolation. Following the grid standards set in
        the diffusion program, polar angle is defined from the south pole.
        In other words, 0 is the south pole, 180 the north.

    Returns
    -------
    insolation : numpy array
        Insolation returned over the input latitudes.
    '''

    # Constants:
    max_tilt = 23.5   # tilt of earth in degrees

    # Create an array to hold insolation:
    insolation = np.zeros(lats.size)

    #  Daily rotation of earth reduces solar constant by distributing the sun
    #  energy all along a zonal band
    dlong = 0.01  # Use 1/100 of a degree in summing over latitudes
    angle = np.cos(np.pi/180. * np.arange(0, 360, dlong))
    angle[angle < 0] = 0
    total_solar = S0 * angle.sum()
    S0_avg = total_solar / (360/dlong)

    # Accumulate normalized insolation through a year.
    # Start with the spin axis tilt for every day in 1 year:
    tilt = [max_tilt * np.cos(2.0*np.pi*day/365) for day in range(365)]

    # Apply to each latitude zone:
    for i, lat in enumerate(lats):
        # Get solar zenith; do not let it go past 180. Convert to latitude.
        zen = lat - 90. + tilt
        zen[zen > 90] = 90
        # Use zenith angle to calculate insolation as function of latitude.
        insolation[i] = S0_avg * np.sum(np.cos(np.pi/180. * zen)) / 365.

    # Average over entire year; multiply by S0 amplitude:
    insolation = S0_avg * insolation / 365

    return insolation


def snowball_earth(nbins=18, dt=1., tstop=10000, lam=100., spherecorr=True,
                   debug=False, albedo=0.3, emiss=1, S0=1370, initial_temp=None):
    '''
    Perform snowball earth simulation.

    Parameters
    ----------
    nbins : int, defaults to 18
        Number of latitude bins.
    dt : float, defaults to 1
        Timestep in units of years
    tstop : float, defaults to 10,000
        Stop time in years
    lam : float, defaults to 100
        Diffusion coefficient of ocean in m^2/s
    spherecorr : bool, defaults to True
        Use the spherical coordinate correction term. This should always be
        true except for testing purposes.
    debug : bool, defaults to False
        Turn  on or off debug print statements.
    albedo : float, defaults to 0.3
        Set the Earth's albedo.
    emiss : float, defaults to 1.0
        Set ground emissivity. Set to zero to turn off radiative cooling.
    S0 : float, defaults to 1370
        Set incoming solar forcing constant. Change to zero to turn off
        insolation.

    Returns
    -------
    lats : Numpy array
        Latitude grid in degrees where 0 is the south pole.
    Temp : Numpy array
        Final temperature as a function of latitude.
    '''
    # Get time step in seconds:
    dt_sec = 365 * 24 * 3600 * dt  # Years to seconds.

    # Generate grid:
    dlat, lats = gen_grid(nbins)

    # Get grid spacing in meters.
    dy = radearth * np.pi * dlat / 180.

    # Generate insolation:
    insol = insolation(S0, lats)

    # Create initial condition:
    Temp = temp_warm(lats)
    if debug:
        print('Initial temp = ', Temp)

    # Get number of timesteps:
    nstep = int(tstop / dt)

    # Debug for problem initialization
    if debug:
        print("DEBUG MODE!")
        print(f"Function called for nbins={nbins}, dt={dt}, tstop={tstop}")
        print(f"This results in nstep={nstep} time step")
        print(f"dlat={dlat} (deg); dy = {dy} (m)")
        print("Resulting Lat Grid:")
        print(lats)

    # Build A matrix:
    if debug:
        print('Building A matrix...')
    A = np.identity(nbins) * -2  # Set diagonal elements to -2
    A[np.arange(nbins-1), np.arange(nbins-1)+1] = 1  # Set off-diag elements
    A[np.arange(nbins-1)+1, np.arange(nbins-1)] = 1  # Set off-diag elements
    # Set boundary conditions:
    A[0, 1], A[-1, -2] = 2, 2

    # Build "B" matrix for applying spherical correction:
    B = np.zeros((nbins, nbins))
    B[np.arange(nbins-1), np.arange(nbins-1)+1] = 1  # Set off-diag elements
    B[np.arange(nbins-1)+1, np.arange(nbins-1)] = -1  # Set off-diag elements
    # Set boundary conditions:
    B[0, :], B[-1, :] = 0, 0

    # Set the surface area of the "side" of each latitude ring at bin center.
    Axz = np.pi * ((radearth+50.0)**2 - radearth**2) * np.sin(np.pi/180.*lats)
    dAxz = np.matmul(B, Axz) / (Axz * 4 * dy**2)

    if debug:
        print('A = ', A)
    # Set units of A derp
    A /= dy**2

    # Get our "L" matrix:
    L = np.identity(nbins) - dt_sec * lam * A
    L_inv = np.linalg.inv(L)

    if debug:
        print('Time integrating...')
    for i in range(nstep):
        # Add spherical correction term:
        if spherecorr:
            Temp += dt_sec * lam * dAxz * np.matmul(B, Temp)

        # Apply insolation and radiative losses:
        # print('T before insolation:', Temp)
        radiative = (1-albedo) * insol - emiss*sigma*(Temp+273.15)**4
        # print('\t Rad term = ', dt_sec * radiative / (rho*C*mxdlyr))
        Temp += dt_sec * radiative / (rho*C*mxdlyr)
        # print('\t T after rad:', Temp)

        Temp = np.matmul(L_inv, Temp)

    return lats, Temp


def test_snowball(tstop=10000):
    '''
    Reproduce example plot in lecture/handout.

    Using our DEFAULT values (grid size, diffusion, etc.) and a warm-Earth
    initial condition, plot:
        - Initial condition
        - Plot simple diffusion only
        - Plot simple diffusion + spherical correction
        - Plot simple diff + sphere corr + insolation
    '''
    nbins = 18

    # Generate very simple grid
    # Generate grid:
    dlat, lats = gen_grid(nbins)

    # Create initial condition:
    initial = temp_warm(lats)

    # Get simple diffusion solution:
    lats, t_diff = snowball_earth(tstop=tstop, spherecorr=False, S0=0, emiss=0)

    # Get diffusion + spherical correction:
    lats, t_sphe = snowball_earth(tstop=tstop, S0=0, emiss=0)

    # Get diffusion + sphercorr + radiative terms:
    lats, t_rad = snowball_earth(tstop=tstop)

    # Create figure and plot!
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.plot(lats, initial, label='Warm Earth Init. Cond.')
    ax.plot(lats, t_diff, label='Simple Diffusion')
    ax.plot(lats, t_sphe, label='Diffusion + Sphere. Corr.')
    ax.plot(lats, t_rad, label='Diffusion + Sphere. Corr. + Radiative')

    ax.set_xlabel('Latitude (0=South Pole)')
    ax.set_ylabel('Temperature ($^{\circ} C$)')

    ax.legend(loc='best')

    fig.tight_layout()
    plt.show()

test_snowball()

# Experiment 2 Code

# Constants
lam_range = np.linspace(0, 150, 35)  # Range of diffusivity values
emiss_range = np.linspace(0, 1, 35)  # Range of emissivity values
nbins = 18

# Generate grid
_, lats = gen_grid(nbins)

# Get the "warm Earth" profile for comparison
warm_earth_temp = temp_warm(lats)

# Store results
mse_results = []

for lam in lam_range:
    for emiss in emiss_range:
        # Run the simulation
        _, temp = snowball_earth(nbins=nbins, lam=lam, emiss=emiss, tstop=10000)
        
        # Calculate Mean Squared Error
        mse = np.mean((temp - warm_earth_temp) ** 2)
        mse_results.append((lam, emiss, mse))

# Find the best-fit parameters
best_fit = min(mse_results, key=lambda x: x[2])
best_lam, best_emiss, best_mse = best_fit

# Plot the best fit
_, best_temp = snowball_earth(nbins=nbins, lam=best_lam, emiss=best_emiss, tstop=10000)

plt.figure(figsize=(10, 6))
plt.plot(lats, warm_earth_temp, label="Warm Earth Curve", linestyle="--")
plt.plot(lats, best_temp, label=f"Best Fit (λ={best_lam:.2f}, ε={best_emiss:.2f})")
plt.xlabel("Latitude (0=South Pole)")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.title("Best Fit to Warm Earth Curve")
plt.show()

# Print the results
print(f"Best Fit Parameters:\n Diffusivity (λ): {best_lam:.2f}\n Emissivity (ε): {best_emiss:.2f}\n MSE: {best_mse:.4f}")


# Experiment 3 Code 

def dynamic_albedo(T):
    """
    Calculate albedo based on temperature.
    T < -10°C: snow/ice covered (high albedo)
    T > 0°C: ice-free (low albedo)
    -10°C to 0°C: linear transition
    """
    albedo_snow = 0.6  # snow/ice albedo
    albedo_water = 0.3  # water albedo
    
    alpha = np.zeros_like(T)
    
    # Ice covered regions
    alpha[T <= -10] = albedo_snow
    
    # Ice-free regions
    alpha[T >= 0] = albedo_water
    
    # Transition zone
    mask = (T > -10) & (T < 0)
    alpha[mask] = albedo_snow + (albedo_water - albedo_snow) * (T[mask] + 10) / 10
    
    return alpha

def modified_snowball_earth(nbins=18, dt=1., tstop=10000, lam=100., 
                          initial_temp=None, dynamic_alb=False, fixed_albedo=0.3):
    """
    Modified snowball earth simulation with support for different initial conditions
    and dynamic albedo.
    """
    # Get time step in seconds
    dt_sec = 365 * 24 * 3600 * dt
    
    # Generate grid
    dlat = 180 / nbins
    lats = np.arange(0, 180, dlat) + dlat/2.
    
    # Grid spacing in meters
    dy = radearth * np.pi * dlat / 180.
    
    # Generate insolation
    insol = insolation(S0=1370, lats=lats)
    
    # Set initial condition
    if initial_temp is None:
        Temp = temp_warm(lats)
    else:
        Temp = np.full_like(lats, initial_temp)
    
    # Get number of timesteps
    nstep = int(tstop / dt)
    
    # Build matrices for diffusion
    A = np.identity(nbins) * -2
    A[np.arange(nbins-1), np.arange(nbins-1)+1] = 1
    A[np.arange(nbins-1)+1, np.arange(nbins-1)] = 1
    A[0, 1], A[-1, -2] = 2, 2
    A /= dy**2
    
    # Spherical correction matrices
    B = np.zeros((nbins, nbins))
    B[np.arange(nbins-1), np.arange(nbins-1)+1] = 1
    B[np.arange(nbins-1)+1, np.arange(nbins-1)] = -1
    B[0, :], B[-1, :] = 0, 0
    
    # Surface area calculations
    Axz = np.pi * ((radearth+50.0)**2 - radearth**2) * np.sin(np.pi/180.*lats)
    dAxz = np.matmul(B, Axz) / (Axz * 4 * dy**2)
    
    # Get "L" matrix
    L = np.identity(nbins) - dt_sec * lam * A
    L_inv = np.linalg.inv(L)
    
    # Store temperature history
    temp_history = np.zeros((nstep//100 + 1, nbins))
    temp_history[0] = Temp
    
    for i in range(nstep):
        # Spherical correction
        Temp += dt_sec * lam * dAxz * np.matmul(B, Temp)
        
        # Calculate albedo
        if dynamic_alb:
            albedo = dynamic_albedo(Temp)
        else:
            albedo = fixed_albedo
        
        # Apply insolation and radiative losses
        radiative = (1-albedo) * insol - sigma*(Temp+273.15)**4
        Temp += dt_sec * radiative / (rho*C*mxdlyr)
        
        Temp = np.matmul(L_inv, Temp)
        
        # Store temperature every 100 steps
        if i % 100 == 0:
            temp_history[i//100 + 1] = Temp
    
    return lats, Temp, temp_history

# Run simulations with different initial conditions
tstop = 10000
nbins = 18
_, lats = gen_grid(nbins)

# 1. Hot Earth initial condition (60°C everywhere)
_, hot_final, hot_history = modified_snowball_earth(
    initial_temp=60, dynamic_alb=True, tstop=tstop)

# 2. Cold Earth initial condition (-60°C everywhere)
_, cold_final, cold_history = modified_snowball_earth(
    initial_temp=-60, dynamic_alb=True, tstop=tstop)

# 3. Flash freeze (warm Earth with high albedo)
_, flash_final, flash_history = modified_snowball_earth(
    initial_temp=None, fixed_albedo=0.6, tstop=tstop)

# Create visualization
plt.style.use('fivethirtyeight')
fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))

# Plot final states
ax1.plot(lats, hot_final, label='Hot Earth Initial (60°C)', linewidth=2)
ax1.plot(lats, cold_final, label='Cold Earth Initial (-60°C)', linewidth=2)
ax1.plot(lats, flash_final, label='Flash Freeze (α=0.6)', linewidth=2)
ax1.plot(lats, temp_warm(lats), '--', label='Reference Warm Earth', alpha=0.5)

ax1.set_xlabel('Latitude (0=South Pole)')
ax1.set_ylabel('Temperature (°C)')
ax1.set_title('Final Temperature Distributions')
ax1.legend()

# Plot temperature evolution at equator (lat ≈ 90°)
fig, ax2 = plt.subplots(1, 1, figsize=(8, 6))
eq_idx = nbins//2
times = np.linspace(0, tstop, hot_history.shape[0])
ax2.plot(times, hot_history[:, eq_idx], label='Hot Earth Initial (60°C)', linewidth=2)
ax2.plot(times, cold_history[:, eq_idx], label='Cold Earth Initial (-60°C)', linewidth=2)
ax2.plot(times, flash_history[:, eq_idx], label='Flash Freeze (α=0.6)', linewidth=2)

ax2.set_xlabel('Time (years)')
ax2.set_ylabel('Equatorial Temperature (°C)')
ax2.set_title('Temperature Evolution at Equator')
ax2.legend()

plt.tight_layout()
plt.show()

# Experiment 4 Code

def global_average_temp(temps, lats):
    """
    Calculate global average temperature weighted by latitude
    """
    weights = np.cos(np.deg2rad(lats - 90))  # Weight by cos(latitude)
    return np.average(temps, weights=weights)

def hysteresis_analysis(nbins=18, dt=1., tstop=5000):
    """
    Perform hysteresis analysis of snowball Earth under varying solar forcing
    """
    # Generate base grid
    dlat = 180 / nbins
    lats = np.arange(0, 180, dlat) + dlat/2.
    
    # Initialize arrays for storing results
    gamma_up = np.arange(0.4, 1.41, 0.05)
    gamma_down = np.arange(1.4, 0.39, -0.05)
    gamma_full = np.concatenate([gamma_up, gamma_down])
    temps_up = []
    temps_down = []
    
    # Start with cold Earth - explicitly use float dtype
    initial_temp = np.full(nbins, -60.0, dtype=np.float64)
    
    # Increasing gamma phase
    print("Running increasing gamma phase...")
    for gamma in gamma_up:
        # Run simulation
        _, final_temp, _ = modified_snowball_earth_more(
            nbins=nbins, dt=dt, tstop=tstop,
            initial_temp=initial_temp,
            dynamic_alb=True,
            gamma=gamma
        )
        
        # Store results
        avg_temp = global_average_temp(final_temp, lats)
        temps_up.append(avg_temp)
        
        # Use final state as initial condition for next run
        initial_temp = final_temp.copy()  # Make sure to copy the array
        print(f"γ = {gamma:.2f}, Average Temperature = {avg_temp:.1f}°C")
    
    # Decreasing gamma phase
    print("\nRunning decreasing gamma phase...")
    for gamma in gamma_down:
        # Run simulation
        _, final_temp, _ = modified_snowball_earth_more(
            nbins=nbins, dt=dt, tstop=tstop,
            initial_temp=initial_temp,
            dynamic_alb=True,
            gamma=gamma
        )
        
        # Store results
        avg_temp = global_average_temp(final_temp, lats)
        temps_down.append(avg_temp)
        
        # Use final state as initial condition for next run
        initial_temp = final_temp.copy()
        print(f"γ = {gamma:.2f}, Average Temperature = {avg_temp:.1f}°C")
    
    return gamma_up, gamma_down, temps_up, temps_down

def modified_snowball_earth_more(nbins=18, dt=1., tstop=5000, lam=100., 
                          initial_temp=None, dynamic_alb=False, 
                          fixed_albedo=0.3, gamma=1.0):
    """
    Modified snowball earth simulation with solar forcing multiplier
    """
    # Get time step in seconds
    dt_sec = 365 * 24 * 3600 * dt
    
    # Generate grid
    dlat = 180 / nbins
    lats = np.arange(0, 180, dlat) + dlat/2.
    
    # Grid spacing in meters
    dy = radearth * np.pi * dlat / 180.
    
    # Generate insolation with gamma multiplier
    insol = gamma * insolation(S0=1370, lats=lats)
    
    # Set initial condition
    if initial_temp is None:
        Temp = temp_warm(lats)
    else:
        # Make sure to copy and convert to float64
        Temp = np.array(initial_temp, dtype=np.float64)
    
    # Get number of timesteps
    nstep = int(tstop / dt)
    
    # Build matrices for diffusion
    A = np.identity(nbins) * -2
    A[np.arange(nbins-1), np.arange(nbins-1)+1] = 1
    A[np.arange(nbins-1)+1, np.arange(nbins-1)] = 1
    A[0, 1], A[-1, -2] = 2, 2
    A = A.astype(np.float64)  # Ensure float64 type
    A /= dy**2
    
    # Spherical correction matrices
    B = np.zeros((nbins, nbins), dtype=np.float64)  # Ensure float64 type
    B[np.arange(nbins-1), np.arange(nbins-1)+1] = 1
    B[np.arange(nbins-1)+1, np.arange(nbins-1)] = -1
    B[0, :], B[-1, :] = 0, 0
    
    # Surface area calculations
    Axz = np.pi * ((radearth+50.0)**2 - radearth**2) * np.sin(np.pi/180.*lats)
    dAxz = np.matmul(B, Axz) / (Axz * 4 * dy**2)
    
    # Get "L" matrix
    L = np.identity(nbins, dtype=np.float64) - dt_sec * lam * A  # Ensure float64 type
    L_inv = np.linalg.inv(L)
    
    # Store temperature history
    temp_history = np.zeros((nstep//100 + 1, nbins), dtype=np.float64)  # Ensure float64 type
    temp_history[0] = Temp
    
    for i in range(nstep):
        # Spherical correction
        Temp += dt_sec * lam * dAxz * np.matmul(B, Temp)
        
        # Calculate albedo
        if dynamic_alb:
            albedo = dynamic_albedo(Temp)
        else:
            albedo = fixed_albedo
        
        # Apply insolation and radiative losses
        radiative = (1-albedo) * insol - sigma*(Temp+273.15)**4
        Temp += dt_sec * radiative / (rho*C*mxdlyr)
        
        Temp = np.matmul(L_inv, Temp)
        
        # Store temperature every 100 steps
        if i % 100 == 0:
            temp_history[i//100 + 1] = Temp
    
    return lats, Temp, temp_history

# Run the hysteresis analysis
gamma_up, gamma_down, temps_up, temps_down = hysteresis_analysis()

# Create visualization
plt.figure(figsize=(12, 8))
plt.plot(gamma_up, temps_up, 'b-o', label='Increasing Solar Forcing', linewidth=2)
plt.plot(gamma_down, temps_down, 'r-o', label='Decreasing Solar Forcing', linewidth=2)

plt.xlabel('Solar Forcing Multiplier (γ)')
plt.ylabel('Global Average Temperature (°C)')
plt.title('Hysteresis in the Snowball Earth System')
plt.grid(True)
plt.legend()

# Add arrows to show direction
arrow_props = dict(arrowstyle='->', color='gray', lw=2)
plt.annotate('', xy=(1.0, temps_up[12]), xytext=(0.6, temps_up[4]), arrowprops=arrow_props)
plt.annotate('', xy=(0.8, temps_down[12]), xytext=(1.2, temps_down[4]), arrowprops=arrow_props)

plt.tight_layout()
plt.show()

#!/usr/bin/env python3
'''
Tools and methods for solving our heat equation/diffusion.
This is for my favorite class with Dr. Welling who is the best professor ever. What a cool guy Im going to give him $10.
'''

import numpy as np
import matplotlib.pyplot as plt

# Kangerlussuaq average temperature equation from lab manual
t_kanger = np.array([-19.7, -21.0, -17., -8.4, 2.3, 8.4,
10.7, 8.5, 3.1, -6.0, -12.0, -16.9])

def temp_kanger(t, offset_temp):
    '''
    For an array of times in days, return timeseries of temperature for
    Kangerlussuaq, Greenland.
    '''
    t_amp = (t_kanger - t_kanger.mean()).max()
    return t_amp*np.sin(np.pi/180 * t - np.pi/2) + t_kanger.mean() + offset_temp

def heatdiff(xmax, tmax, dx, dt, c2=.0216, debug=False, file_prefix="output", offset_temp=0, validation_case = False):
    '''
    Parameters:
    -----------
     xmax : float
        Maximum value of x in the grid.
    tmax : float
        Maximum value of t in the grid.
    dx : float
        Step size for x.
    dt : float
        Step size for t.
    c2 : float, optional
        Diffusion coefficient (default is 1).
    debug : bool, optional
        If True, prints debug information (default is True).
    file_prefix : str, optional
        Prefix for the filenames of the output .png files.
    offset_temp : float, optional
        Temperature offset applied to initial conditions
    validation_case : bool, optional
        If True, uses a specific set of initial and boundary conditions for validation purposes 
        (default is False). 

    Returns:
    --------
    xgrid : numpy.ndarray
        Array representing the spatial grid.
    tgrid : numpy.ndarray
        Array representing the time grid.
    U : numpy.ndarray
        2D array representing the solution over the grid.
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
    U = np.zeros((M,N))

    # Setting Initial and Boundary Conditions
    if(validation_case):
        U[:, 0] = 4*xgrid - 4*xgrid**2
        U[0, :] = 0
        U[-1, :] = 0
    else:
        U[0, :] = temp_kanger(tgrid, offset_temp) # Surface, due to Kangerlussuaq heating equation
        U[-1, :] = 5 # Lowest depth, due to geothermal 

    # Set our "r" constant.
    r = c2 * dt / dx**2

    # Solve! Forward differnce ahoy.
    for j in range(N-1):
        U[1:-1, j+1] = (1-2*r) * U[1:-1, j] + \
            r*(U[2:, j] + U[:-2, j])

    # Return grid and result:
    return xgrid, tgrid, U

def plot_heat_map(xgrid, tgrid, U, file_prefix):
    """
    Plots a space-time heat map showing temperature variation over time and depth.
    Useful for visualizing how temperature diffuses through the ground layers over time.
    
    Parameters:
    -----------
    xgrid : numpy.ndarray
        Array representing the depth grid in meters.
    tgrid : numpy.ndarray
        Array representing the time grid in days.
    U : numpy.ndarray
        2D array of calculated temperatures over depth and time.
    file_prefix : str
        Prefix for the filename to save the plot as a PNG image.
    """
    # Initialize the figure and axis for the heatmap
    fig, ax1 = plt.subplots()
    
    # Plotting the heatmap using pseudocolor (pcolor) for smooth shading
    # tgrid / 365 to convert time from days to years for better readability
    heatmap = ax1.pcolor(tgrid / 365, xgrid, U, cmap='seismic', vmin=-25, vmax=25)
    
    # Set plot title and axis labels for clarity
    ax1.set_title('Ground Temperature Heat Map')
    ax1.set_xlabel('Time (Years)')
    ax1.set_ylabel('Depth (m)')
    
    # Invert the y-axis to have the surface (depth = 0) at the top
    ax1.invert_yaxis()
    
    # Add a color bar to the plot for a temperature scale reference
    plt.colorbar(heatmap, ax=ax1, label='Temperature (°C)')
    
    # Save the figure as a high-resolution PNG file with the given file prefix
    fig.savefig(f"{file_prefix}_heat_map.png", dpi=300, bbox_inches='tight')
    
    # Display the plot
    plt.show()

def plot_seasonal_profiles(xgrid, tgrid, U, file_prefix):
    """
    Plots seasonal temperature profiles for winter and summer, showing
    temperature distribution as a function of depth in the ground.
    
    Parameters:
    -----------
    xgrid : numpy.ndarray
        Array representing the depth grid in meters.
    tgrid : numpy.ndarray
        Array representing the time grid in days.
    U : numpy.ndarray
        2D array of calculated temperatures over depth and time.
    file_prefix : str
        Prefix for the filename to save the plot as a PNG image.
    """
    # Determine number of timesteps in a typical year by dividing by the time step size
    loc = int(-365 / (tgrid[1] - tgrid[0]))  
    
    # Extract seasonal temperature profiles by finding min/max for each depth
    winter = U[:, loc:].min(axis=1)  
    summer = U[:, loc:].max(axis=1)  
    
    # Initialize the figure and axis for the seasonal profile plot
    fig, ax2 = plt.subplots()
    
    # Plotting the winter profile in blue and summer profile in red with a dashed line
    ax2.plot(winter, xgrid, label='Winter', color='blue')
    ax2.plot(summer, xgrid, label='Summer', color='red', linestyle='--')
    
    # Invert y-axis to display depth increasing downwards
    ax2.invert_yaxis()
    
    # Set axis labels and title for the seasonal profile plot
    ax2.set_xlabel('Temperature (°C)')
    ax2.set_ylabel('Depth (m)')
    ax2.set_title('Seasonal Temperature Profiles')
    
    # Add a legend to distinguish winter and summer
    ax2.legend()
    
    # Save the figure with the specified file prefix
    fig.savefig(f"{file_prefix}_seasonal_profile.png", dpi=300, bbox_inches='tight')
    
    # Display the plot
    plt.show()

# Main execution
if __name__ == "__main__":

    # NOTE: this reference test executes with arbitrary input to ensure that the 
    # plots are generating with the form generally expected with the lab. Their 
    # correct appearance gives assurance that the underlying code is working 
    # correctly. This means you can feel free to create other experiments / tests
    # that tweak the input parameters, while being confident that their result
    # is correct. (i.e. now that code verification is complete, next you can 
    # USE the code in different ways for new discoveries and learnings)

    x, t, U = heatdiff(100, 3650, 1, 0.2, debug=False, file_prefix="test")
    plot_heat_map(x, t, U, "test_heat")
    plot_seasonal_profiles(x, t, U, "test_seasonal")

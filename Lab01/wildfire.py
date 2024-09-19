#!/usr/bin/env python3

'''
This file contains tools and scripts for completing Lab 1 for CLaSP 410. 
We explore the questions found below:

How does the spread of wildfire depend on the
probability of spread of fire and initial forest density?.

To reproduce the plots shown in the lab report, do this...
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Set grid size and number of iterations
nNorth = 3
nEast = 3
maxiter = 4

# Set probabilities for different events

prob_spread = 1.0 # Chance to spread to adjacent cells.
prob_bare = 0.0 # Chance of cell to start as bare patch.
prob_start = 1.0 # Chance of cell to start on fire.
prob_fatal = 0.5 # Chance that an infected cell has passed on

# Function to simulate fire spread in n,e,s,w cardinal directions   
             
def full_fire_spread(nNorth = nNorth, nEast = nEast, maxiter = maxiter, prob_spread = prob_spread,
                      prob_bare = prob_bare, prob_start = prob_start):
    
    # Initialize the forest grid, set all cells to 2 (default healthy state)

    forest = np.zeros((maxiter,nNorth,nEast), dtype=int) + 2
    istart, jstart = nNorth//2, nEast//2  # Determine the center of the grid

    # Determine which cells are bare patches

    isbare = np.random.rand(nNorth, nEast)
    isbare = isbare <= prob_bare # logical mask for bare cells
    forest[0,isbare] = 1 # Set bare patches to 1
    
    # Randomly start a fire in the center cell if it's not bare
    
    if (np.random.rand() <= prob_start):
        forest[0, istart, jstart] = 3

    # Visualize the initial state
        
    # fig,ax = plt.subplots(1,1)
    # forest_cmap = ListedColormap(['tan', 'darkgreen', 'crimson'])
    # ax.pcolor(forest[0,:,:], cmap=forest_cmap, vmin=1, vmax=3)
    # ax.set_title(f'Forest Status (i-step = {0:03d})')
    # ax.set_xlabel('X km')
    # ax.set_ylabel('Y km')
    # fig.savefig(f'fig{0:04d}.png')
    #plt.show()

    # Store the number of burning cells at each iteration
    burning_cells = [(forest[0, :, :] == 3).sum()]

    # Loop through each iteration

    for k in range(maxiter-1) :
        ignite = np.random.rand(nNorth, nEast) # Random spread probability

        forest[k+1,:,:] = forest[k,:,:] # Copy the previous state into the next state

        # Spread to the North using logical masking
        
        doburnN = (forest[k, :-1, :] == 3) & (forest[k, 1:, :] == 2) & (ignite[1:, :] <= prob_spread)
        forest[k+1, 1:, :][doburnN] = 3

        # Spread to the South using logical masking

        doburnS = (forest[k, 1:, :] == 3) & (forest[k, :-1, :] == 2) & (ignite[:-1, :] <= prob_spread)
        forest[k+1, :-1, :][doburnS] = 3

        # Spread to the East using logical masking

        doburnE = (forest[k, :, :-1] == 3) & (forest[k, :, 1:] == 2) & (ignite[:, 1:] <= prob_spread)
        forest[k+1, :, 1:][doburnE] = 3

        # Spread to the West using logical masking

        doburnW = (forest[k, :, 1:] == 3) & (forest[k, :, :-1] == 2) & (ignite[:, :-1] <= prob_spread)
        forest[k+1, :, :-1][doburnW] = 3

        # Update previously burning cells to bare ground (state 1)

        wasburn = forest[k,:,:] == 3
        forest[k+1, wasburn] = 1
        
        # Visualize the current state

        # fig,ax = plt.subplots(1,1)
        # forest_cmap = ListedColormap(['tan', 'darkgreen', 'crimson'])
        # ax.pcolor(forest[k+1,:,:], cmap=forest_cmap, vmin=1, vmax=3)
        # ax.set_title(f'Forest Status (i-step = {k+1:03d})')
        # ax.set_xlabel('X km')
        # ax.set_ylabel('Y km')
        # fig.savefig(f'fig{k+1:04d}.png')
        #plt.show()

        # Quit if all fires are out
        nBurn = (forest[k+1,:,:] == 3).sum()
        burning_cells.append(nBurn)

        if nBurn == 0:
            break
    
    # ****_COMMENT THIS LINE WHEN RUNNING scatter_initial_density_sim() SIMULATION_*****
    # ****_UNCOMMENT WHEN RUNNING line_graph_fire_spread_sim() SIMULATION_*****
    #return burning_cells
    
    # Calculate total area burned (cells changed to state 1 from state 3)

    total_burned = np.sum(forest == 1)
    
    return total_burned

#---------------------------------------

# Function to simulate disease spread

def full_disease_spread(nNorth = nNorth, nEast = nEast, maxiter = maxiter, prob_spread = prob_spread,
                      prob_bare = prob_bare, prob_start = prob_start, prob_fatal = prob_fatal):
    
    # Initialize the disease grid, set all cells to 2 (default healthy state)

    disease = np.zeros((maxiter,nNorth,nEast), dtype=int) + 2
    istart, jstart = nNorth//2, nEast//2  # Determine the center of the grid

    # Determine which cells are bare patches

    isbare = np.random.rand(nNorth, nEast)
    isbare = isbare <= prob_bare # logical mask for immune cells
    disease[0,isbare] = 1 # Set immune patches to 1

    # Determine which cells will turn fatal if infected

    isfatal = np.random.rand(nNorth, nEast)
    isfatal = isfatal <= prob_fatal
    
    # Randomly start an infection in the center cell if it's not bare

    if (np.random.rand() <= prob_start):
        disease[0, istart, jstart] = 3

    infected_counts = []
    immune_counts = []
    deceased_counts = []

    # Visualize the initial state

    # fig,ax = plt.subplots(1,1)
    # forest_cmap = ListedColormap(['black','lightgreen', 'darkgreen', 'crimson'])
    # ax.pcolor(disease[0,:,:], cmap=forest_cmap, vmin=0, vmax=3)
    # ax.set_title(f'Disease Status (i-step = {0:03d})')
    # ax.set_xlabel('X km')
    # ax.set_ylabel('Y km')
    # fig.savefig(f'fig{0:04d}.png')
    # plt.show()

    for k in range(maxiter-1) :
        ignite = np.random.rand(nNorth, nEast) # Random spread probability

        disease[k+1,:,:] = disease[k,:,:] # Copy the previous state into the next state

        # Spread to the North using logical masking

        doburnN = (disease[k, :-1, :] == 3) & (disease[k, 1:, :] == 2) & (ignite[1:, :] <= prob_spread)
        disease[k+1, 1:, :][doburnN] = 3

        # Spread to the South using logical masking

        doburnS = (disease[k, 1:, :] == 3) & (disease[k, :-1, :] == 2) & (ignite[:-1, :] <= prob_spread)
        disease[k+1, :-1, :][doburnS] = 3

        # Spread to the East using logical masking

        doburnE = (disease[k, :, :-1] == 3) & (disease[k, :, 1:] == 2) & (ignite[:, 1:] <= prob_spread)
        disease[k+1, :, 1:][doburnE] = 3

        # Spread to the West using logical masking
        
        doburnW = (disease[k, :, 1:] == 3) & (disease[k, :, :-1] == 2) & (ignite[:, :-1] <= prob_spread)
        disease[k+1, :, :-1][doburnW] = 3
        
        # Identify the cells that were infected in the current iteration

        wasinfected = disease[k,:,:] == 3
        
        # Determine if infected cells become fatal (state 0)

        is_fatal_now = (disease[k, :, :] == 3) & (isfatal)

        # Update the disease state: tiles that meet the fatal condition turn into 0

        disease[k+1, is_fatal_now] = 0

        # Otherwise, the infected tiles turn back to 1 (recover + immune)

        disease[k+1, wasinfected & ~is_fatal_now] = 1
        
        # Visualize the current state

        # fig,ax = plt.subplots(1,1)
        # ax.set_title(f'Disease Status (i-step = {k+1:03d})')
        # ax.set_xlabel('X units')
        # ax.set_ylabel('Y units')
        # forest_cmap = ListedColormap(['black','lightgreen', 'darkgreen', 'crimson'])
        # ax.pcolor(disease[k+1,:,:], cmap=forest_cmap, vmin=0, vmax=3)
        # fig.savefig(f'fig{k+1:04d}.png')
        # plt.show()

        # Count the number of susceptible, infected, immune, and deceased cells
        infected_count = np.sum(disease[k + 1, :, :] == 3)
        immune_count = np.sum(disease[k + 1, :, :] == 1)
        deceased_count = np.sum(disease[k + 1, :, :] == 0)

        # Store these counts

        infected_counts.append(infected_count)
        immune_counts.append(immune_count)
        deceased_counts.append(deceased_count)

        # Quit if all diseases are out
        if infected_count == 0:
            break
 
    return infected_counts, immune_counts, deceased_counts

#------------------------------

def line_graph_fire_spread_sim():
    """
    Run simulations for different fire spread probabilities and plot a line graph
    showing the number of trees burnt over multiple iterations for each probability.
    """
    spread_probs = [0.2, 0.4, 0.6, 0.8, 1.0]  # List of different fire spread probabilities
    max_iterations = 20  # Number of iterations to run for each simulation

    # Dictionary to store the number of trees burnt for each spread probability
    results = {}

    # Run simulations for each spread probability
    for prob in spread_probs:
        # Call the simulation function and store the result
        results[prob] = full_fire_spread(nNorth=10, nEast=10, maxiter=max_iterations, prob_spread=prob)

    # Plot the results
    plt.figure(figsize=(10, 6))  # Set the size of the figure
    for prob, burns in results.items():
        # Plot the number of burning cells over iterations for each probability
        plt.plot(range(len(burns)), burns, label=f'Pspread = {prob}')

    # Label the axes and title the plot
    plt.xlabel('Iteration')
    plt.ylabel('Number of Burning Cells')
    plt.title('Fire Spread Over Time for Different Spread Probabilities')
    plt.legend()  # Add a legend to differentiate between spread probabilities
    plt.grid(True)  # Add a grid for better readability
    plt.show()  # Display the plot

def scatter_initial_density_sim():
    """
    Run simulations for different initial forest densities and plot a scatter plot
    showing the total area burned as a function of initial forest density.
    """
    bare_probs = np.linspace(0, 1, 20)  # Range of initial forest densities (from 0 to 1)
    total_burned_areas = []  # List to store the total burned areas for each density

    # Run simulations for each initial forest density
    for pbare in bare_probs:
        # Call the simulation function and store the total burned area
        total_burned = full_fire_spread(nNorth=20, nEast=20, maxiter=50, prob_spread=1.0, prob_bare=pbare)
        total_burned_areas.append(total_burned)

    # Plot the scatter plot
    plt.figure(figsize=(10, 6))  # Set the size of the figure
    plt.scatter(bare_probs, total_burned_areas, color='orange')  # Create scatter plot
    plt.xlabel('Initial Forest Density (Pbare)')  # Label x-axis
    plt.ylabel(f'Total Area Burned (km\u00b2)')  # Label y-axis
    plt.title('Effect of Forest Density on Total Area Burned')  # Title of the plot
    plt.grid(True)  # Add a grid for better readability
    plt.show()  # Display the plot

#------------------------------

def line_graph_mortality_rate_sim():
    """
    Run simulations to model disease spread while varying the disease mortality rate
    and plot line graphs showing the number of individuals in different states over time.
    """
    # Define different mortality rates (probability of death) from 0 to 1
    mortality_rates = np.linspace(0, 1, 5)  
    maxiter = 50  # Number of iterations for the simulation

    # Dictionary to store the results for each mortality rate
    results = {
        rate: full_disease_spread(
            nNorth=5, nEast=10, maxiter=maxiter, prob_spread=1.0, prob_bare=0.2, prob_start=1.0, prob_fatal=rate
        ) for rate in mortality_rates
    }

    # Create a figure for plotting
    plt.figure(figsize=(15, 10))

    # Plot results for each mortality rate
    for rate, (infected, immune, deceased) in results.items():
        plt.plot(range(len(infected)), infected, label=f'Mortality Rate: {rate:.2f} (Infected)', linestyle='-')
        plt.plot(range(len(immune)), immune, label=f'Mortality Rate: {rate:.2f} (Immune)', linestyle='--')
        plt.plot(range(len(deceased)), deceased, label=f'Mortality Rate: {rate:.2f} (Deceased)', linestyle=':')

    # Label the axes and title the plot
    plt.xlabel('Time (Iterations)')
    plt.ylabel('Number of Individuals (people)')
    plt.title('Disease Spread Dynamics for Different Mortality Rates')
    plt.legend()  # Add legend to differentiate between mortality rates
    plt.grid(True)  # Add grid for better readability
    plt.show()  # Display the plot

def bar_graph_initial_vaccination_sim():
    """
    Run simulations for different initial vaccination rates and plot a bar graph
    showing the peak infection count as a function of initial vaccination rate.
    """
    # List of different initial vaccination rates
    prob_bare_list = [0.2, 0.4, 0.6, 0.8, 1.0]

    peak_infection_counts = []  # List to store peak infection counts for each vaccination rate

    # Run simulations for each initial vaccination rate
    for prob_bare in prob_bare_list:
        # Run the disease spread simulation
        infected_counts, _, _ = full_disease_spread(
            nNorth=8, 
            nEast=10, 
            maxiter=20, 
            prob_spread=1.0, 
            prob_bare=prob_bare,
            prob_start=1.0, 
            prob_fatal=0.0
        )
        # Find and store the peak infection count
        peak_infection = max(infected_counts)
        peak_infection_counts.append(peak_infection)

    # Create a bar chart for the peak infection counts
    plt.figure(figsize=(10, 6))
    plt.bar([str(p) for p in prob_bare_list], peak_infection_counts, color='skyblue')
    plt.xlabel('Initial Vaccination Rate (proportion of immune population)')
    plt.ylabel('Peak Infection Count (people)')
    plt.title('Effect of Early Vaccination on Peak Infection Count')
    plt.show()  # Display the plot

# UNCOMMENT THIS LINE TO CREATE A 3x3 grid, 100% CHANCE OF
# SPREAD, ZERO BARE INITIAL SPOTS, AND ONLY THE CENTER CELL ON FIRE
#full_fire_spread()
#-----------------

# UNCOMMENT THIS LINE TO CREATE A 3x5 grid, 100% CHANCE OF
# SPREAD, ZERO BARE INITIAL SPOTS, AND ONLY THE CENTER CELL ON FIRE
#full_fire_spread(3,5,5)
#----------------

# UNCOMMENT THIS LINE TO CREATE A 4x6 grid, 100% CHANCE OF
# SPREAD, ZERO IMMUNE INITIAL SPOTS, AND ONLY THE CENTER CELL INFECTED
#full_disease_spread(4,6,10)
#------------------

# UNCOMMENT THIS LINE TO CREATE A 4x6 grid, 100% CHANCE OF
# SPREAD, 50% PROB OF IMMUNE INITIAL SPOTS, AND ONLY THE CENTER CELL INFECTED
#full_disease_spread(4,6,10,1.0,0.5,1.0)
#--------------------
    
##****FOR ALL THE BELOW FUNCTION CALLS PLEASE COMMENT OUT ALL THE PLOT CODE IN THE FULL
#FIRE SPREAD AND DISEASE SPREAD FUNCTIONS LINES (51-58, 97-104, 158-165, 210-217)****##
    
# UNCOMMENT THIS LINE TO CREATE A LINE GRAPH TO ANALYZE THE FIRE SPREAD OVER TIME WITH 
# VARIATION IN THE PROBABILITY OF SPREAD:
#line_graph_fire_spread_sim()
#----------------------

# UNCOMMENT THIS LINE TO CREATE A SCATTER PLOT TO ANALYZE THE FIRE SPREAD OVER TIME WITH 
# VARIATION IN THE INITIAL FOREST DENSITY (INITIAL BARE SPOTS):
#scatter_initial_density_sim()
#----------------------

# UNCOMMENT THIS LINE TO CREATE A LINE GRAPH TO ANALYZE THE DISEASE SPREAD OVER TIME WITH 
# VARIATION IN THE MORTALITY RATE (PROB OF FATALITY):
line_graph_mortality_rate_sim()
#----------------------

# UNCOMMENT THIS LINE TO CREATE A BAR GRAPH TO ANALYZE THE DISEASE SPREAD OVER TIME WITH 
# VARIATION IN THE INITIAL IMMUNE POPULATION:
#bar_graph_initial_vaccination_sim()
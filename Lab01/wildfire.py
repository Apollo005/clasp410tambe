#!/usr/bin/env python3

'''
This file contains tools and scripts for completing Lab 1 for CLaSP 410.
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
prob_bare = 0.5 # Chance of cell to start as bare patch.
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
        
    fig,ax = plt.subplots(1,1)
    forest_cmap = ListedColormap(['tan', 'darkgreen', 'crimson'])
    ax.pcolor(forest[0,:,:], cmap=forest_cmap, vmin=1, vmax=3)
    ax.set_title(f'iteration = {0:03d}')
    fig.savefig(f'fig{0:04d}.png')

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

        fig,ax = plt.subplots(1,1)
        forest_cmap = ListedColormap(['tan', 'darkgreen', 'crimson'])
        ax.pcolor(forest[k+1,:,:], cmap=forest_cmap, vmin=1, vmax=3)
        ax.set_title(f'iteration = {k+1:03d}')
        fig.savefig(f'fig{k+1:04d}.png')

        # Quit if all fires are out
        nBurn = (forest[k+1,:,:] == 3).sum()
        if nBurn == 0:
            print(f"burn completed in {k+1} steps")
            break

    return k+1

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

    # Visualize the initial state

    fig,ax = plt.subplots(1,1)
    forest_cmap = ListedColormap(['black','lightgreen', 'darkgreen', 'crimson'])
    ax.pcolor(disease[0,:,:], cmap=forest_cmap, vmin=0, vmax=3)
    ax.set_title(f'iteration = {0:03d}')
    fig.savefig(f'fig{0:04d}.png')

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

        fig,ax = plt.subplots(1,1)
        ax.set_title(f'iteration = {k+1:03d}')
        forest_cmap = ListedColormap(['black','lightgreen', 'darkgreen', 'crimson'])
        ax.pcolor(disease[k+1,:,:], cmap=forest_cmap, vmin=0, vmax=3)
        fig.savefig(f'fig{k+1:04d}.png')

        # Quit if all diseases are out

        ndiseases = (disease[k+1,:,:] == 3).sum()
        if ndiseases == 0:
            print(f"burn completed in {k+1} steps")
            break

    return k+1

def explore_burnrate():
    prob = np.arrange(0,1,.05)
    nsteps = np.zeros(prob.size)
    for i,p in enumerate(prob):
        print(f"Burning for pspread = {p}")
        #nsteps[i] = full__fire_spread()

    plt.plot(prob,nsteps)

full_fire_spread(3,5,6)
full_disease_spread(3,5,6)
#!/usr/bin/env python3

'''
This file contains tools and scripts for completing Lab 1 for CLaSP 410.
To reproduce the plots shown in the lab report, do this...
'''
import numpy as np
import matplotlib.pyplot as plt

# # 1 - barren, 2 - forest, 3 - fire
# nstep = 4 # Number of iterations
# nx, ny = 3, 3 # Number of cells in X and Y direction.


# # Create an initial grid, set all values to "2". dtype sets the value
# # type in our array to integers only.
# forest = np.zeros((nstep,nx,ny), dtype=int) + 2

# # Create an array of randomly generated numbers of range [0,1)  
# isbare = np.random.rand(nx,ny)

# # IsBare into array of T/F values
# isbare = isbare < prob_bare
# forest[0,isbare] = 1

# # Set the center cell to "burning":
# forest[0, 1, 1] = 3


# # Loop over every cell in the x and y directions.
# for k in range(nstep) :
#     for i in range(nx) :
#         for j in range(ny) :
#         # Roll our "dice" to see if we get a bare spot:
#             if np.random.rand() < prob_bare:
#                 forest[k,i,j] = 1  
            
######################

nNorth = 3
nEast = 3
maxiter = 4

prob_spread = 1.0 # Chance to spread to adjacent cells.
prob_bare = 1.0 # Chance of cell to start as bare patch.
prob_start = 1.0 # Chance of cell to start on fire.
                
def full_fire_spread(nNorth = nNorth, nEast = nEast, maxiter = maxiter, prob_spread = prob_spread,
                      prob_bare = prob_bare, prob_start = prob_start):
    #this function description, later p_bare, p_spread
    #1) create forest
    forest = np.zeros((maxiter,nNorth,nEast), dtype=int) + 2
    istart, jstart = nNorth//2, nEast//2

    isbare = np.random.rand(nNorth, nEast)
    isbare = isbare <= prob_bare
    forest[0,isbare] = 1
    #random probability to start a fire
    
    if (np.random.rand() <= prob_start and forest[0, istart, jstart] != 1):
        forest[0, istart, jstart] = 3

    fig,ax = plt.subplots(1,1)
    contour = ax.matshow(forest[0,:,:], vmin=1, vmax=3)
    ax.set_title(f'iteration = {0:03d}')
    plt.colorbar(contour, ax=ax)

    for k in range(maxiter-1) :
        ignite = np.random.rand(nNorth, nEast)

        forest[k+1,:,:] = forest[k,:,:]

       # North
        doburnN = (forest[k, :-1, :] == 3) & (forest[k, 1:, :] == 2) & (ignite[1:, :] <= prob_spread)
        forest[k+1, 1:, :][doburnN] = 3

        # South
        doburnS = (forest[k, 1:, :] == 3) & (forest[k, :-1, :] == 2) & (ignite[:-1, :] <= prob_spread)
        forest[k+1, :-1, :][doburnS] = 3

        # East
        doburnE = (forest[k, :, :-1] == 3) & (forest[k, :, 1:] == 2) & (ignite[:, 1:] <= prob_spread)
        forest[k+1, :, 1:][doburnE] = 3

        # West
        doburnW = (forest[k, :, 1:] == 3) & (forest[k, :, :-1] == 2) & (ignite[:, :-1] <= prob_spread)
        forest[k+1, :, :-1][doburnW] = 3

        wasburn = forest[k,:,:] == 3
        forest[k+1, wasburn] = 1

        fig,ax = plt.subplots(1,1)
        contour = ax.matshow(forest[k+1,:,:], vmin=1, vmax=3)
        ax.set_title(f'iteration = {k+1:03d}')
        plt.colorbar(contour, ax=ax)

        fig.savefig(f'fig{k:04d}.png')
        plt.show()
        plt.close('all')

        #quit once all fires are out
        nBurn = (forest[k+1,:,:] == 3).sum()
        if nBurn == 0:
            print(f"burn completed in {k+1} steps")
            break

    return k+1

def explore_burnrate():
    prob = np.arrange(0,1,.05)
    nsteps = np.zeros(prob.size)
    for i,p in enumerate(prob):
        print(f"Burning for pspread = {p}")
        nsteps[i] = full_fire_spread()

    plt.plot(prob,nsteps)

full_fire_spread(3,5,6)
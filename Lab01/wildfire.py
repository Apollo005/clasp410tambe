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
# prob_spread = 1.0 # Chance to spread to adjacent cells.
# prob_bare = 0.0 # Chance of cell to start as bare patch.
# prob_start = 0.0 # Chance of cell to start on fire.

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

                
def fire_spread(nNorth = nNorth, nEast = nEast, maxiter = maxiter):
    #this function description, later p_bare, p_spread
    #1) create forest
    forest = np.zeros((maxiter,nNorth,nEast), dtype=int) + 2
    forest[0, 1, 1] = 3

    fig,ax = plt.subplots(1,1)
    contour = ax.matshow(forest[0,:,:], vmin=1, vmax=3)
    ax.set_title(f'iteration = {0:03d}')
    plt.colorbar(contour, ax=ax)

    for k in range(maxiter-1) :

        forest[k+1,:,:] = forest[k,:,:]

        for i in range(nNorth):
            for j in range(nEast):
                if forest[k, i, j] == 3:
                    if i < nNorth - 1 and forest[k, i + 1, j] == 2:
                        forest[k+1, i + 1, j] = 3
                    if i > 0 and forest[k, i - 1, j] == 2:
                        forest[k+1, i - 1, j] = 3
                    if j < nEast - 1 and forest[k, i, j + 1] == 2:
                        forest[k+1, i, j + 1] = 3
                    if j > 0 and forest[k, i, j - 1] == 2:
                        forest[k+1, i, j - 1] = 3

        wasburn = forest[k,:,:] == 3
        forest[k+1, wasburn] = 1

        fig,ax = plt.subplots(1,1)
        contour = ax.matshow(forest[k+1,:,:], vmin=1, vmax=3)
        ax.set_title(f'iteration = {k+1:03d}')
        plt.colorbar(contour, ax=ax)
        plt.show()

fire_spread(3,3,4)
#!/usr/bin/env python3

'''
This file produces the solution to the coffee problem which is really important to everyone.

To use this file....

This file solves Newton's law of cooling for ...
'''

import numpy as np

# Define constants for our problem

k = 1/300 # Coefficient of cooling in units of 1/s
T_init, T_env = 90, 20 # Initial and Environmental in units of C.

def solve_temp(time, k=k, T_init=T_init, T_env=T_env):
    '''    
    Given an array of times in seconds, calculate temperature using Newton's Law of Cooling.
    '''

    return T_env + (T_init - T_env) * np.exp(-k*time)

print(solve_temp(np.array([0,10])))
    

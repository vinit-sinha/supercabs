#from Env import CabDriver
# Import routines

import numpy as np
import math
import random

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
    
        self.state_space = [(loc,time,day) for loc in range(1,m+1) for time in range(0,t) for day in range(0,d)]
        self.action_space = [(start, end) 
                             for start in range(1,m+1) 
                             for end in range(1,m+1) if start != end or start != 0 or end != 0]
        self.state_init = random.choice(self.state_space)
        self.average_requests = {
            1:2,
            2:12,
            3:4,
            4:7,
            5:8
        }
        
        # Constants
        self.LOCATION_INDEX = 0
        self.TIME_INDEX = 1
        self.DAY_INDEX = 2
        
        self.PICKUP_INDEX = 0
        self.DROP_INDEX = 1
        
        # Start the first round
        self.reset()
    

    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. 
        This method converts a given state into a vector format. 
        Hint: The vector is of size m + t + d."""
        
        state_encod = [0 for _ in range(m+t+d)]
        state_encod[state[self.LOCATION_INDEX]] = 1
        state_encod[m+state[self.TIME_INDEX]] = 1
        state_encod[m+t+state[self.DAY_INDEX]] = 1

        return state_encod


    # Use this function if you are using architecture-2 
    def state_encod_arch2(self, state, action):
        """convert the (state-action) into a vector so that it can be fed to the NN. 
        This method converts a given state-action pair into a vector format. 
        Hint: The vector is of size m + t + d + m + m."""

        state_encod = [0 for _ in range(m+t+d+m+m)]
        state_encod[state[self.LOCATION_INDEX]] = 1
        state_encod[m+state[self.TIME_INDEX]] = 1
        state_encod[m+t+state[self.DAY_INDEX]] = 1
        
        if (action[self.PICKUP_INDEX] != 0):
            state_encod[m+t+d+action[self.PICKUP_INDEX]] = 1
        if (action[self.DROP_INDEX] != 0):
            state_encod[m+t+d+m+action[self.DROP_INDEX]] = 1
        
        return state_encod


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[self.LOCATION_INDEX]
        requests = np.random.poisson(self.average_requests[location])

        if requests >15:
            requests =15

        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests) # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_index]
        
        actions.append([0,0])

        return possible_actions_index,actions   

    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        return reward

    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        return next_state

    def reset(self):
        return self.action_space, self.state_space, self.state_init



#from Env import CabDriver
# Import routines

import math
import random
import numpy as np

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self, time_matrix):
        """initialise your state and define your action space and state space"""
    
        self.state_space = [(loc,time,day) for loc in range(0,m) for time in range(0,t) for day in range(0,d)]
        self.action_space = [(start, end) 
                             for start in range(0,m) 
                             for end in range(0,m) if start != 0 or end != 0]
        self.state_init = random.choice(self.state_space)
        self.average_requests = {
            0:2,
            1:12,
            2:4,
            3:7,
            4:8
        }
        
        self.time_matrix = time_matrix
        self.chargable_time = 0
        self.non_chargable_time = 0
        
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

        possible_actions_index = random.sample(range(0, (m-1)*m +1), requests) # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_index]
        
        actions.append((0,0))
        possible_actions_index.append(0)
        
        return possible_actions_index,actions   

    def _is_offline(self, pickup_loc, drop_loc):
        return pickup_loc == 0 and drop_loc == 0
    
    def _already_at_pickup(self, curr_loc, pickup_loc):
        return curr_loc == pickup_loc
    
    def _travel(self, from_loc, to_loc, curr_time, curr_day):
        time_taken = self.time_matrix[from_loc][to_loc][curr_time][curr_day]
        new_day = int(( curr_day + ( time_taken  % 24 ) ) % 7)
        new_time = int(( curr_day + time_taken ) // 24)
        return time_taken, [to_loc, new_time, new_day]
    
    def reward_func(self, state, action):
        """Takes in state, action and Time-matrix and returns the reward"""
        revenue_generating_time = self.chargable_time
        total_time = self.non_chargable_time + self.chargable_time
        revenue = R * revenue_generating_time
        cost = C * total_time
        return revenue - cost, total_time
            
    def next_state_func(self, state, action):
        """Takes state and action as input and returns next state"""
        
        curr_loc, curr_time, curr_day = state 
        pickup_loc, drop_loc = action
        
       
        if self._is_offline(pickup_loc, drop_loc):
            wait_time, next_state = self._travel(curr_loc, curr_loc, curr_time, curr_day)
            transit_time = 0
            ride_time = 0
            
        elif self._already_at_pickup(curr_loc, pickup_loc):
            wait_time = 0
            transit_time = 0
            ride_time, next_state = self._travel(curr_loc, drop_loc, curr_time, curr_day)
            
        else: # Cab driver is neither offline nor already at pickup. 
            wait_time = 0
            transit_time, [curr_loc, curr_time, curr_day] = self._travel(curr_loc, pickup_loc, curr_time, curr_day)
            ride_time, next_state = self._travel(pickup_loc, drop_loc, curr_time, curr_day)
        
        self.non_chargable_time = wait_time + transit_time
        self.chargable_time = ride_time
        
        
        return next_state

    def reset(self):
        self.non_chargable_time = 0
        self.chargable_time = 0
        return self.action_space, self.state_space, self.state_init



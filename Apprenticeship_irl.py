"""
Implements Apprenticeship Learning via IRL (Abbeel & Ng, 2004)

Ce Ju, 2018
juce.sysu@gmail.com
"""

import numpy as np
from . import value_iteration
import time 

def compute_feature_expectations(feature_matrix, trajectories):
    '''
    feature expectations comes from the formula (4) on Abbeel & Ng's paper.
    '''
    feature_expectations = np.zeros(feature_matrix.shape[1])
    for trajectory in trajectories:
    	for state, _, _ in trajectory:
    		feature_expectations += feature_matrix[state]
    feature_expectations /= trajectories.shape[0]

    return feature_expectations
    

def irl(gridworld, n_trajectories, learning_rate, epsilon = 0.1):
    '''
    Generate the expert's trajectories according to the optimal policy
    and compute the feature expectations. 
    '''   


    time_start = time.time()
    feature_matrix = gridworld.feature_matrix()
    n_states, d_states = feature_matrix.shape
    trajectory_length = 3*gridworld.grid_size
    expert_trajectories = gridworld.generate_trajectories(n_trajectories, trajectory_length, gridworld.optimal_policy)
    expert_feature_expectations = compute_feature_expectations(feature_matrix, expert_trajectories)
    
    ''' 
    Pick the initial policy
    generate the trajectories according to the initial policy
    compute the feature expectations
    '''                      
    np.random.seed(10)
    w = np.random.uniform(size=(d_states,))
    R = feature_matrix.dot(w)
    Q_table = value_iteration.find_policy(gridworld.n_states, gridworld.n_actions, gridworld.transition_probability, R, gridworld.discount, stochastic = True)  
    updated_policy_trajectories = gridworld.new_generate_trajectories(n_trajectories, trajectory_length, Q_table)
    feature_expectations_bar = compute_feature_expectations(feature_matrix, updated_policy_trajectories)
    #print('Initial weight: \n', w.reshape((n_states,)))
    

    '''
    while loop for policy iteration: In this loop, we apply the computation trik in section 3.1 of 
    Ng & Abeel's paper. E.g. the projection margin method.
    '''
    w = expert_feature_expectations - feature_expectations_bar
    t = np.linalg.norm(w, 2)
    print("Initial threthod: ", t)
    i = 0

    while t > epsilon:
        
    	R = feature_matrix.dot(w)
    	Q_table = value_iteration.find_policy(gridworld.n_states, gridworld.n_actions, gridworld.transition_probability, R, gridworld.discount)
    	updated_policy_trajectories = gridworld.new_generate_trajectories(n_trajectories, trajectory_length, Q_table)
    	feature_expectations = compute_feature_expectations(feature_matrix, updated_policy_trajectories)
    	updated_loss = feature_expectations-feature_expectations_bar
    	feature_expectations_bar += updated_loss*updated_loss.dot(w)/np.square(updated_loss).sum()
    	w = expert_feature_expectations-feature_expectations_bar
    	t = np.linalg.norm(w, 2)
    	i += 1
    	#print distance t every 100 iterations. 
    	if i % 100 == 0: print('The '+ str(i) +'th threthod is '+ str(t)+'.')

    time_elasped = time.time() -  time_start
    print('Total Apprenticeship computational time is /n', time_elasped)
       

    return feature_matrix.dot(w).reshape((n_states,))






"""
Implements Apprenticeship Learning via IRL (Abbeel & Ng, 2004)

Ce Ju, 2018
juce.sysu@gmail.com
"""

import numpy as np
import time 

def optimal_value(n_states, n_actions, transition_probabilities, reward,
                  discount, threshold=1e-2):
    
    v = np.zeros(n_states)

    diff = float("inf")
    while diff > threshold:
        diff = 0
        for s in range(n_states):
            max_v = float("-inf")
            for a in range(n_actions):
                tp = transition_probabilities[s, a, :]
                max_v = max(max_v, np.dot(tp, reward + discount*v))

            new_diff = abs(v[s] - max_v)
            if new_diff > diff:
                diff = new_diff
            v[s] = max_v

    return v

def find_policy(n_states, n_actions, transition_probabilities, reward, discount,
                threshold=1e-2, v=None, stochastic=True):
 
    if v is None:
        v = optimal_value(n_states, n_actions, transition_probabilities, reward,
                          discount, threshold)

    if stochastic:
        # Get Q using equation 9.2 from Ziebart's thesis.
        Q = np.zeros((n_states, n_actions))
        for i in range(n_states):
            for j in range(n_actions):
                p = transition_probabilities[i, j, :]
                Q[i, j] = p.dot(reward + discount*v)
        Q -= Q.max(axis=1).reshape((n_states, 1))  # For numerical stability.
        Q = np.exp(Q)/np.exp(Q).sum(axis=1).reshape((n_states, 1))
        return Q

    def _policy(s):
        return max(range(n_actions),
                   key=lambda a: sum(transition_probabilities[s, a, k] *
                                     (reward[k] + discount * v[k])
                                     for k in range(n_states)))
    policy = np.array([_policy(s) for s in range(n_states)])
    return policy

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






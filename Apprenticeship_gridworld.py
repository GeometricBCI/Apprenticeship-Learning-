"""
Run Apprenticeship Learning via IRL on the gridworld MDP.

Ce Ju, 2018
juce.sysu@gmail.com
"""

import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt

import irl.Apprenticeship_irl as Apprenticeship
import irl.mdp.gridworld as gridworld
import time


def main(grid_size, discount, n_trajectories, learning_rate):

	wind = 0.3
	gw = gridworld.Gridworld(grid_size, wind, discount)

	ground_r = np.array([gw.reward(s) for s in range(gw.n_states)])
	r = Apprenticeship.irl(gw, n_trajectories, learning_rate)

	plt.subplot(1, 2, 1)
	plt.pcolor(ground_r.reshape((grid_size, grid_size)))
	plt.colorbar()
	plt.title("Groundtruth reward")
	plt.subplot(1, 2, 2)
	plt.pcolor(r.reshape((grid_size, grid_size)))
	plt.colorbar()
	plt.title("Recovered reward")
	plt.show()



if __name__ == '__main__':
    main(3, 0.001, 20, 0.01)

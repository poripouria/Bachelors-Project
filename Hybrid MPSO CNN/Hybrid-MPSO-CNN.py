"""
Description:
    Hybrid MPSO-CNN: Multi-level Particle Swarm optimized hyperparameters of Convolutional Neural Network
    Using this algorithm for tuning hyper-parameters of deep unfolding network
"""

import random

# Range of hyperparameters (Based on Table 1)
search_space = {
    'nC':   (1, 5),     # Number of convolutional layers
    'nP':   (1, 5),     # Number of pooling layers
    'nF':   (1, 5),     # Number of fully connected layers
    'c_nf': (1, 64),    # Number of filters
    'c_fs': (1, 13),    # Filter Size (odd) 
    'c_pp': (0, 1),     # Padding pixels (0: "valid", 1: "same")
    'c_ss': (1, 5),     # Stride Size (< 𝑐_fs)
    'p_fs': (1, 13),    # Filter Size (odd)
    'p_ss': (1, 5),     # Stride Size
    'p_pp': (0, 1),     # Padding pixels (< 𝑝_fs) (0: "valid", 1: "same")
    'op':   (1, 1024)   # Number of neurons
}

# Values of parameters (Based on Table 2)
values = {
    'c1': 2,  # Social coefficient
    'c2': 2,  # Cognitive coefficient
    'omega': 
}

class Particle_Swarm_L1:
    def __init__(self, search_space):
        self.search_space = search_space
        

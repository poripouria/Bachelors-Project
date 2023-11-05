"""
Description:
    Implementation of "Hybrid MPSO-CNN: Multi-level Particle Swarm optimized hyperparameters of Convolutional Neural Network"
    doi: https://doi.org/10.1016/j.swevo.2021.100863
    Using this algorithm for tuning hyper-parameters of deep unfolding network
"""

import random
import math
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

search_space = { """ Range of hyperparameters (Based on Table 1) """
    'nC':   (1, 5),         # Number of convolutional layers
    'nP':   (1, 5),         # Number of pooling layers
    'nF':   (1, 5),         # Number of fully connected layers
    'c_nf': (1, 64),        # Number of filters
    'c_fs': (1, 13),        # Filter Size (odd) 
    'c_pp': (0, 1),         # Padding pixels (0: "valid", 1: "same")
    'c_ss': (1, 5),         # Stride Size (< 𝑐_fs)
    'p_fs': (1, 13),        # Filter Size (odd)
    'p_ss': (1, 5),         # Stride Size
    'p_pp': (0, 1),         # Padding pixels (< 𝑝_fs) (0: "valid", 1: "same")
    'op':   (1, 1024)       # Number of neurons
}

def calculate_omega(t, t_max, alpha=0.2):
    if t < alpha * t_max:
        return 0.9
    return 1 / (1 + math.e ** ((10 * t - t_max) / t_max))

class Hybrid_MPSO_CNN:
    def __init__(PSL1, PSL2):
        self.c1 = 2,                                # Social coefficient
        self.c2 = 2,                                # Cognitive coefficient
        self.omega = calculate_omega(t, t_max),     # Inertia weight (𝜔)
        self.r1 = random.uniform(0,1),              # Random binary variable
        self.r2 = random.uniform(0,1),              # Random binary variable
        self.swarm_size_lvl1 = 5*3,                 # Swarm size at Swarm Level-1 (nP≤ nC, nF ≤ nC)
        self.swarm_size_lvl2 = 5*PSL1.nC*8,          # Swarm size at Swarm Level-2
        self.max_iter_lvl1 = random.randint(5,8),   # Maximum iterations at Swarm Level-1
        self.max_iter_lvl2 = 5                      # Maximum iterations at Swarm Level-2


class Particle_Swarm_L1:
    def __init__(self, search_space):
        self.search_space = search_space
        self.nC = randint(search_space['nC'])
        self.nP = randint(search_space['nP'])
        self.nF = randint(search_space['nF'])
        

class Particle_Swarm_L2:
    def __init__(self, search_space):
        self.search_space = search_space
        self.c_nf = randint(search_space['c_nf'])
        self.c_fs = randint(search_space['c_fs'])
        self.c_pp = randint(search_space['c_pp'])
        self.c_ss = randint(search_space['c_ss'])
        self.p_fs = randint(search_space['p_fs'])
        self.p_ss = randint(search_space['p_ss'])
        self.p_pp = randint(search_space['p_pp'])
        self.op = randint(search_space['op'])


class CNN:
    def __init__(self, hyperparameters):
        self.hyperparameters = hyperparameters
        self.nC = hyperparameters['nC']        # number of convolution layers
        self.nP = hyperparameters['nP']        # number of pooling layers
        self.nF = hyperparameters['nF']        # number of fully connected layers
        self.c_nf = hyperparameters['c_nf']    # number of filters in the convolutional layer
        self.c_fs = hyperparameters['c_fs']    # size of filter/kernel in the convolutional layer
        self.c_pp = hyperparameters['c_pp']    # padding (valid or same) requirement in the convolutional layer
        self.c_ss = hyperparameters['c_ss']    # size of stride in the convolutional layer
        self.p_fs = hyperparameters['p_fs']    # size of a filter in the max pooling layer
        self.p_ss = hyperparameters['p_ss']    # size of stride in the max-pooling layer
        self.p_pp = hyperparameters['p_pp']    # padding pixels in pooling layer
        self.op = hyperparameters['op']        # number of output neurons in the fully connected layer

        self.model = Sequential()

    def buid_model(self):
        # Add convolution and pooling layers according to the hyperparameters
        for i in range(self.nC):
            self.model.add(layers.Conv2D(filters = self.c_nf[i], 
                                         kernel_size = self.c_fs[i], 
                                         strides = self.c_ss[i], 
                                         padding = self.c_pp[i], 
                                         activation = 'relu'))
            if i < self.nP:
                self.model.add(layers.MaxPooling2D(pool_size = self.p_fs[i], 
                                                   strides = self.p_ss[i], 
                                                   padding = self.p_pp[i]))
        # Flatten the output before adding fully connected layers
        self.model.add(layers.Flatten())
        # Add fully connected layers according to the hyperparameters
        for j in range(self.nF):
            self.model.add(layers.Dense(units = self.op[j], 
                                   activation = 'relu'))
                      
        self.model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

        def fitness(self, x_test, y_test):
            test_loss, test_acc = self.model.evaluate(x_test, y_test)
            return test_acc


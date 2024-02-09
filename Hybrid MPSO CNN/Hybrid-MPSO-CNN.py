"""
Description:
    Implementation of "Hybrid MPSO-CNN: Multi-level Particle Swarm optimized hyperparameters of Convolutional Neural Network"
    doi: https://doi.org/10.1016/j.swevo.2021.100863
    Using this algorithm for tuning hyper-parameters of deep unfolding network
"""

import random
import math
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

class Particle_Swarm_L1:
    def __init__(self, search_space):
        self.nC = random.randint(search_space['nC'])
        self.nP = random.randint(search_space['nP'], self.nC)
        self.nF = random.randint(search_space['nF'], self.nC)

        self.pos_i = [self.nC, self.nP, self.nF]        # Particle position 
        self.vel_i = [0, 0, 0]                          # Particle velocity
        
        self.pbest_i = self.pos_i                       # Personal best position
        self.gbest_i = None                             # Global best position

        self.F_pbest_i = float("inf")                   # Personal best fitness
        self.F_gbest_i = float("inf")                   # Global best fitness
        self.F_i = None                                 # Current fitness

    def evaluate(self, cnn):
        """Evaluate particle fitness using CNN"""
        cnn.build_model(self.pos_i)
        self.F_i = cnn.fitness()
        
        # Check personal best
        if self.F_i < self.F_pbest_i:
            self.pbest_i = self.pos_i
        # Check global best
        if self.gbest_i is None or self.F_i < self.F_gbest_i:
            self.gbest_i = self.pos_i

    def update_velocity(self, c1, c2, r1, r2):
        """Update particle velocity"""
        w = 0.9
        for i in range(len(self.vel_i)):
            self.vel_i[i] = w*self.vel_i[i] + c1*r1*(self.pbest_i[i] - self.pos_i[i]) + c2*r2*(self.gbest_i[i] - self.pos_i[i])
                             
    def update_position(self, bounds):
        """Update particle position within bounds"""
        for i in range(len(self.pos_i)):
            self.pos_i[i] += self.vel_i[i]
            self.pos_i[i] = max(bounds[i][0], min(self.pos_i[i], bounds[i][1]))
        

class Particle_Swarm_L2:
    def __init__(self, search_space):
        self.c_nf = random.randint(search_space['c_nf'])
        self.c_fs = random.randint(search_space['c_fs'])
        self.c_pp = random.randint(search_space['c_pp'])
        self.c_ss = random.randint(search_space['c_ss'])
        self.p_fs = random.randint(search_space['p_fs'])
        self.p_ss = random.randint(search_space['p_ss'])
        self.p_pp = random.randint(search_space['p_pp'])
        self.op   = random.randint(search_space['op'])

        self.pos_ij = [self.c_nf, self.c_fs, self.c_pp, self.c_ss,
                       self.p_fs, self.p_ss, self.p_pp, self.op]        # Particle position
        self.vel_ij = [0] * len(self.pos_ij)                            # Particle velocity 
        
        self.pbest_ij = self.pos_ij                                     # Personal best position
        self.gbest_ij = None                                            # Global best position 
        
        self.F_pbest_ij = float("inf")                                  # Personal best fitness 
        self.F_gbest_ij = float("inf")                                  # Global best fitness
        self.F_ij = None                                                # Current fitness

    def evaluate(self, cnn):
        """Evaluate particle fitness using CNN"""
        cnn.build_model(self.pos_ij)  
        self.F_ij = cnn.fitness()

        # Check personal best
        if self.F_ij < self.F_pbest_ij:
            self.pbest_ij = self.pos_ij
            self.F_pbest_ij = self.F_ij
        # Check global best
        if self.F_ij < self.F_gbest_ij:
            self.gbest_ij = self.pos_ij
            self.F_gbest_ij = self.F_ij

    def update_velocity(self, c1, c2, r1, r2, w):
        """Update particle velocity"""
        for i in range(len(self.vel_ij)):
            self.vel_ij[i] = w*self.vel_ij[i] + c1*r1*(self.pbest_ij[i] - self.pos_ij[i]) + c2*r2*(self.gbest_ij[i] - self.pos_ij[i])
                             
    def update_position(self, bounds):
        """Update particle position within bounds"""
        for i in range(len(self.pos_ij)):
            self.pos_ij[i] += self.vel_ij[i]
            self.pos_ij[i] = max(bounds[i][0], min(self.pos_ij[i], bounds[i][1]))


class Hybrid_MPSO_CNN:
    def __init__(self, PSL1, PSL2):
        self.c1 = 2,                                        # Social coefficient
        self.c2 = 2,                                        # Cognitive coefficient
        self.omega = 0.9,                                   # Inertia weight (𝜔)
        self.r1 = random.uniform(0,1),                      # Random binary variable
        self.r2 = random.uniform(0,1),                      # Random binary variable
        self.swarm_size_lvl1 = 5*3,                         # Swarm size at Swarm Level-1 (nP≤ nC, nF ≤ nC)
        self.swarm_size_lvl2 = 5*PSL1.nC*8,                 # Swarm size at Swarm Level-2
        self.max_iter_lvl1 = random.random.randint(5,8),    # Maximum iterations at Swarm Level-1
        self.max_iter_lvl2 = 5
        self.search_space = { """ Range of hyperparameters (Based on Table 1) """
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
        
        self.swarm_lvl1 = [PSL1 for _ in range(self.swarm_size_lvl1)]
        self.swarm_lvl2 = [[] for _ in range(self.swarm_size_lvl2)]
        self.gbest = None

    def level1_optimize(self):
        
        for i in range(self.max_iter_lvl1):
            w = self.calculate_inertia(i, self.max_iter_lvl1)

            for j in range(self.swarm_size_lvl1):
                gbest_ij = self.level2_optimize(self.swarm_lvl1[j])

                self.swarm_lvl1[j].evaluate(gbest_ij, CNN)
                self.swarm_lvl1[j].update_velocity(w, self.c1, self.c2, self.r1, self.r2) 
                self.swarm_lvl1[j].update_position(self.search_space)

                if self.swarm_lvl1[j].F_i < self.gbest.F_i:
                    self.gbest = self.swarm_lvl1[j]

        return self.gbest
        
        
    def level2_optimize(self, particle_l1):

        for _ in range(self.swarm_size_lvl2):
            particle_l2 = Particle_Swarm_L2(self.search_space)
            self.swarm_lvl2[particle_l1].append(particle_l2)

        for i in range(self.max_iter_lvl2):
            w = self.calculate_inertia(i, self.max_iter_lvl2)
            
            for particle in self.swarm_lvl2[particle_l1]:
                particle.evaluate(CNN) 

                particle.update_velocity(w, self.c1, self.c2, self.r1, self.r2)
                particle.update_position(self.search_space)

            gbest_ij = min(self.swarm_lvl2[particle_l1], key=lambda x: x.F_ij)

        return gbest_ij   

    def calculate_omega(t, t_max, alpha=0.2):
        if t < alpha * t_max:
            return 0.9
        return 1 / (1 + math.e ** ((10 * t - t_max) / t_max))   # Maximum iterations at Swarm Level-2


class CNN:
    def __init__(self, hyperparameters):
        self.hyperparameters = hyperparameters
        self.nC = hyperparameters['nC']             # number of convolution layers
        self.nP = hyperparameters['nP']             # number of pooling layers
        self.nF = hyperparameters['nF']             # number of fully connected layers
        self.c_nf = hyperparameters['c_nf']         # number of filters in the convolutional layer
        self.c_fs = hyperparameters['c_fs']         # size of filter/kernel in the convolutional layer
        self.c_pp = hyperparameters['c_pp']         # padding (valid or same) requirement in the convolutional layer
        self.c_ss = hyperparameters['c_ss']         # size of stride in the convolutional layer
        self.p_fs = hyperparameters['p_fs']         # size of a filter in the max pooling layer
        self.p_ss = hyperparameters['p_ss']         # size of stride in the max-pooling layer
        self.p_pp = hyperparameters['p_pp']         # padding pixels in pooling layer
        self.op = hyperparameters['op']             # number of output neurons in the fully connected layer

        self.model = Sequential()

    def buid_model(self):
        for i in range(self.nC):
            self.model.add(keras.layers.Conv2D(filters = self.c_nf[i], 
                                         kernel_size = self.c_fs[i], 
                                         strides = self.c_ss[i], 
                                         padding = self.c_pp[i], 
                                         activation = 'relu'))
            if i < self.nP:
                self.model.add(keras.layers.MaxPooling2D(pool_size = self.p_fs[i], 
                                                   strides = self.p_ss[i], 
                                                   padding = self.p_pp[i]))
                                                   
        self.model.add(keras.layers.Flatten())
        
        for j in range(self.nF):
            self.model.add(keras.layers.Dense(units = self.op[j], 
                                        activation = 'relu'))
                      
        self.model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    def fitness(self, x_test, y_test):
        test_loss, test_acc = self.model.evaluate(x_test, y_test)
        return test_acc

    def __repr__(self):
        return str(self.model.summary())

    def __str__(self):
        model_summary = (f"nC: {self.nC}\n"
                        f"nP: {self.nP}\n"
                        f"nF: {self.nF}\n"
                        f"c_nf: {self.c_nf}\n"
                        f"c_fs: {self.c_fs}\n"
                        f"c_pp: {self.c_pp}\n"
                        f"c_ss: {self.c_ss}\n"
                        f"p_fs: {self.p_fs}\n"
                        f"p_ss: {self.p_ss}\n"
                        f"p_pp: {self.p_pp}\n"
                        f"op: {self.op}\n")
        return model_summary, str(self.model.summary())

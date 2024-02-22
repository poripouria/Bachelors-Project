"""
Description:
    Implementation of "Hybrid MPSO-CNN: Multi-level Particle Swarm optimized hyperparameters of Convolutional Neural Network"
    doi: https://doi.org/10.1016/j.swevo.2021.100863
    Using this algorithm for tuning hyper-parameters of deep unfolding network
"""

import random
import math
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

class Particle_L1:
    def __init__(self, search_space):
        self.nC = random.randint(search_space['nC'])
        self.nP = random.randint(search_space['nP'], self.nC)
        self.nF = random.randint(search_space['nF'], self.nC)

        self.pos_i = [self.nC, self.nP, self.nF]        # Particle position 
        self.vel_i = [0, 0, 0]                          # Particle velocity
        
        self.pbest_i = self.pos_i                       # Personal best position
        self.gbest_i = None                             # Global best position
        self.F_i = -float("inf")                        # Current fitness

        self.swarm_size_lvl2 = 5*self.nC*8              # Swarm size at Swarm Level-2
        self.swarm_lvl2 = [Particle_L2 for _ in range(self.swarm_size_lvl2)]

    def evaluate(self):
        """Evaluate particle fitness by """
        F_i = 
        
        # Check personal best

    def update_velocity(self, w, c1, c2):
        """Update particle velocity"""
        r1 = random.uniform(0,1)
        r2 = random.uniform(0,1)
        for i in range(len(self.vel_i)):
            self.vel_i[i] = w*self.vel_i[i] + c1*r1*(self.pbest_i[i] - self.pos_i[i]) + c2*r2*(self.gbest_i[i] - self.pos_i[i])

    def update_position(self, bounds):
        """Update particle position within bounds"""
        for i in range(len(self.pos_i)):
            self.pos_i[i] += self.vel_i[i]
            self.pos_i[i] = max(bounds[i][0], min(self.pos_i[i], bounds[i][1]))
        
class Particle_L2:
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
        self.F_ij = -float("inf")                                       # Current fitness

    def evaluate(self, cnn, x_train, y_train):
        """Evaluate particle fitness using CNN"""
        _, F_ij = cnn.fitness(x_train, y_train)

        if F_ij > self.F_ij:
            self.F_ij = F_ij
            self.pbest_ij = self.pos_ij.copy()   

    def update_velocity(self, w, c1, c2):
        """Update particle velocity"""
        r1 = random.uniform(0,1)
        r2 = random.uniform(0,1)
        for i in range(len(self.vel_ij)):
            self.vel_ij[i] = w*self.vel_ij[i] + c1*r1*(self.pbest_ij[i] - self.pos_ij[i]) + c2*r2*(self.gbest_ij[i] - self.pos_ij[i])
                             
    def update_position(self, bounds):
        """Update particle position within bounds"""
        for i in range(len(self.pos_ij)):
            self.pos_ij[i] += self.vel_ij[i]
            self.pos_ij[i] = max(bounds[i][0], min(self.pos_ij[i], bounds[i][1]))


class Hybrid_MPSO_CNN:
    def __init__(self):
        self.c1 = 2                                         # Social coefficient
        self.c2 = 2                                         # Cognitive coefficient
        self.m = 5                                          # m
        self.n = 5                                          # n
        self.max_iter_lvl1 = random.randint(5,8)            # Maximum iterations at Swarm Level-1
        self.max_iter_lvl2 = 5                              # Maximum iterations at Swarm Level-2
        self.swarm_size_lvl1 = 5*3                          # Swarm size at Swarm Level-1 (nP ≤ nC, nF ≤ nC)
        self.swarm_lvl1 = [Particle_L1 for _ in range(self.swarm_size_lvl1)]
        self.search_space = {                               """ Range of hyperparameters (Based on Table 1) """
                                'nC':   (1, 5),             # Number of convolutional layers
                                'nP':   (1, 5),             # Number of pooling layers
                                'nF':   (1, 5),             # Number of fully connected layers
                                'c_nf': (1, 64),            # Number of filters
                                'c_fs': (1, 13),            # Filter Size (odd) 
                                'c_pp': (0, 1),             # Padding pixels (0: "valid", 1: "same")
                                'c_ss': (1, 5),             # Stride Size (< 𝑐_fs)
                                'p_fs': (1, 13),            # Filter Size (odd)
                                'p_ss': (1, 5),             # Stride Size
                                'p_pp': (0, 1),             # Padding pixels (< 𝑝_fs) (0: "valid", 1: "same")
                                'op':   (1, 1024)           # Number of neurons
                            }
        

    def calculate_omega(t, t_max, alpha=0.2):
        """Calculate inertia weight"""
        if t < alpha * t_max:
            return 0.9
        return 1 / (1 + math.e ** ((10 * t - t_max) / t_max))

    def level1_optimize(self):
        """Optimize hyperparameters at Swarm Level-1"""
        sl1_gbest = max(self.swarm_lvl1, key=lambda x: x.F_i)
        for t in range(self.max_iter_lvl1):
            w = self.calculate_omega(t, self.max_iter_lvl1)
            for i, particle_l1 in enumerate(self.swarm_lvl1):
                gbest_i, fitness_i = self.level2_optimize(particle_l1, w)
                if fitness_i > particle_l1.F_i:
                    particle_l1.F_i = fitness_i
                    particle_l1.pbest_i = particle_l1.pos_i.copy()
                    if fitness_i > sl1_gbest.F_i:
                        sl1_gbest = particle_l1.pos_i.copy()
                
        return sl1_gbest
        
    def level2_optimize(self, particle_l1, w):
        """Optimize hyperparameters at Swarm Level-2"""
        for t in range(self.max_iter_lvl2):
            for i, particle_L2 in enumerate(particle_l1.swarm_lvl2):
                particle_L2.update_velocity(w, self.c1, self.c2)
                particle_L2.update_position()
                particle_L2.evaluate()
                if particle_L2.F_ij > particle_L2.F_ij:
                    particle_L2.F_ij = particle_L2.F_ij
                    particle_L2.pbest_ij = particle_L2.pos_ij.copy()
                    if particle_L2.F_ij > particle_L2.gbest_ij:
                        particle_L2.gbest_ij = particle_L2.pos_ij.copy()
        
        return particle_L2.gbest_ij, particle_L2.F_ij
    
    def run(self):
        """Run the algorithm"""
        gbest, cnn = self.level1_optimize()
        return gbest, cnn

class CNN:
    def __init__(self, hyperparameters):
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

    def buid_model(self, input_shape, output_shape):
        for i in range(self.nC):
            self.model.add(Conv2D(filters = self.c_nf[i], 
                                  kernel_size = self.c_fs[i], 
                                  strides = self.c_ss[i], 
                                  padding = self.c_pp[i], 
                                  activation = 'relu'),
                                  input_shape = input_shape if i == 0 else None)
            if i < self.nP:
                self.model.add(MaxPooling2D(pool_size = self.p_fs[i], 
                                            strides = self.p_ss[i], 
                                            padding = self.p_pp[i]))
            
                                                   
        self.model.add(Flatten())
        
        for i in range(self.nF):
            self.model.add(Dense(units = self.op[i] if i < self.nF-1 else output_shape, 
                                 activation = 'relu' if i < self.nF-1 else 'softmax'))
        
        return self.model

    def train_model(self, x_train, y_train, batch_size=32, epochs=20):
        self.model.compile(optimizer='adam', 
                           loss='mse', 
                           metrics=['accuracy'])
        
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    def fitness(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)

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
        return model_summary

"""
Description:
    Implementation of "Deep Unfolding of a Proximal Interior Point Method for Image Restoration"
    doi: https://doi.org/10.48550/arXiv.1812.04276
"""

import numpy as np
from scipy import signal
from tensorflow import keras

class UnfoldingBlock(keras.layers.Layer):
    def __init__(self, H, lambda_init, mu_init, gamma_init):
        super().__init__()
        self.H = H 
        self.lambda_var = keras.backend.variable(lambda_init)
        self.mu_var = keras.backend.variable(mu_init)
        self.gamma_var = keras.backend.variable(gamma_init)
        
    def proximal_op(self, x):
        # Implements barrier proximity operator
        return x  

    def call(self, inputs):  
        lambda_var = keras.backend.exp(self.lambda_var) 
        mu = keras.backend.exp(self.mu_var)
        gamma = keras.backend.exp(self.gamma_var)
        
        grad = self.H.T @ (self.H @ inputs - self.y) + lambda_var*self.D.T@self.D@inputs
        
        prox_input = inputs - gamma*grad
        output = self.proximal_op(prox_input, mu, gamma)
        
        return output

def iRestNet(y, H, D, K):
    x = keras.layers.Input(shape=y.shape[0])  
    
    # Stack K unfolding blocks
    for k in range(K):
        block = UnfoldingBlock(H, lambda_init, mu_init, gamma_init) 
        x = block(x)
        
    model = keras.Model(inputs=x, outputs=x)
    
    return model
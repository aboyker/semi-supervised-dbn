# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 16:08:59 2018

@author: Alexandre Boyker
"""
from rbm import RBM
import os
import numpy as np

class DBN():
    
    """
    Deep belief network implementation.
    
    A deep belief network is composed of stacked Restricted Boltzmann machines.
    This class is actually a wrapper of the RBM class.
    
    
    """
    def __init__(self, n_epochs=50,layers_size = [784, 128, 128,128, 10],learning_rates=[0.01,0.01,0.01,0.01], batch_size = 200, model_name ="my_dbn", embedding_dimensions=None):

        self.model_stack = []
        self.input_size = layers_size[0]
        self.embedding_sizes = layers_size[1:]
        self.n_hidden_layers = len(self.embedding_sizes)
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rates = learning_rates
        self.model_directory = os.path.join("saved_dbn_models", model_name)
        self.model_stack.append(RBM(input_size=self.input_size , embedding_size = self.embedding_sizes[0], batch_size = self.batch_size, learning_rate = 0.1, n_epochs = self.n_epochs, model_id = 0, saved_model_directory=self.model_directory ))
        
        for rbm_index in range(self.n_hidden_layers-1):
            
            rbm_i = RBM(input_size=self.embedding_sizes[rbm_index] , embedding_size = self.embedding_sizes[rbm_index+1 ], batch_size = self.batch_size, learning_rate = self.learning_rates[rbm_index], n_epochs = self.n_epochs, model_id = rbm_index+1 , saved_model_directory=self.model_directory )
            self.model_stack.append(rbm_i)
            
           
    def train(self, X_train):
        
        """
        Train the DBN. This can be donc by training each RBM, starting from the
        bottom.
        
        positional argument:
            
            -- X_train: numpy ndarray of training data. X_train.shape[1] is expected to be the input_size of the 
            bottom RBM
        
        """
        for rbm in self.model_stack:
          
            rbm.train(X_train)
            _, X_train = rbm.predict(X_train)
            
    def predict(self, X_predict):
        
        """
        Returns two lists containing the 'synthetic' input for each layer as well as the embedding layers.
        The first element of each list corresponds to the bottom layer, and the last to the top layers.
        
        positional argument:
            
            -- X_predict: numpy ndarray of training data. X_predict.shape[1] is expected to be the input_size of the 
            bottom RBM
        
        """
        h_list = []
        synthetic_V_list = []
        
        for rbm in self.model_stack:
          
            synthetic_v, X_predict = rbm.predict(X_predict)
            h_list.append(X_predict)
            synthetic_V_list.append(synthetic_v)

        return synthetic_V_list, h_list
        

    
    def get_weights(self):
        
        """
        
        Returns a dict containing the weights associated to each layer as numpy ndarrays, with an 
        explicit naming
        
        """
        weights_dict = {}
        
        for layer in os.listdir(self.model_directory):
            
            W = np.load(os.path.join(self.model_directory, layer, "W.npy"))
            b = np.load(os.path.join(self.model_directory, layer, "b.npy"))
            a = np.load(os.path.join(self.model_directory, layer, "a.npy"))
            
            weights_dict[layer] = {}
            weights_dict[layer]["W"] = W
            weights_dict[layer]["a"] = a
            weights_dict[layer]["b"] = b
            
        return weights_dict
    
    
    
    
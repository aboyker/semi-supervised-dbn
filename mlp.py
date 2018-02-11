# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 16:13:04 2018

@author: Alexandre Boyker
"""
from helper import  plot_confusion_matrix
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt #for plotting


class MLP(object):
    
    """
    This class implements a simple multi-layer perceptron.
    
    """
    
    def __init__(self, input_size= 784, n_epochs=50, layers_size = [784,128, 128,128, 10], learning_rate=.01, batch_size = 200, initial_weights=None, initial_bias=None, model_name ="my_mlp", embedding_dimensions=None):
        
        self.layers_size = layers_size
        self.n_hidden_layers = len(layers_size) - 1
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.model_directory = os.path.join(os.getcwd(),model_name)
        
        if not os.path.exists(self.model_directory):
            os.makedirs(self.model_directory)
            
        self.initial_weights = initial_weights
        self.initial_bias = initial_bias
        
        if self.initial_weights is None:
            
            self.initial_weights = ['default' for i in range(self.n_hidden_layers) ]
            
        if self.initial_bias is None:
            
            self.initial_bias = ['default' for i in range(self.n_hidden_layers) ]
        
 
    def _get_weight(self, input_size, output_size, ini_weight="default", name="weight"):
        """
        Returns a weight variable of size (input_size * output_size)
        
        positional arguments:
            
            -- input_size: int, dimension of the input
            
            -- output_size: int, dimension of the output
            
        keyword argument:
            
            -- ini_weight (default value: 'default'): initial weight for the variable matrix
            
            if default, each element of the matrix is sampled from a Standard Gaussian distribution
            
            -- name: str, name of the tensor
        
        """
        #if ini_weight == 'default':
        if isinstance(ini_weight, str):
            
             ini_weight = tf.random_normal([input_size, output_size], stddev=0.1)
             
        weight = tf.Variable(ini_weight, name=name)
      
        return weight
    
    def _get_bias(self, input_size, ini_bias= "default", name="bias"):
        """
        Returns a bias variable
        
        positional arguments:
            
            -- input_size: int, dimension of the input
            
            
        keyword argument:
            
            -- ini_bias (default value: 'default'): initial value for the bias vector
            
            if default, each element of the matrix is sampled from a Standard Gaussian distribution
            
            -- name: str, name of the tensor
        """
        #if ini_bias == 'default':
        if isinstance(ini_bias, str):
            
            ini_bias = tf.random_normal([input_size], stddev=.1)
            
        bias = tf.Variable(ini_bias, name=name)
        
        return bias
    
    def _get_fully_connected_layer(self, input_tensor, W, b):
        
        """
        
        Returns a fully connected layer, with a Relu activation gate
        
        positional arguments:
            
            -- input_tensor: tensor object of size (p * q)
            -- W: weight tensor of size (q * r)
            -- b: bias term of size (p * r)
            
        """
        
        return tf.nn.relu(tf.add(tf.matmul(input_tensor, W), b))
        
        
    
    def _build_model(self):
        
        """
        
        Builds the tensorflow computational graph and returns relevant objects for training and prediction
        
        """
        variable_list = [self._get_weight( self.layers_size[i],  self.layers_size[i+1], self.initial_weights[i], name="weight_"+str(i)) for i in range(self.n_hidden_layers)]  
        bias_list = [self._get_bias( self.layers_size[i], self.initial_bias[i-1], name="bias_"+str(i)) for i in range(1,self.n_hidden_layers+1)]  
        
        X_plac = tf.placeholder(tf.float32, [None, self.layers_size[0]], name="X_plac")
        y_plac = tf.placeholder(tf.float32, [None, self.layers_size[-1]], name="X_plac")
        
        hidden_layer = self._get_fully_connected_layer(X_plac, variable_list[0], bias_list[0])
        
        for i in range(1, self.n_hidden_layers-1):
            
             hidden_layer = self._get_fully_connected_layer(hidden_layer, variable_list[i], bias_list[i])
        
        logits =tf.add(tf.matmul(hidden_layer, variable_list[-1]), bias_list[-1])
        y_pred = tf.argmax(logits, axis=1)
        
        with tf.name_scope('cross_entropy'):
            
             cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_plac, logits=logits))
             
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        loss = optimizer.minimize(cross_entropy)
        return X_plac, y_plac, cross_entropy, loss, y_pred
      
   
    def train(self, X_train, y_train, X_val, y_val):
        
        """
        
        Train the MLP
        
        positional arguments:
            
            -- X_train, X_val: numpy ndarray of input samples
            
            -- y_train, y_val: nu^mpy ndarray of labels, one-hot-encoded
            
        """
        X_plac, y_plac,cross_entropy, loss,y_pred = self._build_model()
        
        init = tf.global_variables_initializer()

        
        with tf.Session() as sess:

            sess.run(init)
            
            for i in range(self.n_epochs):
                
                for batch_number in range(0, X_train.shape[0], self.batch_size):
                    batch_X = X_train[batch_number:batch_number + self.batch_size]
                    batch_y = y_train[batch_number:batch_number + self.batch_size]

                    ce, predi, _ = sess.run([cross_entropy,y_pred, loss], feed_dict={X_plac:batch_X, y_plac:batch_y})

            predi_val = sess.run(y_pred , feed_dict={X_plac:X_val, y_plac: y_val})

            plot_confusion_matrix(np.argmax(y_val, axis=1),predi_val, range(self.layers_size[-1]))
            plt.show()
            
            

            
            

            
                    
    


    
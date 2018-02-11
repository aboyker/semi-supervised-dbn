# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 10:46:09 2018

@author: Alexandre Boyker
"""
from helper import get_MNIST_data 
from tf_helper import reset_graph
from dbn import DBN
from mlp import MLP
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("-bt", "--train_bool", dest="train_bool",
                    help=" training for dbn boolean, default=False", type=bool, default=False)

args = parser.parse_args()
train_bool = args.train_bool




def main():
    
    reset_graph()
    #get MNIST dataset, validation set is 10% of total samples (each class is equally represented)
    X_train, X_val, y_train, y_val = get_MNIST_data()
    # parameters for Deep Belief Network training
    # We use layers of size 784 - 500 - 500 - 2000, as they are known to work best
    param = {'batch_size':100,'n_epochs':25, 'model_name':"dbn_MNIST", 'layers_size': [X_train.shape[1] ,500, 500, 2000]}
    dbn = DBN(**param)
    # train the dbn
    if train_bool:
        
        dbn.train(X_train)
        
    # get the weights of the DBN
    weights_dict = dbn.get_weights()
    initial_weights = [weights_dict["hidden_layer_0"]["W"], weights_dict["hidden_layer_1"]["W"], weights_dict["hidden_layer_2"]["W"], "default"]
    initial_bias = [weights_dict["hidden_layer_0"]["b"], weights_dict["hidden_layer_1"]["b"], weights_dict["hidden_layer_2"]["b"], "default"]
    
    
    # we split the validation set of MNIST 2% for training and 98 % for validation
    X_train, X_val, y_train, y_val = train_test_split(X_val, y_val, test_size=0.98, random_state=23)
    
    # MLP trainng with DBN weights for initialization
    print("\n\n MLP trained with DBN weights")
    param_dict = {'n_epochs':400,'layers_size' :[784, 500, 500, 2000, 10],'initial_bias':initial_bias, 'initial_weights':initial_weights, 'model_name':'mlp_dbn_ini'}
    mlp = MLP(**param_dict)
    

    mlp.train(X_train, y_train, X_val, y_val)
    
    # MLP training with random Gaussian weights for initialization
    print("\n\nMLP trained with random standard Gaussian weights")

    param_dict = {'n_epochs':400,'layers_size' :[784, 500, 500, 2000, 10],'initial_bias':None, 'initial_weights':None, 'model_name':'mlp_random_ini'}
    mlp = MLP(**param_dict)
    mlp.train(X_train, y_train, X_val, y_val)

if __name__ == '__main__':
    
    main()
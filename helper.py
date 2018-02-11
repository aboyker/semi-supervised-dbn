# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 13:28:03 2018

@author: Alexandre Boyker

Miscallenaous methods and classes

"""
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools


def plot_confusion_matrix(y_test, y_pred, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = confusion_matrix(np.array(y_test), np.array(y_pred))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      
    print("accuracy: {}".format(accuracy_score(y_test, y_pred)))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



def get_MNIST_data():
    
    data = pd.read_csv(os.path.join("data","train.csv"))
    X_train = (data.ix[:,1:].values).astype('float32') # all pixel values
    y_train = data.ix[:,0].values.astype('int32') # only labels i.e targets digits   
    b = np.zeros((X_train.shape[0],10))
    b[np.arange(X_train.shape[0]), y_train] = 1
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, b, test_size = 0.1, random_state=42)
    
    
    return     X_train/255.0, X_val/255.0, Y_train, Y_val 
#
#class MbtiParser(object):
#    
#    def __init__(self):
#        
#        pass
#    
#    def get_label(self,x):
#        if x=='ISTJ':
#            return np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
#        if x=='ISTP':
#            return np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
#        if x=='ESTP':
#            return np.array([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0])
#        if x=='ESTJ':
#            return np.array([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0])
#        if x=='ISFJ':
#            return np.array([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0])
#        if x=='ISFP':
#            return np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0])
#        if x=='ESFP':
#            return np.array([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0])
#        if x=='ESFJ':
#            return np.array([0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0])
#        if x=='INFJ':
#            return np.array([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0])
#        if x=='INFP':
#            return np.array([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0])
#        if x=='ENFP':
#            return np.array([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0])
#        if x=='ENFJ':
#            return np.array([0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0])
#        if x=='INTJ':
#            return np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0])
#        if x=='INTP':
#            return np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0])
#        if x=='ENTP':
#            return np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0])
#        if x=='ENTJ':
#            return np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])
#        
#    def parse_post(self, line):
#        row = ((line.split(",")))
#        #label = np.array([1,0]) if row[0][0]=='I' else np.array([0,1])
#        row = ' '.join(row[1:])
#        return re.sub(r'\W+', ' ', row)
#    
#    def get_MBTI_data(self, max_features=10000):
#        
#        vectorizer = CountVectorizer(stop_words='english', max_features=max_features)
#        data = pd.read_csv(os.path.join("data","mbti_1.csv"))
#        data.posts = data.posts.apply((self.parse_post))
#        X = vectorizer.fit_transform(data.posts.values)
#        X_train, X_val, Y_train, Y_val = train_test_split(X, data.type.values, test_size = 0.1, random_state=42)
#        X_train[X_train>0] = 1
#        X_val[X_val>0] = 1
#        Y_train = np.array([self.get_label(x) for x in Y_train])
#        Y_val = np.array([self.get_label(x) for x in Y_val])
#        return X_train.todense(), X_val.todense(), Y_train, Y_val
#     
#
#
#class StsaParser(object):
#    def __init__(self):
#        pass
#    def transform_label_to_numeric(self, y):
#            if '1' in y:
#                return np.array([1,0])
#            else:
#                return np.array([0,1])
#
#
#    def parse_line(self, row):
#        
#        row = row.split(' ')
#        text = (' '.join(row[1:]))
#        label = self.transform_label_to_numeric(row[0])
#        return (re.sub(r'\W+', ' ', text), label) 
#    
#    def get_data(self, max_features=1000):
#        X_train = []
#        y_train = []
#        X_val = []
#        y_val = []
#        
#        train = open(os.path.join("data","stsa","train","stsa-train.txt"), "r").read().split('\n')
#        for line in train:
#            x, y = self.parse_line(line)
#            X_train.append(x)
#            y_train.append(y)
#            
#        valid = open(os.path.join("data","stsa","test","stsa-test.txt"), "r").read().split('\n')
#        for line in valid:
#            x, y = self.parse_line(line)
#            X_val.append(x)
#            y_val.append(y)
#        
#        vectorizer = CountVectorizer(stop_words='english', max_features=max_features)
#        X = vectorizer.fit_transform(X_train+X_val)
#        X_train = X[:len(y_train)]
#        X_val = X[len(y_train):]
#        
#        return X_train.todense(), np.array(y_train), X_val.todense(), np.array(y_val)
#    
#   

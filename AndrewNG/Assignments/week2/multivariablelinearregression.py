# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 18:16:19 2017

@author: Mahe
"""

import numpy as np
from matplotlib import pyplot as plt

data=np.loadtxt('ex1data2.txt',delimiter=',')
X=data[:,:2]
Y=data[:,-1]

training_x=X[0:40,:]
training_y=Y[0:40]

test_x=X[41:46,:]
test_y=Y[41:46]

num_rows,num_coloumns=training_x.shape;

for i in range(num_coloumns):
    mean=np.mean(training_x[:,i])
    max=np.max(training_x[:,i])
    min=np.min(training_x[:,i])
    training_x[:,i]=training_x[:,i]-mean
    training_x[:,i]=training_x[:,i]/(max-min)
    test_x[:,i]=test_x[:,i]-mean
    test_x[:,i]=test_x[:,i]/(max-min)
mlr=multiplelinearregression()
mlr.train(training_x,training_y,40,2)
Y_predicted=mlr.predict(test_x,test_y,5,2)

class multiplelinearregression(object):
    def __init__(self):
        self.W=None
    def predict(self,X,Y,n_r,n_c):
        T=np.zeros((n_r,))
        L=np.zeros((n_r,n_c+1))
        L[:,0]=1
        for i in range(n_c):
            L[:,i+1]=X[:,i]
        for c in range(n_r):
            T[c]=self.W.dot(L.T[:,c])
        c=0
        for c,i in enumerate(Y):
            if(T[c] is Y[c]):
                c=c+1
        print(c)
        return T
    
    def train(self,train_x,train_y,n_r,n_c,alpha=0.01):
        self.W=np.random.randn(1,n_c+1)
        X=np.zeros((n_r,n_c+1))
        X[:,0]=1;
        for i in range(n_c):
            X[:,i+1]=train_x[:,i]
        Y=train_y
        for i in range(5000):
            temp=np.zeros((1,n_c+1))
            cost=0
            for c in range(n_r):
                cost=cost+(self.W.dot(X.T[:,c])-Y[c])**2
            cost=cost/(2*n_r)
            print(cost)
            for c in range(n_r):
                for k in range(n_c+1):
                    temp[0,k]=temp[0,k]+(self.W.dot(X.T[:,c])-Y[c])*X[c,k]
            self.W=self.W-(alpha/n_r)*temp
        print(self.W)

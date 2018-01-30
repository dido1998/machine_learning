# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 02:29:35 2017

@author: Mahe
"""

import numpy as np
from matplotlib import pyplot as plt

data=np.loadtxt('ex1data1.txt',delimiter=',')

X=data[:,0]
Y=data[:,1]
x_mean=np.mean(X)
x_max=np.max(X)
x_min=np.min(X)
X=X-x_mean
X=X/(x_max-x_min)
training_x=X[0:80]
training_y=Y[0:80]

validation_x=X[60:80]
validation_y=Y[60:80]

test_x=X[81:96]
test_y=Y[81:96]
lr=onevariableregression()
lr.train(training_x,training_y,80)
lr.predict(test_x,test_y,15)
lr.predict(validation_x,validation_y,20)
plt.scatter(test_x,test_y,color='red')
plt.plot(test_x,lr.predict(test_x,test_y,15),color='blue')
plt.show()

class onevariableregression(object):
    def __init__(self):
        self.W=None
    def predict(self,X,Y,n):
        T=np.zeros((n,))
        L=np.zeros((n,2))
        L[:,0]=1
        L[:,1]=X
        for c,i in enumerate(X):
            T[c]=self.W.dot(L.T[:,c])
        c=0
        for c,i in enumerate(Y):
            if(T[c] is Y[c]):
                c=c+1
        print(c)
        return T
    def train(self,train_x,train_y,n,alpha=0.01):
        self.W=np.random.randn(1,2)
        X=np.zeros((n,2))
        X[:,0]=1
        X[:,1]=train_x
        Y=train_y
        cost=0
        for i in range(5000):
            temp=np.zeros((1,2))
            for c,j in enumerate(X):
                cost=cost+(self.W.dot(X.T[:,c])-Y[c])**2
            cost=cost/(2*n)
            for c,j in enumerate(X):
                #print(self.W.dot(X.T[:,c])-Y[c])
                temp[0,0]=temp[0,0]+(self.W.dot(X.T[:,c])-Y[c])[0]
                #print((self.W.dot(X.T[:,c])-Y[c])*X[c,1])
                temp[0,1]=temp[0,1]+((self.W.dot(X.T[:,c])-Y[c])*X[c,1])
            self.W=self.W-(0.01/n)*temp
            print(cost)
        print(self.W)
        

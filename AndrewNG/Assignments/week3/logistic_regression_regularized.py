# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 19:29:30 2017

@author: Mahe
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 18:16:19 2017

@author: Mahe
"""

import numpy as np
from matplotlib import pyplot as plt

data=np.loadtxt('ex2data2.txt',delimiter=',')

X=data[:,:2]
Y=data[:,-1]
x1=np.copy(X)


num_rows,num_coloumns=X.shape;
new_x=np.zeros((num_rows,27))
c=0
new_x[:,:2]=X
for i in range(5):
    for j in range(i+3):
        new_x[:,c+2]=(X[:,0]**j)*(X[:,1]**(i+3-j))
        c=c+1
n_rows,n_cols=new_x.shape        

for i in range(n_cols):
    mean=np.mean(new_x[:,i])
    max=np.max(new_x[:,i])
    min=np.min(new_x[:,i])
    new_x[:,i]=new_x[:,i]-mean
    new_x[:,i]=new_x[:,i]/(max-min)

from sklearn.cross_validation import train_test_split
new_x_train, new_x_test_1, new_y_train, new_y_test = train_test_split(new_x, Y, test_size = 0.15, random_state = 7)


lr1=logistic_regression()


lr1.train(new_x_train,new_y_train)


Y_predicted_1=lr1.predict(new_x_test_1)
y_train_predict=lr1.predict(new_x_train)
c=0
for i in range(18):
    if Y_predicted_1[i]==new_y_test[i]:
        c=c+1
print(c)

    

class logistic_regression(object):
    def __init__(self):
        self.W=None
    def predict(self,X):
        n_r,n_c=X.shape
        T=np.zeros((n_r,))
        L=np.zeros((n_r,n_c+1))
        L[:,0]=1
        for i in range(n_c):
            L[:,i+1]=X[:,i]
        for c in range(n_r):
            T[c]=self.sigmoid(L.T[:,c])
            #print(T[c])
            if T[c]>0.50:
                T[c]=1
            else:
                T[c]=0
            
           
                    
      
        return T
    def sigmoid(self,X):
        h=1/(1+np.exp(-self.W.dot(X)))
        return h 
    def train(self,train_x,train_y,alpha=1.2):
        n_r,n_c=np.shape(train_x)
        self.W=np.random.randn(1,n_c+1)
        X=np.zeros((n_r,n_c+1))
        X[:,0]=1;
        for i in range(n_c):
            X[:,i+1]=train_x[:,i]
        Y=train_y
        for i in range(5000):
            temp=np.zeros((1,n_c+1))
            cost=0
            reg=0
            for c in range(n_c+1):
                reg=reg+self.W[0,c]**2
            for c in range(n_r):
                cost=cost-(Y[c]*(np.log(self.sigmoid(X.T[:,c])))+(1-Y[c])*np.log(1-self.sigmoid(X.T[:,c])))
            cost=cost/(n_r)
            print(cost)
            for c in range(n_r):
                for k in range(n_c+ 1):
                    temp[0,k]=temp[0,k]+(self.sigmoid(X.T[:,c])-Y[c])*X[c,k]
            self.W=self.W-(alpha/n_r)*temp-(0.001/n_r)*self.W
        print(self.W)

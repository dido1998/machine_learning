# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import scipy.io
data=scipy.io.loadmat('ex3data1.mat')
Raw_X=data['X']
Raw_Y=data['y']
A=np.hstack((Raw_X,Raw_Y))
np.random.shuffle(A)
X=A[:,:399]
Y=A[:,400]
for i in range(5000):
    if Y[i] == 10:
        Y[i]=0
new_Y=np.zeros((5000,10))
for i in range(10):
    print(i)
    indices=(Y==i)
    new_Y[:,i]=indices
train_x=X[:4000,:]
train_y=new_Y[:4000,:]
test_x=X[4001:,:]
test_y=new_Y[4001:,:]

lr=[]
l=classifier()
l.train(train_x,train_y[:,0])
for i in range(10):
    l=classifier()
    l.train(train_x,train_y[:,i])
    lr.append(l)
train_predict_y=np.zeros((4000,10))
for i in range(10):
    train_predict_y[:,i]=lr[i].predict(train_x)

test_predict_y=np.zeros((999,10))

for i in range(10):
    test_predict_y[:,i]=lr[i].predict(test_x)
y1=np.argmax(test_predict_y,axis=1)
y2=np.argmax(test_y,axis=1)
c=0
for i in range(999):
    if(y1[i]==y2[i]):
        c=c+1
print(c)    

"""E=X[9]
im_1=np.resize(E,(20,20))

cv2.imshow('image',im_1)
cv2.resizeWindow('image', 100,100)
cv2.waitKey(0)
cv2.destroyAllWindows()"""


class classifier(object):
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
            print(T[c])
            
                
        return T
        

    
    def sigmoid(self,X):
        h=1/(1+np.exp(-self.W.dot(X)))
        return h 
    def train(self,train_x,train_y,alpha=1.5):
        n_r,n_c=train_x.shape
        self.W = np.random.randn(1, n_c + 1)
        X1 = np.ones((n_r,  1))
        X=np.hstack((X1,train_x))

        Y = np.resize(train_y,(4000,1))
        for i in range(500):
            sigs=np.zeros((n_r,1))
            temp = np.zeros((1, n_c + 1))
            cost = 0
            for c in range(n_r):
                #print(c)
               
                sigs[c,0]=self.sigmoid(X.T[:,c])
                #print(sigs[c,0])
                cost = cost - (Y[c] * (np.log(self.sigmoid(X.T[:, c]))) + (1 - Y[c]) * np.log(1 - self.sigmoid(X.T[:, c])))
            cost = cost / (n_r)

            print(cost)

            temp=np.dot(np.subtract(sigs,Y).T,X)
            self.W = self.W - (alpha / n_r) * temp
        print(self.W)
    
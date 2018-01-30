import numpy as np
import scipy.io
data=scipy.io.loadmat('ex3data1.mat')
Raw_X=data['X']
Raw_Y=data['y']
weights=scipy.io.loadmat('ex3weights.mat')
theta1=weights['Theta1']
theta2=weights['Theta2']
layer1=np.zeros((5000,25))
for i in range(25):
    l=classifier(theta1[i,:])
    layer1[:,i]=l.predict(Raw_X)
layer2=np.zeros((5000,10))
for i in range(10):
    l=classifier(theta2[i,:])
    layer2[:,i]=l.predict(layer1)
y1=np.argmax(layer2,axis=1)
for c in range(5000):
    if y1[c]==Raw_Y[c,0]:
        c=c+1
    
    
    

class classifier(object):
    def __init__(self,Theta):
        self.W=Theta
    def predict(self,X):
        n_r,n_c=X.shape
        T=np.zeros((n_r,))
        L1=np.ones((n_r,1))
        L=np.hstack((L1,X))
            
        for c in range(n_r):
            T[c]=self.sigmoid(L.T[:,c])
            print(T[c])
            
                
        return T
        

    
    def sigmoid(self,X):
        h=1/(1+np.exp(-self.W.dot(X)))
        return h 
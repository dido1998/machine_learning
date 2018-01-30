import numpy as np
import scipy.io
data=scipy.io.loadmat('ex4data1.mat')
Raw_X=data['X']
Raw_Y=data['y']
q=Raw_Y==10
Raw_Y[q]=0
shuffle=np.hstack((Raw_X,Raw_Y))
np.random.shuffle(shuffle)
X=shuffle[:,:400]
Y=shuffle[:,-1]
new_Y=np.zeros((5000,10))
for i in range(10):
    print(i)
    indices=(Y==i)
    new_Y[:,i]=indices
train_x=X[:4000,:]
train_y=new_Y[:4000,:]
test_x=X[4001:,:]
test_y=new_Y[4001:,:]
n1=network()
n1.train(train_x,train_y,25,10)
predicted_y=n1.predict(test_x,25,10)
y1=np.argmax(predicted_y,axis=1)
y2=np.argmax(test_y,axis=1)
c=0
for i in range(999):
    if(y1[i]==y2[i]):
        c=c+1
print(c)    



class network(object):
    def __init__(self):
        self.W=None
    def predict(self,X,u,c):
        n_r,n_c=X.shape
        a1=np.ones((n_r,n_c+1))
        a1[:,1:]=X
        a2=np.ones((n_r,u+1))
        a3=np.zeros((n_r,c))
                       
        l2=classifier(self.W[0])
        a2[:,1:]=l2.predict(a1,u)
        l3=classifier(self.W[1])
        a3=l3.predict(a2,c)
        return a3
        

    def train(self,X,Y,u,c):
        n_r,n_c=X.shape
        self.W=[]
        theta=np.random.randn(u,n_c+1)
        self.W.append(theta)

        theta1=np.random.randn(c,u+1)
        self.W.append(theta1)
        for i in range(2000):
            a1=np.ones((n_r,n_c+1))
            a1[:,1:]=X
            a2=np.ones((n_r,u+1))
            a3=np.zeros((n_r,c))
                        
            l2=classifier(self.W[0])
            a2[:,1:]=l2.predict(a1,u)
            l3=classifier(self.W[1])
            a3=l3.predict(a2,c)
            
            a4=np.copy(a3)
            a5=-Y*np.log(a4)-(1-Y)*np.log(1-a4)
            a6=np.sum(a5,axis=1)
            cost=np.sum(a6)/4000
            print(i,":",cost)
            
            rho2=a3-Y
           
            rho1=np.multiply(np.dot(rho2,self.W[-1]),np.multiply(a2,1-a2))
            
          
            delta2=np.dot(rho2.T,a2)
            delta1=np.dot(rho1[:,1:].T,a1)
            
            self.W[0]=self.W[0]-(1.5/n_r)*delta1
            self.W[1]=self.W[1]-(1.5/n_r)*delta2
            
                
class classifier(object):
    def __init__(self,theta):
        self.W=theta
    def predict(self,X,h):
        n_r,n_c=X.shape
        T=np.zeros((n_r,h))
        
        for i in range(h):
            for c in range(n_r):
                #print(self.sigmoid(X.T[:,c]))
                T[c,i]=self.sigmoid(X.T[:,c],i)
                #print(T[c])
            
                
        return T
        
    def sigmoid(self,X,i):
        h=1/(1+np.exp(-self.W[i,:].dot(X)))
        return h 
   


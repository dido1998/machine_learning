import numpy as np
import scipy.io
import matplotlib.pyplot as plt
data=scipy.io.loadmat('ex5data1.mat')
Raw_X=data['X']
Raw_Y=data['y']
Test_X=data['Xtest']
Test_y=data['ytest']
Val_X=data['Xval']
Val_y=data['yval']
X=np.zeros((12,8))
for i in range(1,9):
    a=np.zeros((12,1))
    a[:,0]=Raw_X[:,0]**i
    a_mean=np.mean(a)
    a-=a_mean
    a_max=np.max(a)
    a_min=np.min(a)
    a=a/(a_max-a_min)
    X[:,i-1]=a[:,0]
V_X=np.zeros((21,8))
for j in range(1,9):
    a=np.zeros((21,1))
    a[:,0]=Val_X[:,0]**j
    a_mean=np.mean(a)
    a-=a_mean
    a_max=np.max(a)
    a_min=np.min(a)
    a=a/(a_max-a_min)
    
    V_X[:,j-1]=a[:,0]
        

m1=model()
m1.train(X[:2],Raw_Y[:2])
pred_Raw_m,err1=m1.predict(X[:2],Raw_Y[:2])
_,err1_c=m1.predict(V_X[:2],Val_y[:2])

m2=model()
m2.train(X[:4],Raw_Y[:4])
pred_Raw_m,err2=m2.predict(X[:4],Raw_Y[:4])
_,err2_c=m2.predict(V_X[:4],Val_y[:4])

m3=model()
m3.train(X[:6],Raw_Y[:6])
pred_Raw_m,err3=m3.predict(X[:6],Raw_Y[:6])
_,err3_c=m3.predict(V_X[:6],Val_y[:6])

m4=model()
m4.train(X[:8],Raw_Y[:8])
pred_Raw_m,err4=m4.predict(X[:8],Raw_Y[:8])
_,err4_c=m4.predict(V_X[:8],Val_y[:8])

m5=model()
m5.train(X[:10],Raw_Y[:10])
pred_Raw_m,err5=m5.predict(X[:10],Raw_Y[:10])
_,err5_c=m5.predict(V_X[:10],Val_y[:10])

m6=model()
m6.train(X[:12],Raw_Y[:12])
pred_Raw_m,err6=m6.predict(X[:12],Raw_Y[:12])
_,err6_c=m6.predict(V_X[:12],Val_y[:12])

x_axis=np.array([2,4,6,8,10,12])
errs=np.array([err1,err2,err3,err4,err5,err6])
errs_c=np.array([err1_c,err2_c,err3_c,err4_c,err5_c,err6_c])

plt.plot(x_axis,errs,color='blue')
plt.plot(x_axis,errs_c,color='red')
plt.show()

plt.scatter(Val_X,Val_y,color='red')
plt.scatter(Val_X,m6.predict(V_X,Val_y)[0],color='blue')
plt.show()
T_X=np.zeros((21,8))
for j in range(1,9):
    a=np.zeros((21,1))
    a[:,0]=Test_X[:,0]**j
    a_mean=np.mean(a)
    a-=a_mean
    a_max=np.max(a)
    a_min=np.min(a)
    a=a/(a_max-a_min)
    
    T_X[:,j-1]=a[:,0]
        


pred_test,err_test=m6.predict(T_X,Test_y)
plt.scatter(Test_X,Test_y,color='red')
plt.scatter(Test_X,pred_test,color='blue')
plt.show()



class model(object):
    def __init__(self):
        self.W=None
    def predict(self,x,y):
        r,c=x.shape
        o=np.ones((r,1),dtype='float64')
        x_final=np.hstack((o,x))
        pred_y=np.dot(x_final,self.W.T)
        l1=pred_y-y
        l1_sq=l1**2
        l1_sum=np.sum(l1_sq)
        error=l1_sum/(2*r)
        return pred_y,error
    def train(self,x,y,learning_rate=0.00095,reg=0.001):
        r,c=x.shape        
        self.W=np.ones((1,c+1),dtype='float64')

        o=np.ones((r,1),dtype='float64')
        x_train=np.hstack((o,x))

        for i in range(10000):
            pred_y=np.dot(x_train,self.W.T)
              
            l1=pred_y-y
            
            l12=l1**2

            l1_sum=np.sum(l12)
        
            w_sq=self.W**2
           
            l2=np.sum(w_sq)
            loss=l1_sum/(2*r)+(reg/(2*r))*l2
            print loss
            
            dW=np.dot(x_train.T,l1)
            self.W-=(learning_rate/r)*(dW.T) -(reg/r)*self.W
        #print self.W


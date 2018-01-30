# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
        return dict
    
    
A1=unpickle('data_batch_1')
A2=unpickle('data_batch_2')
A3=unpickle('data_batch_3')
A4=unpickle('data_batch_4')
A5=unpickle('data_batch_5')
AT=unpickle('test_batch')
B=unpickle('batches.meta')
stringlabels=B[b'label_names']
print(stringlabels)
import numpy as np
import cv2
    
data1=np.asarray(A1[b'data'],dtype='float64')
data2=np.asarray(A2[b'data'],dtype='float64')
data3=np.asarray(A3[b'data'],dtype='float64')
data4=np.asarray(A4[b'data'],dtype='float64')
data5=np.asarray(A5[b'data'],dtype='float64')
Test_data=np.asarray(AT[b'data'],dtype='float64')
label1=np.resize(np.asarray(A1[b'labels'],dtype='int64'),(10000,1))
label2=np.resize(np.asarray(A2[b'labels'],dtype='int64'),(10000,1))
label3=np.resize(np.asarray(A3[b'labels'],dtype='int64'),(10000,1))
label4=np.resize(np.asarray(A4[b'labels'],dtype='int64'),(10000,1))
label5=np.resize(np.asarray(A5[b'labels'],dtype='int64'),(10000,1))
label_test=np.resize(np.asarray(AT[b'labels'],dtype='int64'),(10000,1))
    
    
images=np.vstack((data1,data2,data3,data4,data5))
labels=np.vstack((label1,label2,label3,label4,label5))
data=np.hstack((images,labels))
np.random.shuffle(data)
images=data[:,:3072]
labels=data[:,-1]
    
    
a3=np.reshape(Test_data,(3,32,32,-1))
a4=np.transpose(a3,(1,2,0,3))
Test_data=a4    
    labels=np.reshape(labels,(50000,1))
    labels=np.array(labels,'int64')
    ob=model()
    ob.train(images,labels)
    preds=ob.predict(Test_data,label_test)
    c=0
    for i in range(10000):
        if np.argmax(preds[i,:])==label_test[i,0]:
            
            c=c+1
    print c
class model(object):
    def __init__(self):
        self.W1=None
        self.W2=None
        self.mean=None
        self.learning_rate=0.0000005
        self.reg=0.0001

    def predict(self,x,y):
        r,c=x.shape
        one=np.ones((r,1),dtype='float64')
        x=x-self.mean
        x=np.hstack((x,one))
        h1=np.dot(x,self.W1.T)
        h1_r=np.maximum(h1,0)
        h1_f=np.hstack((h1_r,one))
        op=np.dot(h1_f,self.W2.T)
        return op
    def gradient_check(self,data,labels,dW1,dW2):
        r,c=data.shape
        W1=self.W1
        W2=self.W2
        W1[3,28]=W1[3,28]+0.0001
        
        one=np.ones((r,1))
        h1=np.dot(data,W1.T)
        h1_r=np.maximum(h1,0)
        h1_f=np.hstack((h1_r,one))
        
        op=np.dot(h1_f,W2.T)
        ind=np.arange(r)
        ind_f=np.reshape(ind,(r,1))
        cor_labels=op[ind_f,labels]
        
        svm_loss_each=op-cor_labels+1
        svm_loss_each=np.maximum(svm_loss_each,0)
        svm_loss_each_image=np.sum(svm_loss_each,axis=1)
        
        dataloss=np.sum(svm_loss_each_image)/r
        W1_sq=W1**2
        W2_sq=W2**2
        regloss=(self.reg/(2))*(np.sum(W1_sq)+np.sum(W2_sq))
        loss1=dataloss

        W1[3,28]=W1[3,28]-0.0002
        r,c=data.shape
        one=np.ones((r,1))
        h1=np.dot(data,W1.T)
        h1_r=np.maximum(h1,0)
        h1_f=np.hstack((h1_r,one))
        op=np.dot(h1_f,W2.T)
        ind=np.arange(r)
        ind_f=np.reshape(ind,(r,1))
        cor_labels=op[ind_f,labels]
      
        
        svm_loss_each=op-cor_labels+1
        svm_loss_each=np.maximum(svm_loss_each,0)
        svm_loss_each_image=np.sum(svm_loss_each,axis=1)
        
        dataloss=np.sum(svm_loss_each_image)/r
        W1_sq=W1**2
        W2_sq=W2**2
        regloss=(self.reg/(2))*(np.sum(W1_sq)+np.sum(W2_sq))
        loss2=dataloss
        gradient=(loss1-loss2)/0.0002
        m=0
        if gradient>dW1[3,28]/r:
            m=gradient
        else:
            m=dW1[3,28]/r
        compare=(dW1[3,28]/r-gradient)/m   
        print '------%.8f---------------%.8f--------------%.8f----------'%(gradient,compare,dW1[3,28]/r)
    def batch(self,data,labels,i,j):
        r,c=data.shape
        one=np.ones((r,1))
        h1=np.dot(data,self.W1.T)
        h1_r=np.maximum(h1,0)
        h1_f=np.hstack((h1_r,one))
        op=np.dot(h1_f,self.W2.T)
        ind=np.arange(r)
        ind_f=np.reshape(ind,(r,1))
        cor_labels=op[ind_f,labels]
        
        svm_loss_each=op-cor_labels+1
        svm_loss_each=np.maximum(svm_loss_each,0)
        svm_loss_each_image=np.sum(svm_loss_each,axis=1)
        
        dataloss=np.sum(svm_loss_each_image)/r
        W1_sq=self.W1**2
        W2_sq=self.W2**2
        regloss=(self.reg/(2))*(np.sum(W1_sq)+np.sum(W2_sq))
        loss=dataloss+regloss
        print '%d,%d : %f'%(i,j,loss)
        op_gr_zero=svm_loss_each>0
        dop=np.zeros((r,10))
        dop[op_gr_zero]=1
        dop_sum=np.sum(dop,axis=1)

        dop_sum=np.reshape(dop_sum,(r,1))
        dop_sum=dop_sum-dop[ind_f,labels]
        dop[ind_f,labels]=-dop_sum
        
        dW2=np.dot(dop.T,h1_f)
        dh1_f=np.dot(dop,self.W2)
        dh1_r_in=dh1_f[:,:100]
        h1_r_gr_than_zero=h1>0
        dh1_r=np.zeros((r,100))
        dh1_r[h1_r_gr_than_zero]=1
        dh1_r=dh1_r*dh1_r_in
        dW1=np.dot(dh1_r.T,data)
        if j==5 or j==50 or j==25 or j==35 :
            self.gradient_check(data,labels,dW1,dW2)
        self.W2 =self.W2-self.learning_rate*dW2  
        self.W1 =self.W1-self.learning_rate*dW1  
 



        
    def train(self,data,labels):
        r,c=data.shape
        one=np.ones((r,1),dtype='float64')
        self.W1=0.01*np.random.randn(100,c+1)
        self.W2=0.01*np.random.randn(10,101)
        self.mean=np.mean(data,axis=0)
        self.mean=np.reshape(self.mean,(1,c))
        data=data-self.mean
        train_x=np.hstack((data,one))
  
    
        for i in range(200):
            for j in range(50):
               self.batch(train_x[1000*j:1000*(j+1),:],labels[1000*j:1000*(j+1),:],i,j)
      
              
          
          
   

            
            
            
        
            
           
           
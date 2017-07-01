import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt


'''
===================
========main=======
===================
'''
#output image data each time
#and normalize 0~255 to 0~1
#flatten image from 28x28 to 1x784
class DataFeeder():
    def __init__(self,batch_number=1):
        self.CURRENT_INDEX=0
        self.BATCH_NUMBER=batch_number
        #read image name
        self.image_name=glob.glob('mnist_train/*')
    def feed(self):
        self.CURRENT_INDEX=self.CURRENT_INDEX+self.BATCH_NUMBER
        if self.CURRENT_INDEX > (len(self.image_name)-1):
            self.CURRENT_INDEX=self.BATCH_NUMBER
        try:
            return (np.array([cv2.imread(img,0).flatten()  for img in self.image_name[self.CURRENT_INDEX-self.BATCH_NUMBER:self.CURRENT_INDEX] ]).astype(float))/255.0,\
            np.array([float(img.strip().split('_')[2])  for img in self.image_name[self.CURRENT_INDEX-self.BATCH_NUMBER:self.CURRENT_INDEX] ])
        except:
            print self.CURRENT_INDEX,self.image_name[self.CURRENT_INDEX-self.BATCH_NUMBER:self.CURRENT_INDEX]


#each data
def crossEntropyLoss(y,r):
    loss=-np.sum(one_d_label_to_indicator_array(r)*np.log(y))
    return loss
    
    

def euclideanLoss(r,y):
    return (r-y)*(r-y)
    
    
    
def softmax(a):
    return np.exp(a)/np.sum(np.exp(a),axis=1,keepdims=True)
    
# if OUTPUT_CLASSES is 3   
#[0]     [1,0,0]
#[1] --> [0,1,0]
#[2]     [0,0,1]
OUTPUT_CLASSES=10
def one_d_label_to_indicator_array(r):
    return np.array([[1 if i==cls else 0  for i in  range(0,OUTPUT_CLASSES)] for cls in r.flatten().tolist() ])  
    
#[0.7,0.4,0.1]  -->    [0.7,0.4,0.1,1]   
def appendBias(x):
    x=np.append(x,np.ones(x.shape[:1]+(1,)),1)    
    return x
    

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
#    
#σ'(z)    
def d_sigmoid(z):
    return (1-sigmoid(z))*sigmoid(z)
    
    
def relu(z):
    relu_z=z.copy()
    relu_z[np.where(relu_z<0)]=0
    return relu_z
    
#relu'(z)    
def d_relu(z):
    relu_dz=z.copy()
    relu_dz[np.where(relu_dz>0)]=1
    relu_dz[np.where(relu_dz=<0)]=0
    return relu_dz
    



#data layer
BATCH_NUMBER=1
eta=0.005
df=DataFeeder()
#hidden layer
input_n=784
output_n=32
w0=(1/np.sqrt(output_n))*np.random.random([input_n+1,output_n])-((1/np.sqrt(output_n))*0.5)

#output layer
input_n=32
output_n=10
w1=(1/np.sqrt(output_n))*np.random.random([input_n+1,output_n])-((1/np.sqrt(output_n))*0.5)


mini_batch=32
y_list=np.array([])
r_list=np.array([])

for i in range(0,60000):
    #one update
    batch_loss=0.0
    total_loss=0.0
    #dw1=∂Loss/∂w1
    dw1_total=0.0
    dw0_total=0.0
    for j in range(0,mini_batch):
        d,r=df.feed()
        #r=r/9.999999
        x0=appendBias(d)
        z0=x0.dot(w0)
        a0=relu(z0)
        #
        x1=appendBias(a0)
        z1=x1.dot(w1)
        a1=relu(z1)
        #loss
        y=softmax(a1)
        #y=a1
        if len(y_list)<500:
            y_list=np.append(y_list,np.argmax(y))
            r_list=np.append(r_list,r)
        #euclideanLoss
        #loss=euclideanLoss(y,r)
        loss=crossEntropyLoss(y,r)
        regulation_loss=np.sum(w0*w0)/(w0.shape[0]*w0.shape[1])+np.sum(w1*w1)/(w1.shape[0]*w1.shape[1])
        batch_loss=batch_loss+loss+regulation_loss
        #cross entropy loss
        #deltaL=∂Loss/∂z
        deltaL=(y-one_d_label_to_indicator_array(r))*d_relu(z1)
        #euclideanLoss
        #deltaL=(y-r)*d_relu(z1)
        #dw1=∂Loss/∂w1
        dw1=a0.transpose(1,0)*deltaL
        dw1b=deltaL
        dw1=np.append(dw1,dw1b,axis=0)
        #deltal0
        deltal0=deltaL.dot(w1[0:32,:].transpose(1,0))*d_relu(z0)
        dw0=d.transpose(1,0)*deltal0
        dw0b=deltal0
        dw0=np.append(dw0,dw0b,axis=0)
        #
        dw0_total=dw0_total+dw0
        dw1_total=dw1_total+dw1
    #update
    w1=(1-eta*0.05)*w1-eta*dw1_total/mini_batch
    w0=(1-eta*0.05)*w0-eta*dw0_total/mini_batch
    total_loss=total_loss+batch_loss/mini_batch
    if not i%100:
        print 'iteration:',i,'loss:',total_loss,'accuracy:', np.sum((y_list==r_list).astype(float))/y_list.shape[0]
        print y_list[0:10]
        print r_list[0:10]
        y_list=np.array([])
        r_list=np.array([])
        total_loss=0.0
        
        
        
        
'''        
===================test===================
run over all test dataset,and output accuracy
'''
y_list=np.array([])
r_list=np.array([])
df=DataFeeder()
for j in range(0,10000):
        d,r=df.feed()
        r=r/10.0
        x0=appendBias(d)
        z0=x0.dot(w0)
        a0=sigmoid(z0)
        #
        x1=appendBias(a0)
        z1=x1.dot(w1)
        a1=sigmoid(z1)
        #loss
        #y=softmax(a1)
        y=a1
        y_list=np.append(y_list,y)
        r_list=np.append(r_list,r)
#
print 'accuracy:', np.sum((y_list==r_list).astype(float))/y_list.shape[0]




        
        
                
        
        
        
        
        
        
        
        
        
        
        
'''
===================#visualize result===================
'''
mnist_test=glob.glob('mnist_test/*')
#
for i in range(0,10000):
    d=cv2.imread(mnist_test[i],0).flatten().astype(float)/255.0
    #
    x0=appendBias(d.reshape(1,784))
    z0=x0.dot(w0)
    a0=relu(z0)
    #
    x1=appendBias(a0)
    z1=x1.dot(w1)
    a1=relu(z1)
    #loss
    y=softmax(a1)
    #
    print 'this is ',np.argmax(y)
    #show
    plt.imshow(plt.imread(mnist_test[i]),cmap='gray')
    plt.show()


        
        
        
  

# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 15:17:51 2022


作者：李一邨
人工智能算法案例大全：基于Python
浙大城市学院、杭州伊园科技有限公司
浙江大学 博士
中国民盟盟员
Email:liyicun_yykj@163.com
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import perceptron
from sklearn import metrics
import itertools


def plot_decision_boundary(pred_func, X, y):

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, s=42)
    
#%%

x1= np.concatenate([np.random.rand(100,1)-3,np.random.rand(100,1)+3],axis=0)
x2= np.concatenate([np.random.rand(100,1)-2,np.random.rand(100,1)+2],axis=0)
label=np.concatenate([np.ones((100,1)),np.zeros((100,1))],axis=0)

data1=np.concatenate([x1,x2,label],axis=1)
data1=pd.DataFrame(data1).reset_index(drop=True)

feat_cols = ['x1', 'x2','label']
data1.columns=feat_cols
print(data1.head())


Dtrain, Dtest = train_test_split(data1, test_size=0.2, random_state=1)

Xtrain = Dtrain[['x1', 'x2']].values
Xtest  = Dtest[['x1', 'x2']].values

ytrain = Dtrain['label'].values
ytest  = Dtest['label'].values

#%%



p = perceptron.Perceptron(random_state=1)
p.fit(Xtrain, ytrain)
predicts = p.predict(Xtest)
print("Testing accuracy {}".format(metrics.accuracy_score(ytest, predicts)))

plot_decision_boundary(lambda x: p.predict(x), Xtrain, ytrain)

#%%

x1=6*(np.random.rand(100,1)-0.5)
y1=np.sqrt(9-x1**2)
x1=np.concatenate([x1,x1],axis=0)
y1=np.concatenate([y1,-y1],axis=0)

x2=2*(np.random.rand(100,1)-0.5)
y2=np.sqrt(1-x2**2)
x2=np.concatenate([x2,x2],axis=0)
y2=np.concatenate([y2,-y2],axis=0)

xx= np.concatenate([x1,x2],axis=0)
yy= np.concatenate([y1,y2],axis=0)

label=np.concatenate([np.array(list(itertools.product([[True]],repeat=200))).reshape(200,1),np.array(list(itertools.product([[False]],repeat=200))).reshape(200,1)],axis=0)
data2=np.concatenate([xx,yy],axis=1)

data2=pd.DataFrame(data2).reset_index(drop=True)
data2=pd.concat([data2,pd.DataFrame(label)],axis=1)

feat_cols = ['x1', 'x2','label']
data2.columns=feat_cols
print(data2.head())

Dtrain2, Dtest2 = train_test_split(data2, test_size=0.2, random_state=1)

Xtrain2 = Dtrain2[['x1', 'x2']].values
Xtest2  = Dtest2[['x1', 'x2']].values

ytrain2 = Dtrain2['label'].values
ytest2  = Dtest2['label'].values


#%%

p = perceptron.Perceptron(random_state=1)
p.fit(Xtrain2, ytrain2)
predicts2 = p.predict(Xtest2)
print("Testing accuracy {}".format(metrics.accuracy_score(ytest2, predicts2)))

plot_decision_boundary(lambda x: p.predict(x), Xtrain2, ytrain2)


#%%

num_examples = len(Xtrain2) 
nn_input_dim = 2 
nn_output_dim = 2


epsilon = 0.01
reg_lambda = 0.5

#%%


def build_model(X, y, nn_hdim):
    

    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))


    model = {}
    

    for i in range(0, 20000):

        
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        
        if np.mean(abs(a1))>0.999:
            print(i)
            break
        
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        
        delta3 = probs
        delta3[y, :] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)


        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1


        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2
        

        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    
    return model



def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)


#%%



model = build_model(Xtrain2, ytrain2, 5)

print("Training accuracy: {}".format(metrics.accuracy_score(ytrain2, predict(model,Xtrain2) )))
print("Testing accuracy : {}".format(metrics.accuracy_score(ytest2, predict(model,Xtest2) )))
# Plot the decision boundary
plot_decision_boundary(lambda x: predict(model, x), Xtrain2, ytrain2)
plt.title("Decision Boundary")
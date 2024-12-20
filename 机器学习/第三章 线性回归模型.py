# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 15:08:51 2022

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
from sklearn.linear_model import LinearRegression as LR
import sklearn.metrics as skm


my_dataframe = pd.read_csv('Advertising.csv')
feature_cols = ['TV', 'Radio', 'Newspaper']
X = my_dataframe[feature_cols].values
y = my_dataframe.Sales.values

Xtrain = X[:140]
ytrain = y[:140]

Xtest = X[140:]
ytest = y[140:]


my_model = LR()

my_model.fit(Xtrain, ytrain)

#%%


ypredicts = my_model.predict(Xtest)

print ("The predicted sales:")
print (ypredicts)
print ()
print ("The true sales:")
print (ytest)

plt.plot(ytest,ypredicts,'ok')
plt.ylabel('Predicted sales')
plt.xlabel('True sales')
plt.show()

#%%

mse = np.mean((ypredicts - ytest) ** 2)
print ("MSE: {}".format(mse))

mae_SKL = skm.mean_absolute_error(ytest,ypredicts)
print ("SKL_MAE: {}".format(mae_SKL))

mse_SKL = skm.mean_squared_error(ytest,ypredicts)
print ("SKL_MSE: {}".format(mse_SKL))

#%%

print (my_model.intercept_)
print (my_model.coef_)

zipped_list =  (zip(feature_cols, my_model.coef_))
print(list(zipped_list))

#%%

def getErrorwithSize(model, train_sizes, Xtrain, ytrain, Xtest, ytest):
    
    model_mse   = np.zeros(len(train_sizes))  
    model_wts   = np.zeros([len(train_sizes), 4]) 
    
    for size in train_sizes:    
        Xsubtrain = Xtrain[0:size,:]
        ysubtrain = ytrain[0:size]
        model.fit(Xsubtrain, ysubtrain)
    
        ypredicts = model.predict(Xtest)    
        
        index              = (size//10)-1        
        model_mse[index]  = np.mean((ypredicts - ytest)**2)
        model_wts[index,:] = np.append(model.intercept_, model.coef_)  
    
    return model_mse, model_wts

#%%

train_sizes = np.arange(10,150,10)
print (train_sizes)

mse,weights = getErrorwithSize(my_model,train_sizes, Xtrain, ytrain, Xtest, ytest)


plt.plot(train_sizes, mse)
plt.xlabel('Training Size')
plt.ylabel('Mean Sq Error in Prediction')
plt.title('Effect of Data size on prediction error')




#%%

data = np.c_[X,y]
print (data.shape)
data[0:5,:]

from sklearn.model_selection import train_test_split

Dtrain, Dtest = train_test_split(data, test_size=0.2)


print(data.shape)
print(Dtrain.shape)
print(Dtest.shape)

trials = 100

train_sizes = np.arange(10,150,10) 
final_mse       = np.zeros(len(train_sizes))  
final_model_wts = np.zeros([len(train_sizes), 4]) 

for i in range(0,trials):
    Dtrain, Dtest = train_test_split(data, test_size=0.3)
    Xtrain = Dtrain[:, 0:3]
    ytrain = Dtrain[:,3]

    Xtest = Dtest[:, 0:3]
    ytest = Dtest[:,3]
    
    mse,weights = getErrorwithSize(my_model, train_sizes,Xtrain, ytrain, Xtest, ytest)
    
    final_mse  +=  mse 
   
final_mse/= trials 

plt.plot(train_sizes, final_mse)

plt.xlabel('Training Data Sizes')
plt.ylabel('Model MSE')
plt.title('Effect of Data Size on prediction error')
plt.show()
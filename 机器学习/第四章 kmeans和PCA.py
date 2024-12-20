# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 16:58:54 2022


作者：李一邨
人工智能算法案例大全：基于Python
浙大城市学院、杭州伊园科技有限公司
浙江大学 博士
中国民盟盟员
Email:liyicun_yykj@163.com
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn import preprocessing
from sklearn.cluster import KMeans

#%%
def distance(data, centers):

    dist = np.zeros((data.shape[0], centers.shape[0])) 
    for i in range(len(data)):
        for j in range(len(centers)):
            dist[i, j] = np.sqrt(np.sum((data.iloc[i, :] - centers.iloc[j,:]) ** 2)) 
                                                                        
    return dist

def near_center(data, centers): 
    dist = distance(data, centers)
    near_cen = np.argmin(dist, 1) 
    return near_cen

def kmeans(data, k):

    ind=np.random.choice(len(data), k)
    centers = data.iloc[ind,:].reset_index(drop=True) 
    print(centers)

    for _ in range(10): 

        near_cen = near_center(data, centers)

        for ci in range(k):
            centers.iloc[ci,:] = data.iloc[near_cen == ci,:].mean()
    return centers, near_cen



#%%

df=pd.read_csv("./obesity_levels.csv")
df.head()

#%%

df_=df.drop('NObeyesdad', axis=1)
df_.describe()
le_df=df_
label_encoder1 = preprocessing.LabelEncoder()
le_df['Gender'] = label_encoder1.fit_transform(le_df['Gender']) 
label_encoder2 = preprocessing.LabelEncoder()
le_df['family_history_with_overweight'] = label_encoder2.fit_transform(le_df['family_history_with_overweight']) 
label_encoder3 = preprocessing.LabelEncoder()
le_df['FAVC'] = label_encoder3.fit_transform(le_df['FAVC']) 
label_encoder4 = preprocessing.LabelEncoder()
le_df['NCP'] = label_encoder4.fit_transform(le_df['NCP']) 
label_encoder5 = preprocessing.LabelEncoder()
le_df['CAEC'] = label_encoder5.fit_transform(le_df['CAEC']) 
label_encoder6 = preprocessing.LabelEncoder()
le_df['SMOKE'] = label_encoder6.fit_transform(le_df['SMOKE']) 
label_encoder7 = preprocessing.LabelEncoder()
le_df['SCC'] = label_encoder7.fit_transform(le_df['SCC']) 
label_encoder8 = preprocessing.LabelEncoder()
le_df['CALC'] = label_encoder8.fit_transform(le_df['CALC']) 
label_encoder9 = preprocessing.LabelEncoder()
le_df['MTRANS'] = label_encoder9.fit_transform(le_df['MTRANS'])

le_df.head()

#%% 自编kmeans
import time
time_start = time.time()  # 记录开始时间

centers, near_cen = kmeans(le_df,4)

time_end = time.time()  # 记录结束时间
time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
print(time_sum)

#%%
kmeanModel = KMeans(n_clusters=4)
kmeanModel.fit(le_df)
y_hat=kmeanModel.predict(le_df)

#%%

import sklearn.metrics as sm
print('自编kmeans的轮廓系数：',sm.silhouette_score(le_df, near_cen, sample_size=len(le_df), metric='euclidean'))
print('sklearn的kmeans的轮廓系数：',sm.silhouette_score(le_df, y_hat, sample_size=len(le_df), metric='euclidean'))

#%%


Loss = []
K = range(2,11)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(le_df)
    Loss.append(kmeanModel.inertia_)


import matplotlib.pyplot as plt
plt.figure(figsize=(16,8))
plt.plot(K, Loss, 'bx-')
plt.xlabel('Number of K')
plt.ylabel('Loss')
plt.title('The Elbow Method showing the optimal k')
plt.show()


#%%

data_processed=np.array(le_df)
print(data_processed.shape)

mu = data_processed.mean(axis=0)
sigma = data_processed.std(axis=0)

Xnorm = (data_processed - mu)/sigma
print(Xnorm[0:5,:])

#%%

from sklearn.decomposition import PCA
pca = PCA(n_components=15)
pca.fit(le_df)
Zred = pca.fit_transform(le_df)
Xrec = pca.inverse_transform(Zred)


var=pca.explained_variance_ratio_
print(var)

var_sumed=np.cumsum(np.round(pca.explained_variance_ratio_,decimals=4)*100)
print(var_sumed)
plt.plot(var_sumed)
plt.xlabel("Principal components")
plt.ylabel("Variance captured")

#%%

from sklearn.decomposition import PCA
pca = PCA(n_components=5)
pca.fit(le_df)
Zred = pca.fit_transform(le_df)
Xrec = pca.inverse_transform(Zred)

#%%
import time
time_start = time.time()  # 记录开始时间

centers, near_cen = kmeans(pd.DataFrame(Zred),4)
print('自编kmeans的轮廓系数：',sm.silhouette_score(Zred, near_cen, sample_size=len(Zred), metric='euclidean'))

time_end = time.time()  # 记录结束时间
time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
print(time_sum)



#%%

kmeanModel = KMeans(n_clusters=4)
kmeanModel.fit(Zred)
y_hat=kmeanModel.predict(Zred)
print('sklearn的kmeans的轮廓系数：',sm.silhouette_score(Zred, y_hat, sample_size=len(Zred), metric='euclidean'))


#%%

pca = PCA(n_components=2)
Zred = pca.fit_transform(Xnorm)
print(Zred.shape)


Xrec = pca.inverse_transform(Zred)
print(Xrec.shape)

rec_error=np.linalg.norm(Xnorm-Xrec,"fro")/np.linalg.norm(Xnorm,'fro')
print(rec_error)


nSamples, nDims = Xnorm.shape

n_comp=range(1,nDims+1)
print(n_comp)


rec_error = np.zeros(len(n_comp)+1)

for k in n_comp:
  pca=PCA(n_components=k)
  Zred = pca.fit_transform(Xnorm)
  Xrec = pca.inverse_transform(Zred)
  rec_error[k]=np.linalg.norm(Xnorm-Xrec,"fro")/np.linalg.norm(Xnorm,'fro')
  print("k={}, rec_error={}".format(k, rec_error[k]))

rec_error = rec_error[1:]

plt.plot(n_comp, rec_error)
plt.xlabel("No of princial components (k)")
plt.ylabel("Recontruction Error")
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 21:40:15 2021

作者：李一邨
人工智能算法案例大全：基于Python
浙大城市学院、杭州伊园科技有限公司
浙江大学 博士
中国民盟盟员
Email:liyicun_yykj@163.com

"""

import xlrd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import *  
import pylab as pl
import seaborn as sns
mpl.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus']=False

#%%

data = pd.read_excel(r'高新技术企业行业技术周期数据.xls')
print(data)

col=list(data.columns)

y_list=col[-3:]

y_label=y_list[0]

used_data=data.drop(['StockCode','EndYear'],axis=1)

df=used_data.dropna().reset_index(drop=True)

#check the missing value agian 
print(df.isnull().sum())

df.describe().to_excel('describe.xlsx')

#%%

# 均值。
mean = df[y_label].mean()
# 中位数。
median = df[y_label].median()
# 众数。
s = np.round(df[y_label],4).mode()
print(s)
# 注意，mode方法返回的是Series类型。
mode = s.iloc[0] #将值取出，iloc函数：通过行号来取行数据（如取第二行的数据），loc函数：通过行索引 "Index" 中的具体值来取行数据（如取"Index"为"A"的行）
print("mean, median, mode:",mean, median, mode)


plt.figure(figsize =(15,10))
# 绘制数据的分布（直方图 + 密度图）
sns.distplot(df[y_label])
plt.xlabel(y_label, fontsize=50)
plt.ylabel('Density', fontsize=50)
# 绘制垂直线
plt.axvline(mean, ls = '-', color = 'r', label = "Mean")
plt.axvline(median, ls = '-', color = 'g', label = "Median")
plt.axvline(mode, ls = '-', color = 'indigo', label = "Mode")
plt.legend(fontsize=30)

plt.savefig(y_label+'-均值中位数众数.jpg',bbox_inches = 'tight')

#%%

df['scale']=(df[y_label]-min(df[y_label]))/(max(df[y_label])-min(df[y_label]))

bins=[min(df['scale'])-0.01,0.1,0.3,0.5,0.7,0.9,max(df['scale'])+0.01]
df['cut']=pd.cut(df['scale'],bins,right=False)
labels=['0.1以下','0.1到0.3','0.3到0.5','0.5到0.7','0.7到0.9','0.9以上']
df['cut']=pd.cut(df['scale'],bins,right=False,labels=labels)
df[['scale','cut']]

pie_server = df['cut'].value_counts()
plt.figure(figsize = (20,15))
plt.pie(pie_server.values,
           shadow = True,
        textprops={'fontsize': 20})
plt.title(y_label+'分组占比', fontsize = 50)
plt.legend(labels =pie_server.index,fontsize=30)
centre_circle = plt.Circle((0,0),0.45,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle) 

plt.savefig(y_label+'分组占比.jpg',bbox_inches = 'tight')


#%%

plt.figure(figsize =(15,10))
sns.countplot(x= 'Industry', data =df, order=df['Industry'].value_counts(sort=True).index )
plt.title("Counts of Companies in Different Industries",fontsize =50)
plt.ylabel('Counts', fontsize =30)
plt.xlabel('Industry',fontsize =30)
pl.xticks(rotation=90,fontsize =20)

plt.savefig('行业分布.jpg',bbox_inches = 'tight')



#%%

new_train=data[col[3:-3]]
##数据相关性信息的可视化
import seaborn as sns
plt.figure(figsize=(20,15))
corr = new_train.corr()
ax=sns.heatmap(
    corr,
    vmin = -1,vmax = 1, center = 0 ,square= True,
    cmap = sns.diverging_palette(20,220,n=200),annot=True,annot_kws={"fontsize":10}
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation = 90,
    horizontalalignment = 'right',fontsize =20
)
ax.set_yticklabels(
    ax.get_yticklabels(),
    rotation = 0,
    horizontalalignment = 'right',fontsize =20
)
sns.despine()

plt.savefig('特征相关性图.jpg',bbox_inches = 'tight')


#%%

# 标准正态分布。
standard_normal = pd.Series(np.random.normal(0, 1, size=10000))

plt.figure(figsize=(15,10))
sns.kdeplot(standard_normal, label="Standard Normal Distribution")
sns.kdeplot((data[y_list[0]]-np.mean(data[y_list[0]]))/data[y_list[0]].std(), label=y_list[0])

plt.ylabel('Density', fontsize =30)
plt.xlabel('')
plt.legend(fontsize=25,loc='best')

plt.savefig('正态性观察.jpg',bbox_inches = 'tight')






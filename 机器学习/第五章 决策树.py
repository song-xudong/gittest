# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 10:03:25 2022

作者：李一邨
人工智能算法案例大全：基于Python
浙大城市学院、杭州伊园科技有限公司
浙江大学 博士
中国民盟盟员
Email:liyicun_yykj@163.com

"""


from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing


    
#%%

titanic_df = pd.read_csv("Titanic_cleaned_data.csv")
titanic_df=titanic_df.dropna()
titanic_df.head()

feature_cols = ['Pclass', 'Sex', 'Age', 'Embarked']
X = titanic_df[feature_cols]
label_encoder1 = preprocessing.LabelEncoder()
X['Sex']=label_encoder1.fit_transform(X['Sex']) 
label_encoder2 = preprocessing.LabelEncoder()
X['Embarked']=label_encoder2.fit_transform(X['Embarked']) 

y = titanic_df.Survived
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=42)


#%%


treeclf = DecisionTreeClassifier(max_depth=5, random_state=1)
print(Xtrain.shape)

treeclf.fit(Xtrain, ytrain)
print("Training accuracy: {}".format(accuracy_score(ytrain, treeclf.predict(Xtrain))))
print("Testing accuracy : {}".format(accuracy_score(ytest, treeclf.predict(Xtest))))

    
plt.figure(figsize=(100,50))
tree.plot_tree(treeclf,filled=True)

#%%


from sklearn.model_selection import cross_val_score
scores = cross_val_score(treeclf, Xtrain, ytrain, cv=10, scoring='accuracy')
print("Accuracy for each fold: {}".format(scores))
print("Mean Accuracy: {}".format(np.mean(scores)))
    
#%%

from sklearn.model_selection import validation_curve


max_depth_range = range(1, 11)


train_scores, valid_scores = validation_curve( treeclf, Xtrain, ytrain, param_name="max_depth", param_range=max_depth_range,
    cv=10, scoring="accuracy")


print(train_scores.shape)

#%%


mean_train_score = np.mean(train_scores, axis=1)
mean_val_score   = np.mean(valid_scores, axis=1)

plt.plot(max_depth_range, mean_train_score, color="blue", linewidth=1.5, label="Training")
plt.plot(max_depth_range, mean_val_score, color="red", linewidth=1.5, label="Validation")
plt.legend(loc="upper left")
plt.xlabel("Tree-depth")
plt.ylabel("Model Accuracy")
plt.title("Accuracy comparison of training/validation set")

#%%

treeclf = DecisionTreeClassifier(max_depth=3)
treeclf.fit(Xtrain,ytrain)
print("Training accuracy: {}".format(accuracy_score(ytrain, treeclf.predict(Xtrain))))
print("Testing accuracy : {}".format(accuracy_score(ytest, treeclf.predict(Xtest))))

#%%

clf = DecisionTreeClassifier(random_state=0)
path = clf.cost_complexity_pruning_path(Xtrain, ytrain)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(Xtrain, ytrain)
    clfs.append(clf)
print("Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
      clfs[-1].tree_.node_count, ccp_alphas[-1]))

depth = [clf.tree_.max_depth for clf in clfs]

plt.plot(ccp_alphas, depth, marker='o', drawstyle="steps-post")
plt.xlabel("alpha")
plt.ylabel("depth of tree")
plt.title("Depth vs alpha")
plt.tight_layout()


#%%

train_scores = [clf.score(Xtrain, ytrain) for clf in clfs]
test_scores = [clf.score(Xtest, ytest) for clf in clfs]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker='o', label="train",
        drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
plt.show()

#%%

clf = DecisionTreeClassifier(random_state=0, ccp_alpha=0.04)
clf.fit(Xtrain,ytrain)
pred=clf.predict(Xtest)
accuracy_score(ytest, pred)

#%%

from sklearn import tree
plt.figure(figsize=(15,10))
tree.plot_tree(clf,filled=True)
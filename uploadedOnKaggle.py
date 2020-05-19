# -*- coding: utf-8 -*-
"""
Created on Tue May 19 16:43:47 2020

@author: dimit
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn import metrics

# %% [code]
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# %% [code]
train_data.info()

# %% [code]
train_data = train_data.drop(["PassengerId", "Cabin", "Ticket"], axis=1)

# %% [code]
train_data['Sex'] = train_data['Sex'].apply(lambda data : 0 if (data == 'male') else 1)


# %% [code]
train_data["Embarked"] = train_data["Embarked"].ffill()
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)

# %% [code]
train_data.info()

# %% [code]
embark_dict = {'S':0, 'C':1, 'Q':2}
train_data['Embarked'] = train_data['Embarked'].map(embark_dict)

# %% [code]
import re

# %% [code]
def extract_title(name):
    title = re.search('([A-Za-z]+)\.', name)
    return title[1]
    
train_data['Title'] = train_data['Name'].apply(extract_title)
train_data = train_data.drop(["Name"], axis=1)

# %% [code]
train_data.Title

# %% [code]
title_dict = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Dr': 4, 'Rev': 5, 'Mlle': 6, 'Col': 7, 'Major': 8,
             'Mme': 9, 'Sir': 10, 'Jonkheer': 11, 'Capt': 12, 'Ms': 13, 'Countess': 14, 'Lady': 15, 'Don': 16}
train_data['Title'] = train_data['Title'].map(title_dict)

# %% [code]
train_data.Title

# %% [code]
train_data.head()

# %% [code]
train_data.isnull().values.sum()

# %% [code]
train_data.hist(bins=15)

# %% [code]
train_data.Fare.plot()

# %% [code]
train_data = train_data[train_data.Fare < 300]

# %% [code]
train_data.info()

# %% [code]
train_data.head()

# %% [code]
r = [0, 5, 16, 26, 32, 47, 63, 90]
g = [0, 1, 2, 3, 4, 5, 6]
train_data["Age"] = pd.cut(train_data['Age'], bins=r, labels=g)
train_data.Age = train_data.Age.astype(int)

# %% [code]
train_data.head()

# %% [code]
plt.scatter(train_data.Fare, train_data.Age)

# %% [code]
train_data = train_data[train_data.Fare < 200]

# %% [code]
train_data.info()

# %% [code]
train_data.SibSp.hist() 
train_data.Parch.hist()

# %% [code]
train_data = train_data[(train_data.SibSp < 3) & (train_data.Parch < 3)]

# %% [code]
train_data.info()

# %% [code]
train_data.hist(bins=10)

# %% [code]
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, MinMaxScaler

cols = ["Fare"]
X = train_data.drop(["Survived"],axis=1)

fare_scaled_r = RobustScaler().fit(X[cols])
X[cols] = fare_scaled_r.transform(X[cols])

fare_scaled_ma = MaxAbsScaler().fit(X[cols])
#X[cols] = fare_scaled_ma.transform(X[cols])



# %% [code]
X.head()

# %% [code]
y = train_data["Survived"]

# %% [code]
X.shape, y.shape

# %% [code]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85)


# %% [code]
from sklearn import svm, tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.svm import LinearSVR, SVR
from sklearn.model_selection import KFold, cross_val_score

lr = LogisticRegression()
svm1 = svm.SVC(kernel='linear')
svm2 = svm.SVC(kernel='rbf') 
gb = GaussianNB()
rf = RandomForestClassifier()
knn = KNeighborsClassifier(n_neighbors=15)

models = {"Logistic Regression": lr,"Random Forest": rf, "svm linear": svm1 , "svm rbf": svm2,
          "KNeighborsClassifier": knn ,'GaussianNB': gb}
l=[]
for model in models:
    l.append(models[model])
    
i=0
for Classifier in l:    
    accuracy = cross_val_score(Classifier,X_train,y_train,scoring='accuracy',cv=10)
    print("===", [*models][i] , "===")
    print("accuracy = ",accuracy)
    print("accuracy.mean = ", accuracy.mean())
    print("accuracy.variance = ", accuracy.var())
    i=i+1
    print("")

# %% [code]
model = lr.fit(X_train,y_train)
y_predict = model.predict(X_test)
print("Accuracy :", metrics.accuracy_score(y_test, y_predict))

# %% [markdown]
# Time to update the test dataset and apply the predictions

# %% [code]
test_data = pd.read_csv("test.csv")
test_data = test_data.drop(["PassengerId", "Cabin", "Ticket"], axis=1)
test_data["Embarked"] = test_data["Embarked"].map(embark_dict)
test_data["Sex"] = test_data["Sex"].apply(lambda data : 0 if (data=="male") else 1)
test_data["Age"].fillna(test_data["Age"].median(), inplace = True)
test_data["Fare"] = test_data["Fare"].ffill()
test_data['Title'] = test_data['Name'].apply(extract_title)
test_data['Title'] = test_data['Title'].map(title_dict)
test_data["Title"] = test_data["Title"].ffill()
test_data = test_data.drop(['Name'], axis=1)
test_data = test_data[test_data.Fare < 300]
test_data["Age"] = pd.cut(test_data['Age'], bins=r, labels=g)
test_data.Age = test_data.Age.astype(int)
test_data = test_data[(test_data.SibSp < 3) & (test_data.Parch < 3)]
test_data[cols] = fare_scaled_r.transform(test_data[cols])

# %% [code]
test_data.info()

# %% [code]
test_data.head()

# %% [code]
model = lr.fit(X_train,y_train)
y_predict_final = model.predict(test_data)


# %% [code]
y_predict_final.shape

# %% [code]
df = pd.read_csv("test.csv")
submission = pd.DataFrame({
        "PassengerId": df.PassengerId[:397],
        "Survived": y_predict_final
    })

submission.to_csv('submission.csv', index=False)

# %% [code]
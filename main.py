# -*- coding: utf-8 -*-
"""
Created on Mon May 11 10:16:40 2020

@author: dimit
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import re
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

train_data = train_data.drop(["Cabin"], axis=1)
train_data = train_data.drop(["PassengerId"], axis=1)
train_data["Embarked"] = train_data["Embarked"].ffill()
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
train_data.info()

train_data.head(30)
train_data.Sex.head(15)
train_data['Sex'] = train_data['Sex'].apply(lambda data : 0 if (data == 'male') else 1)

train_data = train_data.drop(["Ticket"],axis=1)
train_data.info()

embark_dict = {'S':0, 'C':1, 'Q':2}
train_data['Embarked'] = train_data['Embarked'].map(embark_dict)
train_data.info()
train_data_save = train_data

test_data = pd.read_csv("test.csv")
test_data = test_data.drop(["PassengerId", "Cabin", "Ticket"], axis=1)
test_data["Embarked"] = test_data["Embarked"].map(embark_dict)
test_data["Sex"] = test_data["Sex"].apply(lambda data : 0 if (data=="male") else 1)
test_data.info()

test_data["Age"].fillna(test_data["Age"].median(), inplace = True)
test_data["Fare"] = test_data["Fare"].ffill()
test_data.info()
test_data_save = test_data
 
def extract_title(name):
    title = re.search('([A-Za-z]+)\.', name)
    return title[1]
    
train_data['Title'] = train_data['Name'].apply(extract_title)
test_data['Title'] = test_data['Name'].apply(extract_title)

title_dict = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Dr': 4, 'Rev': 5, 'Mlle': 6, 'Col': 7, 'Major': 8,
             'Mme': 9, 'Sir': 10, 'Jonkheer': 11, 'Capt': 12, 'Ms': 13, 'Countess': 14, 'Lady': 15, 'Don': 16}
train_data['Title'] = train_data['Title'].map(title_dict)
test_data['Title'] = test_data['Title'].map(title_dict)
train_data.info()

train_data = train_data.drop(['Name'],axis=1)
test_data = test_data.drop(['Name'], axis=1)
test_data_save = test_data
train_data_save = train_data

from sklearn.preprocessing import StandardScaler

numerical_cols = ['Age', 'Fare']
X = train_data.drop(columns=['Survived'],axis=1)
y = train_data['Survived']

r = [0, 5, 16, 26, 32, 47, 63, 90]
g = [0, 1, 2, 3, 4, 5, 6]
train_data["Age"] = pd.cut(train_data['Age'], bins=r, labels=g)
train_data.Age = train_data.Age.astype(int)

scaler = StandardScaler().fit(X[numerical_cols])
X[numerical_cols] = scaler.transform(X[numerical_cols])
X.head(15)

test_scaler = StandardScaler().fit(test_data[numerical_cols])
test_data[numerical_cols] = test_scaler.transform(test_data[numerical_cols])

from scipy import stats
z = np.abs(stats.zscore(train_data["SibSp"]))
threshold = 3
train_data2 = train_data.copy()
train_data2 = train_data2[(z<3)]
train_data2.hist(bins=15)
z = np.abs(stats.zscore(train_data2["Title"]))
train_data2 = train_data2[(z<3)]

#m = len(y)
#theta = np.zeros([2,1])
#iterations = 1500
#alpha = 0.01

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=1)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

svc_model = SVC()

svc_model.fit(X_train, y_train)
svc_predict = svc_model.predict(X_test)

accuracy_score(y_test, svc_predict)
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}

grid_svc = GridSearchCV(estimator=SVC(), param_grid=param_grid, refit=True, verbose=4)
grid_svc.fit(X_train, y_train)
grid_svc.best_params_
grid_svc_prediction = grid_svc.predict(test_data)

submission = pd.DataFrame({
        
        "Survived": grid_svc_prediction
    })

submission.to_csv('submission.csv', index=False)




clf_rr = Ridge(normalize=True)
clf_rr.fit(X_train , y_train)
accuracies = cross_val_score(estimator = clf_rr, X = X_train, y = y_train, cv = 5,verbose = 1)
y_pred = clf_rr.predict(X_test)
print('')
print('###### Ridge Regression ######')
print('Score : %.4f' % clf_rr.score(X_test, y_test))
print(accuracies)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)**0.5
r2 = r2_score(y_test, y_pred)

print('')
print('MSE    : %0.2f ' % mse)
print('MAE    : %0.2f ' % mae)
print('RMSE   : %0.2f ' % rmse)
print('R2     : %0.2f ' % r2)

R2_Scores.append(r2)



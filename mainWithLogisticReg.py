# -*- coding: utf-8 -*-
"""
Created on Mon May 11 14:47:36 2020

@author: dimit
"""
import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt
import seaborn as sn
import re
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

from sklearn.neighbors import KNeighborsClassifier  
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.metrics import accuracy_score, make_scorer,f1_score, precision_score, recall_score, confusion_matrix

from sklearn.impute import SimpleImputer


train_data = pd.read_csv("train.csv")
train_data.info()

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

X = train_data.drop(['Survived'], axis=1)

y = train_data['Survived']
m =len(y)
# add the sigmoid function for logistic reg

def sigmoid(x):
  return 1/(1+np.exp(-x))

# cost function including the sigmoid func

def costFunction(theta, X, y):
    J = (-1/m) * np.sum(np.multiply(y, np.log(sigmoid(X @ theta))) 
        + np.multiply((1-y), np.log(1 - sigmoid(X @ theta))))
    return J

# gradient function

def gradient(theta, X, y):
    return ((1/m) * X.T @ (sigmoid(X @ theta) - y))

# calling functions using initial parameters
    
(m, n) = X.shape
X = np.hstack((np.ones((m,1)), X))
y = y[:, np.newaxis]
theta = np.zeros((n+1,1)) # intializing theta with all zeros
J = costFunction(theta, X, y)
print(J)

# using scipy's built in function fmin_tnc to optimise for theta

temp = opt.fmin_tnc(func = costFunction, 
                    x0 = theta.flatten(),fprime = gradient, 
                    args = (X, y.flatten()))

# the output of above function is a tuple whose first element
#contains the optimized values of theta

theta_optimized = temp[0]
print(theta_optimized)

J = costFunction(theta_optimized[:,np.newaxis], X, y)
print(J)

def accuracy(X, y, theta, cutoff):
    pred = [sigmoid(np.dot(X, theta)) >= cutoff]
    acc = np.mean(pred == y)
    print(acc * 100)
    
accuracy(X, y.flatten(), theta_optimized, 0.5)

# ----------------------------------------------------------------- #
# the shorter approach to the above is as follows:
from sklearn.linear_model import LogisticRegression

scaled_features_df = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

logreg = LogisticRegression()
logreg.fit(X_train,y_train)

test_data = test_data.fillna(0)
y_pred = logreg.predict(test_data)

df = pd.DataFrame(y_pred)

df.T.to_csv('final.csv')

print("Accuracy:",metrics.accuracy_score(y_test, df.T))

ln = LinearRegression()
svm1 = svm.SVC(kernel='linear',random_state = 42)
svm2 = svm.SVC(kernel='rbf',random_state = 42) 
lr = LogisticRegression(random_state = 42)
gb = GaussianNB()
rf = RandomForestClassifier(random_state = 42)
knn = KNeighborsClassifier(n_neighbors=15)
tree = tree.DecisionTreeClassifier()
models = {"Logistic Regression": lr, 'DecisionTreeClassifier' : tree, 
          "Random Forest": rf, "svm linear": svm1 , "svm rbf": svm2,
          "KNeighborsClassifier": knn ,'GaussianNB': gb, 
          'Linear Regression' :ln}
l=[]
for model in models:
    l.append(make_pipeline(SimpleImputer(),  models[model]))
    
i=0
for Classifier in l:    
    accuracy = cross_val_score(Classifier,X_train,Y_train,scoring='accuracy',cv=5)
    print("===", [*models][i] , "===")
    print("accuracy = ",accuracy)
    print("accuracy.mean = ", accuracy.mean())
    print("accuracy.variance = ", accuracy.var())
    i=i+1
    print("")

model1 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model1.fit(scaled_features_df, y)
y1_test = model1.predict()


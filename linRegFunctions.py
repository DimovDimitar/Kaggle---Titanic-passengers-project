# -*- coding: utf-8 -*-
"""
Created on Sun May 10 16:57:12 2020

@author: dimitar
"""
import numpy as np
import matplotlib.pyplot as plt

# an array of ones of same dimension as x
# ones = np.ones_like(x) 

# Add a column of ones to x. hstack means stacking horizontally i.e. columnwise
# X = np.hstack((ones,x)) 

def plotData(x, y):
    
    fig, ax = plt.subplots() 
    ax.plot(x,y,'rx',markersize=10)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")

    return fig

def normalEquation(X,y):
    
    return np.dot((np.linalg.inv(np.dot(X.T,X))),np.dot(X.T,y))

def featureNormalize(X):
    return np.divide((X - np.mean(X,axis=0)),np.std(X,axis=0))

def batchGradientDescent(theta, alpha, num_iters, h, X, y, n):
    cost = np.ones(num_iters)
    for i in range(0,num_iters):
        theta[0] = theta[0] - (alpha/X.shape[0]) * sum(h - y)
        for j in range(1,n+1):
            theta[j] = theta[j] - (alpha/X.shape[0])*sum((h-y) * X.transpose()[j])
        h = hypothesis(theta, X, n)
        cost[i] = (1/X.shape[0]) * 0.5 * sum(np.square(h - y))
    theta = theta.reshape(1,n+1)
    return theta, cost

def gdWithCostFunction(X, y, theta, alpha, num_iters):
    
    m = len(y) # number of training examples
    J_history = np.zeros(num_iters)

    for i in range(num_iters):
        theta = theta - (alpha/m)*np.sum((np.dot(X,theta)-y)[:,None]*X,axis=0)
        J_history[i] = computeCost(X, y, theta)
        print('Cost function: ',J_history[i])
    
    return (theta, J_history)

def gradientDescent(x, y, theta, m, alpha, num_iters):
    for iteration in range(num_iters):
        for j in range(len(theta)):
            gradient = 0
            for i in range(m):
                gradient += (hypothesis(x[i], theta) - y[i]) * x[i][j]
        gradient *= 1/m
        theta[j] = theta[j] -  (alpha * gradient)
        print(theta)
    return theta

def stocashtic_gradient_descent(X,y,theta,learning_rate=0.01,iterations=10):
    '''
    X    = Matrix of X with added bias units
    y    = Vector of Y
    theta=Vector of thetas np.random.randn(j,1)
    learning_rate 
    iterations = no of iterations
    
    Returns the final theta vector and array of cost history over no of iterations
    '''
    m = len(y)
    cost_history = np.zeros(iterations)
    
    
    for it in range(iterations):
        cost =0.0
        for i in range(m):
            rand_ind = np.random.randint(0,m)
            X_i = X[rand_ind,:].reshape(1,X.shape[1])
            y_i = y[rand_ind].reshape(1,1)
            prediction = np.dot(X_i,theta)

            theta = theta -(1/m)*learning_rate*( X_i.T.dot((prediction - y_i)))
            cost += computeCost(X, y, theta)
        cost_history[it]  = cost
        
    return theta, cost_history


def computeCost(X, y, theta):
    m = len(y)
    J = (np.sum((np.dot(X,theta) - y)**2))/(2*m)
    return J

def hypothesis(x, theta):
	return np.dot(np.transpose(theta),x)



def generateZValues(x, theta):
	z_values = []
	for i in range(len(x)):
		z_values.append(hypothesis(x[i], theta))
	return np.asarray(z_values) 

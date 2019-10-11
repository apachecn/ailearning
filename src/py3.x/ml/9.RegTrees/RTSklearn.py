#!/usr/bin/python
# coding:utf8

# '''
# Created on 2017-03-10
# Update on 2017-03-10
# author: jiangzhonglian
# content: 回归树
# '''

# print(__doc__)


# # Import the necessary modules and libraries
# import numpy as np
# from sklearn.tree import DecisionTreeRegressor
# import matplotlib.pyplot as plt


# # Create a random dataset
# rng = np.random.RandomState(1)
# X = np.sort(5 * rng.rand(80, 1), axis=0)
# y = np.sin(X).ravel()
# print X, '\n\n\n-----------\n\n\n', y
# y[::5] += 3 * (0.5 - rng.rand(16))


# # Fit regression model
# regr_1 = DecisionTreeRegressor(max_depth=2, min_samples_leaf=5)
# regr_2 = DecisionTreeRegressor(max_depth=5, min_samples_leaf=5)
# regr_1.fit(X, y)
# regr_2.fit(X, y)


# # Predict
# X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
# y_1 = regr_1.predict(X_test)
# y_2 = regr_2.predict(X_test)


# # Plot the results
# plt.figure()
# plt.scatter(X, y, c="darkorange", label="data")
# plt.plot(X_test, y_1, color="cornflowerblue", label="max_depth=2", linewidth=2)
# plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
# plt.xlabel("data")
# plt.ylabel("target")
# plt.title("Decision Tree Regression")
# plt.legend()
# plt.show()








'''
Created on 2017-03-10
Update on 2017-03-10
author: jiangzhonglian
content: 模型树
'''

print(__doc__)

# Author: Noel Dawe <noel.dawe@gmail.com>
#
# License: BSD 3 clause

# importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

# Create the dataset
rng = np.random.RandomState(1)
X = np.linspace(0, 6, 100)[:, np.newaxis]
y = np.sin(X).ravel() + np.sin(6 * X).ravel() + rng.normal(0, 0.1, X.shape[0])

# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=4)

regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                          n_estimators=300, random_state=rng)

regr_1.fit(X, y)
regr_2.fit(X, y)

# Predict
y_1 = regr_1.predict(X)
y_2 = regr_2.predict(X)

# Plot the results
plt.figure()
plt.scatter(X, y, c="k", label="training samples")
plt.plot(X, y_1, c="g", label="n_estimators=1", linewidth=2)
plt.plot(X, y_2, c="r", label="n_estimators=300", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Boosted Decision Tree Regression")
plt.legend()
plt.show()

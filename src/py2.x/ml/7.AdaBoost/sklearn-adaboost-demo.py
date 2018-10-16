#!/usr/bin/python
# coding:utf8
"""
Created on 2017-07-10
Updated on 2017-07-10
Author: 片刻／Noel Dawe
GitHub: https://github.com/apachecn/AiLearning
sklearn-AdaBoost译文链接: http://cwiki.apachecn.org/pages/viewpage.action?pageId=10813457
"""
from __future__ import print_function

import matplotlib.pyplot as plt
# importing necessary libraries
import numpy as np
from sklearn import metrics
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

print(__doc__)


# Create the dataset
rng = np.random.RandomState(1)
X = np.linspace(0, 6, 100)[:, np.newaxis]
y = np.sin(X).ravel() + np.sin(6 * X).ravel() + rng.normal(0, 0.1, X.shape[0])
# dataArr, labelArr = loadDataSet("db/7.AdaBoost/horseColicTraining2.txt")


# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=4)
regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=300, random_state=rng)

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

print('y---', type(y[0]), len(y), y[:4])
print('y_1---', type(y_1[0]), len(y_1), y_1[:4])
print('y_2---', type(y_2[0]), len(y_2), y_2[:4])

# 适合2分类
y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
print('y_scores---', type(y_scores[0]), len(y_scores), y_scores)
print(metrics.roc_auc_score(y_true, y_scores))

# print "-" * 100
# print metrics.roc_auc_score(y[:1], y_2[:1])

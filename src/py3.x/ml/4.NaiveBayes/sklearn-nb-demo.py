#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
Created on 2017-06-28
Updated on 2017-06-28
NaiveBayes：朴素贝叶斯
Author: 小瑶
GitHub: https://github.com/apachecn/AiLearning
"""


# GaussianNB_高斯朴素贝叶斯
import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X, Y)
print(clf.predict([[-0.8, -1]]))
clf_pf = GaussianNB()
clf_pf.partial_fit(X, Y, np.unique(Y))
print(clf_pf.predict([[-0.8, -1]]))

# MultinomialNB_多项朴素贝叶斯
'''
import numpy as np
X = np.random.randint(5, size=(6, 100))
y = np.array([1, 2, 3, 4, 5, 6])
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X, y)
print clf.predict(X[2:3])
'''

# BernoulliNB_伯努利朴素贝叶斯
'''
import numpy as np
X = np.random.randint(2, size=(6, 100))
Y = np.array([1, 2, 3, 4, 4, 5])
from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB()
clf.fit(X, Y)
print clf.predict(X[2:3])
'''

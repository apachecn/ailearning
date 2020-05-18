#!/usr/bin/python
# coding:utf8

"""
Created on 2017-06-28
Updated on 2017-06-28
SVM: 最大边距分离超平面
Author: 片刻
GitHub: https://github.com/apachecn/AiLearning
sklearn-SVM译文链接: http://cwiki.apachecn.org/pages/viewpage.action?pageId=10031359
"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

print(__doc__)


# 创建40个分离点
np.random.seed(0)
# X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
# Y = [0] * 20 + [1] * 20


def loadDataSet(fileName):
    """
    对文件进行逐行解析，从而得到第行的类标签和整个数据矩阵
    Args:
        fileName 文件名
    Returns:
        dataMat  数据矩阵
        labelMat 类标签
    """
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


X, Y = loadDataSet('data/6.SVM/testSet.txt')
X = np.mat(X)

print(("X=", X))
print(("Y=", Y))

# 拟合一个SVM模型
clf = svm.SVC(kernel='linear')
clf.fit(X, Y)

# 获取分割超平面
w = clf.coef_[0]
# 斜率
a = -w[0]/w[1]
# 从-5到5，顺序间隔采样50个样本，默认是num=50
# xx = np.linspace(-5, 5)  # , num=50)
xx = np.linspace(-2, 10)  # , num=50)
# 二维的直线方程
yy = a * xx - (clf.intercept_[0]) / w[1]
print(("yy=", yy))

# plot the parallels to the separating hyperplane that pass through the support vectors
# 通过支持向量绘制分割超平面
print(("support_vectors_=", clf.support_vectors_))
b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = clf.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

# plot the line, the points, and the nearest vectors to the plane
plt.plot(xx, yy, 'k-')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')

plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80, facecolors='none')
plt.scatter([X[:, 0]], [X[:, 1]], c=Y, cmap=plt.cm.Paired)

plt.axis('tight')
plt.show()

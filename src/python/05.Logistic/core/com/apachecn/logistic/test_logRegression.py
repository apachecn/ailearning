#!/usr/bin/env python
# encoding: utf-8
import os
import sys
sys.path.append("C:\Python27")

from numpy import *



import matplotlib.pyplot as plt

from  logRegression import *
"""
@version: 
@author: yangjf
@license: ApacheCN
@contact: highfei2011@126.com
@site: https://github.com/apachecn/MachineLearning
@software: PyCharm
@file: test_logRegression.py
@time: 2017/3/3 22:09
@test result: ok
"""

def loadData():
    train_x = []
    train_y = []
    # 获取当前文件所在路径
    project_dir = os.getcwdu()
    # 截取字符串至项目名：Test\
    project_dir = project_dir[:project_dir.find("MachineLearning\\") + 15]
    print project_dir
    fileIn = open("%s/testData/testSet.txt" % project_dir)
    for line in fileIn.readlines():
        lineArr = line.strip().split()
        train_x.append([1.0, float(lineArr[0]), float(lineArr[1])])
        train_y.append(float(lineArr[2]))
    return mat(train_x), mat(train_y).transpose()


##第一步: 加载数据
print "step 1: load data..."
train_x, train_y = loadData()
test_x = train_x; test_y = train_y

##第二步: 训练数据...
print "step 2: training..."
opts = {'alpha': 0.01, 'maxIter': 20, 'optimizeType': 'smoothStocGradDescent'}
optimalWeights =  trainLogRegres(train_x, train_y, opts)

##第三步: 测试
print "step 3: testing..."
accuracy =  testLogRegres(optimalWeights, test_x, test_y)

##第四步: 显示结果
print "step 4: show the result..."
print 'The classify accuracy is: %.3f%%' % (accuracy * 100)
showLogRegres(optimalWeights, train_x, train_y)
#!/usr/bin/env python
# encoding: utf-8
from numpy import *
import matplotlib.pyplot as plt
import time


"""
@version: 
@author: yangjf
@license: ApacheCN
@contact: highfei2011@126.com
@site: https://github.com/apachecn/MachineLearning
@software: PyCharm
@file: logRegression01.py
@time: 2017/3/3 22:03
@test result:not pass
"""

# sigmoid函数
def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))

def trainLogRegres(train_x, train_y, opts):
    # 计算训练时间
    startTime = time.time()

    numSamples, numFeatures = shape(train_x)
    alpha = opts['alpha']; maxIter = opts['maxIter']
    weights = ones((numFeatures, 1))

    # 通过梯度下降算法优化
    for k in range(maxIter):
        if opts['optimizeType'] == 'gradDescent': # 梯度下降算法
            output = sigmoid(train_x * weights)
            error = train_y - output
            weights = weights + alpha * train_x.transpose() * error
        elif opts['optimizeType'] == 'stocGradDescent': # 随机梯度下降
            for i in range(numSamples):
                output = sigmoid(train_x[i, :] * weights)
                error = train_y[i, 0] - output
                weights = weights + alpha * train_x[i, :].transpose() * error
        elif opts['optimizeType'] == 'smoothStocGradDescent': # 光滑随机梯度下降
            # 随机选择样本以优化以减少周期波动
            dataIndex = range(numSamples)
            for i in range(numSamples):
                alpha = 4.0 / (1.0 + k + i) + 0.01
                randIndex = int(random.uniform(0, len(dataIndex)))
                output = sigmoid(train_x[randIndex, :] * weights)
                error = train_y[randIndex, 0] - output
                weights = weights + alpha * train_x[randIndex, :].transpose() * error
                del(dataIndex[randIndex]) # 在一次交互期间，删除优化的样品
        else:
            raise NameError('Not support optimize method type!')


    print 'Congratulations, training complete! Took %fs!' % (time.time() - startTime)
    return weights


#测试给定测试集的训练Logistic回归模型
def testLogRegres(weights, test_x, test_y):
    numSamples, numFeatures = shape(test_x)
    matchCount = 0
    for i in xrange(numSamples):
        predict = sigmoid(test_x[i, :] * weights)[0, 0] > 0.5
        if predict == bool(test_y[i, 0]):
            matchCount += 1
    accuracy = float(matchCount) / numSamples
    return accuracy


# 显示你的训练逻辑回归模型只有2-D数据可用
def showLogRegres(weights, train_x, train_y):
    # 注意：train_x和train_y是垫数据类型
    numSamples, numFeatures = shape(train_x)
    if numFeatures != 3:
        print "抱歉! 我不能绘制，因为你的数据的维度不是2！"
        return 1

    # 画出所有抽样数据
    for i in xrange(numSamples):
        if int(train_y[i, 0]) == 0:
            plt.plot(train_x[i, 1], train_x[i, 2], 'or')
        elif int(train_y[i, 0]) == 1:
            plt.plot(train_x[i, 1], train_x[i, 2], 'ob')

    # 画图操作
    min_x = min(train_x[:, 1])[0, 0]
    max_x = max(train_x[:, 1])[0, 0]
    weights = weights.getA()  # 将mat转换为数组
    y_min_x = float(-weights[0] - weights[1] * min_x) / weights[2]
    y_max_x = float(-weights[0] - weights[1] * max_x) / weights[2]
    plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')
    plt.xlabel('X1'); plt.ylabel('X2')
    #显示图像
    plt.show()
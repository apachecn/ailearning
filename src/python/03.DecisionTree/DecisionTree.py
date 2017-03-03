#!/usr/bin/python
# coding:utf8

'''
Created on Oct 12, 2010
Update on 2017-02-27
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington/jiangzhonglian
'''
from math import log
import operator
import DecisionTreePlot as dtPlot


def createDataSet():
    """DateSet 基础数据集

    Args:
        无需传入参数
    Returns:
        返回数据集和对应的label标签
    Raises:

    """
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    # dataSet = [['yes'],
    #         ['yes'],
    #         ['no'],
    #         ['no'],
    #         ['no']]
    labels = ['no surfacing', 'flippers']
    # change to discrete values
    return dataSet, labels


def calcShannonEnt(dataSet):
    """calcShannonEnt(calculate Shannon entropy 计算label分类标签的香农熵)

    Args:
        dataSet 数据集
    Returns:
        返回香农熵的计算值
    Raises:

    """
    # 求list的长度，表示计算参与训练的数据量
    numEntries = len(dataSet)
    # print type(dataSet), 'numEntries: ', numEntries

    # 计算分类标签label出现的次数
    labelCounts = {}
    # the the number of unique elements and their occurance
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
        # print '-----', featVec, labelCounts

    # 对于label标签的占比，求出label标签的香农熵
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        # log base 2
        shannonEnt -= prob * log(prob, 2)
        # print '---', prob, prob * log(prob, 2), shannonEnt
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    """splitDataSet(通过遍历dataSet数据集，求出axis对应的colnum列的值为value的行)

    Args:
        dataSet 数据集
        axis 表示每一行的axis列
        value 表示axis列对应的value值
    Returns:
        axis列为value的数据集【该数据集需要排除axis列】
    Raises:

    """
    retDataSet = []
    for featVec in dataSet:
        # axis列为value的数据集【该数据集需要排除axis列】
        if featVec[axis] == value:
            # chop out axis used for splitting
            reducedFeatVec = featVec[:axis]
            '''
            请百度查询一下： extend和append的区别
            '''
            reducedFeatVec.extend(featVec[axis+1:])
            # 收集结果值 axis列为value的行【该行需要排除axis列】
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    """chooseBestFeatureToSplit(选择最好的特征)

    Args:
        dataSet 数据集
    Returns:
        bestFeature 最优的特征列
    Raises:

    """
    # 求第一行有多少列的 Feature
    numFeatures = len(dataSet[0]) - 1
    # label的信息熵
    baseEntropy = calcShannonEnt(dataSet)
    # 最优的信息增益值, 和最优的Featurn编号
    bestInfoGain, bestFeature = 0.0, -1
    # iterate over all the features
    for i in range(numFeatures):
        # create a list of all the examples of this feature
        # 获取每一个feature的list集合
        featList = [example[i] for example in dataSet]
        # get a set of unique values
        # 获取剔重后的集合
        uniqueVals = set(featList)
        # 创建一个临时的信息熵
        newEntropy = 0.0
        # 遍历某一列的value集合，计算该列的信息熵
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        # gain[信息增益] 值越大，意味着该分类提供的信息量越大，该特征对分类的不确定程度越小
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    """majorityCnt(选择出线次数最多的一个结果)

    Args:
        classList label列的集合
    Returns:
        bestFeature 最优的特征列
    Raises:

    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    # 倒叙排列classCount得到一个字典集合，然后取出第一个就是结果（yes/no）
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    # print 'sortedClassCount:', sortedClassCount
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # 如果数据集的最后一列的第一个值出现的次数=整个集合的数量，也就说只有一个类别，就只直接返回结果就行
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 如果数据集只有1列，那么最初出现label次数最多的一类，作为结果
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    # 选择最优的列，得到最有列对应的label含义
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    # 初始化myTree
    myTree = {bestFeatLabel: {}}
    # 注：labels列表是可变对象，在PYTHON函数中作为参数时传址引用，能够被全局修改
    # 所以这行代码导致函数外的同名变量被删除了元素，造成例句无法执行，提示'no surfacing' is not in list
    del(labels[bestFeat])
    # 取出最优列，然后它的branch做分类
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        # 求出剩余的标签label
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
        # print 'myTree', value, myTree
    return myTree


def classify(inputTree, featLabels, testVec):
    # 获取tree的第一个节点对应的key值
    firstStr = inputTree.keys()[0]
    # 获取第一个节点对应的value值
    secondDict = inputTree[firstStr]
    # 判断根节点的索引值，然后根据testVec来获取对应的树分枝位置
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    print '+++', firstStr, 'xxx', secondDict, '---', key, '>>>', valueOfFeat
    # 判断分枝是否结束
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel


def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)


if __name__ == "__main__":

    # 1.创建数据和结果标签
    myDat, labels = createDataSet()
    # print myDat, labels

    # # 计算label分类标签的香农熵
    # calcShannonEnt(myDat)

    # # 求第0列 为 1/0的列的数据集【排除第0列】
    # print '1---', splitDataSet(myDat, 0, 1)
    # print '0---', splitDataSet(myDat, 0, 0)

    # # 计算最好的信息增益的列
    # print chooseBestFeatureToSplit(myDat)

    import copy
    myTree = createTree(myDat, copy.deepcopy(labels))
    print myTree
    # [1, 1]表示要取的分支上的节点位置，对应的结果值
    # print classify(myTree, labels, [1, 1])

    # 画图可视化展现
    dtPlot.createPlot(myTree)

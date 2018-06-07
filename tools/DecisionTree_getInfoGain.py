#!/usr/bin/python
# coding: utf8

from math import log


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
    # print(type(dataSet), 'numEntries: ', numEntries)

    # 计算分类标签label出现的次数
    labelCounts = {}
    # the the number of unique elements and their occurance
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
        # print('-----', featVec, labelCounts)

    # 对于label标签的占比，求出label标签的香农熵
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        # log base 2
        shannonEnt -= prob * log(prob, 2)
        # print('---', prob, prob * log(prob, 2), shannonEnt)
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


def getFeatureShannonEnt(dataSet, labels):
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
    bestInfoGain, bestFeature, endEntropy = 0.0, -1, 0.0
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
        # gain[信息增益]=0, 表示与类别相同，无需其他的分类
        # gain[信息增益]=baseEntropy, 表示分类和没分类没有区别
        infoGain = baseEntropy - newEntropy
        # print(infoGain)
        if (infoGain > bestInfoGain):
            endEntropy = newEntropy
            bestInfoGain = infoGain
            bestFeature = i
    else:
        if numFeatures < 0:
            labels[bestFeature] = 'null'

    return labels[bestFeature], baseEntropy, endEntropy, bestInfoGain


if __name__ == '__main__':
    labels = ['no surfacing', 'flippers']
    dataSet1 = [['yes'], ['yes'], ['no'], ['no'], ['no']]
    dataSet2 = [['a', 1, 'yes'], ['a', 2, 'yes'], ['b', 3, 'no'], ['c', 4, 'no'], ['c', 5, 'no']]
    dataSet3 = [[1, 'yes'], [1, 'yes'], [1, 'no'], [3, 'no'], [3, 'no']]
    infoGain1 = getFeatureShannonEnt(dataSet1, labels)
    infoGain2 = getFeatureShannonEnt(dataSet2, labels)
    infoGain3 = getFeatureShannonEnt(dataSet3, labels)
    print('信息增益: \n\t%s, \n\t%s, \n\t%s' % (infoGain1, infoGain2, infoGain3))


#!/usr/bin/python
# coding:utf8
'''
Created on Feb 4, 2011
Update on 2017-05-18
Tree-Based Regression Methods Source Code for Machine Learning in Action Ch. 9
@author: Peter Harrington/片刻
《机器学习实战》更新地址：https://github.com/apachecn/MachineLearning
'''
print(__doc__)
from numpy import *


# 默认解析的数据是用tab分隔，并且是数值类型
# general function to parse tab -delimited floats
def loadDataSet(fileName):
    """loadDataSet(解析每一行，并转化为float类型)

    Args:
        fileName 文件名
    Returns:
        dataMat 每一行的数据集array类型
    Raises:
    """
    # 假定最后一列是结果值
    # assume last column is target value
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # 将所有的元素转化为float类型
        # map all elements to float()
        fltLine = map(float, curLine)
        dataMat.append(fltLine)
    return dataMat


def binSplitDataSet(dataSet, feature, value):
    """binSplitDataSet(将数据集，按照feature列的value进行 二元切分)

    Args:
        dataMat 数据集
        feature 特征列
        value 特征列要比较的值
    Returns:
        mat0 小于的数据集在左边
        mat1 大于的数据集在右边
    Raises:
    """
    # # 测试案例
    # print 'dataSet[:, feature]=', dataSet[:, feature]
    # print 'nonzero(dataSet[:, feature] > value)[0]=', nonzero(dataSet[:, feature] > value)[0]
    # print 'nonzero(dataSet[:, feature] <= value)[0]=', nonzero(dataSet[:, feature] <= value)[0]

    # dataSet[:, feature] 取去每一行中，第1列的值(从0开始算)
    # nonzero(dataSet[:, feature] > value)  返回结果为true行的index下标
    mat0 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
    mat1 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
    return mat0, mat1


# 返回每一个叶子结点的均值
# returns the value used for each leaf
def regLeaf(dataSet):
    return mean(dataSet[:, -1])


# 计算总方差=方差*样本数
def regErr(dataSet):
    # shape(dataSet)[0] 表示行数
    return var(dataSet[:, -1]) * shape(dataSet)[0]


# 1.用最佳方式切分数据集
# 2.生成相应的叶节点
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    """chooseBestSplit(用最佳方式切分数据集 和 生成相应的叶节点)

    Args:
        dataSet   加载的原始数据集
        leafType  建立叶子点的函数
        errType   误差计算函数(求总方差)
        ops       [容许误差下降值，切分的最少样本数]
    Returns:
        bestIndex feature的index坐标
        bestValue 切分的最优值
    Raises:
    """
    tolS = ops[0]
    tolN = ops[1]
    # 如果结果集(最后一列为1个变量)，就返回推出
    # .T 对数据集进行转置
    # .tolist()[0] 转化为数组并取第0列
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        #  exit cond 1
        return None, leafType(dataSet)
    # 计算行列值
    m, n = shape(dataSet)
    # 无分类误差的总方差和
    # the choice of the best feature is driven by Reduction in RSS error from mean
    S = errType(dataSet)
    # inf 正无穷大
    bestS, bestIndex, bestValue = inf, 0, 0
    # 循环处理每一列对应的feature值
    for featIndex in range(n-1):
        # [0]表示这一列的[所有行]，不要[0]就是一个array[[所有行]]
        for splitVal in set(dataSet[:, featIndex].T.tolist()[0]):
            # 对该列进行分组，然后组内的成员的val值进行 二元切分
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            # 判断二元切分的方式的元素数量是否符合预期
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)
            # 如果二元切分，算出来的误差在可接受范围内，那么就记录切分点，并记录最小误差
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    # 判断二元切分的方式的元素误差是否符合预期
    # if the decrease (S-bestS) is less than a threshold don't do the split
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    # 对整体的成员进行判断，是否符合预期
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex, bestValue


# assume dataSet is NumPy Mat so we can array filtering
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    """createTree(获取回归树)

    Args:
        dataSet      加载的原始数据集
        leafType     建立叶子点的函数
        errType      误差计算函数
        ops=(1, 4)   [容许误差下降值，切分的最少样本数]
    Returns:
        retTree    决策树最后的结果
    """
    # 选择最好的切分方式： feature索引值，最优切分值
    # choose the best split
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    # if the splitting hit a stop condition return val
    if feat is None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    # 大于在右边，小于在左边，分为2个数据集
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    # 递归的进行调用
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


# 判断节点是否是一个字典
def isTree(obj):
    return (type(obj).__name__ == 'dict')


# 计算左右枝丫的均值
def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0


# 检查是否适合合并分枝
def prune(tree, testData):
    # 判断是否测试数据集没有数据，如果没有，就直接返回tree本身的均值
    if shape(testData)[0] == 0:
        return getMean(tree)

    # 判断分枝是否是dict字典，如果是就将测试数据集进行切分
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    # 如果是左边分枝是字典，就传入左边的数据集和左边的分枝，进行递归
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    # 如果是右边分枝是字典，就传入左边的数据集和左边的分枝，进行递归
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)

    # 如果左右两边同时都不是dict字典，那么分割测试数据集。
    # 1. 如果正确 
    #   * 那么计算一下总方差 和 该结果集的本身不分枝的总方差比较
    #   * 如果 合并的总方差 < 不合并的总方差，那么就进行合并
    # 注意返回的结果： 如果可以合并，原来的dict就变为了 数值
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        # power(x, y)表示x的y次方
        errorNoMerge = sum(power(lSet[:, -1] - tree['left'], 2)) + sum(power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right'])/2.0
        errorMerge = sum(power(testData[:, -1] - treeMean, 2))
        # 如果 合并的总方差 < 不合并的总方差，那么就进行合并
        if errorMerge < errorNoMerge:
            print "merging"
            return treeMean
        else:
            return tree
    else:
        return tree


# 得到模型的ws系数：f(x) = x0 + x1*featrue1+ x3*featrue2 ...
# create linear model and return coeficients
def modelLeaf(dataSet):
    ws, X, Y = linearSolve(dataSet)
    return ws


# 计算线性模型的误差值
def modelErr(dataSet):
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    # print corrcoef(yHat, Y, rowvar=0)
    return sum(power(Y - yHat, 2))


 # helper function used in two places
def linearSolve(dataSet):
    m, n = shape(dataSet)
    # 产生一个关于1的矩阵
    X = mat(ones((m, n)))
    Y = mat(ones((m, 1)))
    # X的0列为1，常数项，用于计算平衡误差
    X[:, 1: n] = dataSet[:, 0: n-1]
    Y = dataSet[:, -1]

    # 转置矩阵*矩阵
    xTx = X.T * X
    # 如果矩阵的逆不存在，会造成程序异常
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\ntry increasing the second value of ops')
    # 最小二乘法求最优解:  w0*1+w1*x1=y
    ws = xTx.I * (X.T * Y)
    return ws, X, Y


# 回归树测试案例
def regTreeEval(model, inDat):
    return float(model)


# 模型树测试案例
def modelTreeEval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1, n+1)))
    X[:, 1: n+1] = inDat
    # print X, model
    return float(X * model)


# 计算预测的结果
def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree):
        return modelEval(tree, inData)
    if inData[tree['spInd']] <= tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)


# 预测结果
def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = mat(zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat


if __name__ == "__main__":
    # # 测试数据集
    # testMat = mat(eye(4))
    # print testMat
    # print type(testMat)
    # mat0, mat1 = binSplitDataSet(testMat, 1, 0.5)
    # print mat0, '\n-----------\n', mat1

    # # 回归树
    # myDat = loadDataSet('input/9.RegTrees/data1.txt')
    # # myDat = loadDataSet('input/9.RegTrees/data2.txt')
    # # print 'myDat=', myDat
    # myMat = mat(myDat)
    # # print 'myMat=',  myMat
    # myTree = createTree(myMat)
    # print myTree

    # # 1. 预剪枝就是：提起设置最大误差数和最少元素数
    # myDat = loadDataSet('input/9.RegTrees/data3.txt')
    # myMat = mat(myDat)
    # myTree = createTree(myMat, ops=(0, 1))
    # print myTree

    # # 2. 后剪枝就是：通过测试数据，对预测模型进行合并判断
    # myDatTest = loadDataSet('input/9.RegTrees/data3test.txt')
    # myMat2Test = mat(myDatTest)
    # myFinalTree = prune(myTree, myMat2Test)
    # print '\n\n\n-------------------'
    # print myFinalTree

    # # --------
    # # 模型树求解
    # myDat = loadDataSet('input/9.RegTrees/data4.txt')
    # myMat = mat(myDat)
    # myTree = createTree(myMat, modelLeaf, modelErr)
    # print myTree

    # 回归树 VS 模型树 VS 线性回归
    trainMat = mat(loadDataSet('input/9.RegTrees/bikeSpeedVsIq_train.txt'))
    testMat = mat(loadDataSet('input/9.RegTrees/bikeSpeedVsIq_test.txt'))
    # 回归树
    myTree1 = createTree(trainMat, ops=(1, 20))
    print myTree1
    yHat1 = createForeCast(myTree1, testMat[:, 0])
    print "回归树:", corrcoef(yHat1, testMat[:, 1],rowvar=0)[0, 1]

    # 模型树
    myTree2 = createTree(trainMat, modelLeaf, modelErr, ops=(1, 20))
    yHat2 = createForeCast(myTree2, testMat[:, 0], modelTreeEval)
    print myTree2
    print "模型树:", corrcoef(yHat2, testMat[:, 1],rowvar=0)[0, 1]

    # 线性回归
    ws, X, Y = linearSolve(trainMat)
    print ws
    m = len(testMat[:, 0])
    yHat3 = mat(zeros((m, 1)))
    for i in range(shape(testMat)[0]):
        yHat3[i] = testMat[i, 0]*ws[1, 0] + ws[0, 0]
    print "线性回归:", corrcoef(yHat3, testMat[:, 1],rowvar=0)[0, 1]

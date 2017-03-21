#!/usr/bin/python
# encoding: utf-8

from numpy import *
from numpy import linalg as la


def loadExData():
    return[[1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [1, 1, 1, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1]]


# 欧氏距离相似度，假定inA和inB 都是列向量
# 计算向量的第二范式，相当于计算了欧氏距离
def ecludSim(inA, inB):
    return 1.0/(1.0 + la.norm(inA - inB))


# pearsSim()函数会检查是否存在3个或更多的点。
# corrcoef直接计算皮尔逊相关系数
def pearsSim(inA, inB):
    # 如果不存在，该函数返回1.0，此时两个向量完全相关。
    if len(inA) < 3:
        return 1.0
    return 0.5 + 0.5*corrcoef(inA, inB, rowvar=0)[0][1]


# 计算余弦相似度
def cosSim(inA, inB):
    num = float(inA.T*inB)
    denom = la.norm(inA)*la.norm(inB)
    return 0.5 + 0.5*(num/denom)


# 基于物品相似度的推荐引擎
# standEst()函数，用来计算在给定相似度计算方法的条件下，用户对物品的估计评分值。
    # standEst()函数的参数包括数据矩阵、用户编号、物品编号和相似度计算方法
def standEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0:
            continue
        # 寻找两个用户都评级的物品
        overLap = nonzero(logical_and(dataMat[:, item].A>0, dataMat[:, j].A>0))[0]
        if len(overLap) == 0:similarity =0
        else: similarity = simMeas(dataMat[overLap,item], \
                                    dataMat[overLap,j])
        #print 'the %d and %d similarity is : %f'(iten,j,similarity)
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal


#recommend()函数，就是推荐引擎，它会调用standEst()函数。 
def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    # 寻找未评级的物品
    unratedItems = nonzero(dataMat[user, :].A == 0)[1]
    if len(unratedItems) == 0:
        return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        # 寻找前N个未评级物品
        itemScores.append((item, estimatedScore))
        return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[: N]

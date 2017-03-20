# coding=utf-8
def loadExData():
    return[[1,1,1,0,0],
           [2,2,2,0,0],
           [1,1,1,0,0],
           [5,5,5,0,0],
           [1,1,0,2,2],
           [0,0,0,3,3],
           [0,0,0,1,1]]

from numpy import * 
from numpy import linalg as la
# 欧氏距离相似度，假定inA和inB 都是列向量
# 计算向量的第二范式，相当于计算了欧氏距离
def ecludSim(inA,inB):
    return 1.0/(1.0 + la.norm(inA - inB))

# pearsSim()函数会检查是否存在3个或更多的点。
# corrcoef直接计算皮尔逊相关系数
def pearsSim(inA,inB):
    # 如果不存在，该函数返回1.0，此时两个向量完全相关。
    if len(inA)< 3 :return 1.0
    return 0.5 + 0.5*corrcoef(inA,inB,rowvar = 0)[0][1]

# 计算余弦相似度
def cosSim(inA,inB):
    num = float(inA.T*inB)
    denom = la.norm(inA)*la.norm(inB)
    return 0.5 +0.5*(num/denom)


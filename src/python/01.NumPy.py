#!/usr/bin/python
# coding:utf8

from numpy import random, mat, eye

'''
# NumPy 矩阵和数字的区别
NumPy存在2中不同的数据类型:
    1. 矩阵 matrix
    2. 数组 array
相似点：
    都可以处理行列表示的数字元素
不同点：
    1. 2个数据类型上执行相同的数据运算可能得到不同的结果。
    2. NumPy函数库中的 matrix 与 MATLAB中 matrices 等价。
'''

# 生成一个 4*4 的随机数组
randArray = random.rand(4, 4)

# 转化关系， 数组转化为矩阵
randMat = mat(randArray)
# .I表示对矩阵求逆(可以利用矩阵的初等变换
#    # 意义：逆矩阵是一个判断相似性的工具。逆矩阵A与列向量p相乘后，将得到列向量q，q的第i个分量表示p与A的第i个列向量的相似度。
#    # 参考案例链接：
#    # https://www.zhihu.com/question/33258489
#    # http://blog.csdn.net/vernice/article/details/48506027
# .T表示对矩阵转置(行列颠倒)
invRandMat = randMat.I
# 输出结果
print randArray, '\n---\n', randMat, '\n+++\n', invRandMat
# 矩阵和逆矩阵 进行求积 (单位矩阵，对角线都为1嘛，理论上4*4的矩阵其他的都为0)
myEye = randMat*invRandMat
# 误差
print myEye - eye(4)

'''
如果上面的代码运行没有问题，说明numpy安装没有问题
'''

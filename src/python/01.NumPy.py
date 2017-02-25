#!/usr/bin/python
# coding:utf8

from numpy import random

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
print random.rand(4, 4)



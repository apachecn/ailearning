#!/usr/bin/python
# coding:utf-8

'''
Created on 2017-04-06
Update  on 2017-11-17
Author: Peter/ApacheCN-xy/片刻
GitHub: https://github.com/apachecn/AiLearning
'''

from __future__ import print_function

import sys

'''
    mapper 接受原始的输入并产生中间值传递给 reducer。
    很多的mapper是并行执行的，所以需要将这些mapper的输出合并成一个值。
    即: 将中间的 key/value 对进行组合。
'''


def read_input(file):
    for line in file:
        yield line.rstrip()						# 返回值中包含输入文件的每一行的数据的一个大的List


input = read_input(sys.stdin)					# 创建一个输入的数据行的列表list

# 将输入行分割成单独的项目并存储在列表的列表中
mapperOut = [line.split('\t') for line in input]
# 输入 数据的个数，n个数据的均值，n个数据平方之后的均值
print (mapperOut)

# 累计样本总和，总和 和 平分和的总和
cumN, cumVal, cumSumSq = 0.0, 0.0, 0.0
for instance in mapperOut:
    nj = float(instance[0])
    cumN += nj
    cumVal += nj*float(instance[1])
    cumSumSq += nj*float(instance[2])

# 计算均值( varSum是计算方差的展开形式 )
mean_ = cumVal/cumN
varSum = (cumSumSq - 2*mean_*cumVal + cumN*mean_*mean_)/cumN
# 输出 数据总量，均值，平方的均值（方差）
print("数据总量: %d\t均值: %f\t方差: %f" % (cumN, mean_, varSum))
print("reduce report: still alive", file=sys.stderr)

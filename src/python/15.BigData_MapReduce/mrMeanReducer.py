#!/usr/bin/python
# coding:utf8

'''
Created on 2017-04-06
Machine Learning in Action Chapter 18
Map Reduce Job for Hadoop Streaming 
@author: Peter Harrington/ApacheCn-xy
'''


'''
    mapper 接受原始的输入并产生中间值传递给 reducer。
    很多的mapper是并行执行的，所以需要将这些mapper的输出合并成一个值。
    即：将中间的 key/value 对进行组合。
'''
import sys
from numpy import mat, mean, power

def read_input(file):
    for line in file:
        yield line.rstrip()						# 返回值中包含输入文件的每一行的数据的一个大的List
       
input = read_input(sys.stdin)					# 创建一个输入的数据行的列表list

# 将输入行分割成单独的项目并存储在列表的列表中
mapperOut = [line.split('\t') for line in input]
print (mapperOut)

# 累计样本总和，总和 和 总和 sq
cumVal=0.0
cumSumSq=0.0
cumN=0.0
for instance in mapperOut:
    nj = float(instance[0])
    cumN += nj
    cumVal += nj*float(instance[1])
    cumSumSq += nj*float(instance[2])
    
#计算均值
mean = cumVal/cumN
meanSq = cumSumSq/cumN

#输出 数据总量，均值，平方的均值（方差）
print ("%d\t%f\t%f" % (cumN, mean, meanSq))
print >> sys.stderr, "report: still alive"

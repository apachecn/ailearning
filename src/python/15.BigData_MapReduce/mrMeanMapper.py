#!/usr/bin/python
# coding:utf8
'''
Created on 2017-04-06
Machine Learning in Action Chapter 18
Map Reduce Job for Hadoop Streaming 
@author: Peter Harrington/ApacheCn-xy
'''


'''
	这个mapper文件按行读取所有的输入并创建一组对应的浮点数，然后得到数组的长度并创建NumPy矩阵。
	再对所有的值进行平方，最后将均值和平方后的均值发送出去。这些值将用来计算全局的均值和方差。

	Args：
		file 输入数据
	Return：
		
'''
import sys
from numpy import mat, mean, power

def read_input(file):
    for line in file:
        yield line.rstrip()				# 返回值中包含输入文件的每一行的数据的一个大的List
        
input = read_input(sys.stdin)			# 创建一个输入的数据行的列表list
input = [float(line) for line in input] # 将得到的数据转化为 float 类型
numInputs = len(input)					# 获取数据的个数，即输入文件的数据的行数
input = mat(input)						# 将 List 转换为矩阵
sqInput = power(input,2)				# 将矩阵的数据分别求 平方，即 2次方

# 输出 数据的个数，n个数据的均值，n个数据平方之后的均值
print ("%d\t%f\t%f" % (numInputs, mean(input), mean(sqInput))) #计算均值
print >> sys.stderr, "report: still alive"

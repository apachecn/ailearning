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
from numpy import mat, mean, power

'''
    这个mapper文件按行读取所有的输入并创建一组对应的浮点数，然后得到数组的长度并创建NumPy矩阵。
    再对所有的值进行平方，最后将均值和平方后的均值发送出去。这些值将用来计算全局的均值和方差。

    Args：
        file 输入数据
    Return：
'''


def read_input(file):
    for line in file:
        yield line.rstrip()             # 返回一个 yield 迭代器，每次获取下一个值，节约内存。


input = read_input(sys.stdin)            # 创建一个输入的数据行的列表list
input = [float(line) for line in input]  # 将得到的数据转化为 float 类型
numInputs = len(input)                   # 获取数据的个数，即输入文件的数据的行数
input = mat(input)                       # 将 List 转换为矩阵
sqInput = power(input, 2)                # 将矩阵的数据分别求 平方，即 2次方

# 输出 数据的个数，n个数据的均值，n个数据平方之后的均值
# 第一行是标准输出，也就是reducer的输出
# 第二行识标准错误输出，即对主节点作出的响应报告，表明本节点工作正常。
# 【这不就是面试的装逼重点吗？如何设计监听架构细节】注意：一个好的习惯是想标准错误输出发送报告。如果某任务10分钟内没有报告输出，则将被Hadoop中止。
print("%d\t%f\t%f" % (numInputs, mean(input), mean(sqInput)))  # 计算均值
print("map report: still alive", file=sys.stderr)

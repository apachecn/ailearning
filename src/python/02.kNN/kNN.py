'''
导入科学计算包numpy和运算符模块operator
@author: geekidentity
'''
from numpy import *
import operator

'''
 创建数据集和标签

 调用方式
 import kNN
 group, labels = createDateSet()11
'''
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels
#!/usr/bin/python
# coding:utf8

'''
Created on 2017-04-07

@author: Peter/ApacheCN-xy
'''
from mrjob.job import MRJob


class MRmean(MRJob):
    def __init__(self, *args, **kwargs): # 对数据初始化
        super(MRmean, self).__init__(*args, **kwargs)
        self.inCount = 0
        self.inSum = 0
        self.inSqSum = 0

    def map(self, key, val): # 需要 2 个参数，求数据的和与平方和
        if False: yield
        inVal = float(val)
        self.inCount += 1
        self.inSum += inVal
        self.inSqSum += inVal*inVal

    def map_final(self): # 计算数据的平均值，平方的均值，并返回
        mn = self.inSum/self.inCount
        mnSq = self.inSqSum/self.inCount
        yield (1, [self.inCount, mn, mnSq])

    def reduce(self, key, packedValues):
        cumVal=0.0; cumSumSq=0.0; cumN=0.0
        for valArr in packedValues: # 从输入流中获取值
            nj = float(valArr[0])
            cumN += nj
            cumVal += nj*float(valArr[1])
            cumSumSq += nj*float(valArr[2])
        mean = cumVal/cumN
        var = (cumSumSq - 2*mean*cumVal + cumN*mean*mean)/cumN
        yield (mean, var) # 发出平均值和方差

    def steps(self):
        return ([self.mr(mapper=self.map, mapper_final=self.map_final, reducer=self.reduce,)])


if __name__ == '__main__':
    MRmean.run()

#!/usr/bin/python
# coding:utf-8

'''
Created on 2017-04-07
Update  on 2017-11-17
Author: Peter/ApacheCN-xy/片刻
GitHub: https://github.com/apachecn/AiLearning
'''

import pickle
from numpy import *
from mrjob.job import MRJob
from mrjob.step import MRStep


class MRsvm(MRJob):
    DEFAULT_INPUT_PROTOCOL = 'json_value'

    def __init__(self, *args, **kwargs):
        super(MRsvm, self).__init__(*args, **kwargs)
        self.data = pickle.load(open('/opt/git/MachineLearnidb/15.BigData_MapReduce/svmDat27', 'r'))
        self.w = 0
        self.eta = 0.69
        self.dataList = []
        self.k = self.options.batchsize
        self.numMappers = 1
        self.t = 1  # iteration number

    def configure_args(self):
        super(MRsvm, self).configure_args()
        self.add_passthru_arg(
            '--iterations', dest='iterations', default=2, type=int,
            help='T: number of iterations to run')
        self.add_passthru_arg(
            '--batchsize', dest='batchsize', default=100, type=int,
            help='k: number of data points in a batch')

    def map(self, mapperId, inVals):  # 需要 2 个参数
        # input: nodeId, ('w', w-vector) OR nodeId, ('x', int)
        if False:
            yield
        if inVals[0] == 'w':                  # 积累 w向量
            self.w = inVals[1]
        elif inVals[0] == 'x':
            self.dataList.append(inVals[1])   # 累积数据点计算
        elif inVals[0] == 't':                # 迭代次数
            self.t = inVals[1]
        else:
            self.eta = inVals                 # 这用于 debug， eta未在map中使用

    def map_fin(self):
        labels = self.data[:, -1]
        X = self.data[:, :-1]                # 将数据重新形成 X 和 Y
        if self.w == 0:
            self.w = [0.001] * shape(X)[1]   # 在第一次迭代时，初始化 w
        for index in self.dataList:
            p = mat(self.w)*X[index, :].T    # calc p=w*dataSet[key].T
            if labels[index]*p < 1.0:
                yield (1, ['u', index])      # 确保一切数据包含相同的key
        yield (1, ['w', self.w])             # 它们将在同一个 reducer
        yield (1, ['t', self.t])

    def reduce(self, _, packedVals):
        for valArr in packedVals:            # 从流输入获取值
            if valArr[0] == 'u':
                self.dataList.append(valArr[1])
            elif valArr[0] == 'w':
                self.w = valArr[1]
            elif valArr[0] == 't':
                self.t = valArr[1]

        labels = self.data[:, -1]
        X = self.data[:, 0:-1]
        wMat = mat(self.w)
        wDelta = mat(zeros(len(self.w)))

        for index in self.dataList:
            wDelta += float(labels[index]) * X[index, :]  # wDelta += label*dataSet
        eta = 1.0/(2.0*self.t)       # calc new: eta
        # calc new: w = (1.0 - 1/t)*w + (eta/k)*wDelta
        wMat = (1.0 - 1.0/self.t)*wMat + (eta/self.k)*wDelta
        for mapperNum in range(1, self.numMappers+1):
            yield (mapperNum, ['w', wMat.tolist()[0]])    # 发出 w
            if self.t < self.options.iterations:
                yield (mapperNum, ['t', self.t+1])        # 增量 T
                for j in range(self.k/self.numMappers):   # emit random ints for mappers iid
                    yield (mapperNum, ['x', random.randint(shape(self.data)[0])])

    def steps(self):
        return [MRStep(mapper=self.map, reducer=self.reduce, mapper_final=self.map_fin)] * self.options.iterations


if __name__ == '__main__':
    MRsvm.run()

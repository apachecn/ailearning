'''
Created on Feb 27, 2011
MapReduce version of Pegasos SVM
Using mrjob to automate job flow
@author: Peter
'''
from mrjob.job import MRJob

import pickle
from numpy import *

class MRsvm(MRJob):
                                                 
    def map(self, mapperId, inVals): #needs exactly 2 arguments
        if False: yield
        yield (1, 22)

    def reduce(self, _, packedVals):
        yield "fuck ass" 
        
    def steps(self):
        return ([self.mr(mapper=self.map, reducer=self.reduce)])

if __name__ == '__main__':
    MRsvm.run()

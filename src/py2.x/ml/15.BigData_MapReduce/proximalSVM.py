#!/usr/bin/python
# coding:utf8
'''
Created on 2011-02-25
Update  on 2017-06-20
Author: Peter/ApacheCN-xy/片刻
GitHub: https://github.com/apachecn/AiLearning
'''
from __future__ import print_function
import base64
import pickle

import numpy


def map(key, value):
   # input key= class for one training example, e.g. "-1.0"
   classes = [float(item) for item in key.split(",")]   # e.g. [-1.0]
   D = numpy.diag(classes)

   # input value = feature vector for one training example, e.g. "3.0, 7.0, 2.0"
   featurematrix = [float(item) for item in value.split(",")]
   A = numpy.matrix(featurematrix)

   # create matrix E and vector e
   e = numpy.matrix(numpy.ones(len(A)).reshape(len(A), 1))
   E = numpy.matrix(numpy.append(A, -e, axis=1)) 

   # create a tuple with the values to be used by reducer
   # and encode it with base64 to avoid potential trouble with '\t' and '\n' used
   # as default separators in Hadoop Streaming
   producedvalue = base64.b64encode(pickle.dumps((E.T*E, E.T*D*e)))

   # note: a single constant key "producedkey" sends to only one reducer
   # somewhat "atypical" due to low degree of parallism on reducer side
   print("producedkey\t%s" % (producedvalue))

def reduce(key, values, mu=0.1):
  sumETE = None
  sumETDe = None

  # key isn't used, so ignoring it with _ (underscore).
  for _, value in values:
    # unpickle values
    ETE, ETDe = pickle.loads(base64.b64decode(value))
    if sumETE == None:
      # create the I/mu with correct dimensions
      sumETE = numpy.matrix(numpy.eye(ETE.shape[1])/mu)
    sumETE += ETE

    if sumETDe == None:
      # create sumETDe with correct dimensions
      sumETDe = ETDe
    else:
      sumETDe += ETDe

    # note: omega = result[:-1] and gamma = result[-1]
    # but printing entire vector as output
    result = sumETE.I*sumETDe
    print("%s\t%s" % (key, str(result.tolist())))

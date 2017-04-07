'''
Created on 2017-04-07
Sequential Pegasos 
the input T is k*T in Batch Pegasos
@author: Peter/ApacheCN-xy
'''

from numpy import *

def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        #dataMat.append([float(lineArr[0]), float(lineArr[1]), float(lineArr[2])])
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def seqPegasos(dataSet, labels, lam, T):
    m,n = shape(dataSet); w = zeros(n)
    for t in range(1, T+1):
        i = random.randint(m)
        eta = 1.0/(lam*t)
        p = predict(w, dataSet[i,:])
        if labels[i]*p < 1:
            w = (1.0 - 1/t)*w + eta*labels[i]*dataSet[i,:]
        else:
            w = (1.0 - 1/t)*w
        print w
    return w
        
def predict(w, x):
    return w*x.T

def batchPegasos(dataSet, labels, lam, T, k):
    m,n = shape(dataSet); w = zeros(n); 
    dataIndex = range(m)
    for t in range(1, T+1):
        wDelta = mat(zeros(n)) # 重置 wDelta
        eta = 1.0/(lam*t)
        random.shuffle(dataIndex)
        for j in range(k):# 全部的训练集
            i = dataIndex[j]
            p = predict(w, dataSet[i,:])        # mapper 代码
            if labels[i]*p < 1:                 # mapper 代码
                wDelta += labels[i]*dataSet[i,:].A # 累积变化  
        w = (1.0 - 1/t)*w + (eta/k)*wDelta       # 在每个 T上应用更改
    return w

datArr,labelList = loadDataSet('testSet.txt')
datMat = mat(datArr)
#finalWs = seqPegasos(datMat, labelList, 2, 5000)
finalWs = batchPegasos(datMat, labelList, 2, 50, 100)
print finalWs

import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
x1=[]; y1=[]; xm1=[]; ym1=[]
for i in range(len(labelList)):
    if labelList[i] == 1.0:
        x1.append(datMat[i,0]); y1.append(datMat[i,1])
    else:
        xm1.append(datMat[i,0]); ym1.append(datMat[i,1])
ax.scatter(x1, y1, marker='s', s=90)
ax.scatter(xm1, ym1, marker='o', s=50, c='red')
x = arange(-6.0, 8.0, 0.1)
y = (-finalWs[0,0]*x - 0)/finalWs[0,1]
#y2 = (0.43799*x)/0.12316
y2 = (0.498442*x)/0.092387 #2 iterations
ax.plot(x,y)
ax.plot(x,y2,'g-.')
ax.axis([-6,8,-4,5])
ax.legend(('50 Iterations', '2 Iterations') )
plt.show()
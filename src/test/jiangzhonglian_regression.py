#!/usr/bin/python
# coding: utf8

'''
Created on Jan 8, 2011

@author: Peter
'''

import os
from numpy import *
import matplotlib.pylab as plt


def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) - 1 #get number of fields
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat


def standRegres(xArr,yArr):
    # >>> A.T  # transpose, 转置
    xMat = mat(xArr); yMat = mat(yArr).T
    # 转置矩阵*矩阵
    xTx = xMat.T*xMat
    if linalg.det(xTx) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return
    # >>> print A.I  # inverse, 逆矩阵
    # print xTx.I, "*"*10, xMat.T, "*"*10, yMat
    ws = xTx.I * (xMat.T*yMat)  # 最小二乘法求最优解
    return ws


def plotBestFit(xArr, yArr, ws):

    xMat = mat(xArr)
    yMat = mat(yArr)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0])

    yHat = xMat*ws
    # 再计算相关系数
    print "相关系数\n", corrcoef(yHat.T, yMat)

    xMat.sort(0)
    yHat = xMat*ws
    n = shape(xMat)[0]
    xcord = []; ycord = []
    for i in range(n):
        xcord.append(xMat[i, 1]); ycord.append(yHat[i, 0])

    ax.plot(xcord, ycord, c='red')
    plt.xlabel('X'); plt.ylabel('Y')
    plt.show()


def main1():
    # w0*x0+w1*x1+w2*x2=f(x)
    project_dir = os.path.dirname(os.path.dirname(os.getcwd()))
    # 1.收集并准备数据
    xArr, yArr = loadDataSet("%s/resources/ex0.txt" % project_dir)
    # print xArr, '---\n', yArr
    # 2.训练模型，  f(x)=a1*x1+b2*x2+..+nn*xn中 (a1,b2, .., nn).T的矩阵值
    ws = standRegres(xArr, yArr)
    print '*'*30, '---\n', ws

    # 数据可视化
    plotBestFit(xArr, yArr, ws)


def lwlr(testPoint, xArr, yArr,k=1.0):
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))
    for j in range(m):                      #next 2 lines create weights matrix
        diffMat = testPoint - xMat[j,:]
        # 高斯核对应的加权
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return

    # 加权的回归系数求解
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


def lwlrTest(testArr,xArr,yArr,k=1.0):  #loops over all the data points and applies lwlr to each one
    m = shape(testArr)[0]
    # m*1的矩阵
    # 函数 zeros 创建一个全0的数组
    yHat = zeros(m)
    print "shape(yHat)", shape(yHat)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat


def lwlrTestPlot(xArr, yArr, yHat):

    xMat = mat(xArr)
    yMat = mat(yArr)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0])

    # 再计算相关系数
    print "相关系数\n", corrcoef(yHat.T, yMat)

    n = shape(xMat)[0]
    xcord = []; ycord = []
    for i in range(n):
        xcord.append(xMat[i, 1]), ycord.append(yHat[i])

    xcord.sort(), ycord.sort()
    # print xcord, "------\n", ycord
    ax.plot(xcord, ycord, c='red')
    plt.xlabel('X'); plt.ylabel('Y')
    plt.show()


def main2():
    # w0*x0+w1*x1+w2*x2=f(x)
    # project_dir = os.path.dirname(os.path.dirname(os.getcwd()))
    # 1.收集并准备数据
    # xArr, yArr = loadDataSet("%s/resources/ex0.txt" % project_dir)
    xArr, yArr = loadDataSet("testData/Regression_data.txt")
    # print xArr, '---\n', yArr
    # 2.训练模型，  f(x)=a1*x1+b2*x2+..+nn*xn中 (a1,b2, .., nn).T的矩阵值
    yHat = lwlrTest(xArr, xArr, yArr, 0.003)
    print xArr, '---\n', yHat[1]

    # 数据可视化
    lwlrTestPlot(xArr, yArr, yHat)


if __name__ == "__main__":
    # 线性回归
    # main1()
    # 局部加权线性回归
    main2()


def rssError(yArr,yHatArr): #yArr and yHatArr both need to be arrays
    return ((yArr-yHatArr)**2).sum()

def ridgeRegres(xMat,yMat,lam=0.2):
    xTx = xMat.T*xMat
    denom = xTx + eye(shape(xMat)[1])*lam
    if linalg.det(denom) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return
    ws = denom.I * (xMat.T*yMat)
    return ws

def ridgeTest(xArr,yArr):
    xMat = mat(xArr); yMat=mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean     #to eliminate X0 take mean off of Y
    #regularize X's
    xMeans = mean(xMat,0)   #calc mean then subtract it off
    xVar = var(xMat,0)      #calc variance of Xi then divide by it
    xMat = (xMat - xMeans)/xVar
    numTestPts = 30
    wMat = zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,exp(i-10))
        wMat[i,:]=ws.T
    return wMat

def regularize(xMat):#regularize by columns
    inMat = xMat.copy()
    inMeans = mean(inMat,0)   #calc mean then subtract it off
    inVar = var(inMat,0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat

def stageWise(xArr,yArr,eps=0.01,numIt=100):
    xMat = mat(xArr); yMat=mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean     #can also regularize ys but will get smaller coef
    xMat = regularize(xMat)
    m,n=shape(xMat)
    #returnMat = zeros((numIt,n)) #testing code remove
    ws = zeros((n,1)); wsTest = ws.copy(); wsMax = ws.copy()
    for i in range(numIt):
        print ws.T
        lowestError = inf;
        for j in range(n):
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yTest = xMat*wsTest
                rssE = rssError(yMat.A,yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        #returnMat[i,:]=ws.T
        #return returnMat

def scrapePage(inFile,outFile,yr,numPce,origPrc):
   from BeautifulSoup import BeautifulSoup
   fr = open(inFile); fw=open(outFile,'a') #a is append mode writing
   soup = BeautifulSoup(fr.read())
   i=1
   currentRow = soup.findAll('table', r="%d" % i)
   while(len(currentRow)!=0):
       title = currentRow[0].findAll('a')[1].text
       lwrTitle = title.lower()
       if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
           newFlag = 1.0
       else:
           newFlag = 0.0
       soldUnicde = currentRow[0].findAll('td')[3].findAll('span')
       if len(soldUnicde)==0:
           print "item #%d did not sell" % i
       else:
           soldPrice = currentRow[0].findAll('td')[4]
           priceStr = soldPrice.text
           priceStr = priceStr.replace('$','') #strips out $
           priceStr = priceStr.replace(',','') #strips out ,
           if len(soldPrice)>1:
               priceStr = priceStr.replace('Free shipping', '') #strips out Free Shipping
           print "%s\t%d\t%s" % (priceStr,newFlag,title)
           fw.write("%d\t%d\t%d\t%f\t%s\n" % (yr,numPce,newFlag,origPrc,priceStr))
       i += 1
       currentRow = soup.findAll('table', r="%d" % i)
   fw.close()

from time import sleep
import json
import urllib2
def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    sleep(10)
    myAPIstr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (myAPIstr, setNum)
    pg = urllib2.urlopen(searchURL)
    retDict = json.loads(pg.read())
    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]
            if currItem['product']['condition'] == 'new':
                newFlag = 1
            else: newFlag = 0
            listOfInv = currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice = item['price']
                if  sellingPrice > origPrc * 0.5:
                    print "%d\t%d\t%d\t%f\t%f" % (yr,numPce,newFlag,origPrc, sellingPrice)
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except: print 'problem with item %d' % i

def setDataCollect(retX, retY):
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)

def crossValidation(xArr,yArr,numVal=10):
    m = len(yArr)
    indexList = range(m)
    errorMat = zeros((numVal,30))#create error mat 30columns numVal rows
    for i in range(numVal):
        trainX=[]; trainY=[]
        testX = []; testY = []
        random.shuffle(indexList)
        for j in range(m):#create training set based on first 90% of values in indexList
            if j < m*0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX,trainY)    #get 30 weight vectors from ridge
        for k in range(30):#loop over all of the ridge estimates
            matTestX = mat(testX); matTrainX=mat(trainX)
            meanTrain = mean(matTrainX,0)
            varTrain = var(matTrainX,0)
            matTestX = (matTestX-meanTrain)/varTrain #regularize test with training params
            yEst = matTestX * mat(wMat[k,:]).T + mean(trainY)#test ridge results and store
            errorMat[i,k]=rssError(yEst.T.A,array(testY))
            #print errorMat[i,k]
    meanErrors = mean(errorMat,0)#calc avg performance of the different ridge weight vectors
    minMean = float(min(meanErrors))
    bestWeights = wMat[nonzero(meanErrors==minMean)]
    #can unregularize to get model
    #when we regularized we wrote Xreg = (x-meanX)/var(x)
    #we can now write in terms of x not Xreg:  x*w/var(x) - meanX/var(x) +meanY
    xMat = mat(xArr); yMat=mat(yArr).T
    meanX = mean(xMat,0); varX = var(xMat,0)
    unReg = bestWeights/varX
    print "the best model from Ridge Regression is:\n",unReg
    print "with constant term: ",-1*sum(multiply(meanX,unReg)) + mean(yMat)
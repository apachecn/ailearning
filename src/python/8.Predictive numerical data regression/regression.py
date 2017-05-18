#!/usr/bin/python
# coding:utf8

'''
Created on Jan 8, 2011
Update  on 2017-05-18
@author: Peter Harrington/ApacheCN-小瑶
《机器学习实战》更新地址：https://github.com/apachecn/MachineLearning
'''


from numpy import *
import matplotlib.pylab as plt

def loadDataSet(fileName):                 #解析以tab键分隔的文件中的浮点数
    numFeat = len(open(fileName).readline().split('\t')) - 1 #获得每一行的输入数据，最后一个代表真实值 
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():            #读取每一行
        lineArr =[]
        curLine = line.strip().split('\t') #删除一行中以tab分隔的数据前后的空白符号
        for i in range(numFeat):           #从0到2，不包括2
            lineArr.append(float(curLine[i]))#将数据添加到lineArr List中，每一行数据测试数据组成一个行向量
        dataMat.append(lineArr)            #将测试数据的输入数据部分存储到dataMat矩阵中
        labelMat.append(float(curLine[-1]))#将每一行的最后一个数据，即真实的目标变量存储到labelMat矩阵中
    return dataMat,labelMat

def standRegres(xArr,yArr):               #线性回归
    xMat = mat(xArr); yMat = mat(yArr).T  #mat()函数将xArr，yArr转换为矩阵
    xTx = xMat.T*xMat                     #矩阵乘法的条件是左矩阵的列数等于右矩阵的行数
    if linalg.det(xTx) == 0.0:            #因为要用到xTx的逆矩阵，所以事先需要确定计算得到的xTx是否可逆，条件是矩阵的行列式不为0
        print ("This matrix is singular, cannot do inverse")
        return
    # 最小二乘法
    # http://www.apache.wiki/pages/viewpage.action?pageId=5505133
    ws = xTx.I * (xMat.T*yMat)            #书中的公式，求得w的最优解
    return ws

def lwlr(testPoint,xArr,yArr,k=1.0):      #局部加权线性回归
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]                    #获得xMat矩阵的行数
    weights = mat(eye((m)))               #eye()返回一个对角线元素为1，其他元素为0的二维数组，创建权重矩阵
    for j in range(m):                      #下面两行创建权重矩阵
        diffMat = testPoint - xMat[j,:]     #遍历数据集，计算每个样本点对应的权重值
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))#k控制衰减的速度
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print ("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat)) #计算出回归系数的一个估计
    return testPoint * ws

def lwlrTest(testArr,xArr,yArr,k=1.0):  #循环所有的数据点，并将lwlr运用于所有的数据点
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

def lwlrTestPlot(xArr,yArr,k=1.0):  #首先将 X 排序，其余的都与lwlrTest相同，这样更容易绘图
    yHat = zeros(shape(yArr))       
    xCopy = mat(xArr)
    xCopy.sort(0)
    for i in range(shape(xArr)[0]):
        yHat[i] = lwlr(xCopy[i],xArr,yArr,k)
    return yHat,xCopy

def rssError(yArr,yHatArr): #yArr 和 yHatArr 两者都需要是数组
    return ((yArr-yHatArr)**2).sum()

def ridgeRegres(xMat,yMat,lam=0.2):  #岭回归
    xTx = xMat.T*xMat
    denom = xTx + eye(shape(xMat)[1])*lam   #按照书上的公式计算计算回归系数
    if linalg.det(denom) == 0.0:            #检查行列式是否为零，即矩阵是否可逆
        print ("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T*yMat)
    return ws

def ridgeTest(xArr,yArr):
    xMat = mat(xArr); yMat=mat(yArr).T
    yMean = mean(yMat,0)    #计算Y均值
    yMat = yMat - yMean     #Y的所有的特征减去均值
                            #标准化 x
    xMeans = mean(xMat,0)   #X计算平均值
    xVar = var(xMat,0)      #然后计算 X的方差
    xMat = (xMat - xMeans)/xVar
    numTestPts = 30
    wMat = zeros((numTestPts,shape(xMat)[1]))#创建30 * m 的全部数据为0 的矩阵
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,exp(i-10))#exp返回e^x
        wMat[i,:]=ws.T
    return wMat


def regularize(xMat):#按列进行规范化
    inMat = xMat.copy()
    inMeans = mean(inMat,0)   #计算平均值然后减去它
    inVar = var(inMat,0)      #计算除以Xi的方差
    inMat = (inMat - inMeans)/inVar
    return inMat

def stageWise(xArr,yArr,eps=0.01,numIt=100):
    xMat = mat(xArr); yMat=mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean     #也可以规则化ys但会得到更小的coef
    xMat = regularize(xMat)
    m,n=shape(xMat)
    #returnMat = zeros((numIt,n)) #测试代码删除
    ws = zeros((n,1)); wsTest = ws.copy(); wsMax = ws.copy()
    for i in range(numIt):
        print (ws.T)
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

#def scrapePage(inFile,outFile,yr,numPce,origPrc):
#    from BeautifulSoup import BeautifulSoup
#    fr = open(inFile); fw=open(outFile,'a') #a is append mode writing
#    soup = BeautifulSoup(fr.read())
#    i=1
#    currentRow = soup.findAll('table', r="%d" % i)
#    while(len(currentRow)!=0):
#        title = currentRow[0].findAll('a')[1].text
#        lwrTitle = title.lower()
#        if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
#            newFlag = 1.0
#        else:
#            newFlag = 0.0
#        soldUnicde = currentRow[0].findAll('td')[3].findAll('span')
#        if len(soldUnicde)==0:
#            print "item #%d did not sell" % i
#        else:
#            soldPrice = currentRow[0].findAll('td')[4]
#            priceStr = soldPrice.text
#            priceStr = priceStr.replace('$','') #strips out $
#            priceStr = priceStr.replace(',','') #strips out ,
#            if len(soldPrice)>1:
#                priceStr = priceStr.replace('Free shipping', '') #strips out Free Shipping
#            print "%s\t%d\t%s" % (priceStr,newFlag,title)
#            fw.write("%d\t%d\t%d\t%f\t%s\n" % (yr,numPce,newFlag,origPrc,priceStr))
#        i += 1
#        currentRow = soup.findAll('table', r="%d" % i)
#    fw.close()
 
'''   
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
                    print ("%d\t%d\t%d\t%f\t%f" % (yr,numPce,newFlag,origPrc, sellingPrice))
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except: print ('problem with item %d' % i)
    
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
    errorMat = zeros((numVal,30))#create error mat 30columns numVal rows创建error mat 30columns numVal 行
    for i in range(numVal):
        trainX=[]; trainY=[]
        testX = []; testY = []
        random.shuffle(indexList)
        for j in range(m):#create training set based on first 90% of values in indexList
                          #基于indexList中的前90%的值创建训练集
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
    print ("the best model from Ridge Regression is:\n",unReg)
    print ("with constant term: ",-1*sum(multiply(meanX,unReg)) + mean(yMat))
'''










    #test for standRegression
def regression1():
    xArr, yArr = loadDataSet("input/8.Regression/data.txt")
    xMat = mat(xArr)
    yMat = mat(yArr)
    ws = standRegres(xArr, yArr)
    fig = plt.figure()
    ax = fig.add_subplot(111)               #add_subplot(349)函数的参数的意思是，将画布分成3行4列图像画在从左到右从上到下第9块
    ax.scatter(xMat[:, 1].flatten(), yMat.T[:, 0].flatten().A[0]) #scatter 的x是xMat中的第二列，y是yMat的第一列
    xCopy = xMat.copy() 
    xCopy.sort(0)
    yHat = xCopy * ws
    ax.plot(xCopy[:, 1], yHat)
    plt.show()
    



    #test for LWLR
def regression2():
    xArr, yArr = loadDataSet("input/8.Regression/data.txt")
    yHat = lwlrTest(xArr, xArr, yArr, 0.003)
    xMat = mat(xArr)
    srtInd = xMat[:,1].argsort(0)           #argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出
    xSort=xMat[srtInd][:,0,:]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:,1], yHat[srtInd])
    ax.scatter(xMat[:,1].flatten().A[0], mat(yArr).T.flatten().A[0] , s=2, c='red')
    plt.show()


#test for ridgeRegression
def regression3():
    abX,abY = loadDataSet("input/8.Regression/abalone.txt")
    ridgeWeights = ridgeTest(abX, abY)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.show()


#test for stageWise
def regression4():
    xArr,yArr=loadDataSet("input/8.Regression/abalone.txt")
    stageWise(xArr,yArr,0.01,200)
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xMat = regularize(xMat)
    yM = mean(yMat,0)
    yMat = yMat - yM
    weights = standRegres(xMat, yMat.T)
    print (weights.T)

if __name__ == "__main__":
    # regression1()
    regression2()
    # regression3()
    # regression4()

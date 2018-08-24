#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = "Ng WaiMing"

from numpy import *
from time import sleep
import matplotlib
from matplotlib import pyplot as plt


def loadDataSet(fileName):
    '''
    加载数据集
    :param fileName:
    :return:
    '''
    # 初始化一个空列表
    dataSet = []
    # 读取文件
    fr = open(fileName)
    # 循环遍历文件所有行
    for line in fr.readlines():
        # 切割每一行的数据
        curLine = line.strip().split('\t')
        # 将数据转换为浮点类型,便于后面的计算
        # fltLine = [float(x) for x in curLine]
        # 将数据追加到dataMat
        fltLine = list(map(float,curLine))    # 映射所有的元素为 float（浮点数）类型
        dataSet.append(fltLine)
    # 返回dataMat
    return dataSet


def distEclud(vecA, vecB):
    '''
    欧氏距离计算函数
    :param vecA:
    :param vecB:
    :return:
    '''
    return sqrt(sum(power(vecA - vecB, 2)))


def randCent(dataMat, k):
    '''
    为给定数据集构建一个包含K个随机质心的集合,
    随机质心必须要在整个数据集的边界之内,这可以通过找到数据集每一维的最小和最大值来完成
    然后生成0到1.0之间的随机数并通过取值范围和最小值,以便确保随机点在数据的边界之内
    :param dataMat:
    :param k:
    :return:
    '''
    # 获取样本数与特征值
    m, n = shape(dataMat)
    # 初始化质心,创建(k,n)个以零填充的矩阵
    centroids = mat(zeros((k, n)))
    # 循环遍历特征值
    for j in range(n):
        # 计算每一列的最小值
        minJ = min(dataMat[:, j])
        # 计算每一列的范围值
        rangeJ = float(max(dataMat[:, j]) - minJ)
        # 计算每一列的质心,并将值赋给centroids
        centroids[:, j] = mat(minJ + rangeJ * random.rand(k, 1))
    # 返回质心
    return centroids


def kMeans(dataMat, k, distMeas=distEclud, createCent=randCent):
    '''
    创建K个质心,然后将每个店分配到最近的质心,再重新计算质心。
    这个过程重复数次,直到数据点的簇分配结果不再改变为止
    :param dataMat: 数据集
    :param k: 簇的数目
    :param distMeans: 计算距离
    :param createCent: 创建初始质心
    :return:
    '''
    # 获取样本数和特征数
    m, n = shape(dataMat)
    # 初始化一个矩阵来存储每个点的簇分配结果
    # clusterAssment包含两个列:一列记录簇索引值,第二列存储误差(误差是指当前点到簇质心的距离,后面会使用该误差来评价聚类的效果)
    clusterAssment = mat(zeros((m, 2)))
    # 创建质心,随机K个质心
    centroids = createCent(dataMat, k)
    # 初始化标志变量,用于判断迭代是否继续,如果True,则继续迭代
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        # 遍历所有数据找到距离每个点最近的质心,
        # 可以通过对每个点遍历所有质心并计算点到每个质心的距离来完成
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                # 计算数据点到质心的距离
                # 计算距离是使用distMeas参数给出的距离公式,默认距离函数是distEclud
                distJI = distMeas(centroids[j, :], dataMat[i, :])
                # 如果距离比minDist(最小距离)还小,更新minDist(最小距离)和最小质心的index(索引)
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            # 如果任一点的簇分配结果发生改变,则更新clusterChanged标志
            if clusterAssment[i, 0] != minIndex: clusterChanged = True
            # 更新簇分配结果为最小质心的index(索引),minDist(最小距离)的平方
            clusterAssment[i, :] = minIndex, minDist ** 2
        # print(centroids)
        # 遍历所有质心并更新它们的取值
        for cent in range(k):
            # 通过数据过滤来获得给定簇的所有点
            ptsInClust = dataMat[nonzero(clusterAssment[:, 0].A == cent)[0]]
            # 计算所有点的均值,axis=0表示沿矩阵的列方向进行均值计算
            centroids[cent, :] = mean(ptsInClust, axis=0)
    # 返回所有的类质心与点分配结果
    return centroids, clusterAssment


def biKmeans(dataMat, k, distMeas=distEclud):
    '''
    在给定数据集,所期望的簇数目和距离计算方法的条件下,函数返回聚类结果
    :param dataMat:
    :param k:
    :param distMeas:
    :return:
    '''
    m, n = shape(dataMat)
    # 创建一个矩阵来存储数据集中每个点的簇分配结果及平方误差
    clusterAssment = mat(zeros((m, 2)))
    # 计算整个数据集的质心,并使用一个列表来保留所有的质心
    centroid0 = mean(dataMat, axis=0).tolist()[0]
    centList = [centroid0]
    # 遍历数据集中所有点来计算每个点到质心的误差值
    for j in range(m):
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataMat[j, :]) ** 2
    # 对簇不停的进行划分,直到得到想要的簇数目为止
    while (len(centList) < k):
        # 初始化最小SSE为无穷大,用于比较划分前后的SSE
        lowestSSE = inf
        # 通过考察簇列表中的值来获得当前簇的数目,遍历所有的簇来决定最佳的簇进行划分
        for i in range(len(centList)):
            # 对每一个簇,将该簇中的所有点堪称一个小的数据集
            ptsInCurrCluster = dataMat[nonzero(clusterAssment[:, 0].A == i)[0], :]
            # 将ptsInCurrCluster输入到函数kMeans中进行处理,k=2,
            # kMeans会生成两个质心(簇),同时给出每个簇的误差值
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            # 将误差值与剩余数据集的误差之和作为本次划分的误差
            sseSplit = sum(splitClustAss[:, 1])
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print('sseSplit, and notSplit: ', sseSplit, sseNotSplit)
            # 如果本次划分的SSE值最小,则本次划分被保存
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        # 找出最好的簇分配结果
        # 调用kmeans函数并且指定簇数为2时,会得到两个编号分别为0和1的结果簇
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        # 更新为最佳质心
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print('the bestCentToSplit is: ', bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        # 更新质心列表
        # 更新原质心list中的第i个质心为使用二分kMeans后bestNewCents的第一个质心
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]
        # 添加bestNewCents的第二个质心
        centList.append(bestNewCents[1, :].tolist()[0])
        # 重新分配最好簇下的数据(质心)以及SSE
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
    return mat(centList), clusterAssment


def distSLC(vecA, vecB):
    '''
    返回地球表面两点间的距离,单位是英里
    给定两个点的经纬度,可以使用球面余弦定理来计算亮点的距离
    :param vecA:
    :param vecB:
    :return:
    '''
    # 经度和维度用角度作为单位,但是sin()和cos()以弧度为输入.
    # 可以将江都除以180度然后再诚意圆周率pi转换为弧度
    a = sin(vecA[0, 1] * pi / 180) * sin(vecB[0, 1] * pi / 180)
    b = cos(vecA[0, 1] * pi / 180) * cos(vecB[0, 1] * pi / 180) * \
        cos(pi * (vecB[0, 0] - vecA[0, 0]) / 180)
    return arccos(a + b) * 6371.0


def clusterClubs(fileName, imgName, numClust=5):
    '''
    将文本文件的解析,聚类以及画图都封装在一起
    :param fileName: 文本数据路径
    :param imgName: 图片路径
    :param numClust: 希望得到的簇数目
    :return:
    '''
    # 创建一个空列表
    datList = []
    # 打开文本文件获取第4列和第5列,这两列分别对应维度和经度,然后将这些值封装到datList
    for line in open(fileName).readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = mat(datList)
    # 调用biKmeans并使用distSLC函数作为聚类中使用的距离计算方式
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    # 创建一幅图和一个举行,使用该矩形来决定绘制图的哪一部分
    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    # 构建一个标记形状的列表用于绘制散点图
    scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    # 使用imread函数基于一幅图像来创建矩阵
    imgP = plt.imread(imgName)
    # 使用imshow绘制该矩阵
    ax0.imshow(imgP)
    # 再同一幅图上绘制一张新图,允许使用两套坐标系统并不做任何缩放或偏移
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    # 遍历每一个簇并将它们一一画出来,标记类型从前面创建的scatterMarkers列表中得到
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:, 0].A == i)[0], :]
        # 使用索引i % len(scatterMarkers)来选择标记形状,这意味这当有更多簇时,可以循环使用这标记
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        # 使用十字标记来表示簇中心并在图中显示
        ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0], ptsInCurrCluster[:, 1].flatten().A[0], marker=markerStyle,
                    s=90)
    ax1.scatter(myCentroids[:, 0].flatten().A[0], myCentroids[:, 1].flatten().A[0], marker='+', s=300)
    plt.show()

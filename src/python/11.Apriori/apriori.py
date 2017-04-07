#!/usr/bin/python
# coding: utf8

'''
Created on Mar 24, 2011
Update on 2017-03-16
Ch 11 code
@author: Peter/片刻
'''
print(__doc__)
from numpy import *


def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def createC1(dataSet):
    """createC1(创建一个数据集)

    Args:
        dataSet   加载的原始数据集
    Returns:
        frozenset 返回一个元素为frozenset格式的list
    """
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                # 遍历所有的元素，然后append到C1中
                C1.append([item])
    # 对数组进行 从小到大 的排序
    print 'sort前=', C1
    C1.sort()
    # frozenset表示冻结的set集合，元素无改变；可以把它当字典的key来使用
    print 'sort后=', C1
    print 'frozenset=', map(frozenset, C1)
    return map(frozenset, C1)


def scanD(D, Ck, minSupport):
    """scanD(计算支持support， retList表示满足support的key, supportData表示全集的集合)

    Args:
        D    原始数据集, D用来判断，CK中的元素，是否存在于原数据D中
        Ck   所有key的元素集合
    Returns:
        retList      支持度大于阈值的集合
        supportData  全量key的字典集合
    """
    # ssCnt 临时存放Ck的元素集合，查看Ck每个元素 并 计算元素出现的次数 生成相应的字典
    ssCnt = {}
    for tid in D:
        for can in Ck:
            # s.issubset(t)  测试是否 s 中的每一个元素都在 t 中
            if can.issubset(tid):
                if not ssCnt.has_key(can):
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    # 元素有多少行
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        # 计算支持度
        support = ssCnt[key]/numItems
        if support >= minSupport:
            # 在retList的首位插入元素，只存储支持度满足频繁项集的值
            retList.insert(0, key)
        # 存储所有的key和对应的support值
        supportData[key] = support
    return retList, supportData


# creates Ck
def aprioriGen(Lk, k):
    """aprioriGen(循环数据集，然后进行两两合并)

    Args:
        Lk 频繁项集的list
        k  元素的前k-2相同，就进行合并
    Returns:
        retList 元素两两合并的数据集
    """
    retList = []
    lenLk = len(Lk)
    # 循环Lk这个数组
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            L1 = list(Lk[i])[: k-2]
            L2 = list(Lk[j])[: k-2]
            # print '-----i=', i, k-2, Lk, Lk[i], list(Lk[i])[: k-2]
            # print '-----j=', j, k-2, Lk, Lk[j], list(Lk[j])[: k-2]
            L1.sort()
            L2.sort()
            # 第一次L1,L2为空，元素直接进行合并，返回元素两两合并的数据集
            # if first k-2 elements are equal
            if L1 == L2:
                # set union
                # print 'union=', Lk[i] | Lk[j], Lk[i], Lk[j]
                retList.append(Lk[i] | Lk[j])
    return retList


def apriori(dataSet, minSupport=0.5):
    """apriori

    Args:
        dataSet 原始数据集
        minSupport 支持度的阈值
    Returns:
        L 频繁项集的全集
        supportData 所有元素和支持度的全集
    """
    # C1表示让元素转化为frozenset，并按照顺序存放到list中
    C1 = createC1(dataSet)
    # 对每一行进行set转换，然后存放到集合中
    D = map(set, dataSet)
    print 'D=', D
    # 计算支持support， 计算在数据集D中，C1集合中元素是否满足minSupport
    # L1表示满足support的key, supportData表示全集的集合
    L1, supportData = scanD(D, C1, minSupport)
    # print "L1=", L1, "\n", "outcome: ", supportData

    # L加了一层list, L一共2曾list
    L = [L1]
    k = 2
    # 判断L的第一层list是否有元素，就说说：L1是否有元素
    while (len(L[k-2]) > 0):
        # 如果L的每个元素子项和后面任意一个元素项，进行对比， 就对前后2个频繁项集的合并，得到CK
        print 'k=', k, L, L[k-2]
        Ck = aprioriGen(L[k-2], k)
        print 'Ck', Ck

        # 计算合并后的数据集的支持度
        # Lk满足支持度的key的list， supK表示key全集
        Lk, supK = scanD(D, Ck, minSupport)
        # 如果字典没有，就追加元素，如果有，就更新元素
        supportData.update(supK)
        if len(Lk) == 0:
            break
        # Lk表示满足频繁子项的集合，L元素在增加
        # l=[[set(1), set(2), set(3)]]
        # l=[[set(1), set(2), set(3)]  [set(1, 2), set(2, 3)]]
        L.append(Lk)
        k += 1
        # print 'k=', k, len(L[k-2])
    return L, supportData


def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    """calcConf(对2元素的频繁项，计算置信度，例如： {1,2}/{1} 或者{1,2}/{2} 看是否满足条件)

    Args:
        freqSet 每一组的各个元素
        H 将元素变成set集合
        supportData 所有元素的支持度全集
        brl bigRuleList的空数组
        minConf 置信度的阈值
    Returns:
        prunedH 记录 置信度大于阈值的集合
    """
    # 记录 可信度大于阈值的集合
    prunedH = []
    for conseq in H:
        # 计算自信度的值，例如元素 H=set(1, 2)， 分别求：supportData[1] 和 supportData[2]
        # 例如： confidence= frozenset([2, 3, 5]) frozenset([2, 3]) frozenset([5])
        # print 'confidence=', freqSet, conseq, freqSet-conseq
        conf = supportData[freqSet]/supportData[freqSet-conseq]
        if conf >= minConf:
            # 意思是说： 只要买了freqSet-conseq集合，一定会买 conseq集合【freqSet-conseq集合 和 conseq集合 是全集】
            print freqSet-conseq, '-->', conseq, 'conf:', conf
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH


def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    """rulesFromConseq

    Args:
        freqSet 每一组的各个元素
        H 将元素变成set集合
        supportData 所有元素的支持度全集
        brl bigRuleList的空数组
        minConf 置信度的阈值
    Returns:
        prunedH 记录 可信度大于阈值的集合
    """
    # H[0]是freqSet的元素组合的第一个元素
    m = len(H[0])
    # 判断，freqSet的长度是否>组合的长度+1, 避免过度匹配 例如：计算过一边{1,2,3} 和 {1, 2} {1, 3}，就没必要再计算了进一步合并来计算 {1,2,3}和{1,2,3}的组合关系
    if (len(freqSet) > (m + 1)):
        print 'freqSet******************', len(freqSet), m + 1, freqSet, H, H[0]
        # 合并数据集集合，组合为2/3/..n的集合
        Hmp1 = aprioriGen(H, m+1)
        # 返回记录 可信度大于阈值的集合
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        # 如果有2个结果都可以，直接返回结果就行，下面这个判断是多余，我个人觉得
        print 'Hmp1=', Hmp1
        if (len(Hmp1) > 1):
            print '----------------------', Hmp1
            # print len(freqSet),  len(Hmp1[0]) + 1
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)


def generateRules(L, supportData, minConf=0.7):
    """generateRules

    Args:
        L 频繁项集的全集
        supportData 所有元素和支持度的全集
        minConf 置信度的阈值
    Returns:
        bigRuleList 关于 (A->B+置信度) 3个字段的组合
    """
    bigRuleList = []
    # 循环L频繁项集，所有的统一大小组合（2/../n个的组合，从第2组开始）
    for i in range(1, len(L)):
        # 获取频繁项集中每个组合的所有元素
        # [[frozenset([1]), frozenset([3]), frozenset([2]), frozenset([5])], [frozenset([1, 3]), frozenset([2, 5]), frozenset([2, 3]), frozenset([3, 5])], [frozenset([2, 3, 5])]]
        for freqSet in L[i]:
            # 假设：freqSet=frozenset([1, 3])  H1=[1, 3]
            # 组合总的元素并遍历子元素，并转化为冻结的set集合，再存放到list列表中
            H1 = [frozenset([item]) for item in freqSet]
            # 2个的组合，走else, 2个以上的组合，走if
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList


def getActionIds():
    from time import sleep
    from votesmart import votesmart
    # votesmart.apikey = 'get your api key first'
    votesmart.apikey = 'a7fa40adec6f4a77178799fae4441030'
    actionIdList = []
    billTitleList = []
    fr = open('input/11.Apriori/recent20bills.txt')
    for line in fr.readlines():
        billNum = int(line.split('\t')[0])
        try:
            billDetail = votesmart.votes.getBill(billNum) # api call
            for action in billDetail.actions:
                if action.level == 'House' and (action.stage == 'Passage' or action.stage == 'Amendment Vote'):
                    actionId = int(action.actionId)
                    print 'bill: %d has actionId: %d' % (billNum, actionId)
                    actionIdList.append(actionId)
                    billTitleList.append(line.strip().split('\t')[1])
        except:
            print "problem getting bill %d" % billNum
        sleep(1)                                      # delay to be polite
    return actionIdList, billTitleList


def getTransList(actionIdList, billTitleList): #this will return a list of lists containing ints
    itemMeaning = ['Republican', 'Democratic']#list of what each item stands for
    for billTitle in billTitleList:#fill up itemMeaning list
        itemMeaning.append('%s -- Nay' % billTitle)
        itemMeaning.append('%s -- Yea' % billTitle)
    transDict = {}#list of items in each transaction (politician)
    voteCount = 2
    for actionId in actionIdList:
        sleep(3)
        print 'getting votes for actionId: %d' % actionId
        try:
            voteList = votesmart.votes.getBillActionVotes(actionId)
            for vote in voteList:
                if not transDict.has_key(vote.candidateName):
                    transDict[vote.candidateName] = []
                    if vote.officeParties == 'Democratic':
                        transDict[vote.candidateName].append(1)
                    elif vote.officeParties == 'Republican':
                        transDict[vote.candidateName].append(0)
                if vote.action == 'Nay':
                    transDict[vote.candidateName].append(voteCount)
                elif vote.action == 'Yea':
                    transDict[vote.candidateName].append(voteCount + 1)
        except:
            print "problem getting actionId: %d" % actionId
        voteCount += 2
    return transDict, itemMeaning


# 暂时没用上
# def pntRules(ruleList, itemMeaning):
#     for ruleTup in ruleList:
#         for item in ruleTup[0]:
#             print itemMeaning[item]
#         print "           -------->"
#         for item in ruleTup[1]:
#             print itemMeaning[item]
#         print "confidence: %f" % ruleTup[2]
#         print       #print a blank line


def main():
    # # 以前的测试
    # project_dir = os.path.dirname(os.path.dirname(os.getcwd()))
    # # 收集并准备数据
    # dataMat, labelMat = loadDataSet("%s/resources/Apriori_testdata.txt" % project_dir)

    # 现在的的测试
    # 1. 加载数据
    dataSet = loadDataSet()
    print dataSet
    # 调用 apriori 做购物篮分析
    # 支持度满足阈值的key集合L，和所有元素和支持度的全集suppoerData
    L, supportData = apriori(dataSet, minSupport=0.5)
    print L, '\n', supportData
    print '\ngenerateRules\n'
    rules = generateRules(L, supportData, minConf=0.25)
    print rules

    # # 项目实战
    # # 构建美国国会投票记录的事务数据集
    # actionIdList, billTitleList = getActionIds()
    # # 测试前2个
    # # transDict, itemMeaning = getTransList(actionIdList[: 2], billTitleList[: 2])
    # # transDict 表示 action_id的集合，transDict[key]这个就是action_id对应的选项，例如 [1, 2, 3]
    # transDict, itemMeaning = getTransList(actionIdList, billTitleList)
    # # 得到全集的数据
    # dataSet = [transDict[key] for key in transDict.keys()]
    # L, supportData = apriori(dataSet, minSupport=0.3)
    # rules = generateRules(L, supportData, minConf=0.95)
    # print rules

    # # 项目实战
    # # 发现毒蘑菇的相似特性
    # # 得到全集的数据
    # dataSet = [line.split() for line in open("input/11.Apriori/mushroom.dat").readlines()]
    # L, supportData = apriori(dataSet, minSupport=0.3)
    # # 2表示毒蘑菇，1表示可食用的蘑菇
    # # 找出关于2的频繁子项出来，就知道如果是毒蘑菇，那么出现频繁的也可能是毒蘑菇
    # for item in L[1]:
    #     if item.intersection('2'):
    #         print item

    # for item in L[2]:
    #     if item.intersection('2'):
    #         print item


if __name__ == "__main__":
    main()

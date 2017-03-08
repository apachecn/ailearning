######
## author：wenhuanhuan
##
######

def loadDataSet():
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]] #list集合
###构建集合#####
def createC1(dataSet):
    c1=[]
    for transaction in dataSet:
        for item in transaction:
            if not [item] in c1:   ##对C1中每个项构建一个不变集合
                #print (item);
                c1.append([item])   ##得到一个不重复项的list
                #print(c1);
    c1.sort()              ##排序C1
    return map(frozenset,c1) #返回一个不可变的set集合

def scanD(D,CK,minSupport):
    ssCnt = {}  #定义一个字典
    for tid in D:
        for can in CK:
            if can.issubset(tid):                ##测试是否 tid 中的每一个元素都在 can 中
                if not can  in ssCnt:       ##key在字典的键中，则返回True
                    ssCnt[can]=1
                else:ssCnt[can] +=1
    numItems = float(len(list(D)))+1.0
    retList = []
    supportData = {}
    for key in ssCnt:
        if numItems==0:
            numItems=1
        support = ssCnt[key]/numItems         #计算置项集的支持度
        if support >= minSupport:
            retList.insert(0,key)
        supportData[key]=support
    return retList,supportData
##
    ##输入参数：频繁项集列表，项集元素个数
    ##返回值：候选项集 CK
##
def aprioriGen(LK,k):      #creates CK
    retList = []
    lenLK = len(LK)
    for i in range(lenLK):
        for j in range(i+1,lenLK):
            L1 = list(LK[i])[:k-2]
            L2 = list(LK[j]) [:k-2]
            print('L1=',L1)
            L1.sort()
            print('L1Sort='L1)
            print('L2=',L2)
            L2.sort()
            print(L2)
            if L1==L2:
                retList.append(LK[[i] | LK[j]])
    return retList

def apriori(dataSet,minSupport=0.5):
    C1=createC1(dataSet)
    D = map(set,dataSet)
    L1,supportData = scanD(D,C1,minSupport)
    L = [L1]
    k=2
    while (len(L[k-2]) >0):
        CK = aprioriGen(L[k-2],k)
        LK,supK = scanD(D,CK,minSupport)
        supportData.update(supK)
        L.append(LK)
        k += 1
    return L,supportData

def generateRules(L,supportData,minConf=0.7):
    bigRuleList = []
    for i in range(1,len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i>1):
                rulesFromConseq(freqSet,H1,supportData,bigRuleList,minConf)
            else:
                calcConf(freqSet,H1,supportData,bigRuleList,minConf)
    return bigRuleList
def calcConf(freqset,H,supportData,br1,minConf=0.7):
    prunedH = []
    for conseq in H:
        conf = supportData[freqset]/supportData[freqset-conseq]
        if conf >= minConf:
            print(freqset-conseq,'-->',conseq,'conf:',conf)
            br1.append((freqset-conseq,conseq,conf))
            prunedH.append(conseq)
    return prunedH
def rulesFromConseq(freSet,H,supportData,br1,minConf=0.7):
    m = len(H[0])
    if (len(freSet) > (m+1)):
        Hmp1 = aprioriGen(H,m + 1)
        Hmp1 = calcConf(freSet,Hmp1,supportData,br1,minConf)
        if (len(Hmp1) > 1):
            rulesFromConseq(freSet,Hmp1,supportData,br1,minConf)


def pntRules(ruleList, itemMeaning):
    for ruleTup in ruleList:
        for item in ruleTup[0]:
            print(itemMeaning[item])
        print("           -------->")
        for item in ruleTup[1]:
            print
            itemMeaning[item]
        print("confidence: %f" % ruleTup[2])
        print  # print a blank line


from time import sleep

votesmart.apikey = 'a7fa40adec6f4a77178799fae4441030'


# votesmart.apikey = 'get your api key first'
def getActionIds():
    actionIdList = [];
    billTitleList = []
    fr = open('recent20bills.txt')
    for line in fr.readlines():
        billNum = int(line.split('\t')[0])
        try:
            billDetail = votesmart.votes.getBill(billNum)  # api call
            for action in billDetail.actions:
                if action.level == 'House' and \
                        (action.stage == 'Passage' or action.stage == 'Amendment Vote'):
                    actionId = int(action.actionId)
                    print('bill: %d has actionId: %d' % (billNum, actionId))
                    actionIdList.append(actionId)
                    billTitleList.append(line.strip().split('\t')[1])
        except:
            print("problem getting bill %d" % billNum)
        sleep(1)  # delay to be polite
    return actionIdList, billTitleList


def getTransList(actionIdList, billTitleList):  # this will return a list of lists containing ints
    itemMeaning = ['Republican', 'Democratic']  # list of what each item stands for
    for billTitle in billTitleList:  # fill up itemMeaning list
        itemMeaning.append('%s -- Nay' % billTitle)
        itemMeaning.append('%s -- Yea' % billTitle)
    transDict = {}  # list of items in each transaction (politician)
    voteCount = 2
    for actionId in actionIdList:
        sleep(3)
        print('getting votes for actionId: %d' % actionId)
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
            print("problem getting actionId: %d",actionId)
        voteCount += 2
    return transDict, itemMeaning

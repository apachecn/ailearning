#!/usr/bin/python
# coding:utf8

'''
Created on Jun 14, 2011
FP-Growth FP means frequent pattern
the FP-Growth algorithm needs:
1. FP-tree (class treeNode)
2. header table (use dict)
This finds frequent itemsets similar to apriori but does not find association rules.
@author: Peter/片刻
'''
print(__doc__)


class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        # needs to be updated
        self.parent = parentNode
        self.children = {}

    def inc(self, numOccur):
        """inc(对count变量增加给定值)

        """
        self.count += numOccur

    def disp(self, ind=1):
        """disp(用于将树以文本形式显示)

        """
        print '  '*ind, self.name, ' ', self.count
        for child in self.children.values():
            child.disp(ind+1)


def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat


def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict


# this version does not use recursion
def updateHeader(nodeToTest, targetNode):
    """updateHeader(更新头指针，添加targetNode到nodeToTest的nodeLink上面)

    从头指针的nodeLink开始，一直沿着nodeLink直到到达链表末尾。这就是链表。
    性能：如果链表很长可能会遇到迭代调用的次数限制。

    Args:
        nodeToTest  头节点
        targetNode  目标节点
    """
    # Do not use recursion to traverse a linked list!
    while (nodeToTest.nodeLink is not None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode


def updateTree(items, inTree, headerTable, count):
    """updateTree(更新FP-tree，第二次遍历)

    Args:
        items      满足minSup 排序后的元素数组（大到小的排序）
        inTree     空的Tree对象
        headerTable   满足minSup {所有的元素+(value, treeNode)}
        count      原数据集中每一组Kay出现的次数
    """
    # 判断满足minSup排序后的第一个元素，是否是inTree的子节点
    if items[0] in inTree.children:
        # 如果是，那么这个子节点的key元素添加count次
        inTree.children[items[0]].inc(count)
    else:
        # 如果不存在子节点，我们为该inTree添加子节点
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        # 如果满足minSup的dist字典的value值第二位为null， 我们就设置该元素为 本节点对应的tree节点
        # 如果元素第二位不为null，我们就更新header节点
        if headerTable[items[0]][1] is None:
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:
        # print 'items[1::]=', items[1::]
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)


def createTree(dataSet, minSup=1):
    """createTree(生成FP-tree，第一次遍历)

    Args:
        dataSet  dist字典对象
        minSup   最小的支持度
    Returns:
        retTree  FP-tree
        headerTable 满足minSup {所有的元素+(value, treeNode)}
    """
    # 创建一个满足支持度>=minSup的dist字典
    headerTable = {}
    # 循环得到dist字典所有的key
    for trans in dataSet:
        # 对所有的key进行循环，得到key里面的所有元素
        for item in trans:
            # 存储每个元素和它对应的次数： 本身+dataSet该元素出现的次数
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    # 循环所有元素出现的次数，然后remove到小于minSup的元素
    for k in headerTable.keys():
        if headerTable[k] < minSup:
            del(headerTable[k])

    # 求出满足minSup元素的集合
    freqItemSet = set(headerTable.keys())
    # 如果不存在满足minSup的元素就直接返回None
    if len(freqItemSet) == 0:
        return None, None
    for k in headerTable:
        # reformat headerTable to use Node link
        # value值为一个元组
        headerTable[k] = [headerTable[k], None]

    # create tree
    retTree = treeNode('Null Set', 1, None)
    for tranSet, count in dataSet.items():
        localD = {}
        for item in tranSet:
            # 判断是否在满足minSup的集合中
            if item in freqItemSet:
                # print 'headerTable[item][0]=', headerTable[item][0], headerTable[item]
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            # p=key,value; 所以是通过value值的大小，进行从大到小进行排序
            # orderedItems表示取出元组的key值，也就是字母本身，但是字母本身是存在顺序的
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
            # print 'sorted(localD.items(), key=lambda p: p[1], reverse=True)]=', sorted(localD.items(), key=lambda p: p[1], reverse=True)
            # print 'orderedItems=', orderedItems

            # 使用有序freq项集来填充树
            updateTree(orderedItems, retTree, headerTable, count)

    return retTree, headerTable


def ascendTree(leafNode, prefixPath): #ascends from leaf node to root
    if leafNode.parent is not None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)


def findPrefixPath(basePat, treeNode): #treeNode comes from header table
    condPats = {}
    while treeNode is not None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1: 
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats


if __name__ == "__main__":
    rootNode = treeNode('pyramid', 9, None)
    rootNode.children['eye'] = treeNode('eye', 13, None)
    rootNode.children['phoenix'] = treeNode('phoenix', 3, None)
    # 将树以文本形式显示
    # print rootNode.disp()

    # load样本数据
    simpDat = loadSimpDat()
    # print simpDat, '\n'
    # 重新装载 frozen set 格式化样本数据，用dist存储数据和对应的次数
    initSet = createInitSet(simpDat)
    # print initSet

    # 创建FP树
    myFPtree, myHeaderTab = createTree(initSet, 3)
    myFPtree.disp()
    # print myHeaderTab





def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1])]#(sort header table)
    for basePat in bigL:  #start from bottom of header table
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        #print 'finalFrequent Item: ',newFreqSet    #append to set
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        #print 'condPattBases :',basePat, condPattBases
        #2. construct cond FP-tree from cond. pattern base
        myCondTree, myHead = createTree(condPattBases, minSup)
        #print 'head from conditional tree: ', myHead
        if myHead != None: #3. mine cond. FP-tree
            #print 'conditional tree for: ',newFreqSet
            #myCondTree.disp(1)            
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)




import twitter
from time import sleep
import re


def textParse(bigString):
    urlsRemoved = re.sub('(http:[/][/]|www.)([a-z]|[A-Z]|[0-9]|[/.]|[~])*', '', bigString)    
    listOfTokens = re.split(r'\W*', urlsRemoved)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def getLotsOfTweets(searchStr):
    CONSUMER_KEY = ''
    CONSUMER_SECRET = ''
    ACCESS_TOKEN_KEY = ''
    ACCESS_TOKEN_SECRET = ''
    api = twitter.Api(consumer_key=CONSUMER_KEY, consumer_secret=CONSUMER_SECRET,
                      access_token_key=ACCESS_TOKEN_KEY, 
                      access_token_secret=ACCESS_TOKEN_SECRET)
    #you can get 1500 results 15 pages * 100 per page
    resultsPages = []
    for i in range(1,15):
        print "fetching page %d" % i
        searchResults = api.GetSearch(searchStr, per_page=100, page=i)
        resultsPages.append(searchResults)
        sleep(6)
    return resultsPages


def mineTweets(tweetArr, minSup=5):
    parsedList = []
    for i in range(14):
        for j in range(100):
            parsedList.append(textParse(tweetArr[i][j].text))
    initSet = createInitSet(parsedList)
    myFPtree, myHeaderTab = createTree(initSet, minSup)
    myFreqList = []
    mineTree(myFPtree, myHeaderTab, minSup, set([]), myFreqList)
    return myFreqList


#minSup = 3
#simpDat = loadSimpDat()
#initSet = createInitSet(simpDat)
#myFPtree, myHeaderTab = createTree(initSet, minSup)
#myFPtree.disp()
#myFreqList = []
#mineTree(myFPtree, myHeaderTab, minSup, set([]), myFreqList)

######
## author：wenhuanhuan
##
######
import apriori


class Test:
    if __name__ == "__main__":
        #fza=frozenset(['a','bc'])
        #adict={fza:1,'b':2}
        #print(adict)
       # print (isinstance('36521dyht', str)) ##可以判断变量 x 是否是字符串；
        #cc= loadDataSet()
        #createC1(cc)
        #c=[6,5,4,9,8,3,5,6,8,1]
        #c.sort()
        #print(c)
        dataSet = apriori.loadDataSet()
        print(dataSet)
        C1=apriori.createC1(dataSet)
        C1
        D=map(set,dataSet)
        L1=[]
        supportData=[]
        (L1,supportData)=apriori.scanD(D, C1, 0.5)
        print(L1)
        print(supportData)
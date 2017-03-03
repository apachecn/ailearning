class treeNode:
    def __init__(self,nameValue,numOccur,parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}
    def inc(self,numOccur):
        self.count += numOccur
    def disp(self,ind=1):
        print(' '*ind,self.name,' ',self.count)
        for child in self.children.values():
            child.disp(ind+1)

    if __name__ == "__main__":
        import fpGrowth
        rootNode = fpGrowth.treeNode('pyramid',9,None)
        rootNode.children['eye']=fpGrowth.treeNode('eye',13,None)
        rootNode.disp()
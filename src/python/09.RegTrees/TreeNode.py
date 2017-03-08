#!/usr/bin/python
# coding:utf8

'''
Created on 2017-03-06
Update on 2017-03-06
@author: jiangzhonglian
'''


class treeNode():
    def __init__(self, feat, val, right, left):
        self.featureToSplitOn = feat
        self.valueOfSplit = val
        self.rightBranch = right
        self.leftBranch = left

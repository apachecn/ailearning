#!/usr/bin/env python
# coding: utf-8

"""
Created on 2018-05-09
Updated on 2017-05-09
Author: /片刻
GitHub: https://github.com/apachecn/AiLearning
"""


"""
the first example for nltk book
"""
from __future__ import print_function

from nltk.book import *


# 查找特定词语上下文
text1.concordance("monstrous")

# 相关词查找
text1.similar("monstrous")

# 查找多个词语的共同上下文
text2.common_contexts(["monstrous", "very"])

# 画出词语的离散图
text4.dispersion_plot(["citizens", "democracy", "freedom", "duties", "America"])

# 产生随机文本
text3.generate()
# TOTO: Clean up the following
# Traceback (most recent call last):
#   File "E:/nlp/eg1.py", line 25, in <module>
#     text3.generate()
# TypeError: generate() missing 1 required positional argument: 'words'

# 单词数量 标识符总数
print(len(text3))

# 词汇的种类及数量 用集合set显示
print(sorted(set(text3)))
print(len(set(text3)))

# 测量平均每类词语被使用的次数
from __future__ import division #本命令必须放在文件的开始之初
print(len(text3)/len(set(text3)))

# 统计特定单词在文本中出现的次数，并计算其占比
print(text3.count("smote"))
print(100*text4.count('a')/len(text4))


# 计算一个文本中，平均一个字出现的次数（词汇多样性）
def lexical_diversity(text):
  return len(text) / len(set(text))


def percentage(count, total):
  return 100 * count / total


"""
测试案例

In [32]: ex1 = ['Monty', 'Python', 'and', 'the', 'Holy', 'Grail']
In [34]: sorted(ex1)
Out[34]: ['Grail', 'Holy', 'Monty', 'Python', 'and', 'the']

In [35]: len(set(ex1))
Out[35]: 6

In [36]: ex1.count("the")
Out[36]: 1

In [37]: ['Monty', 'Python'] + ['and', 'the', 'Holy', 'Grail']
Out[37]: ['Monty', 'Python', 'and', 'the', 'Holy', 'Grail']
"""

# # 词的频率分布
fdist1 = FreqDist(text1)
# # 输出总的词数
print(fdist1)
# In Python 3 dict.keys() returns an iteratable but not indexable object.
vac1 = list(fdist1.keys())
# # 输出词数最多的前五十个词
print(vac1[:50])
# # 输出whale的次数
print(fdist1["whale"])
# # 输出前五十个词的累积频率图

fdist1.plot(50)

# 查找长度超过15个字符的词
V = set(text1)
long_words = [w for w in V if len(w)>15]
print(sorted(long_words))

# 查找长度超过7的词且频率超过7
fdist5 = FreqDist(text5)
print(sorted([ w for w in set(text5) if len(w)>7 and fdist5[w]>7]))

# 双连词的使用
from nltk import bigrams
# # 查了一下nltk官网上的函数说明，要加list()函数，结果才是书上的情况
print(list(bigrams(['more', 'is', 'said', 'than', 'done'])))

# 文本中常用的连接词
print(text4.collocations())

print([len(w) for w in text1])
fdist = FreqDist([len(w) for w in text1])
print(fdist)
print(fdist.keys())
print(fdist.items())
print(fdist.max())
print(fdist[3])
print(fdist.freq(3))

print(sorted([w for w in set(text1) if w.endswith('ableness')]))

print(babelize_shell())

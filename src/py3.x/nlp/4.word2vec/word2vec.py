# coding: utf-8

"""
Word2Vec 模型:
* Word2Vec 通过训练，可以把对文本内容的处理简化为K维向量空间中的向量运算.(而向量空间上的相似度可以用来表示文本语义上的相似度)
    * 采用的模型有CBOW(Continuous Bag-Of-Words，即连续的词袋模型)和 Skip-Gram 两种.
    * 因此，Word2Vec 输出的词向量可以被用来做很多NLP相关的工作，比如聚类、找同义词、词性分析等等.
* CBOW 模型: 能够根据输入周围n-1个词来预测出这个词本身.
    * 也就是说，CBOW模型的输入是某个词A周围的n个单词的词向量之和，输出是词A本身的词向量.
* Skip-gram 模型: 能够根据词本身来预测周围有哪些词.
    * 也就是说，Skip-gram模型的输入是词A本身，输出是词A周围的n个单词的词向量.
"""

import pandas as pd
from gensim.models import word2vec

# 加载语料
sentences = word2vec.Text8Corpus(u"/opt/data/NLP/4.word2vec/text8.txt")
model = word2vec.Word2Vec(sentences, size=200)  # 训练skip-gram模型; 默认window=5

# 计算两个词的相似度/相关程度
y1 = model.similarity("woman", "man")
print(u"woman和man的相似度为：", y1)
print("--------\n")

# 计算某个词的相关词列表
y2 = model.most_similar("good", topn=20)  # 20个最相关的
print(pd.Series(y2))
# print(u"和good最相关的词有：\n")
# for item in y2:
#     print(item[0], item[1])
print("--------\n")

# 寻找对应关系
print(' "boy" is to "father" as "girl" is to ...? \n')
y3 = model.most_similar(['girl', 'father'], ['boy'], topn=3)
for item in y3:
    print(item[0], item[1])
print("--------\n")

more_examples = ["he his she", "big bigger bad", "going went being"]
for example in more_examples:
    a, b, x = example.split()
    predicted = model.most_similar([x, b], [a])[0][0]
    print("'%s' is to '%s' as '%s' is to '%s'" % (a, b, x, predicted))
print("--------\n")

# 寻找不合群的词
y4 = model.doesnt_match("breakfast cereal dinner lunch".split())
print(u"不合群的词：", y4)
print("--------\n")

# 保存模型，以便重用
# model.save("text8.model")
# 对应的加载方式
# model_2 = word2vec.Word2Vec.load("text8.model")

# 以一种C语言可以解析的形式存储词向量
# model.save_word2vec_format("text8.model.bin", binary=True)
# 对应的加载方式
# model_3 = word2vec.Word2Vec.load_word2vec_format("text8.model.bin", binary=True)

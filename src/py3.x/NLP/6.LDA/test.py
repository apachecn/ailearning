#!/usr/bin/python
# coding:utf8
"""
Created on 2018-03-13
Updated on 2018-03-13
Author: 片刻
GitHub: https://github.com/apachecn/AiLearning
Coding: http://blog.csdn.net/github_36299736/article/details/54966460
"""

import gensim
from gensim import corpora, models
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words

# 创建示例文档
doc_a = "Brocolli is good to eat. My brother likes to eat good brocolli, but not my mother."
doc_b = "My mother spends a lot of time driving my brother around to baseball practice."
doc_c = "Some health experts suggest that driving may cause increased tension and blood pressure."
doc_d = "I often feel pressure to perform well at school, but my mother never seems to drive my brother to do better."
doc_e = "Health professionals say that brocolli is good for your health."
# 将示例文档编译成列表
doc_set = [doc_a, doc_b, doc_c, doc_d, doc_e]


# 创建PorterStemmer类的p_stemmer
p_stemmer = PorterStemmer()
# 分词：将文档转化为其原子元素
tokenizer = RegexpTokenizer(r'\w+')
# 创建英文停用词列表
en_stop = get_stop_words('en')


# 循环中标记的文档列表
texts = []
# 遍历文档列表
for i in doc_set:

    # 清理并标记文档字符串
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)
    print(tokens)

    # 从令牌中删除停用词(停用词处理：移除无意义的词)
    stopped_tokens = [i for i in tokens if not i in en_stop]

    # 词干令牌(词干提取：将同义词合并)
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    # 添加令牌列表
    texts.append(stemmed_tokens)

# 把我们的标记文档转换成一个id <-> 词条字典
dictionary = corpora.Dictionary(texts)
print(dictionary.token2id)  # 来查看每个单词的id
# print(dictionary.roken2id)  # 显示 brocolli 的 id 是 0

# 将标记文档转换为文档术语矩阵
# doc2bow() 方法将 dictionary 转化为一个词袋
corpus = [dictionary.doc2bow(text) for text in texts]


# 生成LDA模型
"""
LdaModel 类的详细描述可以在 gensim 文档中查看。我们的实例中用到的参数：

参数：
num_topics: 必须。LDA 模型要求用户决定应该生成多少个主题。由于我们的文档集很小，所以我们只生成三个主题。
id2word: 必须。LdaModel 类要求我们之前的 dictionary 把 id 都映射成为字符串。
passes: 可选。模型遍历语料库的次数。遍历的次数越多，模型越精确。但是对于非常大的语料库，遍历太多次会花费很长的时间。
"""
ldamodel = gensim.models.ldamodel.LdaModel(
    corpus, num_topics=3, id2word=dictionary, passes=20)

print(dir(ldamodel))
print(ldamodel.print_topics(num_topics=3, num_words=3))  

"""
这是什么意思呢？每一个生成的主题都用逗号分隔开。每个主题当中有三个该主题当中最可能出现的单词。即使我们的文档集很小，这个模型依旧是很可靠的。还有一些需要我们考虑的问题：

- health, brocolli 和 good 在一起时有很好的含义。
- 第二个主题有点让人疑惑，如果我们重新查看源文档，可以看到 drive 有很多种含义：driving a car 意思是开车，driving oneself to improve 是激励自己进步。这是我们在结果中需要注意的地方。
- 第三个主题包含 mother 和 brother，这很合理。

调整模型的主题数和遍历次数对于得到一个好的结果是很重要的。两个主题看起来更适合我们的文档。
"""

lda = gensim.models.ldamodel.LdaModel(
    corpus, num_topics=2, id2word=dictionary, passes=20)

print(lda.print_topics(num_topics=3, num_words=3))  


# # LDA主题模型的保存
# from gensim import corpora, models

# # # 语料导入
# id2word = corpora.Dictionary.load_from_text('zhwiki_wordids.txt')
# mm = corpora.MmCorpus('zhwiki_tfidf.mm')

# # # 模型训练，耗时28m
# lda = models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=100)

# # 打印前20个topic的词分布
# lda.print_topics(20)
# # 打印id为20的topic的词分布
# lda.print_topic(20)

# # 模型的保存/ 加载
# lda.save('zhwiki_lda.model')
# lda = models.ldamodel.LdaModel.load('zhwiki_lda.model')


# # LDA 主题模型的使用
# # 对新文档，转换成bag-of-word后，可进行主题预测。
# test_doc = list(jieba.cut(test_doc))     # 新文档进行分词
# doc_bow = id2word.doc2bow(test_doc)      # 文档转换成bow
# *** # doc_lda = lda[doc_bow]             # 得到新文档的主题分布
# #输出新文档的主题分布
# print doc_lda
# for topic in doc_lda:
#     print("%s\t%f\n" % (lda.print_topic(topic[0]), topic[1]))

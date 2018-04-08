# coding: utf-8

"""
词汇表模型：
* 袋模型可以很好的表现文本由哪些单词组成，但是却无法表达出单词之间的前后关系，
* 于是人们借鉴了词袋模型的思想，使用生成的词汇表对原有句子按照单词逐个进行编码。
"""

import numpy as np
from tensorflow.contrib.learn import preprocessing as pc

# 语料
corpus = ['i love you', 'me too']

vocab = pc.VocabularyProcessor(max_document_length=4)
"""
VocabularyProcessor 参数
* max_document_length: 文档的最大长度。如果文本的长度大于最大长度，那么它会被剪切，反之则用0填充。
* min_frequency: 词频的最小值，出现次数小于最小词频则不会被收录到词表中。
* vocabulary: CategoricalVocabulary 对象。
* tokenizer_fn: 分词函数。
"""
# 创建词汇表，创建后不能更改
vocab.fit(corpus)

# 获取 Iterator 对象, next 进行遍历
print("Encoding: \n", next(vocab.transform(['i me too'])).tolist())

# 获取 预料 编码后的矩阵向量
mat_corpus = np.array(list(vocab.fit_transform(corpus)))
print("mat_corpus: \n", mat_corpus)

# 保存和加载词汇表
# vocab.save('vocab.pickle')
# vocab = pc.VocabularyProcessor.restore('vocab.pickle')

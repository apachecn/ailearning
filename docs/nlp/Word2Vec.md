# Word2Vec 讲解

## 介绍

**需要复习** 手写 Word2Vec 源码: https://blog.csdn.net/u014595019/article/details/51943428

* 2013年，Google开源了一款用于词向量计算的工具—— `word2vec`，引起了工业界和学术界的关注。
* `word2vec` 算法或模型的时候，其实指的是其背后用于计算 **word vector** 的 `CBoW` 模型和 `Skip-gram` 模型
* 很多人以为 `word2vec` 指的是一个算法或模型，这也是一种谬误。
* 因此通过 Word2Vec 技术 输出的词向量可以被用来做很多NLP相关的工作，比如聚类、找同义词、词性分析等等.

> 适用场景

1. cbow适用于小规模，或者主题比较散的语料，毕竟他的向量产生只跟临近的字有关系，更远的语料并没有被采用。
2. 而相反的skip-gram可以处理基于相同语义，义群的一大批语料。

## CBoW 模型（Continuous Bag-of-Words Model）

* 连续词袋模型（CBOW）常用于NLP深度学习。
* 这是一种模式，它试图根据目标词 `之前` 和 `之后` 几个单词的背景来预测单词（CBOW不是顺序）。
* CBOW 模型: 能够根据输入周围n-1个词来预测出这个词本身.
    * 也就是说，CBOW模型的输入是某个词A周围的n个单词的词向量之和，输出是词A本身的词向量.

![CBoW 模型/img/NLP/Word2Vce/CBoW.png)

## Skip-gram 模型

* skip-gram与CBOW相比，只有细微的不同。skip-gram的输入是当前词的词向量，而输出是周围词的词向量。
* Skip-gram 模型: 能够根据词本身来预测周围有哪些词.
    * 也就是说，Skip-gram模型的输入是词A本身，输出是词A周围的n个单词的词向量.

![Skip-gram 模型/img/NLP/Word2Vce/Skip-gram.png)


明天看看这个案例: https://blog.csdn.net/lyb3b3b/article/details/72897952


## 补充: NPLM - Ngram 模型

* n-gram 模型是一种近似策略，作了一个马尔可夫假设: 认为目标词的条件概率只与其之前的 n 个词有关
* NPLM基于 n-gram, 相当于目标词只有上文。


* * *

参考资料: 

1. https://www.cnblogs.com/iloveai/p/word2vec.html

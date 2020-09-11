# 【入门须知】必须了解

实体： 抽取
关系： 图谱
意图： 分类

* **【入门须知】必须了解**: <https://github.com/apachecn/AiLearning/tree/master/docs/nlp>
* **【入门教程】强烈推荐: PyTorch 自然语言处理**: <https://github.com/apachecn/NLP-with-PyTorch>
* Python 自然语言处理 第二版: <https://usyiyi.github.io/nlp-py-2e-zh>
* 推荐一个[liuhuanyong大佬](https://github.com/liuhuanyong)整理的nlp全面知识体系: <https://liuhuanyong.github.io>

## nlp 学习书籍和工具: 

* 百度搜索: Python自然语言处理
* 读书笔记: <https://wnma3mz.github.io/hexo_blog/2018/05/13/《Python自然语言处理》阅读笔记（一）>
* Python自然语言处理工具汇总: <https://blog.csdn.net/sa14023053/article/details/51823122>

## nlp 全局介绍视频: （简单做了解就行）

地址链接:  http://bit.baidu.com/Course/detail/id/56.html
 
1. 自然语言处理知识入门
2. 百度机器翻译
3. 篇章分析
4. UNIT: 语言理解与交互技术

## 中文 NLP

> 开源 - 词向量库集合

* <https://github.com/Embedding/Chinese-Word-Vectors>
* <https://github.com/brightmart/nlp_chinese_corpus>
* <https://github.com/codemayq/chinese_chatbot_corpus>
* <https://github.com/candlewill/Dialog_Corpus>

> 深度学习必学

1. [反向传递](/docs/dl/反向传递.md): https://www.cnblogs.com/charlotte77/p/5629865.html
2. [CNN原理](/docs/dl/CNN原理.md): http://www.cnblogs.com/charlotte77/p/7759802.html
3. [RNN原理](/docs/dl/RNN原理.md): https://blog.csdn.net/qq_39422642/article/details/78676567
4. [LSTM原理](/docs/dl/LSTM原理.md): https://blog.csdn.net/weixin_42111770/article/details/80900575

> [Word2Vec 原理](/docs/nlp/Word2Vec.md):

1. 负采样

介绍:
    自然语言处理领域中，判断两个单词是不是一对上下文词（context）与目标词（target），如果是一对，则是正样本，如果不是一对，则是负样本。
    采样得到一个上下文词和一个目标词，生成一个正样本（positive example），生成一个负样本（negative example），则是用与正样本相同的上下文词，再在字典中随机选择一个单词，这就是负采样（negative sampling）。

案例:
    比如给定一句话“这是去上学的班车”，则对这句话进行正采样，得到上下文“上”和目标词“学”，则这两个字就是正样本。
    负样本的采样需要选定同样的“上”，然后在训练的字典中任意取另一个字，如“我”、“梦”、“目”，这一对就构成负样本。
    训练需要正样本和负样本同时存在。

优势:
    负采样的本质: 每次让一个训练样本只更新部分权重，其他权重全部固定；减少计算量；（一定程度上还可以增加随机性）

## nlp 操作流程

[本项目](https://pytorch.apachecn.org/docs/1.0/#/char_rnn_classification_tutorial) 试图通过名字分类问题给大家描述一个基础的深度学习中自然语言处理模型，同时也向大家展示了Pytorch的基本玩法。 其实对于大部分基础的NLP工作，都是类似的套路: 

1. 收集数据 
2. 清洗数据 
3. 为数据建立字母表或词表（vocabulary或者叫look-up table） 
4. 根据字母表或者词表把数据向量化 
5. 搭建神经网络，深度学习中一般以LSTM或者GRU为主，按照需求结合各种其他的工具，包括embedding，注意力机制，双向RNN等等常见算法。 
6. 输入数据，按需求得到输出，比如分类模型根据类别数来得到输出，生成模型根据指定的长度或者结束标志符来得到输出等等。 
7. 把输出的结果进行处理，得到最终想要的数据。常需要把向量化的结果根据字母表或者词表变回文本数据。 
8. 评估模型。

如果真的想要对自然语言处理或者序列模型有更加全面的了解，建议大家去网易云课堂看一看吴恩达深度学习微专业中的序列模型这一板块，可以说是讲的非常清楚了。 此外极力推荐两个blog:  

1. 讲述RNN循环神经网络在深度学习中的各种应用场景。http://karpathy.github.io/2015/05/21/rnn-effectiveness/ 
2. 讲述LSTM的来龙去脉。http://colah.github.io/posts/2015-08-Understanding-LSTMs/

最后，本文参考整合了:

* Pytorch中文文档: https://pytorch.apachecn.org
* Pytorch官方文档: http://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html 
* Ngarneau小哥的博文: https://github.com/ngarneau/understanding-pytorch-batching-lstm
* 另外，本项目搭配Sung Kim的Pytorch Zero To All的第13讲rnn_classification会更加方便食用喔，视频可以在油管和b站中找到。

## nlp - 比赛链接

* https://competitions.codalab.org/competitions/12731
* https://sites.ualberta.ca/%7Emiyoung2/COLIEE2018/
* https://visualdialog.org/challenge/2018
+ 人机对话 NLP
    - http://jddc.jd.com 
+ 司法数据文本的 NLP
    - http://cail.cipsc.org.cn
+ “达观杯” 文本智能处理挑战赛   
    - http://www.dcjingsai.com/common/cmpt/“达观杯”文本智能处理挑战赛_竞赛信息.html
+ 中文论文摘要数据
    - https://biendata.com/competition/smpetst2018
+ 中文问答任务
    - https://biendata.com/competition/CCKS2018_4/
+ 第二届讯飞杯中文机器阅读理解评测 
    - http://www.hfl-tek.com/cmrc2018
+ 2018机器阅读理解技术竞赛  这也是结束了的 NLP
    - http://mrc2018.cipsc.org.cn
+ 句子文本相似度计算
    - https://www.kaggle.com/c/quora-question-pairs


* * * 

【比赛收集平台】: https://github.com/iphysresearch/DataSciComp

# 入门教程需看资料

## nlp 学习书籍：

* 百度搜索：Python自然语言处理
* 读书笔记：https://wnma3mz.github.io/hexo_blog/2018/05/13/《Python自然语言处理》阅读笔记（一）

## nlp 全局介绍视频：（简单做了解就行）

地址链接： http://bit.baidu.com/Course/detail/id/56.html
 
1. 自然语言处理知识入门
2. 百度机器翻译
3. 篇章分析
4. UNIT：语言理解与交互技术
 
## 中文 nlp 词向量

https://github.com/Embedding/Chinese-Word-Vectors

> 深度学习必学

1. 反向传递: https://www.cnblogs.com/charlotte77/p/5629865.html
2. CNN原理: http://www.cnblogs.com/charlotte77/p/7759802.html
3. RNN原理: https://blog.csdn.net/qq_39422642/article/details/78676567
4. LSTM深入浅出的好文: https://blog.csdn.net/roslei/article/details/61912618

## nlp 操作流程

[本项目](http://pytorch.apachecn.org/cn/tutorials/intermediate/char_rnn_classification_tutorial.html) 试图通过名字分类问题给大家描述一个基础的深度学习中自然语言处理模型，同时也向大家展示了Pytorch的基本玩法。 其实对于大部分基础的NLP工作，都是类似的套路： 

1. 收集数据 
2. 清洗数据 
3. 为数据建立字母表或词表（vocabulary或者叫look-up table） 
4. 根据字母表或者词表把数据向量化 
5. 搭建神经网络，深度学习中一般以LSTM或者GRU为主，按照需求结合各种其他的工具，包括embedding，注意力机制，双向RNN等等常见算法。 
6. 输入数据，按需求得到输出，比如分类模型根据类别数来得到输出，生成模型根据指定的长度或者结束标志符来得到输出等等。 
7. 把输出的结果进行处理，得到最终想要的数据。常需要把向量化的结果根据字母表或者词表变回文本数据。 
8. 评估模型。

如果真的想要对自然语言处理或者序列模型有更加全面的了解，建议大家去网易云课堂看一看吴恩达深度学习微专业中的序列模型这一板块，可以说是讲的非常清楚了。 此外极力推荐两个blog： 

1. 讲述RNN循环神经网络在深度学习中的各种应用场景。http://karpathy.github.io/2015/05/21/rnn-effectiveness/ 
2. 讲述LSTM的来龙去脉。http://colah.github.io/posts/2015-08-Understanding-LSTMs/

最后，本文参考整合了:

* Pytorch中文教程：http://pytorch.apachecn.org/cn/tutorials
* Pytorch中文文档：http://pytorch.apachecn.org/cn/docs/0.3.0
* Pytorch官方文档：http://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html 
* Ngarneau小哥的博文：https://github.com/ngarneau/understanding-pytorch-batching-lstm
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
    - http://www.dcjingsai.com/common/cmpt/%E2%80%9C%E8%BE%BE%E8%A7%82%E6%9D%AF%E2%80%9D%E6%96%87%E6%9C%AC%E6%99%BA%E8%83%BD%E5%A4%84%E7%90%86%E6%8C%91%E6%88%98%E8%B5%9B_%E7%AB%9E%E8%B5%9B%E4%BF%A1%E6%81%AF.html
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

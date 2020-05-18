# Bert 项目实战

```
题外话: 
我感觉的确很惭愧
那些在NLP领域一直都说自己很牛逼的公司？
为啥中文通用的预训练模型是Google，而不是我们热熟能详的国内互联网公司？
```

## 基本介绍

BERT的全称为Bidirectional Encoder Representation from Transformers，是一个预训练的语言表征模型。

该模型有以下主要优点: 

1. 采用MLM对`双向的Transformers`进行预训练，以生成深层的双向语言表征，可以更好的理解上下文信息
2. 预训练后，只需要添加一个额外的输出层进行fine-tune，就可以在各种各样的下游任务中取得state-of-the-art的表现。在这过程中并不需要对BERT进行任务特定的结构修改。


![](/img/tf_2.0/bert.png)

总结一下:

1. token embeddings 表示的是词向量，第一个单词是CLS，可以用于之后的分类任务；SEP分割句子的
2. segment embeddings 用来区别两种句子，因为预训练不光做LM还要做以两个句子为输入的分类任务
3. position embeddings 表示位置信息（索引ID）

例如: 

这里包含keras-bert的三个例子，分别是文本分类、关系抽取和主体抽取，都是在官方发布的预训练权重基础上进行微调来做的。

---

## 提问环节:

> 1.bert中的[CLS]甚意思？

bert论文中提到: “GPT uses a sentence separator ([SEP]) and classifier token ([CLS]) which are only introduced at fine-tuning time; BERT learns [SEP], [CLS] and sentence A/B embeddings during pre-training.”

说明[CLS]不是bert原创，在GPT中就有。在GPT中用于句子的分类，判断[CLS]开头的句子是不是下一句。

> 2.单向语言模型和双向语言模型

* 单向语言模型是指: 从左到右或者从右到左，使其只能获取单方向的上下文信息
* 双向语言模型是指: 不受单向的限制，可以更好的理解上下文信息


> 3.预训练模型

例如我们今天利用A数据训练了一个A模型，第二次我们有新的预料，在A的基础上，做了微调(fine-tuning)，得到一个更好的B模型。而模型在这个过程中，是预先训练好，方便下一个预料进行迭代式训练。

---

* https://www.cnblogs.com/dogecheng/p/11617940.html
* https://blog.csdn.net/zkq_1986/article/details/100155596
* https://blog.csdn.net/yangfengling1023/article/details/84025313
* https://zhuanlan.zhihu.com/p/98855346
* https://blog.csdn.net/weixin_42598761/article/details/104592171
* https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
* pip install git+https://www.github.com/keras-team/keras-contrib.git
* pip install keras-bert


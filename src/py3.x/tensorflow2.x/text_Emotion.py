# *-* coding:utf-8 *-*
# 词向量: 
#   https://www.cnblogs.com/Darwin2000/p/5786984.html
# 数据集:
#   https://blog.csdn.net/alip39/article/details/95891321
# 参考代码:
#   https://blog.csdn.net/u012052268/article/details/90238282
# Attention:
#   https://github.com/philipperemy/keras-attention-mechanism
import re
import os
import keras
import random
import gensim
import numpy as np
import pandas as pd
import jieba
from sklearn.model_selection import train_test_split
from keras import Model
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout, Dense, Flatten, Bidirectional, Embedding, GRU, Input, multiply
"""
# padding: pre(默认) 向前补充0  post 向后补充0
# truncating: 文本超过 pad_num,  pre(默认) 删除前面  post 删除后面
# x_train = pad_sequences(x, maxlen=pad_num, value=0, padding='post', truncating="post")
# print("--- ", x_train[0][:20])
"""
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
from config import Config
import pickle
import matplotlib.pyplot as plt


# 存储模型: 持久化
def load_pkl(filename):
    with open(filename, 'rb') as fr:
        model = pickle.load(fr)
    return model


def save_pkl(model, filename):
    with open(filename, 'wb') as fw:
        pickle.dump(model, fw)


## 训练自己的词向量，并保存。
def trainWord2Vec(infile, outfile):
    sentences =  gensim.models.word2vec.LineSentence(infile) # 读取分词后的 文本
    model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=1, workers=4) # 训练模型
    model.save(outfile)


def loadMyWord2Vec(outfile):
    # 导入 预训练的词向量
    Word2VecModel = gensim.models.Word2Vec.load(outfile)
    return Word2VecModel


def load_embeding():
    # 训练词向量(用空格隔开的文本)
    infile = "./CarCommentAll_cut.csv"
    outfile = "/opt/data/nlp/开源词向量/gensim_word2vec_60/Word60.model"
    # trainWord2Vec(infile, outfile)
    # 加载词向量
    Word2VecModel = loadMyWord2Vec(outfile)

    print('空间的词向量（60 维）:', Word2VecModel.wv['空间'].shape, Word2VecModel.wv['空间'])
    print('打印与空间最相近的5个词语: ', Word2VecModel.wv.most_similar('空间', topn=5))

    ## 2 构造包含所有词语的 list，以及初始化 “词语-序号”字典 和 “词向量”矩阵
    vocab_list = [word for word, Vocab in Word2VecModel.wv.vocab.items()]# 存储 所有的 词语

    word_index = {" ": 0}# 初始化 `[word : token]` ，后期 tokenize 语料库就是用该词典。
    word_vector = {} # 初始化`[word : vector]`字典

    # 初始化存储所有向量的大矩阵，留意其中多一位（首行），词向量全为 0，用于 padding补零。
    # 行数 为 所有单词数+1 比如 10000+1 ； 列数为 词向量“维度”比如60。
    embeddings_matrix = np.zeros((len(vocab_list) + 1, Word2VecModel.vector_size))

    ## 3 填充 上述 的字典 和 大矩阵
    for i in range(len(vocab_list)):
        # print(i)
        word = vocab_list[i]  # 每个词语
        word_index[word] = i + 1 # 词语: 序号
        word_vector[word] = Word2VecModel.wv[word] # 词语: 词向量
        embeddings_matrix[i + 1] = Word2VecModel.wv[word]  # 词向量矩阵
    print("加载词向量结束..")
    return vocab_list, word_index, embeddings_matrix


def plot_history(history):
    history_dict = history.history
    print(history_dict.keys())
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    epochs = range(1, len(acc) + 1)
    # “bo”代表 "蓝点"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b代表“蓝色实线”
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('Emotion_loss.png')
    # plt.show()

    plt.clf()   # 清除数字

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('Emotion_acc.png')
    # plt.show()


class EmotionModel(object):
    def __init__(self, config):
        self.model = None
        self.config = config
        self.pre_num = self.config.pre_num
        self.data_file = self.config.data_file
        self.vocab_list = self.config.vocab_list
        self.word_index = self.config.word_index
        self.EMBEDDING_DIM = self.config.EMBEDDING_DIM
        self.MAX_SEQUENCE_LENGTH = self.config.MAX_SEQUENCE_LENGTH

        # 如果模型文件存在则直接加载模型，否则开始训练
        if os.path.exists(self.config.model_file):
            self.model = load_model(self.config.model_file)
            self.model.summary()
        else:
            self.train()

    def build_model(self, embeddings_matrix):
        ## 4 在 keras的Embedding层中使用 预训练词向量
        embedding_layer = Embedding(
            input_dim = len(embeddings_matrix), # 字典长度
            output_dim = self.EMBEDDING_DIM, # 词向量 长度（60）
            weights = [embeddings_matrix], # 重点: 预训练的词向量系数
            input_length = self.MAX_SEQUENCE_LENGTH, # 每句话的 最大长度（必须padding） 
            trainable = False # 是否在 训练的过程中 更新词向量
        )
        # 如果不加载外界的，可以自己训练
        # 可以看出在使用 Keras的中Embedding层时候，不指定参数 weights=[embeddings_matrix] 即可自动生成词向量。
        # embedding_layer = Embedding(
        #     input_dim = len(word_index) + 1, # 由于 没有预训练，设置+1 
        #     output_dim = EMBEDDING_DIM, # 设置词向量的维度
        #     input_length=MAX_SEQUENCE_LENGTH
        # ) #设置句子的最大长度
        print("开始训练模型.....")
        sequence_input = Input(shape=(self.MAX_SEQUENCE_LENGTH,), dtype='int32')  # 返回一个张量，长度为1000，也就是模型的输入为batch_size*1000
        embedded_sequences = embedding_layer(sequence_input)  # 返回batch_size*1000*100
        # 添加 注意力(本质上是通过加入  一个随机向量 作为 权重 来优化 输入的值 - 与全链接不同的是，这个还会作为输入项 和 输入做点乘 )
        attention_probs = Dense(self.EMBEDDING_DIM, activation='softmax', name='attention_probs')(embedded_sequences)
        attention_mul = multiply([embedded_sequences, attention_probs], name='attention_mul')
        x = Bidirectional(GRU(self.EMBEDDING_DIM, return_sequences=True, dropout=0.5))(attention_mul)
        x = Dropout(0.5)(x)
        x = Flatten()(x)
        # x = BatchNormalization()(x)
        preds = Dense(self.pre_num, activation='softmax')(x)
        self.model = Model(sequence_input, preds)
        # 设置优化器
        optimizer = Adam(lr=self.config.learning_rate, beta_1=0.95, beta_2=0.999,epsilon=1e-08)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.summary()

    def load_word2jieba(self):
        vocab_list = load_pkl(self.vocab_list)
        if vocab_list != []:
            print("加载词的总量: ", len(vocab_list))
            for word in vocab_list:
                jieba.add_word(word)

    def predict(self, line):
        '''预测'''
        word_index = load_pkl(self.word_index)
        STOPWORDS = ["-", "\t", "\n", ".", "。", ",", "，", ";", "!", "！", "?", "？", "%"]
        words = [word for word in jieba.cut(str(line), cut_all=False) if word not in STOPWORDS]
        indexs = [word_index.get(word, 0) for word in words]
        x_pred = pad_sequences([indexs], maxlen=self.MAX_SEQUENCE_LENGTH)
        res = self.model.predict(x_pred, verbose=0)[0]
        return res

    def load_data(self, word_index, vocab_list, test_size=0.25):
        STOPWORDS = ["-", "\t", "\n", ".", "。", ",", "，", ";", "!", "！", "?", "？", "%"]
        if vocab_list != []:
            for word in vocab_list:
                jieba.add_word(word)

        def func(line):
            # 将文本 ['1, 2, 3', '1, 2, .., n'] 分解为: [[1, 2, 3], [1, 2, .., n]]
            words = [word for word in jieba.cut(str(line), cut_all=False) if word not in STOPWORDS]
            indexs = [word_index.get(word, 0) for word in words]
            return indexs

        df = pd.read_excel(self.data_file, header=0, error_bad_lines=False, encoding="utf_8_sig")
        x = df["comment"].apply(lambda line: func(line)).tolist()
        x = pad_sequences(x, maxlen=self.MAX_SEQUENCE_LENGTH)
        y = df["label"].tolist()
        # 按照大小和顺序，生成 label(0,1,2...自然数类型)
        """
        In [7]: to_categorical(np.asarray([1,1,0,1,3]))
        Out[7]:
        array([[0., 1., 0., 0.],
            [0., 1., 0., 0.],
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 0., 1.]], dtype=float32)
        """
        y = to_categorical(np.asarray(y))
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=10000)
        return (x_train, y_train), (x_test, y_test) 

    def train(self):
        '''训练模型'''
        vocab_list, word_index, embeddings_matrix = load_embeding()
        save_pkl(vocab_list, self.vocab_list)
        save_pkl(word_index, self.word_index)
        (x_train, y_train), (x_test, y_test) = self.load_data(word_index, vocab_list)
        print("---------")
        print(x_train[:3], "\n", y_train[:3])
        print("\n")
        print(x_test[:3], "\n", y_test[:3])
        print("---------")
        self.build_model(embeddings_matrix)

        # 画相关的 loss 和 accuracy=(预测正确-正or负/总预测的)
        history = self.model.fit(x_train, y_train, batch_size=60, epochs=40, validation_split=0.2, verbose=0)
        plot_history(history)

        # self.model.fit(x_train, y_train, batch_size=60, epochs=40)
        self.model.evaluate(x_test, y_test, verbose=2)
        self.model.save(self.config.model_file)


if __name__ == '__main__':
    # 测试加载外界word2vec词向量
    # vocab_list, word_index, embeddings_matrix = load_embeding()
    model = EmotionModel(Config)
    status = False
    while 1:
        text = input("text:")
        if text in ["exit", "quit"]:
            break
        # 首次启动加载jieba词库
        if not status:
            model.load_word2jieba()
            status = True
        res = model.predict(text)
        label_dic = {0:"消极的", 1:"中性的", 2:"积极的"}
        print(res, " : ", label_dic[np.argmax(res)])

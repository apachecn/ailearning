# *-* coding:utf-8 *-*
# https://blog.csdn.net/u012052268/article/details/90238282
# 词向量: 
#   https://blog.csdn.net/xiezj007/article/details/85073890
#   https://www.cnblogs.com/Darwin2000/p/5786984.html
#   https://ai.tencent.com/ailab/nlp/embedding.html
import re
import os
import keras
import random
import gensim
import numpy as np
from keras import Model
from keras.models import load_model
from keras.layers import Dropout, Dense, Flatten, Bidirectional, Embedding, GRU, Input
from keras.optimizers import Adam
# 该目录下的 config.py文件， 数据文件是: poetry.txt
from config import Config


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
    outfile = "/opt/data/开源词向量/gensim_word2vec_60/Word60.model"
    # trainWord2Vec(infile, outfile)
    # 加载词向量
    Word2VecModel = loadMyWord2Vec(outfile)

    print('空间的词向量（60 维）:', Word2VecModel.wv['空间'].shape, Word2VecModel.wv['空间'])
    print('打印与空间最相近的5个词语：', Word2VecModel.wv.most_similar('空间', topn=5))

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
        word_index[word] = i + 1 # 词语：序号
        word_vector[word] = Word2VecModel.wv[word] # 词语：词向量
        embeddings_matrix[i + 1] = Word2VecModel.wv[word]  # 词向量矩阵
    print("加载词向量结束..")
    return embeddings_matrix


class EmotionModel(object):
    def __init__(self, config):
        self.model = None
        self.config = config
        self.pre_num = self.config.pre_num

        # 如果模型文件存在则直接加载模型，否则开始训练
        if os.path.exists(self.config.weight_file):
            self.model = load_model(self.config.weight_file)
            self.model.summary()
        else:
            self.train()

    def build_model(self, embeddings_matrix):
        MAX_SEQUENCE_LENGTH = 1000  # 每个文本或者句子的截断长度，只保留1000个单词

        ## 4 在 keras的Embedding层中使用 预训练词向量
        EMBEDDING_DIM = 100 # 词向量维度
        embedding_layer = Embedding(
            input_dim = len(embeddings_matrix), # 字典长度
            output_dim = EMBEDDING_DIM, # 词向量 长度（100）
            weights = [embeddings_matrix], # 重点：预训练的词向量系数
            input_length = MAX_SEQUENCE_LENGTH, # 每句话的 最大长度（必须padding） 
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
        # 使用
        sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')  # 返回一个张量，长度为1000，也就是模型的输入为batch_size*1000
        embedded_sequences = embedding_layer(sequence_input)  # 返回batch_size*1000*100
        x = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
        x = Dropout(0.6)(x)
        x = Flatten()(x)
        preds = Dense(len(self.pre_num), activation='softmax')(x)
        model = Model(sequence_input, preds)
        # 设置优化器
        optimizer = Adam(lr=self.config.learning_rate)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()

    def predict(self, x_pred):
        '''预测'''
        # x_pred
        res = self.model.predict(x_pred, verbose=0)[0] 
        return res

    def load_data():
        pass

    def train(self):
        '''训练模型'''
        embeddings_matrix = load_embeding()
        x_train, y_train, x_val, y_val = load_data()
        self.build_model(embeddings_matrix)
        self.model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_val, y_val))
        self.model.save(self.config.weight_file)


if __name__ == '__main__':
    # 测试加载外界word2vec词向量
    load_embeding()

    # model = EmotionModel(Config)
    # while 1:
    #     text = input("text:")
    #     res = model.predict(text)
    #     print(res)


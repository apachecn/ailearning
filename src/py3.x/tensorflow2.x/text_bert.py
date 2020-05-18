# *-* coding:utf-8 *-*
# 预训练模型 bert: 
#   https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
# 参考代码:
#   https://blog.csdn.net/qq_32796253/article/details/98844242
import json
import numpy as np
import pandas as pd
from random import choice
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import re, os
import codecs
from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam
from config import Config


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]') # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]') # 剩余的字符是[UNK]
        return R


class data_generator:
    def __init__(self, data, tokenizer, batch_size=16):
        self.data = data
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:Config.bert.maxlen]
                x1, x2 = self.tokenizer.encode(first=text)
                y = d[1]
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y
                    [X1, X2, Y] = [], [], []



def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


if __name__ == "__main__":
    tb = TextBert()
    model = tb.build_model()
    tokenizer = OurTokenizer(tb.token_dict)

    train_data, valid_data = tb.prepare_data()
    train_D = data_generator(train_data, tokenizer)
    valid_D = data_generator(valid_data, tokenizer)
    model.fit_generator(
        train_D.__iter__(),
        steps_per_epoch=len(train_D),
        epochs=5,
        validation_data=valid_D.__iter__(),
        validation_steps=len(valid_D)
    )


## 文本数据
## bert / Embedding/  + lstm + crt


#%%
# 加载数据
class TextBert():
    def __init__(self):
        self.path_config = Config.bert.path_config
        self.path_checkpoint = Config.bert.path_checkpoint

        self.token_dict = {}
        with codecs.open(Config.bert.dict_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                self.token_dict[token] = len(self.token_dict)


    def prepare_data(self):
        neg = pd.read_excel(Config.bert.path_neg, header=None)
        pos = pd.read_excel(Config.bert.path_pos, header=None)
        data = []
        for d in neg[0]:
            data.append((d, 0))
        for d in pos[0]:
            data.append((d, 1))
        # 按照9:1的比例划分训练集和验证集
        random_order = list(range(len(data)))
        np.random.shuffle(random_order)
        train_data = [data[j] for i, j in enumerate(random_order) if i % 10 != 0]
        valid_data = [data[j] for i, j in enumerate(random_order) if i % 10 == 0]
        return train_data, valid_data

    def build_model(self, m_type="bert"):
        if m_type == "bert":
            bert_model = load_trained_model_from_checkpoint(self.path_config, self.path_checkpoint, seq_len=None)
            for l in bert_model.layers:
                l.trainable = True
            x1_in = Input(shape=(None,))
            x2_in = Input(shape=(None,))
            x = bert_model([x1_in, x2_in])
            x = Lambda(lambda x: x[:, 0])(x)
            p = Dense(1, activation='sigmoid')(x)#根据分类种类自行调节，也可以多加一些层数
            model = Model([x1_in, x2_in], p)
            model.compile(
                loss='binary_crossentropy',
                optimizer=Adam(1e-5), # 用足够小的学习率
                metrics=['accuracy']
            )
        else:
            # 否则用 Embedding
            model = Sequential()
            model.add(Embedding(len(vocab), EMBED_DIM, mask_zero=True))  # Random embedding
            model.add(Bidirectional(LSTM(BiRNN_UNITS // 2, return_sequences=True)))
            crf = CRF(len(chunk_tags), sparse_target=True)
            model.add(crf)
            model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
        
        model.summary()
        return model


#%%
# 加载数据
from keras_bert import Tokenizer
#字典
token_dict = {
    '[CLS]': 0,
    '[SEP]': 1,
    'un': 2,
    '##aff': 3,
    '##able': 4,
    '[UNK]': 5,
}

tokenizer = Tokenizer(token_dict)

# 拆分单词实例
print(tokenizer.tokenize('unaffable')) 
# ['[CLS]', 'un', '##aff', '##able', '[SEP]']

# indices是字对应索引
# segments表示索引对应位置上的字属于第一句话还是第二句话
# 这里只有一句话 unaffable，所以segments都是0
indices, segments = tokenizer.encode('unaffable')
print(indices)  
# [0, 2, 3, 4, 1]
print(segments)  
# [0, 0, 0, 0, 0]



# %%
print(tokenizer.tokenize('unknown')) 
# ['[CLS]', 'un', '##k', '##n', '##o', '##w', '##n', '[SEP]']

indices, segments = tokenizer.encode('unknown')
# [0, 2, 5, 5, 5, 5, 5, 1]
# [0, 0, 0, 0, 0, 0, 0, 0]

# %%
print(tokenizer.tokenize(first='unaffable', second='钢'))
# ['[CLS]', 'un', '##aff', '##able', '[SEP]', '钢', '[SEP]']
indices, segments = tokenizer.encode(first='unaffable', second='钢', max_len=10)
print(indices)  
# [0, 2, 3, 4, 1, 5, 1, 0, 0, 0]
print(segments)  
# [0, 0, 0, 0, 0, 1, 1, 0, 0, 0]

# %%
import keras
from keras_bert import get_base_dict, get_model, compile_model, gen_batch_inputs


# 输入示例
sentence_pairs = [
    [['all', 'work', 'and', 'no', 'play'], ['makes', 'jack', 'a', 'dull', 'boy']],
    [['from', 'the', 'day', 'forth'], ['my', 'arm', 'changed']],
    [['and', 'a', 'voice', 'echoed'], ['power', 'give', 'me', 'more', 'power']],
]

# 构建 token 字典
# 这个字典存放的是【词】
token_dict = get_base_dict()  
# get_base_dict()返回一个字典
# 字典预置了一些特殊token，具体内容如下
# {'': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4}
for pairs in sentence_pairs:
    for token in pairs[0] + pairs[1]:
        if token not in token_dict:
            token_dict[token] = len(token_dict)
# token_dict 是由词组成的字典，大致如下
# {'': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4, 'all': 5, 'work': 6,..., 'me': 26, 'more': 27}

token_list = list(token_dict.keys())


# 构建和训练模型
model = get_model(
    token_num=len(token_dict),
    head_num=5,
    transformer_num=12,
    embed_dim=25,
    feed_forward_dim=100,
    seq_len=20,
    pos_num=20,
    dropout_rate=0.05,
)
compile_model(model)
model.summary()

def _generator():
    while True:
        yield gen_batch_inputs(
            sentence_pairs,
            token_dict,
            token_list,
            seq_len=20,
            mask_rate=0.3,
            swap_sentence_rate=1.0,
        )

model.fit_generator(
# 这里测试集和验证集使用了同样的数据
# 实际中使用时不能这样
    generator=_generator(),
    steps_per_epoch=1000,
    epochs=100,
    validation_data=_generator(),
    validation_steps=100,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    ],
)


# 使用训练好的模型
# 取出 输入层 和 最后一个特征提取层
inputs, output_layer = get_model(
    token_num=len(token_dict),
    head_num=5,
    transformer_num=12,
    embed_dim=25,
    feed_forward_dim=100,
    seq_len=20,
    pos_num=20,
    dropout_rate=0.05,
    training=False,
    trainable=False,
    output_layer_num=4,
)

# %%
import os
from config import Config

# 设置预训练模型的路径
config_path = Config.bert.path_config
checkpoint_path = Config.bert.path_checkpoint
vocab_path = Config.bert.dict_path

# 构建字典
# 也可以用 keras_bert 中的 load_vocabulary() 函数
# 传入 vocab_path 即可
# from keras_bert import load_vocabulary
# token_dict = load_vocabulary(vocab_path)
import codecs
token_dict = {}
with codecs.open(vocab_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

# 加载预训练模型
from keras_bert import load_trained_model_from_checkpoint
model = load_trained_model_from_checkpoint(config_path, checkpoint_path)

# Tokenization
from keras_bert import Tokenizer

tokenizer = Tokenizer(token_dict)
text = '语言模型'
tokens = tokenizer.tokenize(text)
# ['[CLS]', '语', '言', '模', '型', '[SEP]']
indices, segments = tokenizer.encode(first=text, max_len=512)
print(indices[:10])
# [101, 6427, 6241, 3563, 1798, 102, 0, 0, 0, 0]
print(segments[:10])
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# 提取特征
import numpy as np

predicts = model.predict([np.array([indices]), np.array([segments])])[0]
for i, token in enumerate(tokens):
    print(token, predicts[i].tolist()[:5])

# %%
token_dict = {}
with codecs.open(vocab_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

token_dict_rev = {v: k for k, v in token_dict.items()}

model = load_trained_model_from_checkpoint(config_path, checkpoint_path, training=True)

text = '数学是利用符号语言研究数量、结构、变化以及空间等概念的一门学科'
tokens = tokenizer.tokenize(text)
tokens[1] = tokens[2] = '[MASK]'# ['[CLS]', '[MASK]', '[MASK]', '是', '利',..., '学', '科', '[SEP]']

indices = np.array([[token_dict[token] for token in tokens] + [0] * (512 - len(tokens))])
segments = np.array([[0] * len(tokens) + [0] * (512 - len(tokens))])
masks = np.array([[0, 1, 1] + [0] * (512 - 3)])
predicts = model.predict([indices, segments, masks])[0].argmax(axis=-1).tolist()
print('Fill with: ', list(map(lambda x: token_dict_rev[x], predicts[0][1:3])))
# Fill with:  ['数', '学']

# %%

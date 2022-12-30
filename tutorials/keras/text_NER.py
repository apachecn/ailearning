import pickle
import numpy as np
import pandas as pd
import platform
from collections import Counter
import keras
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dropout
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
"""
# padding: pre(默认) 向前补充0  post 向后补充0
# truncating: 文本超过 pad_num,  pre(默认) 删除前面  post 删除后面
# x_train = pad_sequences(x, maxlen=pad_num, value=0, padding='post', truncating="post")
# print("--- ", x_train[0][:20])

使用keras_bert、keras_contrib的crf时bug记录
TypeError: Tensors in list passed to 'values' of 'ConcatV2' Op have types [bool, float32] that don't all match
解决方案, 修改crf.py 516行：
mask2 = K.cast(K.concatenate([mask, K.zeros_like(mask[:, :1])], axis=1),
为:
mask2 = K.cast(K.concatenate([mask, K.cast(K.zeros_like(mask[:, :1]), mask.dtype)], axis=1),
"""
from keras.preprocessing.sequence import pad_sequences
from config.setting import Config


def load_data():
    train = _parse_data(Config.nlp_ner.path_train)
    test  = _parse_data(Config.nlp_ner.path_test)
    print("--- init 数据加载解析完成 ---")
    
    # Counter({'的': 8, '中': 7, '致': 7, '党': 7})
    word_counts = Counter(row[0].lower() for sample in train for row in sample)
    vocab = [w for w, f in iter(word_counts.items()) if f >= 2]
    chunk_tags = Config.nlp_ner.chunk_tags

    # 存储保留的有效个数的 vovab 和 对应 chunk_tags
    with open(Config.nlp_ner.path_config, 'wb') as outp:
        pickle.dump((vocab, chunk_tags), outp)
    print("--- init 配置文件保存成功 ---")

    train = _process_data(train, vocab, chunk_tags)
    test  = _process_data(test , vocab, chunk_tags)
    print("--- init 对数据进行编码，生成训练需要的数据格式 ---")
    return train, test, (vocab, chunk_tags)


def _parse_data(filename):
    """
    以单下划线开头（_foo）的代表不能直接访问的类属性
    用于解析数据，用于模型训练
    :param filename: 文件地址
    :return: data: 解析数据后的结果
    [[['中', 'B-ORG'], ['共', 'I-ORG']], [['中', 'B-ORG'], ['国', 'I-ORG']]]
    """
    with open(filename, 'rb') as fn:
        split_text = '\n'
        # 主要是分句: split_text 默认每个句子都是一行，所以原来换行就需要 两个split_text
        texts = fn.read().decode('utf-8').strip().split(split_text + split_text)
        # 对于每个字需要 split_text, 而字的内部需要用空格分隔
        # len(row) > 0 避免连续2个换行，导致 row 数据为空
        # row.split() 会删除空格或特殊符号，导致空格数据缺失！
        data = [[[" ", "O"] if len(row.split()) != 2 else row.split() for row in text.split(split_text) if len(row) > 0] for text in texts]
        # data = [[row.split() for row in text.split(split_text) if len(row.split()) == 2] for text in texts]
    return data


def _process_data(data, vocab, chunk_tags, maxlen=None, onehot=False):
    if maxlen is None:
        maxlen = max(len(s) for s in data)
    
    # 对每个字进行编码
    word2idx = dict((w, i) for i, w in enumerate(vocab))
    # 如果不在 vocab里面，就给 unk 值为 1
    x = [[word2idx.get(w[0].lower(), 1) for w in s] for s in data]
    y_chunk = [[chunk_tags.index(w[1])  for w in s] for s in data]

    x = pad_sequences(x, maxlen)  # left padding
    y_chunk = pad_sequences(y_chunk, maxlen, value=-1)

    if onehot:
        # 返回一个onehot 编码的多维数组
        y_chunk = np.eye(len(chunk_tags), dtype='float32')[y_chunk]
    else:
        # np.expand_dims:用于扩展数组的形状
        # https://blog.csdn.net/hong615771420/article/details/83448878
        y_chunk = np.expand_dims(y_chunk, 2)
    return x, y_chunk


def process_data(data, vocab, maxlen=100):
    word2idx = dict((w, i) for i, w in enumerate(vocab))
    x = [word2idx.get(w[0].lower(), 1) for w in data]
    length = len(x)
    x = pad_sequences([x], maxlen)  # left padding
    return x, length


def create_model(len_vocab, len_chunk_tags):
    model = Sequential()
    model.add(Embedding(len_vocab, Config.nlp_ner.EMBED_DIM, mask_zero=True))  # Random embedding
    model.add(Bidirectional(LSTM(Config.nlp_ner.BiLSTM_UNITS // 2, return_sequences=True)))
    model.add(Dropout(0.25))
    crf = CRF(len_chunk_tags, sparse_target=True)
    model.add(crf)
    model.summary()
    model.compile('adam', loss=crf_loss, metrics=[crf_viterbi_accuracy])
    # model.compile('rmsprop', loss=crf_loss, metrics=[crf_viterbi_accuracy])

    # from keras.optimizers import Adam
    # adam_lr = 0.0001
    # adam_beta_1 = 0.5
    # model.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1), loss=crf_loss, metrics=[crf_viterbi_accuracy])
    return model


def train():
    (train_x, train_y), (test_x, test_y), (vocab, chunk_tags) = load_data()
    model = create_model(len(vocab), len(chunk_tags))
    # train model
    model.fit(train_x, train_y, batch_size=16, epochs=Config.nlp_ner.EPOCHS, validation_data=[test_x, test_y])
    model.save(Config.nlp_ner.path_model)


def test():
    with open(Config.nlp_ner.path_config, 'rb') as inp:
        (vocab, chunk_tags) = pickle.load(inp)
    model = create_model(len(vocab), len(chunk_tags))
    # predict_text = '造型独特，尺码偏大，估计是钉子头圆的半径的缘故'
    with open(Config.nlp_ner.path_origin, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for predict_text in lines:
            content = predict_text.strip()
            text_EMBED, length = process_data(content, vocab)
            model.load_weights(Config.nlp_ner.path_model)
            raw = model.predict(text_EMBED)[0][-length:]
            pre_result = [np.argmax(row) for row in raw]
            result_tags = [chunk_tags[i] for i in pre_result]

            # 保存每句话的 实体和观点
            result = {}
            tag_list = [i for i in chunk_tags if i not in ["O"]]
            for word, t in zip(content, result_tags):
                # print(word, t)
                if t not in tag_list:
                    continue
                for i in range(0, len(tag_list), 2):
                    if t in tag_list[i:i+2]:
                        # print("\n>>> %s---%s==%s" % (word, t, tag_list[i:i+2]))
                        tag = tag_list[i].split("-")[-1]
                        if tag not in result:
                            result[tag] = ""
                        result[tag] += ' '+word if t==tag_list[i] else word
            print(result)


def main():
    # print("--")
    train()
    test()

# if __name__ == "__main__":
#     train()

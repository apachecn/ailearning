# *-* coding:utf-8 *-*
'''
代码参考: https://github.com/ioiogoo/poetry_generator_Keras
做了一定的简化，作者 @ioiogoo 协议是 MIT
目标: 自动生成歌词的
'''
import re
import os
import keras
import random
import numpy as np
from keras.callbacks import LambdaCallback
from keras.models import load_model
from keras.layers import Dropout, Dense, Flatten, Bidirectional, Embedding, GRU
from keras.optimizers import Adam
# 该目录下的 config.py文件， 数据文件是: poetry.txt
from config import Config


def preprocess_file(Config):
    # 读取文本内容，合并到一个大字符中，用 ] 隔开
    files_content = ''
    with open(Config.poetry_file, 'r', encoding='utf-8') as f:
        for line in f:
            # 每行的末尾加上"]"符号代表一首诗结束
            line = re.sub(r"[\]\[（）(){}《》: ]+", "", line.strip())
            files_content += line + "]"
    
    # 按照字存到字典中，字+频率
    words = [i for i in sorted(list(files_content)) if i != "]"]
    counted_words = {}
    for word in words:
        if word in counted_words:
            counted_words[word] += 1
        else:
            counted_words[word] = 1

    # 去掉低频的字
    # [('。', 567), ('，', 565), ('风', 47), ('花', 42), ('云', 40)]
    wordPairs = sorted([(k,v) for k,v in counted_words.items() if v>=2], key=lambda  x: x[1], reverse=True)
    # print(wordPairs)

    words, _ = zip(*wordPairs)
    # word到id的映射
    word2num = dict((c, i) for i, c in enumerate(words))
    num2word = dict((i, c) for i, c in enumerate(words))
    word2numF = lambda x: word2num.get(x, 0)
    return word2numF, num2word, words, files_content


class PoetryModel(object):
    def __init__(self, config):
        self.model = None
        self.do_train = True
        self.loaded_model = False
        self.config = config

        # 文件预处理
        self.word2numF, self.num2word, self.words, self.files_content = preprocess_file(self.config)

        # 如果模型文件存在则直接加载模型，否则开始训练
        if os.path.exists(self.config.weight_file):
            self.model = load_model(self.config.weight_file)
            self.model.summary()
        else:
            self.train()

        self.do_train = False
        self.loaded_model = True

    def build_model(self):
        '''构建模型'''
        model = keras.Sequential()
        model.add(Embedding(len(self.num2word) + 2, 300, input_length=self.config.max_len))
        model.add(Bidirectional(GRU(128, return_sequences=True)))
        model.add(Dropout(0.6))
        model.add(Flatten())
        model.add(Dense(len(self.words), activation='softmax'))
        # 设置优化器
        optimizer = Adam(lr=self.config.learning_rate)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model

    def predict(self, text):
        '''根据给出的文字，生成诗句'''
        if not self.loaded_model:
            return
        with open(self.config.poetry_file, 'r', encoding='utf-8') as f:
            file_list = f.readlines()
        random_line = random.choice(file_list)
        # 如果给的text不到四个字，则随机补全
        if not text or len(text) != 4:
            for _ in range(4 - len(text)):
                random_str_index = random.randrange(0, len(self.words))
                text += self.num2word.get(random_str_index) \
                        if self.num2word.get(random_str_index) not in [',', '。', '，'] \
                        else self.num2word.get(random_str_index + 1)

        seed = random_line[-(self.config.max_len):-1]

        res = ''
        seed = 'c' + seed

        for c in text:
            seed = seed[1:] + c
            for j in range(5):
                x_pred = np.zeros((1, self.config.max_len))
                for t, char in enumerate(seed):
                    x_pred[0, t] = self.word2numF(char)

                preds = self.model.predict(x_pred, verbose=0)[0]
                next_index = self.sample(preds, 1.0)
                next_char = self.num2word[next_index]
                seed = seed[1:] + next_char
            res += seed
        return res

    def data_generator(self):
        '''生成器生成数据'''
        i = 0
        while 1:
            # 如果越界了，就从0再开始
            if (i + self.config.max_len) > len(self.files_content) -1 :
                i = 0
            x = self.files_content[i: i + self.config.max_len]
            y = self.files_content[i + self.config.max_len]

            puncs = [']', '[', '（', '）', '{', '}', ': ', '《', '》', ':']
            if len([i for i in puncs if i in x]) != 0:
                i += 1
                continue
            if len([i for i in puncs if i in y]) != 0:
                i += 1
                continue

            y_vec = np.zeros(
                shape=(1, len(self.words)),
                dtype=np.bool
            )
            y_vec[0, self.word2numF(y)] = 1.0

            x_vec = np.zeros(
                shape=(1, self.config.max_len),
                dtype=np.int32
            )

            for t, char in enumerate(x):
                x_vec[0, t] = self.word2numF(char)
            yield x_vec, y_vec
            i += 1

    def train(self):
        '''训练模型'''
        number_of_epoch = len(self.files_content) // self.config.batch_size

        if not self.model:
            self.build_model()

        self.model.summary()

        self.model.fit_generator(
            generator=self.data_generator(),
            verbose=True,
            steps_per_epoch=self.config.batch_size,
            epochs=number_of_epoch,
            callbacks=[
                keras.callbacks.ModelCheckpoint(self.config.weight_file, save_weights_only=False),
                LambdaCallback(on_epoch_end=self.generate_sample_result)
            ]
        )


if __name__ == '__main__':
    model = PoetryModel(Config)
    while 1:
        text = input("text:")
        sentence = model.predict(text)
        print(sentence)

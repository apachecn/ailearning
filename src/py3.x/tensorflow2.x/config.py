# *-* coding:utf-8 *-*
'''
@author: ioiogoo
@date: 2018/1/31 19:28
'''


class Config(object):
    poetry_file = 'poetry.txt'
    weight_file = 'poetry_model.h5'
    data_file = 'EmotionData.xlsx'
    model_file = 'EmotionModel.h5'
    vocab_list = 'vocal_list.pkl'
    word_index = 'word_index.pkl'
    # 根据前六个字预测第七个字
    max_len = 6
    batch_size = 512
    learning_rate = 0.005
    pre_num = 3
    MAX_SEQUENCE_LENGTH = 1000  # 每个文本或者句子的截断长度，只保留1000个单词
    EMBEDDING_DIM = 60 # 词向量维度

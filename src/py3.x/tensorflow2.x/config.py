# *-* coding:utf-8 *-*
'''
@author: 片刻
@date: 20200428 11:02
'''


class Bert(object):
    DEBUG = True
    path_root = "/home/wac/jiangzhonglian"
    if DEBUG:
        path_root = "/opt/data/nlp/开源词向量/bert官方版预训练模型"

    dict_path = '%s/chinese_L-12_H-768_A-12/vocab.txt' % path_root
    path_config = '%s/chinese_L-12_H-768_A-12/bert_config.json' % path_root
    path_checkpoint = '%s/chinese_L-12_H-768_A-12/bert_model.ckpt' % path_root
    maxlen = 100
    path_neg = "Emotion/neg.xlsx"
    path_pos = "Emotion/pos.xlsx"


class Config(object):
    poetry_file = 'poetry.txt'
    weight_file = 'poetry_model.h5'
    data_file = 'Emotion/EmotionData.xlsx'
    model_file = 'Emotion/EmotionModel.h5'
    vocab_list = 'Emotion/vocal_list.pkl'
    word_index = 'Emotion/word_index.pkl'
    # 根据前六个字预测第七个字
    max_len = 6
    batch_size = 512
    learning_rate = 0.0005
    pre_num = 3
    MAX_SEQUENCE_LENGTH = 1000  # 每个文本或者句子的截断长度，只保留1000个单词
    EMBEDDING_DIM = 60 # 词向量维度

    bert = Bert()
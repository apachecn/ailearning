# *-* coding:utf-8 *-*
'''
@author: 片刻
@date: 20200901 22:02
'''

class TextNER(object):
    DEBUG = True
    if DEBUG:
        path_root = "data/NER3"
        chunk_tags = ['O', 'B-ORG', 'I-ORG', 
            'B-Po_VIEW', 'I-Po_VIEW', 
            'B-Mi_VIEW', 'I-Mi_VIEW', 
            'B-Ne_VIEW', 'I-Ne_VIEW'
        ]
        # chunk_tags = ['O', 'B-TIME', 'I-TIME', 'B-LOCATION', 'I-LOCATION', 
        #     "B-PERSON_NAME", "I-PERSON_NAME", "B-ORG_NAME", "I-ORG_NAME",
        #     "B-COMPANY_NAME", "I-COMPANY_NAME", "B-PRODUCT_NAME", "I-PRODUCT_NAME"]
        # chunk_tags = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', "B-ORG", "I-ORG"]
    else:
        path_root = "data/NER"
        chunk_tags = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', "B-ORG", "I-ORG"]

    path_origin = '%s/origin.txt'  % path_root
    path_train  = '%s/train.txt'   % path_root
    path_test   = '%s/test.txt'    % path_root
    path_config = '%s/config.pkl'  % path_root
    path_model  = '%s/model.h5'    % path_root

    # 迭代次数
    EPOCHS = 10
    # embedding的维度数
    EMBED_DIM = 128
    # LSTM的维度数
    BiLSTM_UNITS = 128


class Config(object):
    nlp_ner = TextNER()

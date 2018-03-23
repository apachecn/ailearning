import struct
from  BP import *
from datetime import datetime

class Loader(object):
    def __init__(self,path,count):
        '''
        初始化加载器
        path: 数据文件路径
        count: 文件中的样本个数
        '''
        self.path = path
        self.count = count


    def get_file_content(self):
        f = open(self.path,'rb')
        content = f.read()
        f.close()
        return list(content)

    def to_int(self,byte):
         '''
        将unsigned byte字符转换为整数
        '''
        # return struct.unpack('B',byte)[0]
         return byte

class ImageLoader(Loader):
    def get_picture(self,content,index):
        start = index *28 *28+16
        picture = []
        for i in range(28):
            picture.append([])
            for j in range(28):
                picture[i].append(
                    self.to_int(content[start+i*28+j]))
        return picture

    def get_one_sample(self,picture):
        sample = []
        for i in range(28):
            for j in range(28):
                sample.append(picture[i][j])
        return  sample


    def load(self):
        content = self.get_file_content()
        data_set = []
        for index in range(self.count):
            data_set.append(
                self.get_one_sample(
                    self.get_picture(content,index)
                )
            )

        return data_set

class LabelLoader(Loader):
    def load(self):
        '''
            加载数据文件，获得全部样本的标签向量
            '''
        content = self.get_file_content()
        labels= []

        for index in range(self.count):
            labels.append(self.norm(content[index+8]))

        return labels

    def norm(self,label):
        '''
           内部函数，将一个值转换为10维标签向量
           '''
        label_vec =  []
        label_value = self.to_int(label)
        for i in range(10):
            if i == label_value:
                label_vec.append(0.9)
            else:
                label_vec.append(0.1)
        return label_vec


def get_train_data_set():
    '''
    获得训练数据集
    '''
    #源程序文件名为train-images-idx3-ubyte，无论是修改数据文件名或程序均可
    image_loader = ImageLoader('train-images.idx3-ubyte', 60)
    label_loader = LabelLoader('train-labels.idx1-ubyte', 60)
    return image_loader.load(), label_loader.load()

def get_test_data_set():
    '''
    获得测试数据集
    '''
    # 源程序文件名为t10k-images-idx3-ubyte，无论是修改数据文件名或程序均可
    image_loader = ImageLoader('t10k-images.idx3-ubyte',10)
    label_loader = LabelLoader('t10k-labels.idx1-ubyte',10)
    return image_loader.load(), label_loader.load()


def get_result(vec):
    max_value_index = 0
    max_value = 0
    vec = list(vec)#python3中 zip需要强制转换为list否则没有长度
    for i in range(len(vec)):
        if vec[i]>max_value:
            max_value_index = i
    return max_value_index


def evaluate(network,test_data_set,test_labels):
    error = 0
    total = len(test_data_set)
    for i in range(total):
        label = get_result(test_labels[i])
        predict = get_result(network.predict(test_data_set[i]))
        if label != predict:
            error += 1
    return error/total


def now():
    return datetime.now().strftime('%c')

def train_and_evaluate():
    last_error_ratio = 1.0
    epoch = 0
    train_data_set, train_labels = get_train_data_set()
    test_data_set, test_labels = get_test_data_set()
    network = Network([784, 300, 10])
    while True:
        epoch += 1
        network.train(train_labels, train_data_set, 0.3, 1)
        print('%s epoch %d finished' % (now(), epoch))
        if epoch % 10 == 0:
            error_ratio = evaluate(network, test_data_set, test_labels)
            print('%s after epoch %d, error ratio is %f' % (now(), epoch, error_ratio))
            if error_ratio > last_error_ratio:
                break
            else:
                last_error_ratio = error_ratio
if __name__ == '__main__':
    train_and_evaluate()
    '''
    小批数据错误率不下降可能是因为数据输入时没有进行打乱，程序一直在学习某个数字
    如果大批量数据，error ratio仍不下降，问题可能在于文件读取这块。
    建议查看二进制文件读取出的content转化为list后是否正确，
    是否能从文件中看出数字轮廓
    '''
'''
    import numpy as np
    import seaborn as sns
    f = open('train-images.idx3-ubyte', 'rb')
    content = f.read()
    f.close()
    dataarr = np.array(list(content))
    dataarr = dataarr[16:].reshape(28,28,60000)
    dataarr[:,:,1]
'''
from Perceptron import Perceptron
from matplotlib import  pyplot as plt
#定义激活函数f
f = lambda x: x
class LinearUnit(Perceptron):
    def __init__(self, input_num):
        '''初始化线性单元，设置输入参数的个数'''
        Perceptron.__init__(self, input_num, f)


def get_train_dataset():
    input_vecs = [[5],[3],[8],[1.4],[10.1]]
    labels = [5500,2300,7600,1800,11400]
    return input_vecs,labels

def train_linear_unit():
    lu = LinearUnit(1)
    input_vecs,labels = get_train_dataset()
    lu.train(input_vecs,labels,10,0.01)
    return  lu

'''
#画图模块
def plot(linear_unit):
    import matplotlib.pyplot as plt
    input_vecs, labels = get_training_dataset()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(map(lambda x: x[0], input_vecs), labels)
    weights = linear_unit.weights
    bias = linear_unit.bias
    x = range(0,12,1)
    y = map(lambda x:weights[0] * x + bias, x)
    ax.plot(x, y)
    plt.show()
'''

if __name__=='__main__':
    linear_unit = train_linear_unit()
    input_vecs,labels = get_train_dataset()
    print(linear_unit)
    print('Work 3.4 years, monthly salary = %.2f' % linear_unit.predict([3.4]))
    print('Work 15 years, monthly salary = %.2f' % linear_unit.predict([15]))
    print('Work 1.5 years, monthly salary = %.2f' % linear_unit.predict([1.5]))
    print('Work 6.3 years, monthly salary = %.2f' % linear_unit.predict([6.3]))
    print(linear_unit.weights)
    plt.scatter(input_vecs,labels)
    y1 = 0*linear_unit.weights[0]+linear_unit.bias
    y2 = 12*linear_unit.weights[0]+ linear_unit.bias
    plt.plot([0,12],[y1,y2])
    plt.show()
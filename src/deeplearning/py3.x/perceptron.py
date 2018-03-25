from functools import reduce

def add(x,y):
    return  x+y


class Perceptron(object):
    def __init__(self,input_num,activator):
        self.activator = activator
        self.weights = [0.0 for _ in range(input_num)]
        self.bias = 0.0

    def __str__(self):
        return  'weights\t:%s\nbias\t:%f\n' % (self.weights, self.bias)

    def predict(self,input_vec):
        pack = zip(input_vec,self.weights)
        multi = []
        for (x,w) in pack:
            multi.append(x*w)
        activtion = reduce(add, multi)

        return self.activator(activtion + self.bias)

    def train(self,input_vecs,labels,iteration,rate):
        for i in range(iteration):
            self._one_iteration(input_vecs,labels,rate)

    def _one_iteration(self,input_vecs,labels,rate):
        samples = zip(input_vecs,labels)
        for (input_vecs,labels) in samples:
            output = self.predict(input_vecs)
            self._update_weights(input_vecs,output,labels,rate)


    def _update_weights(self,input_vecs,output,labels,rate):
        delta = labels -output
        pack  = zip(input_vecs,self.weights)
        tmp = []
        for (x,w) in pack:
            tmp.append(w+x*delta*rate)
        self.weights = tmp
        self.bias = self.bias + delta*rate

def f(x):
    if x>0:
        return 1
    else:
        return 0

def get_train_dataset():
    input_vecs = [[1,1],[0,0],[1,0],[0,1]]
    labels = [1,0,0,0]
    return input_vecs,labels

def train_and_perception():
    p = Perceptron(2,f)
    input_vecs,labels =get_train_dataset()
    p.train(input_vecs,labels,10,0.1)
    return p

if __name__=='__main__':
    and_perception = train_and_perception()
    print(and_perception)
    print('1 and 1 = %d' % and_perception.predict([1, 1]))
    print('0 and 0 = %d' % and_perception.predict([0, 0]))
    print('1 and 0 = %d' % and_perception.predict([1, 0]))
    print('0 and 1 = %d' % and_perception.predict([0, 1]))
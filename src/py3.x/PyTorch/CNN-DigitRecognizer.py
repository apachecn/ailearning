import os

# third-party library
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision
import matplotlib.pyplot as plt

torch.manual_seed(1)  # reproducible
'''
1. 加载数据
'''
# DOWNLOAD_MNIST = False
# # Mnist digits dataset
# if not(os.path.exists('/opt/data/mnist/')) or not os.listdir('/opt/data/mnist/'):
#     # not mnist dir or mnist is empyt dir
#     DOWNLOAD_MNIST = True

# print('DOWNLOAD_MNIST', DOWNLOAD_MNIST)

# train_data = torchvision.datasets.MNIST(
#     root='/opt/data/mnist/',
#     train=True,                                     # this is training data
#     transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
#                                                     # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
#     download=DOWNLOAD_MNIST,
# )

# # plot one example
# print(train_data.train_data.size())                 # (60000, 28, 28)
# print(train_data.train_labels.size())               # (60000)
# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
# plt.title('%i' % train_data.train_labels[0])
# plt.show()


class CustomedDataSet(Dataset):

    def __init__(self, pd_data, data_type=True):
        self.data_type = data_type
        if self.data_type:
            trainX = pd_data
            trainY = trainX.label.as_matrix().tolist()
            trainX = trainX.drop('label', axis=1).as_matrix().reshape(trainX.shape[0], 1, 28, 28)
            self.datalist = trainX
            self.labellist = trainY
        else:
            testX = pd_data
            testX = testX.as_matrix().reshape(testX.shape[0], 1, 28, 28)
            self.datalist = testX

    def __getitem__(self, index):
        if self.data_type:
            return torch.Tensor(
                self.datalist[index].astype(float)), self.labellist[index]
        else:
            return torch.Tensor(self.datalist[index].astype(float))

    def __len__(self):
        return self.datalist.shape[0]


# 先转换成 torch 能识别的 Dataset
pd_train = pd.read_csv('/opt/data/kaggle/getting-started/digit-recognizer/train.csv', header=0)
pd_pre = pd.read_csv('/opt/data/kaggle/getting-started/digit-recognizer/test.csv', header=0)

# 找到数据分割点
training_ratio = 0.8
row = pd_train.shape[0]
split_num = int(training_ratio*row)

pd_data_train = pd_train[:split_num]
pd_data_test = pd_train[split_num:]

data_train = CustomedDataSet(pd_data_train, data_type=True)
data_test = CustomedDataSet(pd_data_test, data_type=True)
data_pre = CustomedDataSet(pd_pre, data_type=False)

'''
2. 数据读取器 (Data Loader)
'''
BATCH_SIZE = 400
# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
loader_train = DataLoader(dataset=data_train, batch_size=BATCH_SIZE, shuffle=True)
loader_test = DataLoader(dataset=data_test, batch_size=pd_data_test.shape[0], shuffle=True)
loader_pre = DataLoader(dataset=data_pre, batch_size=BATCH_SIZE, shuffle=True)

# # convert test data into Variable, pick 2000 samples to speed up testing
# test_data = torchvision.datasets.MNIST(root='/opt/data/mnist/', train=False)
# test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:2000]/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
# test_y = test_data.test_labels[:2000]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,  # input height
                out_channels=16,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (16, 28, 28)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # input shape (1, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 14, 14)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)  # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x  # return x for visualization


cnn = CNN()
print(cnn)  # net architecture


LR = 0.005  # learning rate
# optimizer 优化器 是训练的工具，lr表示学习率
'''
>>> # lr = 0.05     if epoch < 30
>>> # lr = 0.005    if 30 <= epoch < 80
>>> # lr = 0.0005   if epoch >= 80
'''
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

# following function (plot_with_labels) is for visualization, can be ignored if not interested
from matplotlib import cm
try:
    from sklearn.manifold import TSNE
    HAS_SK = True
except:
    HAS_SK = False
    print('Please install sklearn for layer visualization')


def plot_with_labels(lowDWeights, labels):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9))
        plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.title('Visualize last layer')
    plt.show()
    plt.pause(0.01)


plt.ion()
# Hyper Parameters
EPOCH = 50  # train the training data n times, to save time, we just train 1 epoch
# training and testing
for epoch in range(EPOCH):
    # gives batch data, normalize x when iterate train_loader
    for step, (x, y) in enumerate(loader_train):
        b_x = Variable(x)  # batch x
        b_y = Variable(y)  # batch y

        output = cnn(b_x)[0]  # cnn output
        loss = loss_func(output, b_y)  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        if step % 50 == 0:
            for step, (x_t, y_test) in enumerate(loader_test):
                x_test = Variable(x_t)  # batch x

                test_output, last_layer = cnn(x_test)
                pred_y = torch.max(test_output, 1)[1].data.squeeze()
                accuracy = sum(pred_y == y_test) / float(y_test.size(0))
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.4f' % accuracy)
                if HAS_SK:
                    # Visualization of trained flatten layer (T-SNE)
                    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
                    plot_only = 500
                    low_dim_embs = tsne.fit_transform(
                        last_layer.data.numpy()[:plot_only, :])
                    labels = y_test.numpy()[:plot_only]
                    plot_with_labels(low_dim_embs, labels)
plt.ioff()

# print 10 predictions from test data
test_output, _ = cnn(x_test[:100])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(y_test[:100].numpy(), 'real number')

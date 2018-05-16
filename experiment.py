import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

import numpy as np
from peo import PEO

# 超参数列表
num_epochs = 1 # 多层网络训练轮数
batch_size = 100 # 多层网络每批数据量

max_sample_num = -1 # 样例监督训练样本数 -1代表采用全部样本
each_class_num = 5 # 样例监督每类样例数量

learning_rate = 0.001 # 学习率

# MNIST数据集，训练/测试
train_dataset = dsets.MNIST(root='./mnist',
                            train=True,
                            transform=transforms.ToTensor())

test_dataset = dsets.MNIST(root='./mnist',
                           train=False,
                           transform=transforms.ToTensor())

# 加载训练/测试数据
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

sample_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           shuffle=True)

# 样例数据
Sample_data = torch.FloatTensor(10, each_class_num, 1, 784)

# 多层网络 x->h1->h2->c
class NeuralNet(nn.Module):
    def __init__(self, input_size,hidden1_size,hidden2_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

# 样例监督网络 x->h1->h2->x
class SampleNet(nn.Module):
    def __init__(self,input_size,hidden1_size,hidden2_size):
        super(SampleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, input_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

# Parzen窗损失函数
class ParzenEntropyLoss(nn.Module):
    def __init__(self):
        super(ParzenEntropyLoss, self).__init__()

    def forward(self, train_output, sample_outputs):
        # Parzen窗，窗口大小为1
        out = torch.pow(train_output - sample_outputs, 2)
        out = torch.exp((-0.5)*out)
        out = torch.mean(out,dim=1)
        # 信息熵
        out = (-1)*out*torch.log(out)
        out = torch.mean(out)
        # out = torch.sum(out)
        return out

# 提取样例数据
def extract_sample_data():
    sample_pos = [0 for i in range(10)]
    for _, (image, label) in enumerate(sample_loader):
        digit = int(label)
        pos = sample_pos[digit]
        if pos < each_class_num:
            sample_pos[digit] = pos + 1
            Sample_data[digit, pos] = image
    for i in range(10):
        if sample_pos[i] < each_class_num:
            print('ERROR: %d is not full' % i)


# 多层训练
def train_run(pre_smple_net = False):
    neural_net = NeuralNet(784,128,32,10)
    criterion = nn.CrossEntropyLoss()

    # 用于加载样例监督模型预训练
    if pre_smple_net == True:
        sample_net = torch.load('sample_net.pkl')
        # 固定样例监督网络参数
        for _, p in enumerate(sample_net.parameters()):
                p.requires_grad = False
        print('sample_net.pkl load success.')

    optimizer = torch.optim.Adam(neural_net.parameters(), lr=learning_rate)

    # 训练网络
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images).view(-1,784)
            labels = Variable(labels)

            # Forward + Backward + Optimize
            optimizer.zero_grad()

            if pre_smple_net == True:
                images = sample_net(images)

            outputs = neural_net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step, loss.data[0]))

    # 测试模型准确率
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images).view(-1,784)
        if pre_smple_net == True:
            print('Test  Data mean {} , var {} '.format(torch.mean(images).data[0],torch.var(images).data[0]))
            images = sample_net(images)
            print('After Data mean {} , var {} '.format(torch.mean(images).data[0],torch.var(images).data[0]))
        outputs = neural_net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))

# 样例监督训练
def sample_run():
    sample_net = SampleNet(784,1024,784)
    criterion = ParzenEntropyLoss()
    optimizer = PEO(sample_net.parameters(), lr=learning_rate)
    extract_sample_data() # 抽取样例数据

    # 每次取一个训练数据
    for i, (image, label) in enumerate(sample_loader):
        image = Variable(image.view(-1,784))
        label = Variable(label)

        Diff_Sample_data = torch.from_numpy(np.delete(Sample_data.numpy(),int(label),axis=0)) # 通过numpy移除相同类别数据
        same_sample_outputs = Variable(Sample_data[int(label)].view(-1,784))# 与输入同类样例数据通过样例监督网络输出值 10x784
        diff_sample_outputs = Variable(Diff_Sample_data.view(-1,784)) # 与输入异类样例数据通过样例监督网络输出值  90x784

        train_output = sample_net(image) # 单个训练数据通过样例监督网络输出值

        same_loss = criterion(train_output,same_sample_outputs)
        diff_loss = criterion(train_output,diff_sample_outputs)

        optimizer.zero_grad()
        same_loss.backward(retain_graph=True) # 允许二次计算梯度
        optimizer.cache_grad() # 缓存same_loss的梯度，防止被覆盖

        optimizer.zero_grad()
        diff_loss.backward()
        optimizer.step()

        if (i + 1) % 1000 == 0:
            print('Sample Supervised Train Num: [%d], Same Loss: %.4f, Diff Loss: %.4f' % (i + 1,same_loss.data[0],diff_loss.data[0]))
        if max_sample_num != -1 and i == max_sample_num:
            break

    # 保存样例监督模型
    torch.save(sample_net, 'sample_net.pkl')
    print('sample_net.pkl write success.')


if __name__ == '__main__':
    train_run() # 多层全连接网络
    sample_run() # 样例监督预训练
    train_run(pre_smple_net=True) # 采用样例监督预训练的多层全链接网络
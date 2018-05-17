import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter

# W = Variable(torch.FloatTensor([3]),requires_grad=True)


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.W = Parameter(torch.FloatTensor([3]))
        print('W初始值：', self.W)

    def forward(self, x):
        out = x*self.W
        return out

net = Net()

y1 = net(8)

# net_loc = Net()
for param in net.parameters():
    param.requires_grad = False
y2 = net(6)
for param in net.parameters():
    param.requires_grad = True



y = y1-y2

print(y)

y.backward()

print(net.W.grad)
# print(net_loc.W.grad)

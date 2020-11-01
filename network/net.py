# from https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(kernel_size=5, in_channels=3, out_channels= 6)
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.conv2 = nn.Conv2d(kernel_size=5, in_channels=6, out_channels=16)
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.lin1  = nn.Linear(in_features=400, out_features=120)
    self.lin2  = nn.Linear(in_features=120, out_features= 84)
    self.lin3  = nn.Linear(in_features= 84, out_features=  2)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = self.pool1(x)
    x = F.relu(self.conv2(x))
    x = self.pool2(x)
    x = x.view(-1, 400)
    x = F.relu(self.lin1(x))
    x = F.relu(self.lin2(x))
    x = self.lin3(x)
    return x

# from https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    
    self.conv1 = nn.Conv2d(in_channels=3, out_channels= 6, kernel_size=5)
    self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
    
    self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
    
    self.fc1   = nn.Linear(in_features=400, out_features=120)
    self.fc2   = nn.Linear(in_features=120, out_features= 84)
    self.fc3   = nn.Linear(in_features= 84, out_features=  2)
  
#  def __init__(self):
#    super(Net, self).__init__()
#    self.conv1 = nn.Conv2d(3, 6, 5)
#    self.pool = nn.MaxPool2d(2, 2)
#    self.conv2 = nn.Conv2d(6, 16, 5)
#    self.fc1 = nn.Linear(16 * 5 * 5, 120)
#    self.fc2 = nn.Linear(120, 84)
#    self.fc3 = nn.Linear(84, 2)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = self.pool(x)
    x = F.relu(self.conv2(x))
    x = self.pool(x)
    x = x.view(-1, 400)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

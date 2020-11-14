import torch.nn as nn

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(kernel_size=5, in_channels=3, out_channels=3)
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.conv2 = nn.Conv2d(kernel_size=5, in_channels=3, out_channels=3)
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.lin1  = nn.Linear(in_features=75, out_features=10)
    self.lin2  = nn.Linear(in_features=10, out_features= 2)

  def forward(self, x):
    x = self.conv1(x)
    x = self.pool1(x)
    x = self.conv2(x)
    x = self.pool2(x)
    x = x.view(-1, 75)
    x = self.lin1(x)
    x = self.lin2(x)
    return x

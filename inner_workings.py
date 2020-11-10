#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as patches
import numpy as np
import torch
import os

from network.myimages import load, imshow
from network.net import Net
from network.train import train
from network.validate import predict, validate

def transform_numpy_to_tensor(np_image):
  if len(np_image.shape) == 2: # black and white image
    np_image = np.array([copy.deepcopy(np_image), copy.deepcopy(np_image), copy.deepcopy(np_image)])
  else:
    np_image = np.transpose(np_image, (2, 0, 1))
  tensor_image = torch.from_numpy(np_image)
  tensor_image = tensor_image/128.0 - 1.0 # [0, 256] -> [-1, 1]
  return tensor_image.unsqueeze(0)

count = 0
def block_save_color(A):
  global count
  A -= torch.min(A)
  A /= torch.max(A)
  fig, ax = plt.subplots(1)
  _, m, n = A.shape
  for i in range(m):
    for j in range(n):
      ax.add_patch(patches.Rectangle((i/m, j/m), 1/m, 1/n, facecolor=(A[0,i,n-j-1].item(), A[1,i,n-j-1].item(), A[2,i,n-j-1].item())))
  ax.set_aspect('equal')
  plt.axis('off')
  plt.draw()
  plt.savefig('inner/%04d.jpg' % count, bbox_inches='tight')
  count += 1

def block_save(A):
  global count
  A -= torch.min(A)
  A /= torch.max(A)
  fig, ax = plt.subplots(1)
  m, n = A.shape
  for i in range(m):
    for j in range(n):
      ax.add_patch(patches.Rectangle((i/m, j/m), 1/m, 1/n, facecolor=(A[i,n-j-1].item(), A[i,n-j-1].item(), A[i,n-j-1].item())))
  ax.set_aspect('equal')
  plt.axis('off')
  plt.draw()
  plt.savefig('inner/%04d.jpg' % count, bbox_inches='tight')
  count += 1

torch.manual_seed(0)
np.random.seed(0)

classes = ('franz', 'nina')
training_set, validation_set = load(classes)
net = Net()
train(net, training_set)
accuracy = validate(net, validation_set)


np_image = plt.imread('images/franz/000000.jpg')
x = transform_numpy_to_tensor(np_image)
block_save_color(x[0])

x = net.conv1(x)
for channel in x[0]:
  block_save(channel)

x = net.pool1(x)
for channel in x[0]:
  block_save(channel)

x = net.conv2(x)
for channel in x[0]:
  block_save(channel)

x = net.pool2(x)
for channel in x[0]:
  block_save(channel)

x = x.view(-1, 75)

x = net.lin1(x)
print(x)

x = net.lin2(x)
print(x)

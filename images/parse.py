#!/usr/bin/env python3

# Run with e.g. ./parse.py franz

import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

def prepare_folder(category):
  os.system("rm -rf %s" % category)
  os.system("mkdir %s" % category)

def create_images_handwriting(category):
  big_image = plt.imread('%s.jpg' % category)
  
  nx = 36 #rows
  ny = 41 #columns
  
  count = 0
  for i in range(nx):
    for j in range(ny):
      small_image = big_image[32*i:32*(i+1), 32*j:32*(j+1), :]
      file_path = '%s/%06d' % (category, count)
      plt.imsave(file_path + '.png', small_image)
      os.system('convert %s.png %s.jpg' % (file_path, file_path))
      os.system('rm %s.png' % file_path)
      count += 1

def transform_tensor_to_numpy(image):
  image = image / 2 + 0.5
  np_image = image.numpy()
  return np.transpose(np_image, (1, 2, 0))

def create_images_torchvision(category, idx):
  transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  images_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
  images_loader = torch.utils.data.DataLoader(images_set, batch_size=1, shuffle=False, num_workers=2)
  
  count = 0
  for data in images_loader:
    image, label = data
    
    if label != idx:
      continue
    
    file_path = '%s/%06d' % (category, count)
    plt.imsave(file_path + '.png', transform_tensor_to_numpy(image[0]))
    os.system('convert %s.png %s.jpg' % (file_path, file_path))
    os.system('rm %s.png' % file_path)
    
    count += 1
    if count == 1476:
      break

if __name__ == '__main__':
  category = sys.argv[1]
  if category in ('franz', 'nina', 'robert_scan', 'franz_scan'):
    prepare_folder(category)
    create_images_handwriting(category)
  
  classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
  if category in classes:
    prepare_folder(category)
    create_images_torchvision(category, classes.index(category))
    pass

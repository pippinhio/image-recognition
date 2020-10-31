import copy
import random as rd
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

def transform_numpy_to_tensor(np_image):
  if len(np_image.shape) == 2: # black and white image
    np_image = np.array([copy.deepcopy(np_image), copy.deepcopy(np_image), copy.deepcopy(np_image)])
  else:
    np_image = np.transpose(np_image, (2, 0, 1))
  tensor_image = torch.from_numpy(np_image)
  tensor_image = tensor_image/128.0 - 1.0 # [0, 256] -> [-1, 1]
  return tensor_image.unsqueeze(0)

def load(categories, p=0.1):
  data = []
  for label, category in enumerate(categories):
    directory = os.path.join('images', category)
    for filename in os.listdir(directory):
      filepath = os.path.join(directory, filename)
      np_image = plt.imread(filepath)
      tensor_image = transform_numpy_to_tensor(np_image)
      tensor_label = torch.LongTensor([label])
      data.append((tensor_image, tensor_label))
    
  rd.seed(1)
  rd.shuffle(data)
  i = int((1 - p)*len(data))
  training_set = data[:i]
  validation_set = data[i:]
  return training_set, validation_set
  
def imshow(tensor_image):
  np_image = tensor_image[0].numpy()[0]
  np_image = np_image / 2 + 0.5
  plt.imshow(np.transpose(np_image, (1, 2, 0)))
  plt.show()

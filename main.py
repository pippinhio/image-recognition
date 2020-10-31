#!/usr/bin/env python3

#Run with e.g. ./main.py franz nina

import sys
import warnings

import torch
import numpy as np

from network.myimages import load, imshow
from network.net import Net
from network.train import train
from network.validate import predict, validate

def main():
  categories = sys.argv[1:]
  if len(categories) == 0:
    categories = ["franz", "nina"]
  training_set, validation_set = load(categories)
  net = Net()
  train(net, training_set)
  validate(net, validation_set)

if __name__ == '__main__':
  warnings.filterwarnings("ignore")
  
  torch.manual_seed(0)
  np.random.seed(0)
  
#  import code
#  code.interact(local=locals())
  main()



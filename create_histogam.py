#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import torch

from network.myimages import load, imshow
from network.net import Net
from network.train import train
from network.validate import predict, validate

torch.manual_seed(0)
np.random.seed(0)

classes = ('franz', 'nina')
training_set, validation_set = load(classes)

learning_rates = (0.0001, 0.001, 0.0015)
momentums = (0.5, 0.9, 0.95)
#fig, axs = plt.subplots(len(learning_rates), len(momentums))

for i, lr in enumerate(learning_rates):
  for j, momentum in enumerate(momentums):
    L = []
    for step in range(100):
      print("lr=%f momentum=%f step=%d" % (lr, momentum, step))
      net = Net()
      train(net, training_set, lr, momentum)
      accuracy = validate(net, validation_set)
      L.append(100*accuracy)
#    axs[i, j].hist(L, bins=list(range(0,101,5)), edgecolor='black', linewidth=1.0)
#    axs[i, j].set_ylim([0, 10])

    plt.hist(L, bins=list(range(0,101,5)), edgecolor='black', linewidth=1.0)
    ax = plt.gca()
    ax.set_ylim([0, 100])
    file_path = 'histogram/lr%s_momentum%s' % (str(lr).replace('.','_'), str(momentum).replace('.','_'))
    plt.savefig(file_path + '.jpg')
    plt.clf()

#plt.savefig('histogram/all.jpg')
#plt.clf()

#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as patches
import numpy as np
import torch

from network.myimages import load
from network.net import Net
from network.train import train
from network.validate import validate

torch.manual_seed(0)
np.random.seed(0)

categories = ('franz', 'nina')
training_set, validation_set = load(categories)

learning_rates = (0.0001, 0.001, 0.0015)
momentums = (0.5, 0.9, 0.95)

for i, lr in enumerate(learning_rates):
  for j, momentum in enumerate(momentums):
    L = []
    for step in range(100):
      print("lr=%f momentum=%f step=%d" % (lr, momentum, step))
      net = Net()
      train(net, training_set, lr, momentum)
      accuracy = validate(net, validation_set)
      L.append(accuracy)

    plt.hist(L, bins=[x/100 for x in range(0,101,5)], edgecolor='black', linewidth=1.0)
    ax = plt.gca()
    ax.set_ylim([0, 100])
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1))
    base_path = 'presentation/histogramsOfOptimizerPerformance/'
    file_path = '%s/lr%s_momentum%s' % (base_path, str(lr).replace('.','_'), str(momentum).replace('.','_'))
    plt.savefig(file_path + '.jpg')

    ax.add_patch(patches.Rectangle((0.0, 0.0), 0.6, 100.0, facecolor='red', alpha=0.1))
    ax.add_patch(patches.Rectangle((0.6, 0.0), 0.4, 100.0, facecolor='green', alpha=0.1))
    plt.savefig(file_path + '_rect' + '.jpg')
    plt.clf()

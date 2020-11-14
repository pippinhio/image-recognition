#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import copy
import random as rd

new_image = np.ones((32*36, 32*(41 + 5 + 41 + 30 + 21),3))

for jdx, directory in enumerate(('franz', 'nina')):
  count = 0
  for i in range(36):
    for j in range(41):
      small_image = plt.imread('../../images/%s/%06d.jpg' % (directory, count))
      if len(small_image.shape) == 2: # black and white image
        small_image = np.array([copy.deepcopy(small_image), copy.deepcopy(small_image), copy.deepcopy(small_image)])
        small_image = np.transpose(small_image, (1, 2, 0))
      small_image = small_image / 256
      offset = jdx*32*(41 + 5)
      new_image[32*i:32*(i+1), offset+32*j:offset+32*(j+1), :] = small_image
      count += 1

plt.imsave('full.png', new_image)

L = []
for jdx in range(2):
  knockout = rd.sample(list(range(36*41)), 296//2)
  for k in knockout:
    i = k // 41
    j = k % 41
    offset = jdx*32*(41 + 5)
    small_image = copy.deepcopy(new_image[32*i:32*(i+1), offset+32*j:offset+32*(j+1), :])
    L.append(small_image)
    new_image[32*i:32*(i+1), offset+32*j:offset+32*(j+1), :] = np.ones((32, 32, 3))

for k, small_image in enumerate(L):
  i = 10 + k //21
  j = 41 + 5 + 41 + 30 + k % 21
  new_image[32*i:32*(i+1), 32*j:32*(j+1), :] = small_image
plt.imsave('split.png', new_image)

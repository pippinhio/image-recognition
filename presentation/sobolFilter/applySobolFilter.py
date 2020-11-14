#!/usr/bin/env python3

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

def color_to_blackwhite(image):
  return np.sum(image, axis=2)/3

def blackwhite_to_color(image):
  m, n = image.shape
  image_color = np.zeros((m, n, 3))
  for i in range(m):
    for j in range(n):
      image_color[i, j, :] = np.array([image[i, j], image[i, j], image[i, j]])
  return image_color

def im_save(filename, image_bw):
  plt.imsave(filename + '.png', blackwhite_to_color(image_bw))
  os.system('convert %s.png %s.jpg' % (filename, filename))
  os.system('rm %s.png' % filename)


image_in = plt.imread('raw.jpg')
image_bw = color_to_blackwhite(image_in)/(256)
nx, ny = image_bw.shape
im_save('bw', image_bw)

W = np.array([
  [ 1,  2,  1],
  [ 0,  0,  0],
  [-1, -2, -1],
])

wx, wy = W.shape
image_conv = np.zeros((nx - wx + 1, ny - wy + 1))
for i, j in np.ndindex(image_conv.shape):
  image_conv[i, j] = np.sum(np.multiply(image_bw[i:i+wx, j:j+wy], W)) #/ np.sum(W)
im_save('sobol', (image_conv+5)/10)

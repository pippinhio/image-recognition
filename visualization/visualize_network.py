#!/usr/bin/env python3

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits import mplot3d

extensions = ["jpg", "pdf"]
for folder in extensions:
  if not os.path.exists(folder):
    os.makedirs(folder)
  for item in os.listdir(folder):
    if item.endswith("." + folder):
      os.remove(os.path.join(folder, item))

#------------------------------------------
# Helper functions
#------------------------------------------
count = 0
def save_fig():
  global count
  plt.draw()
  for extension in extensions:
    plt.savefig('%s/%04d.%s' % (extension, count, extension))
  print('saved figure %d' % count)
  count += 1

def delete(L):
  for el in L:
    el.remove()

def set_alpha(A, alpha):
  for el in A:
    plt.setp(el, alpha=alpha)

def set_linestyle(A, linestyle):
  for el in A:
    plt.setp(el, linestyle=linestyle)

# returns n points, equidistant with dx=1/31, centered at 0
def get_linspace(n):
  dx = 1/31
  return np.linspace(-(n-1)/2*dx, (n-1)/2*dx, n)

def set_title(text):
  title = plt.figtext(.5, .9, text, fontsize=50, ha='center')
  return title

# Hack to adjust camera position (elevation and azimuth angle)
def draw_helper(x, y, z):
  helper = ax.scatter([z], [x], [y], color='white', alpha=0.0)
  return helper

#------------------------------------------
# Initialize
#------------------------------------------
fig = plt.figure()
plt.figure(figsize=(20,10))
ax = plt.axes(projection='3d')
ax.set_axis_off()

draw_helper(0.5, 0.5, 1.5)
save_fig()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 1st layer
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
x1 = get_linspace(32)
y1 = get_linspace(32)
z1 = [0.0, 0.1, 0.2]

title = set_title('Input Layer')

image = plt.imread('sample_pic.jpg')/256
image = np.transpose(image, (1, 0, 2))
image = np.flip(image, 1)

def draw_red_plane():
  red_plane = np.empty(shape=(32,32), dtype=object)
  for i in range(32):
    for j in range(32):
      point = ax.scatter([z1[0]], [x1[i]], [y1[j]], color=(image[i, j, 0], 0, 0), s=100)
      red_plane[i, j] = point
  return red_plane

red_plane = draw_red_plane()
save_fig()
set_alpha(red_plane, 0.1)

def draw_green_plane():
  green_plane = np.empty(shape=(32,32), dtype=object)
  for i in range(32):
    for j in range(32):
      point = ax.scatter([z1[1]], [x1[i]], [y1[j]], color=(0, image[i, j, 1], 0), s=100)
      green_plane[i, j] = point
  return green_plane

green_plane = draw_green_plane()
save_fig()
set_alpha(green_plane, 0.1)

def draw_blue_plane():
  blue_plane = np.empty(shape=(32,32), dtype=object)
  for i in range(32):
    for j in range(32):
      point = ax.scatter([z1[2]], [x1[i]], [y1[j]], color=(0, 0, image[i, j, 2]), s=100)
      blue_plane[i, j] = point
  return blue_plane

blue_plane = draw_blue_plane()
save_fig()
set_alpha(blue_plane, 0.1)
title.remove()
save_fig()


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 2nd layer
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
title = set_title('2D Convolution')

x2 = get_linspace(28)
y2 = get_linspace(28)
z2 = np.linspace(1, 1.5, 6)

def draw_line_bundle(x_start_vec, y_start_vec, z_start, x_end, y_end, z_end):
  L = []
  for x_start in x_start_vec:
    for y_start in y_start_vec:
      line, = ax.plot([z_start, z_end], [x_start, x_end], [y_start, y_end], color='black')
      L.append(line)
  return L

def draw_point(x, y, z, color="orange"):
  point = ax.scatter([z], [x], [y], color=color)
  return point

#------------------------------------------
# First neuron
#------------------------------------------
points = []

bundle1 = draw_line_bundle(x1[:5], y1[:5], z1[0], x2[0], y2[0], z2[0])
set_alpha(red_plane[:5,:5], 0.5)
save_fig()

set_alpha(bundle1, 0.3)
set_linestyle(bundle1, 'dotted')
set_alpha(red_plane[:5,:5], 0.2)

bundle2 = draw_line_bundle(x1[:5], y1[:5], z1[1], x2[0], y2[0], z2[0])
set_alpha(green_plane[:5,:5], 0.5)
save_fig()


set_alpha(bundle2, 0.3)
set_linestyle(bundle2, 'dotted')
set_alpha(green_plane[:5,:5], 0.2)

bundle3 = draw_line_bundle(x1[:5], y1[:5], z1[2], x2[0], y2[0], z2[0])
set_alpha(blue_plane[:5,:5], 0.5)
save_fig()

set_alpha(bundle3, 0.3)
set_linestyle(bundle3, 'dotted')
set_alpha(blue_plane[:5,:5], 0.2)
point = draw_point(x2[0], y2[0], z2[0], color='orange')
points.append(point)
save_fig()

set_alpha(  red_plane[:5,:5], 0.1)
set_alpha(green_plane[:5,:5], 0.1)
set_alpha( blue_plane[:5,:5], 0.1)

delete(bundle1)
delete(bundle2)
delete(bundle3)

#------------------------------------------
# First row of neurons
#------------------------------------------

for i in range(1, 4):
  bundle1 = draw_line_bundle(x1[i:i+5], y1[:5], z1[0], x2[i], y2[0], z2[0])
  bundle2 = draw_line_bundle(x1[i:i+5], y1[:5], z1[1], x2[i], y2[0], z2[0])
  bundle3 = draw_line_bundle(x1[i:i+5], y1[:5], z1[2], x2[i], y2[0], z2[0])
  set_alpha(bundle1, 0.3)
  set_alpha(bundle2, 0.3)
  set_alpha(bundle3, 0.3)
  set_linestyle(bundle1, 'dotted')
  set_linestyle(bundle2, 'dotted')
  set_linestyle(bundle3, 'dotted')
  set_alpha(  red_plane[i:i+5,:5], 0.2)
  set_alpha(green_plane[i:i+5,:5], 0.2)
  set_alpha( blue_plane[i:i+5,:5], 0.2)
  point = draw_point(x2[i], y2[0], z2[0], color='orange')
  points.append(point)
  
  save_fig()
  
  set_alpha(  red_plane[i:i+5,:5], 0.1)
  set_alpha(green_plane[i:i+5,:5], 0.1)
  set_alpha( blue_plane[i:i+5,:5], 0.1)
  
  delete(bundle1)
  delete(bundle2)
  delete(bundle3)

for i in range(4, 28):
  point = draw_point(x2[i], y2[0], z2[0], color='orange')
  points.append(point)
save_fig()


#------------------------------------------
# Second row of neurons
#------------------------------------------

for i in range(3):
  bundle1 = draw_line_bundle(x1[i:i+5], y1[1:6], z1[0], x2[i], y2[1], z2[0])
  bundle2 = draw_line_bundle(x1[i:i+5], y1[1:6], z1[1], x2[i], y2[1], z2[0])
  bundle3 = draw_line_bundle(x1[i:i+5], y1[1:6], z1[2], x2[i], y2[1], z2[0])
  set_alpha(bundle1, 0.3)
  set_alpha(bundle2, 0.3)
  set_alpha(bundle3, 0.3)
  set_linestyle(bundle1, 'dotted')
  set_linestyle(bundle2, 'dotted')
  set_linestyle(bundle3, 'dotted')
  set_alpha(  red_plane[i:i+5,1:6], 0.2)
  set_alpha(green_plane[i:i+5,1:6], 0.2)
  set_alpha( blue_plane[i:i+5,1:6], 0.2)
  point = draw_point(x2[i], y2[1], z2[0], color='orange')
  points.append(point)
  
  save_fig()
  
  set_alpha(  red_plane[i:i+5,1:6], 0.1)
  set_alpha(green_plane[i:i+5,1:6], 0.1)
  set_alpha( blue_plane[i:i+5,1:6], 0.1)
  
  delete(bundle1)
  delete(bundle2)
  delete(bundle3)

for i in range(3, 28):
  point = draw_point(x2[i], y2[1], z2[0], color='orange')
  points.append(point)
save_fig()

delete(points)

#------------------------------------------
# First plane
#------------------------------------------
def draw_plane(x_vec, y_vec, z, color='orange'):
  plane = np.empty(shape=(len(x_vec), len(y_vec)), dtype=object)
  for i, x in enumerate(x_vec):
    for j, y in enumerate(y_vec):
      point = ax.scatter([z], [x], [y], color=color)
      plane[i, j] = point
  return plane

planes2 = []
plane = draw_plane(x2, y2, z2[0], 'orange')
planes2.append(plane)
save_fig()

set_alpha(planes2[0], 0.1)
save_fig()

#------------------------------------------
# Second plane
#------------------------------------------
points = []

for i in range(3):
  bundle1 = draw_line_bundle(x1[i:i+5], y1[:5], z1[0], x2[i], y2[0], z2[1])
  bundle2 = draw_line_bundle(x1[i:i+5], y1[:5], z1[1], x2[i], y2[0], z2[1])
  bundle3 = draw_line_bundle(x1[i:i+5], y1[:5], z1[2], x2[i], y2[0], z2[1])
  set_alpha(bundle1, 0.3)
  set_alpha(bundle2, 0.3)
  set_alpha(bundle3, 0.3)
  set_linestyle(bundle1, 'dotted')
  set_linestyle(bundle2, 'dotted')
  set_linestyle(bundle3, 'dotted')
  set_alpha(  red_plane[i:i+5,:5], 0.2)
  set_alpha(green_plane[i:i+5,:5], 0.2)
  set_alpha( blue_plane[i:i+5,:5], 0.2)
  point = draw_point(x2[i], y2[0], z2[1], color='orange')
  points.append(point)
  
  save_fig()
  
  set_alpha(  red_plane[i:i+5,:5], 0.1)
  set_alpha(green_plane[i:i+5,:5], 0.1)
  set_alpha( blue_plane[i:i+5,:5], 0.1)
  
  delete(bundle1)
  delete(bundle2)
  delete(bundle3)

delete(points)

for k in range(1, 6):
  plane = draw_plane(x2, y2, z2[k], 'orange')
  planes2.append(plane)
  save_fig()
  set_alpha(planes2[k], 0.1)
save_fig()
title.remove()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 3nd layer
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
x3 = get_linspace(14)
y3 = get_linspace(14)
z3 = np.linspace(4.5, 5.0, 6)

draw_helper(x3[-1], y3[-1], z3[-1])
save_fig()
title = set_title('2D Max Pooling')

points = []
for i in range(3):
  bundle = draw_line_bundle(x2[2*i:2*i+2], y2[:2], z2[0], x3[i], y3[0], z3[0])
  set_alpha(bundle1, 0.3)
  set_linestyle(bundle1, 'dotted')
  set_alpha(planes2[0][2*i:2*i+2,:2], 1.0)
  point = draw_point(x3[i], y3[0], z3[0], color='red')
  points.append(point)
  save_fig()
  
  set_alpha(planes2[0][2*i:2*i+2,:2], 0.1)
  delete(bundle)
delete(points)

planes3 = []
for k in range(6):
  set_alpha(planes2[k], 1.0)
  plane = draw_plane(x3, y3, z3[k], 'red')
  planes3.append(plane)
  save_fig()
  
  set_alpha(planes2[k], 0.1)
  set_alpha(planes3[k], 0.1)
save_fig()
title.remove()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 4th layer
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
x4 = get_linspace(10)
y4 = get_linspace(10)
z4 = np.linspace(8.5, 10.0, 16)

draw_helper(x4[-1], y4[-1], z4[-1])
save_fig()
title = set_title('2D Convolution')

points = []
for i in range(3):
  bundles = []
  for k in range(6):
    bundle = draw_line_bundle(x3[i:i+5], y3[:5], z3[k], x4[i], y4[0], z4[0])
    bundles.append(bundle)
    set_alpha(bundle, 0.3)
    set_linestyle(bundle, 'dotted')
    
    set_alpha(planes3[k][i:i+5,:5], 0.5)
    point = draw_point(x4[i], y4[0], z4[0], color='orange')
    points.append(point)
  
  save_fig()
  for bundle in bundles:
    delete(bundle)
  for k in range(6):
    set_alpha(planes3[k][i:i+5,:5], 0.1)

delete(points)

planes4 = []
for k in range(16):
  plane = draw_plane(x4, y4, z4[k], 'orange')
  planes4.append(plane)
  if k < 3:
    save_fig()
  set_alpha(planes4[k], 0.1)
save_fig()
title.remove()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 5th layer
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

x5 = get_linspace(5)
y5 = get_linspace(5)
z5 = np.linspace(18.5, 20.0, 16)

draw_helper(x5[-1], y5[-1], z5[-1])
save_fig()
title = set_title('2D Max Pooling')

planes5 = []
for k in range(16):
  set_alpha(planes4[k], 1.0)
  plane = draw_plane(x5, y5, z5[k], 'red')
  planes5.append(plane)
  if k < 3:
    save_fig()
  set_alpha(planes4[k], 0.1)
  set_alpha(planes5[k], 0.1)
save_fig()
title.remove()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 6th
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
x6 = 0
y6 = np.linspace(-0.5, 0.5, 120)
z6 = 40

helper = draw_helper(x6, y6[-1], z6)
save_fig()
helper.remove()
title = set_title('Linear Transformation')

plane6 = draw_plane([x6], y6, z6, 'green')
bundles = []
for y in y6:
  for z in z5:
    bundle = draw_line_bundle(x5, y5, z, x6, y, z6)
    bundles.append(bundle)
    set_alpha(bundle, 0.3)
    set_linestyle(bundle, 'dotted')
save_fig()

for bundle in bundles:
  delete(bundle)

set_alpha(plane6, 0.1)
save_fig()


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 7th
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
x7 = 0
y7 = np.linspace(-0.5*84/120, 0.5*84/120, 84)
z7 = 80

helper = draw_helper(x7, y7[-1], z7)
save_fig()
helper.remove()

plane7 = draw_plane([x7], y7, z7, 'green')
bundles = []
for y in y7:
  bundle = draw_line_bundle([x6], y6, z6, x7, y, z7)
  bundles.append(bundle)
  set_alpha(bundle, 0.3)
  set_linestyle(bundle, 'dotted')
save_fig()

for bundle in bundles:
  delete(bundle)

set_alpha(plane7, 0.1)
save_fig()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 8th
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
x8 = 0
y8 = [-0.2, 0.2]
z8 = 160

helper = draw_helper(x8, y8[-1], z8)
save_fig()
helper.remove()

plane8 = draw_plane([x8], y8, z8, 'green')
bundles = []
for y in y8:
  bundle = draw_line_bundle([x7], y7, z7, x8, y, z8)
  bundles.append(bundle)
  set_alpha(bundle, 0.3)
  set_linestyle(bundle, 'dotted')
save_fig()

for bundle in bundles:
  delete(bundle)

set_alpha(plane8, 1.0)
save_fig()

title.remove()
save_fig()

convolution_patch = mpatches.Patch(color='orange', label='2D Convolution')
maxpool_patch = mpatches.Patch(color='red', label='2D Max Pooling')
linear_patch = mpatches.Patch(color='green', label='Linear Transformation')
plt.legend(handles=[convolution_patch, maxpool_patch, linear_patch], prop={'size': 20})
save_fig()

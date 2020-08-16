from scipy.misc import imsave
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
import cv2
import csv
import os
import random
import pdb
import click

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 60000 samples are too many. Take only 10000 samples from x and y.
x_train = x_train[0:10000]
y_train = y_train[0:10000]
x_test = x_test[0:100]
y_test = y_test[0:100]
x_train = x_train.reshape(-1, 28, 28, 1).astype(np.float32)
x_test = x_test.reshape(-1, 28, 28, 1).astype(np.float32)


def get_rgb(x, y):

	RGB = np.zeros((x.shape[0], 28, 28, 3))
	"""Convert the digits to specific colored-version.
	0: red; 1: blue; 2: green; 3: violet; 4: purple; 5: pink; 
	6: orange; 7: yellow; 8: brown; 9: cornflower blue.""" 
	colors = np.array(
		[[255,   0,   0],
		 [135, 206, 235],
		 [  0, 128,   0],
		 [238, 130, 238],
		 [128,   0, 128],
		 [255, 192, 203],
		 [255, 128,   0],
		 [255, 255,   0],
		 [165,  42,  42],
		 [100, 149, 237]], np.uint8)

	for k in range(x.shape[0]):
		for i in range(28):
			for j in range(28):
				#pdb.set_trace()
				if x[k][i][j] > 0:
					RGB[k][i][j][:] = colors[y[k]]

	return RGB

def get_gray(x):
	"""Convert the digits to 3 channels."""

	gray = np.zeros((x.shape[0], 28, 28, 3))

	for k in range(x.shape[0]):
		for i in range(28):
			for j in range(28):
				gray[k][i][j][:] = x[k][i][j][0]

	return gray

def get_label(x):
	"""Convert the labels to one-hot encodings."""
	label = np.zeros((x.shape[0], 28, 28, 1))
	for k in range(x.shape[0]):
		for i in range(28):
			for j in range(28):
				if x[k][i][j][0] > 0:
					label[k][i][j][:] = 1
				else:
					label[k][i][j][:] = 0

	return label

RGB_train = get_rgb(x_train, y_train)
RGB_test = get_rgb(x_test, y_test)
gray_train = get_gray(x_train)
gray_test = get_gray(x_test)
label_train = get_label(x_train)
label_train = label_train.reshape(-1, 28, 28).astype(np.float32)
label_test = get_label(x_test)
label_test = label_test.reshape(-1, 28, 28).astype(np.float32)

"""Save the gray images, RGB images and the one-hot encoded labels to .png"""

for i in range(x_train.shape[0]):
	imsave('./input/train/gray/G%04d.png' %i, gray_train[i])
	imsave('./input/train/RGB/R%04d.png' %i, RGB_train[i])
	#imsave('./input/train/label/L%04d.png' %i, label_train[i])

for i in range(x_test.shape[0]):
	imsave('./input/test/gray/G%04d.png' %i, gray_test[i])
	imsave('./input/test/RGB/R%04d.png' %i, RGB_test[i])
	#imsave('./input/test/label/L%04d.png' %i, label_test[i])


	








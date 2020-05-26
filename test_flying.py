# ---------------------------------------------------------------------------------------------------------- #
# Author: maups                                                                                              #
# ---------------------------------------------------------------------------------------------------------- #
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import cv2
import random
import logging
from datetime import datetime
import os
import re

# ---------------------------------------------------------------------------------------------------------- #
# Debug                                                                                                      #
# ---------------------------------------------------------------------------------------------------------- #
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s:%(message)s")

# ---------------------------------------------------------------------------------------------------------- #
# List of testing data                                                                                       #
# ---------------------------------------------------------------------------------------------------------- #
path = './images/'
files = [path+f for f in os.listdir(path) if re.match(r'\d\d\d\d-\d\d-\d\d-\d\d-\d\d-\d\d\.png', f)]

# ---------------------------------------------------------------------------------------------------------- #
# Description:                                                                                               #
#         Fully convolutional network architecture for single-scale object detection. Receives images with   #
#         at least NxM RGB pixels (3xNxM torch tensors, N >= 51, M >= 51) and outputs 1x(N-50)x(M-50) torch  #
#         tensors with object probabilities per pixel. For instance, for a 3x51x51 input, the network        #
#         produces a 1x1x1 output.                                                                           #
# ---------------------------------------------------------------------------------------------------------- #
class CONV_RELU_BN_POOL(nn.Module):
	def __init__(self, in_channels, out_channels, conv_kernel_size=(3, 3), conv_stride=(1, 1), pool_kernel_size=(2, 2), pool_stride=(2, 2)):
		super(CONV_RELU_BN_POOL, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=conv_kernel_size, stride=conv_stride, bias=False)
		self.bn = nn.BatchNorm2d(out_channels)
		self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)

	def forward(self, x):
		return self.pool(self.bn(F.relu(self.conv(x))))

class FCN(nn.Module):
	def __init__(self):
		super(FCN, self).__init__()

		self.conv1 = CONV_RELU_BN_POOL(in_channels=3, out_channels=16, conv_kernel_size=(5, 5), conv_stride=(1, 1), pool_kernel_size=(5, 5), pool_stride=(1, 1)) # 3x51x51 -> 16x43x43
		self.conv2 = CONV_RELU_BN_POOL(in_channels=16, out_channels=32, conv_kernel_size=(5, 5), conv_stride=(1, 1), pool_kernel_size=(5, 5), pool_stride=(1, 1)) # 16x43x43 -> 32x35x35
		self.conv3 = CONV_RELU_BN_POOL(in_channels=32, out_channels=64, conv_kernel_size=(5, 5), conv_stride=(1, 1), pool_kernel_size=(5, 5), pool_stride=(1, 1)) # 32x35x35 -> 64x27x27
		self.conv4 = CONV_RELU_BN_POOL(in_channels=64, out_channels=64, conv_kernel_size=(5, 5), conv_stride=(1, 1), pool_kernel_size=(5, 5), pool_stride=(1, 1)) # 64x27x27 -> 64x19x19
		self.conv5 = CONV_RELU_BN_POOL(in_channels=64, out_channels=64, conv_kernel_size=(5, 5), conv_stride=(1, 1), pool_kernel_size=(5, 5), pool_stride=(1, 1)) # 64x19x19 -> 64x11x11
		self.conv6 = nn.Conv2d(64, 1, kernel_size=(11, 11), stride=(1, 1)) # 64x11x11 -> 1x1x1
		self.dropout = nn.Dropout2d(0.2)

	def forward(self, x):
		x = self.conv1(x)
		x = self.dropout(x)
		x = self.conv2(x)
		x = self.dropout(x)
		x = self.conv3(x)
		x = self.dropout(x)
		x = self.conv4(x)
		x = self.dropout(x)
		x = self.conv5(x)
		x = self.dropout(x)
		x = self.conv6(x)
		return torch.sigmoid(x)

# ---------------------------------------------------------------------------------------------------------- #
# Description:                                                                                               #
#         Loads RGB satellite images and classify each 51x51 subwindow into background or parked airplane.   #
# ---------------------------------------------------------------------------------------------------------- #
def process_data(model, bin_threshold, size_threshold):
	# disjoint set union with path compression
	class DSU:
		def __init__(self, n):
			self.parent = [x for x in range(n)]
		def find(self, a):
			if a != self.parent[a]:
				self.parent[a] = self.find(self.parent[a])
			return self.parent[a]
		def join(self, a, b):
			self.parent[self.find(b)] = self.find(a)
		def check(self, a, b):
			return self.find(a) == self.find(b)

	logging.debug("Running data parser.")

	block = 1024
	radius = 25
	size = 51

	count = []
	model.eval()
	for f in files:
		logging.debug("Processing {}".format(f))

		# load satellite image, mask and cloud mask
		img = cv2.imread(f, cv2.IMREAD_COLOR)[:, :, ::-1]/255.0
		mask = cv2.imread(f.replace('.png', '-mask.png'), cv2.IMREAD_GRAYSCALE)/255.0
		cloud_mask = cv2.imread(f.replace('.png', '-cloud.png'), cv2.IMREAD_GRAYSCALE)/255.0

		# divide mask in 9 blocks (3x3 grid) and check which ones are not covered by clouds and have valid pixels
		block_flag = [[True, True, True], [True, True, True], [True, True, True]]

		h_step = [0, int(cloud_mask.shape[0]/3), int(2*cloud_mask.shape[0]/3), cloud_mask.shape[0]]
		w_step = [0, int(cloud_mask.shape[1]/3), int(2*cloud_mask.shape[1]/3), cloud_mask.shape[1]]
		for h in range(3):
			for w in range(3):
				if np.mean(cloud_mask[h_step[h]:h_step[h+1], w_step[w]:w_step[w+1]]) > 0.3:
					block_flag[h][w] = False

		h_step = [0, int(mask.shape[0]/3), int(2*mask.shape[0]/3), mask.shape[0]]
		w_step = [0, int(mask.shape[1]/3), int(2*mask.shape[1]/3), mask.shape[1]]
		for h in range(3):
			for w in range(3):
				if np.mean(mask[h_step[h]:h_step[h+1], w_step[w]:w_step[w+1]]) < 0.9:
					block_flag[h][w] = False

		# compute mask of detection for satellite image in chunks of 1024x1024 pixels
		full_mask = np.zeros((img.shape[0], img.shape[1], 1), np.float32)
		with torch.no_grad():
			for y in range(0, img.shape[0], block-2*radius):
				if y+size > img.shape[0]:
					break
				for x in range(0, img.shape[1], block-2*radius):
					if x+size > img.shape[1]:
						break
					img_crop = torch.from_numpy(np.transpose(img[y:y+block,x:x+block], (2,0,1))).float().unsqueeze(0).cuda()
					mask_crop = model(img_crop)
					full_mask[y+radius:min(y+block-radius,img.shape[0]-radius), x+radius:min(x+block-radius,img.shape[1]-radius)] = mask_crop[0, :, :, :].cpu().numpy().transpose((1,2,0))

		# convert mask to unsigned char and binarize it
		full_mask = (full_mask*255.0).astype(np.uint8)
		_, full_mask = cv2.threshold(full_mask, bin_threshold, 255, cv2.THRESH_BINARY)

		# find connected components of white pixels
		dsu = DSU(full_mask.shape[0]*full_mask.shape[1])
		k = 0
		for y in range(0, full_mask.shape[0]):
			for x in range(0, full_mask.shape[1]):
				if full_mask[y,x] > 0:
					if y > 0 and full_mask[y-1,x] > 0:
						dsu.join(k, k-full_mask.shape[1])
					if x > 0 and full_mask[y,x-1] > 0:
						dsu.join(k, k-1)
				k += 1

		# compute center of mass for blobs
		blobs = {}
		k = 0
		for y in range(0, full_mask.shape[0]):
			for x in range(0, full_mask.shape[1]):
				if full_mask[y,x] > 0:
					label = dsu.find(k)
					if not label in blobs:
						blobs[label] = [y,x,1]
					else:
						blobs[label][0] += y
						blobs[label][1] += x
						blobs[label][2] += 1
				k += 1
		dets = []
		for key, pts in blobs.items():
			if pts[2] > size_threshold:
				dets.append(((int(pts[0]/pts[2]+0.5), int(pts[1]/pts[2]+0.5)), pts[2]))

		# non-maximum suppression
		dets = sorted(dets, key = lambda x: x[1], reverse = True)
		nms = []
		for c, _ in dets:
			flag = True
			for cc in nms:
				if np.sqrt((c[0]-cc[0])**2 + (c[1]-cc[1])**2) <= radius/2:
					flag = False
					break
			if flag:
				nms.append(c)
		dets = nms

		# count detections per block
		block_count = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
		h_step = img.shape[0]//3 + (0 if img.shape[0]%3 == 0 else 1)
		w_step = img.shape[0]//3 + (0 if img.shape[0]%3 == 0 else 1)
		for c in dets:
			block_count[c[0]//h_step][c[1]//w_step] += 1

		block_dets = []
		for h in range(3):
			for w in range(3):
				if block_flag[h][w]:
					block_dets.append(block_count[h][w])
				else:
					block_dets.append(None)					

		count.append((f.split('/')[-1][:10], block_dets))

	logging.debug("Done!\n")

	return count

# ---------------------------------------------------------------------------------------------------------- #
# Description:                                                                                               #
#         Smooth and plot count signal.                                                                      #
# ---------------------------------------------------------------------------------------------------------- #
def temporal_signal(count, start):
	count = sorted(count)

	# convert dates to number of days since start
	signal = []
	dstart = datetime.strptime(start, "%Y-%m-%d")
	for d, c in count:
		dcurr = datetime.strptime(d, "%Y-%m-%d")
		signal.append((abs((dcurr - dstart).days), c))

	# smooth signal
	out = []
	i = 0
	j = 0
	win_size = 60
	win_std = 30
	keep_rate = 0.95
	cc = 0.0
	for t in range(signal[-1][0]+1):
		while i < len(signal) and signal[i][0] < t-win_size:
			i += 1
		while j < len(signal) and signal[j][0] <= t:
			j += 1

		block_sc = 0.0
		for b in range(9):
			sc = 0.0
			sw = 0.0
			for k in range(i,j):
				if signal[k][1][b] is not None:
					w = np.exp(-(signal[k][0]-t)**2.0/(2.0*win_std*win_std))
					sc += signal[k][1][b]*w
					sw += w
			if sw != 0:
				block_sc += sc/sw

		out.append([t, block_sc])

	# save resulting signal
	np.savetxt('flying.csv', np.asarray(out), delimiter=',', fmt='%f')

# ---------------------------------------------------------------------------------------------------------- #
# MAIN                                                                                                       #
# ---------------------------------------------------------------------------------------------------------- #

model = FCN().cuda()
model.load_state_dict(torch.load('./models/flying.pytorch'))
logging.debug("Model created!\n")

# run inference for all test images
count = process_data(model, 127, 9)

# process inference results to create a temporal signal
temporal_signal(count, '2018-01-01')


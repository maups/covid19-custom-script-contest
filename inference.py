# ---------------------------------------------------------------------------------------------------------- #
# Author: maups                                                                                              #
# ---------------------------------------------------------------------------------------------------------- #
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import random
import json
import sys
import cv2
import os
import re

# ---------------------------------------------------------------------------------------------------------- #
# Configuration                                                                                              #
# ---------------------------------------------------------------------------------------------------------- #
path_images = '/home/DATASETS/D2/S2AAD/'
output_folder = './log/'

# ---------------------------------------------------------------------------------------------------------- #
# Fully Convolutional Network architecture                                                                   #
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

		self.conv1 = CONV_RELU_BN_POOL(in_channels=3, out_channels=16, conv_kernel_size=(5, 5), conv_stride=(1, 1), pool_kernel_size=(5, 5), pool_stride=(1, 1))
		self.conv2 = CONV_RELU_BN_POOL(in_channels=16, out_channels=32, conv_kernel_size=(5, 5), conv_stride=(1, 1), pool_kernel_size=(5, 5), pool_stride=(1, 1))
		self.conv3 = CONV_RELU_BN_POOL(in_channels=32, out_channels=64, conv_kernel_size=(5, 5), conv_stride=(1, 1), pool_kernel_size=(5, 5), pool_stride=(1, 1))
		self.conv4 = CONV_RELU_BN_POOL(in_channels=64, out_channels=64, conv_kernel_size=(5, 5), conv_stride=(1, 1), pool_kernel_size=(5, 5), pool_stride=(1, 1))
		self.conv5 = CONV_RELU_BN_POOL(in_channels=64, out_channels=64, conv_kernel_size=(5, 5), conv_stride=(1, 1), pool_kernel_size=(5, 5), pool_stride=(1, 1))
		self.conv6 = nn.Conv2d(64, 1, kernel_size=(11, 11), stride=(1, 1))
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
# Non-maximal suppression                                                                                    #
# ---------------------------------------------------------------------------------------------------------- #
class NMS(nn.Module):
	def __init__(self):
		super(NMS, self).__init__()
		self.maxpool = nn.MaxPool2d(kernel_size=(51, 51), stride=(1, 1), padding=25)

	def forward(self, x):
		y = self.maxpool(x)
		return torch.nonzero(torch.eq(x,y) & (y > 0.5), as_tuple=False)

# ---------------------------------------------------------------------------------------------------------- #
# Detect flying airplanes in one satellite image                                                             #
# ---------------------------------------------------------------------------------------------------------- #
def process_image(det_model, nms_model, img):
	block = 512
	radius = 25
	size = 51

	# compute mask of detection for satellite image in chunks of [block x block] pixels
	full_mask = np.zeros((img.shape[0], img.shape[1], 1), np.float32)
	with torch.no_grad():
		for y in range(0, img.shape[0], block-2*radius):
			if y+size > img.shape[0]:
				break
			for x in range(0, img.shape[1], block-2*radius):
				if x+size > img.shape[1]:
					break
				if np.sum(img[y:y+block,x:x+block]) == 0:
					continue
				img_crop = torch.from_numpy(np.transpose(img[y:y+block,x:x+block], (2,0,1))).float().unsqueeze(0).cuda()
				mask_crop = det_model(img_crop)
				full_mask[y+radius:min(y+block-radius,img.shape[0]-radius), x+radius:min(x+block-radius,img.shape[1]-radius)] = mask_crop[0, :, :, :].cpu().numpy().transpose((1,2,0))

	det_mask = torch.from_numpy(np.transpose(full_mask, (2,0,1))).float().unsqueeze(0).cuda()
	dets = nms_model(det_mask).cpu().numpy()[:,2:].tolist()

	return dets

# ---------------------------------------------------------------------------------------------------------- #
# MAIN                                                                                                       #
# ---------------------------------------------------------------------------------------------------------- #
det_model = FCN().cuda()
det_model.load_state_dict(torch.load('./models/flying41.pytorch'))
det_model.eval()

nms_model = NMS().cuda()
nms_model.eval()

# ---------------------------------------------------------------------------------------------------------- #
# Count airplanes for all testing images and save count per AOI block                                        #
# ---------------------------------------------------------------------------------------------------------- #
airports = sorted(os.listdir(path_images))
for airport in airports:
	files = [f for f in sorted(os.listdir(os.path.join(path_images, airport))) if re.match(r'\d\d\d\d-\d\d-\d\d-\d\d-\d\d-\d\d_image\.png', f)]

	log = {}
	for f in files:
		print(airport, f)

		# load satellite image
		img = cv2.imread(os.path.join(path_images, airport, f), cv2.IMREAD_COLOR)[:, :, ::-1]/255.0

		# detect airplanes
		dets = process_image(det_model, nms_model, img)

		block_height = img.shape[0]//7
		block_width = img.shape[1]//7
		timestamp = f.split('_')[0]

		valid = []
		flag = False

		for i in range(0,img.shape[0],block_height):
			for j in range(0,img.shape[1],block_width):
				if np.sum(img[i:i+block_height,j:j+block_width]) > 0:
					valid.append(0)
				else:
					valid.append(None)

		for c in dets:
			pos = (c[0]//block_height)*7 + c[1]//block_width
			if valid[pos] is not None:
				valid[pos] += 1

		for i in range(49):
			if valid[i] is not None and valid[i] > 5:
				valid[i] = None

		if valid.count(None) < len(valid):
			log[timestamp] = valid

	json.dump(log, open(os.path.join(output_folder, airport+'.log'), 'w'))


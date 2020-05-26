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
import os
import re

# ---------------------------------------------------------------------------------------------------------- #
# Debug                                                                                                      #
# ---------------------------------------------------------------------------------------------------------- #
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s:%(message)s")

# ---------------------------------------------------------------------------------------------------------- #
# List of training data                                                                                      #
# ---------------------------------------------------------------------------------------------------------- #
path_images = './images/'
path_annotations = './annotations/'
files = [f for f in os.listdir(path_annotations) if re.match(r'\d\d\d\d-\d\d-\d\d-\d\d-\d\d-\d\d-parked\.txt', f)]
files = [(path_images+f.replace('parked.txt', 'roi.png'), path_annotations+f) for f in files]

# ---------------------------------------------------------------------------------------------------------- #
# Description:                                                                                               #
#         Fully convolutional network architecture for single-scale object detection. Receives images with   #
#         at least NxM RGB pixels (3xNxM torch tensors, N >= 17, M >= 17) and outputs 1x(N-16)x(M-16) torch  #
#         tensors with object probabilities per pixel. For instance, for a 3x17x17 input, the network        #
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

		self.conv1 = CONV_RELU_BN_POOL(in_channels=3, out_channels=64, conv_kernel_size=(3, 3), conv_stride=(1, 1), pool_kernel_size=(3, 3), pool_stride=(1, 1)) # 3x17x17 -> 64x13x13
		self.conv2 = CONV_RELU_BN_POOL(in_channels=64, out_channels=128, conv_kernel_size=(3, 3), conv_stride=(1, 1), pool_kernel_size=(3, 3), pool_stride=(1, 1)) # 64x13x13 -> 128x9x9
		self.conv3 = CONV_RELU_BN_POOL(in_channels=128, out_channels=256, conv_kernel_size=(3, 3), conv_stride=(1, 1), pool_kernel_size=(3, 3), pool_stride=(1, 1)) # 128x9x9 -> 256x5x5
		self.conv4 = nn.Conv2d(256, 1, kernel_size=(5, 5), stride=(1, 1)) # 256x5x5 -> 1x1x1
		self.dropout = nn.Dropout2d(0.2)

	def forward(self, x):
		x = self.conv1(x)
		x = self.dropout(x)
		x = self.conv2(x)
		x = self.dropout(x)
		x = self.conv3(x)
		x = self.dropout(x)
		x = self.conv4(x)
		return torch.sigmoid(x)

# ---------------------------------------------------------------------------------------------------------- #
# Description:                                                                                               #
#         Loads RGB satellite images and airplane annotations given and crops 17x17 positive and negative    #
#         samples for training. Training is not carried out on entire images to avoid memory issues and lack #
#         of generalization. It returns an augmented set of positive samples and a set of negative samples   #
#         with size not greater than twice the size of the positive set.                                     #
# ---------------------------------------------------------------------------------------------------------- #
def load_data():
	logging.debug("Running data loader.")

	positives = []
	negatives = []
	for img_name, ann_name in files:
		logging.debug("Processing {}".format(img_name))

		# load airplane annotations
		ann_list = []
		with open(ann_name, 'r') as f:
			n = int(next(f))
			for line in f:
				row, col, rad = [int(x) for x in line.split()]
				ann_list.append(((row,col),rad))

		# load satellite image
		img = cv2.imread(img_name, cv2.IMREAD_COLOR)[:, :, ::-1]/255.0

		# crop samples from input image
		# ---------------------
		# |x        x        x| -|
		# |                   |  |
		# |                   |  | r
		# |       * * *       |  |
		# |x      * @ *      x| -| -|
		# |       * * *       |    -| step
		# |                   |
		# |                   |
		# |x        x        x|
		# ---------------------
		# @ -> annotated point
		# * -> augmented positive coordinates
		# x -> negative coordinates
		size = 8
		step = 1
		for cc, r in ann_list:
			for x in range(-1, 2):
				for y in range(-1, 2):
					# positive samples
					c = (cc[0]+y*step, cc[1]+x*step)
					if c[0]-size >= 0 and c[0]+size < img.shape[0] and c[1]-size >= 0 and c[1]+size < img.shape[1]:
						positives.append(img[c[0]-size:c[0]+size+1, c[1]-size:c[1]+size+1].copy())
					# negative samples
					if x != 0 or y != 0:
						c = (cc[0]+y*r, cc[1]+x*r)
						if c[0]-size >= 0 and c[0]+size < img.shape[0] and c[1]-size >= 0 and c[1]+size < img.shape[1]:
							negatives.append(img[c[0]-size:c[0]+size+1, c[1]-size:c[1]+size+1].copy())

		# extra negative samples sampled randomly over the entire image
		for j in range(100):
			c = (np.random.randint(img.shape[0]), np.random.randint(img.shape[1]))
			if c[0]-size >= 0 and c[0]+size < img.shape[0] and c[1]-size >= 0 and c[1]+size < img.shape[1]:
				flag = True
				for cc, r in ann_list:
					if abs(cc[0]-c[0]) <= size or abs(cc[1]-c[1]) <= size:
						flag = False
						break
				# discard if sampled point is too close to an annotated point or if it falls in a blank image region
				if flag and np.sum(img[c[0]-size:c[0]+size+1, c[1]-size:c[1]+size+1]) > 0:
					negatives.append(img[c[0]-size:c[0]+size+1, c[1]-size:c[1]+size+1].copy())

	# keep a 1:2 ratio limit between positive and negtive samples
	if len(negatives) > 2*len(positives):
		negatives = random.sample(negatives, 2*len(positives))

	logging.debug("Done!\n")

	return np.asarray(positives, dtype=np.float32), np.asarray(negatives, dtype=np.float32)

# ---------------------------------------------------------------------------------------------------------- #
# Description:                                                                                               #
#         Loads RGB satellite images and airplane annotations and crops 17x17 negative samples around false  #
#         positive blobs in the detection response from a given model. It returns the detection rate, false  #
#         detection rate, and crops for all false positives.                                                 #
# ---------------------------------------------------------------------------------------------------------- #
def update_data(model):
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

	logging.debug("Running data updater.")

	block = 512
	radius = 8
	size = 17

	model.eval()
	total_annotated = 0
	total_correct = 0
	total_dets = 0
	negatives = []
	for img_name, ann_name in files:
		logging.debug("Processing {}".format(img_name))

		# load airplane annotations
		ann_list = []
		with open(ann_name, 'r') as f:
			n = int(next(f))
			for line in f:
				row, col, rad = [int(x) for x in line.split()]
				ann_list.append(((row,col),rad))

		# load satellite image
		img = cv2.imread(img_name, cv2.IMREAD_COLOR)[:, :, ::-1]/255.0

		# compute mask of detection for satellite image in chunks of [block x block] pixels
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
		_, full_mask = cv2.threshold(full_mask, 127, 255, cv2.THRESH_BINARY)

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

		# find true positives
		flag_crop = [True]*len(dets)
		correct = 0
		for c, r in ann_list:
			# find closest detection to annotation
			d_id = -1
			d_min = 123456.0
			for j in range(len(dets)):
				dist = np.sqrt((c[0]-dets[j][0])**2 + (c[1]-dets[j][1])**2)
				if dist < d_min:
					d_min = dist
					d_id = j

			# if close enough, mark it as true positive
			if d_min <= radius:
				if flag_crop[d_id]:
					flag_crop[d_id] = False
					correct += 1

		total_annotated += len(ann_list)
		total_correct += correct
		total_dets += len(dets)

		logging.debug("Detected {} out of {} airplane(s). Got {} false positive(s).".format(correct, len(ann_list), len(dets)-correct))

		# crop false positives to be added to the negative set
		for j in range(len(dets)):
			if flag_crop[j]:
				negatives.append(img[dets[j][0]-radius:dets[j][0]+radius+1, dets[j][1]-radius:dets[j][1]+radius+1].copy())

	logging.debug("Done!\n")

	return total_correct/total_annotated, (total_dets-total_correct)/total_dets, np.asarray(negatives, dtype=np.float32)

# ---------------------------------------------------------------------------------------------------------- #
# Description:                                                                                               #
#         Computes results per class for a binary classification problem.                                    #
#         TODO: use batch_size to speed up this function.                                                    #
# ---------------------------------------------------------------------------------------------------------- #
def eval(model, imgs, labels):
	model.eval()
	correct = [0, 0]
	total = [0, 0]
	with torch.no_grad():
		for i in range(len(imgs)):
			batch_data = torch.from_numpy(np.transpose(imgs.take([i], axis=0), (0, 3, 1, 2))).cuda()
			batch_target = torch.tensor(labels.take([i], axis=0), dtype=torch.long).cuda()

			batch_pred = model(batch_data)

			label = (batch_pred > 0.5).int()
			if labels[i] == 0:
				correct[0] += label.eq(batch_target.view_as(label)).sum().item()
				total[0] += 1
			else:
				correct[1] += label.eq(batch_target.view_as(label)).sum().item()
				total[1] += 1
	model.train()

	return correct[0]/total[0], correct[1]/total[1]

# ---------------------------------------------------------------------------------------------------------- #
# Description:                                                                                               #
#         Optimizes the @model using a given @optimizer for a given number of @iterations. Each iteration    #
#         consists in getting a random sample from @imgs and @labels with @batch_size elements.              #
# ---------------------------------------------------------------------------------------------------------- #
def train(model, optimizer, imgs, labels, batch_size, iterations):
	logging.debug("Starting training.")

	neg_acc, pos_acc = eval(model, imgs, labels)
	logging.debug("Accuracy before training: POS {} / NEG {}".format(pos_acc, neg_acc))

	model.train()
	for i in range(iterations):
		# randomly select batch_size images from the training set
		batch = np.random.permutation(len(imgs))[:batch_size]

		# data augmentation: horizontal and vertical flips, and 90-degrees rotations
		np_batch_data = imgs.take(batch, axis=0)
		if np.random.randint(2) == 1:
			np_batch_data = np.flip(np_batch_data, 1).copy()
		if np.random.randint(2) == 1:
			np_batch_data = np.flip(np_batch_data, 2).copy()
		if np.random.randint(2) == 1:
			np_batch_data = np.transpose(np_batch_data, (0, 2, 1, 3))

		# create torch tensors for images and labels of the current batch
		batch_data = torch.from_numpy(np.transpose(np_batch_data, (0, 3, 1, 2))).cuda()
		batch_target = torch.tensor(labels.take(batch, axis=0), dtype=torch.long).float().cuda()

		# run one optimization step
		model.zero_grad()
		batch_pred = model(batch_data).squeeze()
		loss = F.binary_cross_entropy(batch_pred, batch_target)
		loss.backward()
		optimizer.step()

		# check performance every 100 iterations
		if i%100 == 99:
			neg_acc, pos_acc = eval(model, imgs, labels)
			logging.debug("Accuracy after {} training iterations: POS {} / NEG {}".format(i+1, pos_acc, neg_acc))

	neg_acc, pos_acc = eval(model, imgs, labels)
	logging.debug("Accuracy after training: POS {} / NEG {}".format(pos_acc, neg_acc))

	logging.debug("Done!\n")

# ---------------------------------------------------------------------------------------------------------- #
# MAIN                                                                                                       #
# ---------------------------------------------------------------------------------------------------------- #

model = FCN().cuda()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
logging.debug("Model created!\n")

# load training data
train_pos, train_neg = load_data()
train_imgs = np.concatenate((train_pos,train_neg), axis=0)
train_labels = np.asarray([1]*len(train_pos) + [0]*len(train_neg), dtype=np.int32)
logging.debug("Training data loaded: {} positive and {} negative samples\n".format(len(train_pos), len(train_neg)))

# run 1st stage of training
train(model, optimizer, train_imgs, train_labels, 256, 3000)

# get current performance and false positive samples
dr, fdr, false_positives = update_data(model)
logging.debug("Current performance: DR {} / FDR {}\n".format(dr, fdr))

# following stages
for i in range(2,7):
	# keep a 1:2 ratio limit between positive and negative samples
	batch1 = train_neg
	batch2 = false_positives
	if len(train_neg) >= len(train_pos) and len(false_positives) >= len(train_pos):
		batch1 = train_neg.take(np.random.permutation(len(train_neg))[:len(train_pos)], axis=0)
		batch2 = false_positives.take(np.random.permutation(len(false_positives))[:len(train_pos)], axis=0)
	elif len(train_neg) >= len(train_pos):
		batch1 = train_neg.take(np.random.permutation(len(train_neg))[:2*len(train_pos)-len(false_positives)], axis=0)
	elif len(false_positives) >= len(train_pos):
		batch2 = false_positives.take(np.random.permutation(len(false_positives))[:2*len(train_pos)-len(train_neg)], axis=0)
	train_neg = np.concatenate((batch1,batch2), axis=0)

	# update training data
	train_imgs = np.concatenate((train_pos,train_neg), axis=0)
	train_labels = np.asarray([1]*len(train_pos) + [0]*len(train_neg), dtype=np.int32)
	logging.debug("Training data rearranged: {} positive and {} negative samples\n".format(len(train_pos), len(train_neg)))

	# run training stage
	train(model, optimizer, train_imgs, train_labels, 256, 3000)

	# get current performance and false positive samples
	dr, fdr, false_positives = update_data(model)
	logging.debug("Current performance: DR {} / FDR {}\n".format(dr, fdr))

# save model
torch.save(model.state_dict(), './models/parked.pytorch')


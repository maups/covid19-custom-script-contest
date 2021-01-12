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
import sys
import cv2
import os
import re

# ---------------------------------------------------------------------------------------------------------- #
# Configuration                                                                                              #
# ---------------------------------------------------------------------------------------------------------- #
path_images = '/home/DATASETS/D2/S2AAD/'
path_annotations = './annotations/'
output_folder = './models/'

# ---------------------------------------------------------------------------------------------------------- #
# List of training images                                                                                    #
# ---------------------------------------------------------------------------------------------------------- #
files = []
airports = sorted(os.listdir(path_annotations))
for airport in airports:
	fs = [f for f in sorted(os.listdir(os.path.join(path_annotations, airport))) if re.match(r'\d\d\d\d-\d\d-\d\d-\d\d-\d\d-\d\d\.txt', f)]
	files += [(os.path.join(path_images, airport, f.replace('.txt', '_image.png')), os.path.join(path_annotations, airport, f)) for f in fs]

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
# Initialize set of image patches                                                                            #
# ---------------------------------------------------------------------------------------------------------- #
def load_data():
	positives = []
	negatives = []
	with tqdm(total=len(files), file=sys.stdout) as pbar:
		pbar.set_description('Parsing training data')
		for img_name, ann_name in files:
			# load airplane annotations
			ann_list = []
			with open(ann_name, 'r') as f:
				for line in f:
					row, col = [int(x) for x in line.split()]
					ann_list.append((row,col))

			# load satellite image
			img = cv2.imread(img_name, cv2.IMREAD_COLOR)[:, :, ::-1]/255.0

			# crop samples from input image
			size = 25
			step = 3
			for cc in ann_list:
				for x in range(-1, 2):
					for y in range(-1, 2):
						# positive samples
						c = (cc[0]+y*step, cc[1]+x*step)
						if c[0]-size >= 0 and c[0]+size < img.shape[0] and c[1]-size >= 0 and c[1]+size < img.shape[1]:
							positives.append(img[c[0]-size:c[0]+size+1, c[1]-size:c[1]+size+1].copy())
						# negative samples
						if x != 0 or y != 0:
							c = (cc[0]+y*size, cc[1]+x*size)
							if c[0]-size >= 0 and c[0]+size < img.shape[0] and c[1]-size >= 0 and c[1]+size < img.shape[1]:
								negatives.append(img[c[0]-size:c[0]+size+1, c[1]-size:c[1]+size+1].copy())

			# extra negative samples sampled randomly over the entire image
			while len(negatives) < 2*len(positives):
				c = (np.random.randint(img.shape[0]), np.random.randint(img.shape[1]))
				if c[0]-size >= 0 and c[0]+size < img.shape[0] and c[1]-size >= 0 and c[1]+size < img.shape[1]:
					flag = True
					for cc in ann_list:
						if abs(cc[0]-c[0]) <= size or abs(cc[1]-c[1]) <= size:
							flag = False
							break
					# discard if sampled point is too close to an annotated point or if it falls in a blank image region
					if flag and np.sum(img[c[0]-size:c[0]+size+1, c[1]-size:c[1]+size+1]) > 0:
						negatives.append(img[c[0]-size:c[0]+size+1, c[1]-size:c[1]+size+1].copy())

			pbar.update(1)

	# keep a 1:2 ratio limit between positive and negtive samples
	if len(negatives) > 2*len(positives):
		negatives = random.sample(negatives, 2*len(positives))

	return np.asarray(positives, dtype=np.float32), np.asarray(negatives, dtype=np.float32)

# ---------------------------------------------------------------------------------------------------------- #
# Update set of image patches and compute detection rate and false discovery rate                            #
# ---------------------------------------------------------------------------------------------------------- #
def update_data(det_model, nms_model):
	block = 512
	radius = 25
	size = 51
	max_falsedet_per_image = 100

	det_model.eval()
	total_annotated = 0
	total_correct = 0
	total_dets = 0
	negatives = []
	with tqdm(total=len(files), file=sys.stdout) as pbar:
		pbar.set_description('Updating training data')

		for img_name, ann_name in files:
			# load airplane annotations
			ann_list = []
			with open(ann_name, 'r') as f:
				for line in f:
					row, col = [int(x) for x in line.split()]
					ann_list.append((row,col))

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
						if np.sum(img[y:y+block,x:x+block]) == 0:
							continue
						img_crop = torch.from_numpy(np.transpose(img[y:y+block,x:x+block], (2,0,1))).float().unsqueeze(0).cuda()
						mask_crop = det_model(img_crop)
						full_mask[y+radius:min(y+block-radius,img.shape[0]-radius), x+radius:min(x+block-radius,img.shape[1]-radius)] = mask_crop[0, :, :, :].cpu().numpy().transpose((1,2,0))

			det_mask = torch.from_numpy(np.transpose(full_mask, (2,0,1))).float().unsqueeze(0).cuda()
			dets = nms_model(det_mask).cpu().numpy()[:,2:].tolist()

			# find true positives
			flag_crop = [True]*len(dets)
			correct = 0
			for c in ann_list:
				# find closest detection to annotation
				d_id = -1
				d_min = 1234567.0
				for j, cc in enumerate(dets):
					dist = np.sqrt((c[0]-cc[0])**2 + (c[1]-cc[1])**2)
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

			# crop false positives to be added to the negative set
			neg = []
			for j, cc in enumerate(dets):
				if flag_crop[j]:
					neg.append(img[cc[0]-radius:cc[0]+radius+1, cc[1]-radius:cc[1]+radius+1].copy())
			if len(neg) > max_falsedet_per_image:
				neg = random.sample(neg, max_falsedet_per_image)
			negatives += neg

			pbar.update(1)

	return total_correct/total_annotated, (total_dets-total_correct)/total_dets, np.asarray(negatives, dtype=np.float32)

# ---------------------------------------------------------------------------------------------------------- #
# Training functions                                                                                         #
# ---------------------------------------------------------------------------------------------------------- #
def eval(det_model, imgs, labels, batch_size):
	det_model.eval()
	confusion_matrix = torch.zeros(2, 2)
	with torch.no_grad():
		for i in range(0, len(imgs), batch_size):
			batch_data = torch.from_numpy(np.transpose(imgs[i:i+batch_size], (0, 3, 1, 2))).cuda()
			batch_target = torch.tensor(labels[i:i+batch_size], dtype=torch.long).cuda()

			batch_pred = det_model(batch_data)
			batch_pred = (batch_pred > 0.5).long()

			for t, p in zip(batch_target.view(-1), batch_pred.view(-1)):
				confusion_matrix[t,p] += 1
	det_model.train()

	return 100.0*confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[0,1]), 100.0*confusion_matrix[1,1]/(confusion_matrix[1,0]+confusion_matrix[1,1])

def train(det_model, optimizer, imgs, labels, batch_size, iterations, epoch):
	neg_acc, pos_acc = eval(det_model, imgs, labels, batch_size)

	with tqdm(total=iterations, file=sys.stdout) as pbar:
		pbar.set_description('Epoch #{} of training (POS {:.2f}% / NEG {:.2f}%)'.format(epoch+1, pos_acc, neg_acc))

		det_model.train()
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
			det_model.zero_grad()
			batch_pred = det_model(batch_data).squeeze()
			loss = F.binary_cross_entropy(batch_pred, batch_target)
			loss.backward()
			optimizer.step()

			# check performance every 100 iterations
			if i%100 == 99:
				neg_acc, pos_acc = eval(det_model, imgs, labels, batch_size)
				pbar.set_description('Epoch #{} of training (POS {:.2f}% / NEG {:.2f}%)'.format(epoch+1, pos_acc, neg_acc))

			pbar.update(1)

# ---------------------------------------------------------------------------------------------------------- #
# MAIN                                                                                                       #
# ---------------------------------------------------------------------------------------------------------- #
num_iter = 3000
batch_size = 256
max_epochs = 50
early_stopping = 10

det_model = FCN().cuda()
optimizer = optim.Adam(det_model.parameters(), lr=0.0001)

nms_model = NMS().cuda()
nms_model.eval()

# load training data
train_pos, train_neg = load_data()

train_imgs = np.concatenate((train_pos,train_neg), axis=0)
train_labels = np.asarray([1]*len(train_pos) + [0]*len(train_neg), dtype=np.int32)
print("Training data loaded: {} positive and {} negative samples\n".format(len(train_pos), len(train_neg)))

# run 1st epoch of training
train(det_model, optimizer, train_imgs, train_labels, batch_size, num_iter, 0)

# get current performance and false positive samples
dr, fdr, false_positives = update_data(det_model, nms_model)
print("Current performance: DR {:.4f} / FDR {:.4f}\n".format(dr, fdr))

best_score = dr*(1.0-fdr)
early_count = 0
torch.save(det_model.state_dict(), os.path.join(output_folder, 'flying.pytorch'))
print('Model saved!\n')

# following stages
for epoch in range(1, max_epochs):
	if len(false_positives) > 0:
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
		print("Training data rearranged: {} positive and {} negative samples\n".format(len(train_pos), len(train_neg)))

	# run training epoch
	train(det_model, optimizer, train_imgs, train_labels, batch_size, num_iter, epoch)

	# get current performance and false positive samples
	dr, fdr, false_positives = update_data(det_model, nms_model)
	print("Current performance: DR {:.4f} / FDR {:.4f}\n".format(dr, fdr))

	score = dr*(1.0-fdr)
	if score > best_score:
		best_score = score
		early_count = 0
		torch.save(det_model.state_dict(), os.path.join(output_folder, 'flying.pytorch'))
		print('Model saved!\n')
	else:
		early_count += 1

	if early_count >= early_stopping:
		break


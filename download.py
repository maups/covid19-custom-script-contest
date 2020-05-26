#################
# Author: maups #
#################
from sentinelhub import BBox
from sentinelhub.constants import CRS
from eolearn.io import S2L1CWCSInput
from eolearn.core import LinearWorkflow
import numpy as np
import datetime
import logging
import cv2
import os

#########
# Debug #
#########
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s:%(message)s")

#################
# Configuration #
#################

path='./images/'
if not os.path.exists(path):
	os.mkdir(path)

# Large area around Charles de Gaulle Airport
# min_lat, max_lon, max_lat, min_lon
bbox = BBox(bbox=[4.45781, 52.52628, 5.05781, 52.12628], crs=CRS.WGS84)
# Time span
time_interval = ('2020-05-01', '2020-05-11')

########################################################################################
# Download cloud classification masks and select timestamps of images to be downloaded #
########################################################################################

# eo-learn request
input_task = S2L1CWCSInput(layer='CLOUD', resx='60m', resy='60m', time_difference=datetime.timedelta(hours=2))
timelapse = LinearWorkflow(input_task)
result = timelapse.execute({input_task: {'bbox': bbox, 'time_interval': time_interval}})

# process response
timestamps = []
l, = [result[key] for key in result.keys()]
for i in range(len(l.data['CLOUD'])):
	# get cloud mask and mask of valid pixels
	cloud_mask = l.data['CLOUD'][i]
	mask = l.mask['IS_DATA'][i]

	# divide mask in 9 blocks (3x3 grid) and check if any of them is not covered by clouds
	h_step = [0, int(cloud_mask.shape[0]/3), int(2*cloud_mask.shape[0]/3), cloud_mask.shape[0]]
	w_step = [0, int(cloud_mask.shape[1]/3), int(2*cloud_mask.shape[1]/3), cloud_mask.shape[1]]
	flag = False
	for h in range(3):
		for w in range(3):
			if np.mean(cloud_mask[h_step[h]:h_step[h+1], w_step[w]:w_step[w+1]]) <= 0.3 and np.mean(mask[h_step[h]:h_step[h+1], w_step[w]:w_step[w+1]]) >= 0.9:
				flag = True

	# save cloud classification mask
	if flag:
		cv2.imwrite(path+str(l.timestamp[i]).replace(' ','-').replace(':','-')+'-cloud.png', cloud_mask*255)
		timestamps.append(str(l.timestamp[i])[:10])
		logging.debug("Queueing {}".format(str(l.timestamp[i])[:10]))
	else:
		logging.debug("Discarding {}".format(str(l.timestamp[i])[:10]))

#############################
# Download satellite images #
#############################

# request satellite images for selected timestamps
for t in timestamps:
	logging.debug("Processing {}".format(t))

	time_interval = (t, t)

	# eo-learn request
	input_task = S2L1CWCSInput(layer='TRUE_COLOR', resx='10m', resy='10m', time_difference=datetime.timedelta(hours=2))
	timelapse = LinearWorkflow(input_task)
	result = timelapse.execute({input_task: {'bbox': bbox, 'time_interval': time_interval}})

	# process response
	l, = [result[key] for key in result.keys()]
	for i in range(len(l.data['TRUE_COLOR'])):
		# get image and mask of valid pixels
		img = l.data['TRUE_COLOR'][i]
		img = img[:,:,::-1]
		mask = l.mask['IS_DATA'][i]

		# save image and mask of valid pixels
		cv2.imwrite(path+str(l.timestamp[i]).replace(' ','-').replace(':','-')+'.png', img*255)
		cv2.imwrite(path+str(l.timestamp[i]).replace(' ','-').replace(':','-')+'-mask.png', mask*255)

logging.debug("Done!")


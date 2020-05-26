#################
# Author: maups #
#################
from urllib.request import urlopen
import xmltodict
import logging
import numpy as np
import cv2
import os
import re

#########
# Debug #
#########
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s:%(message)s")

#################
# Configuration #
#################

path='./images/'
roi_area=[2.25638, 49.20587, 2.85638, 48.80587]

#################################################
# Compute bounding box for the site of interest #
#################################################

# request OpenStreetMap XML of the site of interest (CDG airport)
file = urlopen('https://www.openstreetmap.org/api/0.6/way/589215792')
site = file.read()
file.close()
site = xmltodict.parse(site)

# get name of the site of interest
name = None
for item in site['osm']['way']['tag']:
	if item['@k'] == 'name:en':
		name = item['@v']
		break

logging.debug("Processing {}".format(name))

# update bounding box corners using boundary nodes of the chosen site
min_lat = 123456.0
max_lat = -123456.0
min_lon = 123456.0
max_lon = -123456.0
for item in site['osm']['way']['nd']:
	logging.debug("Parsing node #{}".format(item['@ref']))

	file = urlopen('https://www.openstreetmap.org/api/0.6/node/'+item['@ref'])
	node = file.read()
	file.close()
	node = xmltodict.parse(node)

	lat = float(node['osm']['node']['@lat'])
	lon = float(node['osm']['node']['@lon'])
	min_lat = min(min_lat, lat)
	max_lat = max(max_lat, lat)
	min_lon = min(min_lon, lon)
	max_lon = max(max_lon, lon)

print(min_lat,max_lat,min_lon,max_lon)
quit()

############################################################
# Crop the region of interest enclosed by the bounding box #
############################################################

top_left_alpha = (roi_area[1]-max_lat)/(roi_area[1]-roi_area[3])
top_left_beta = (min_lon-roi_area[0])/(roi_area[2]-roi_area[0])
assert top_left_alpha >= 0.0 and top_left_alpha <= 1.0 and top_left_beta >= 0.0 and top_left_beta <= 1.0

bottom_right_alpha = (roi_area[1]-min_lat)/(roi_area[1]-roi_area[3])
bottom_right_beta = (max_lon-roi_area[0])/(roi_area[2]-roi_area[0])
assert bottom_right_alpha >= 0.0 and bottom_right_alpha <= 1.0 and bottom_right_beta >= 0.0 and bottom_right_beta <= 1.0

files = [f for f in os.listdir(path) if re.match(r'\d\d\d\d-\d\d-\d\d-\d\d-\d\d-\d\d\.png', f)]
for f in files:
	logging.debug("Cropping {}".format(f))

	img = cv2.imread(path+f, cv2.IMREAD_COLOR)
	mask = cv2.imread(path+f.replace('.png', '-mask.png'), cv2.IMREAD_GRAYSCALE)
	cloud = cv2.imread(path+f.replace('.png', '-cloud.png'), cv2.IMREAD_GRAYSCALE)

	img = img[int(img.shape[0]*top_left_alpha+0.5):int(img.shape[0]*bottom_right_alpha+0.5), int(img.shape[1]*top_left_beta+0.5):int(img.shape[1]*bottom_right_beta+0.5)]
	mask = mask[int(mask.shape[0]*top_left_alpha+0.5):int(mask.shape[0]*bottom_right_alpha+0.5), int(mask.shape[1]*top_left_beta+0.5):int(mask.shape[1]*bottom_right_beta+0.5)]
	cloud = cloud[int(cloud.shape[0]*top_left_alpha+0.5):int(cloud.shape[0]*bottom_right_alpha+0.5), int(cloud.shape[1]*top_left_beta+0.5):int(cloud.shape[1]*bottom_right_beta+0.5)]

	_, cloud = cv2.threshold(cloud, int(0.9*255.0), 255, cv2.THRESH_BINARY)
	if np.mean(cloud)/255.0 < 0.01 and np.mean(mask)/255.0 > 0.9:
		cv2.imwrite(path+f.replace('.png', '-roi.png'), img)


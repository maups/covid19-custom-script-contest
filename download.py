# ---------------------------------------------------------------------------------------------------------- #
# Author: maups                                                                                              #
# ---------------------------------------------------------------------------------------------------------- #
from sentinelhub import BBox, CRS, DataCollection, SHConfig
from eolearn.io import SentinelHubInputTask
from eolearn.core import LinearWorkflow, FeatureType
import numpy as np
import datetime
import sys
import cv2
import os

# ---------------------------------------------------------------------------------------------------------- #
# Configuration                                                                                              #
# ---------------------------------------------------------------------------------------------------------- #
output_folder = './images'

# time interval
time_interval = ('2015-01-01', '2020-07-31')

# AOIs names and geographic coordinates (latitude, longitude)
aoi_names = ['AGP','AMS','ARN','ATH','BCN','BRU','CDG','CPH','DUB','DUS','FCO','FRA','HEL','IST','LGW','LHR','LIS','LTN','MAD','MAN','MUC','MXP','ORY','OSL','PMI','STN','TXL','VIE','WAW','ZRH']
aoi_coords = [(36.67166398,-4.492831362), (52.312141474163397,4.757732975709088), (59.6512,17.9178), (37.882337,23.875459), (41.296657915514594,2.085735972162202), (50.9008,4.4840), (49.006976345624935,2.559786183921451), (55.620750,12.650462), (53.4264,-6.2499), (51.2870,6.7667), (41.807101001429608,12.253597669510512), (50.033333,8.570556), (60.3170,24.9580), (40.982555,28.820829), (51.153047759076003,-0.185395073719614), (51.469716296645025,-0.458042395268015), (38.7675,-8.8028), (51.8715,-0.3677), (40.493475736010851,-3.566288968675247), (53.3523,-2.2717), (48.353443982011179,11.781958961375517), (45.62995905,8.723056), (48.7262,2.3652), (60.197552,11.100415), (39.5510,2.7367), (51.904949,0.202641), (52.5558,13.2860), (48.1062,16.5685), (52.1672369,20.9678911), (47.451542,8.564572)]

# defines the region around the AOI; grid_size x grid_size grid of blocks, each with size block_width x block_height in degrees; grid_size must be an odd number
block_width = 0.15
block_height = 0.10
grid_size = 7

# defines when a block is worth being processed
cloud_threshold = 0.9
cloud_max_cover = 0.3
min_valid_area = 0.9

# sentinel-hub credentials
config = SHConfig()
config.instance_id = '***REMOVED***'
config.sh_client_id = '***REMOVED***'
config.sh_client_secret = '***REMOVED***'

# ---------------------------------------------------------------------------------------------------------- #
# Helper functions                                                                                           #
# ---------------------------------------------------------------------------------------------------------- #

# measures the distance in meters between two geographic coordinates
def geo_dist(lat1, lon1, lat2, lon2):
	R = 6378.137
	d_lat = lat2 * np.pi / 180.0 - lat1 * np.pi / 180.0
	d_lon = lon2 * np.pi / 180.0 - lon1 * np.pi / 180.0
	a = np.sin(d_lat/2.0) * np.sin(d_lat/2.0) + np.cos(lat1 * np.pi / 180.0) * np.cos(lat2 * np.pi / 180.0) * np.sin(d_lon/2.0) * np.sin(d_lon/2.0)
	c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0-a))
	d = R * c
	return d * 1000.0

# ---------------------------------------------------------------------------------------------------------- #
# MAIN                                                                                                       #
# ---------------------------------------------------------------------------------------------------------- #

for aoi, target in zip(aoi_names, aoi_coords):
	print('Processing airport', aoi)
	sys.stdout.flush()

	if not os.path.exists(os.path.join(output_folder, aoi)):
		os.mkdir(os.path.join(output_folder, aoi))

	# ---------------------------------------------------------------------------------------------------------- #
	# AOI definitions                                                                                            #
	# ---------------------------------------------------------------------------------------------------------- #

	# central block top-left corner
	lat = target[0] - 0.5 * block_height
	lon = target[1] - 0.5 * block_width

	# block size for Sentinel2 L1C RGB images with 10m resolution
	pixel_size = 10.0
	img_width = int(( geo_dist(lat, lon, lat, lon+block_width)/pixel_size + geo_dist(lat+block_height, lon, lat+block_height, lon+block_width)/pixel_size ) / 2.0 + 0.5)
	img_height = int(( geo_dist(lat, lon, lat+block_height, lon)/pixel_size + geo_dist(lat, lon+block_width, lat+block_height, lon+block_width)/pixel_size ) / 2.0 + 0.5)

	# AOI top-left corner
	lat = target[0] - grid_size/2.0 * block_height
	lon = target[1] - grid_size/2.0 * block_width

	# AOI bounding box
	full_bbox = BBox(bbox=[lon, lat, lon+grid_size*block_width, lat+grid_size*block_height], crs=CRS.WGS84)

	# ---------------------------------------------------------------------------------------------------------- #
	# Cloud-based image filtering to find viable blocks                                                          #
	# ---------------------------------------------------------------------------------------------------------- #

	# eo-learn request
	input_task = SentinelHubInputTask(data_collection=DataCollection.SENTINEL2_L1C, additional_data=[(FeatureType.DATA, 'CLP'), (FeatureType.MASK, 'dataMask')], time_difference=datetime.timedelta(seconds=1), resolution=60, config=config)
	timelapse = LinearWorkflow(input_task)

	try:
		result = timelapse.execute({input_task: {'bbox': full_bbox, 'time_interval': time_interval}})
	except:
		continue

	# parse cloud images and register useful blocks
	blocks_per_timestamp = {}
	l = result.eopatch()
	for i in range(len(l.data['CLP'])):
		print('Parsing image {} of {}.'.format(i+1, len(l.data['CLP'])))
		sys.stdout.flush()

		# get cloud mask and mask of valid pixels
		cloud_mask = np.asarray(l.data['CLP'][i], dtype=np.float32)/255.0
		mask = np.asarray(l.mask['dataMask'][i], dtype=np.int32)

		timestamp = str(l.timestamp[i]).replace(' ','T')

		# check cloud coverage and data coverage for each block
		for j in range(grid_size):
			for k in range(grid_size):
				y1 = int(j * cloud_mask.shape[0]/grid_size + 0.5)
				x1 = int(k * cloud_mask.shape[1]/grid_size + 0.5)
				y2 = int((j+1) * cloud_mask.shape[0]/grid_size + 0.5)
				x2 = int((k+1) * cloud_mask.shape[1]/grid_size + 0.5)

				cloud_coverage = np.mean(np.where(cloud_mask[y1:y2,x1:x2] > cloud_threshold, 1.0, 0.0))
				area_coverage = np.mean(mask[y1:y2,x1:x2])

				if cloud_coverage <= cloud_max_cover and area_coverage >= min_valid_area:
					if timestamp not in blocks_per_timestamp:
						blocks_per_timestamp[timestamp] = []
					blocks_per_timestamp[timestamp].append((j,k))

		if timestamp in blocks_per_timestamp:
			cv2.imwrite(os.path.join(output_folder, aoi, str(l.timestamp[i]).replace(' ','-').replace(':','-')+'_cloud.png'), cloud_mask*255)
			cv2.imwrite(os.path.join(output_folder, aoi, str(l.timestamp[i]).replace(' ','-').replace(':','-')+'_mask.png'), mask*255)

	# ---------------------------------------------------------------------------------------------------------- #
	# Download RGB images for viable blocks                                                                      #
	# ---------------------------------------------------------------------------------------------------------- #
	for timestamp, blocks in blocks_per_timestamp.items():
		print('Downloading image from', timestamp, 'with', len(blocks), 'viable blocks.')
		sys.stdout.flush()

		acquisition_time = (timestamp, timestamp)

		# eo-learn task
		input_task = SentinelHubInputTask(data_collection=DataCollection.SENTINEL2_L1C, bands=['B02','B03','B04'], bands_feature=(FeatureType.DATA, 'L1C_data'), time_difference=datetime.timedelta(seconds=1), resolution=10, config=config)
		timelapse = LinearWorkflow(input_task)

		# create blank image
		image = np.zeros((img_height*grid_size, img_width*grid_size, 3), np.uint8)

		# fill image with valid blocks
		for block in blocks:
			bbox = BBox(bbox=[lon+block[1]*block_width, lat+(grid_size-block[0]-1)*block_height, lon+(block[1]+1)*block_width, lat+(grid_size-block[0])*block_height], crs=CRS.WGS84)

			# eo-learn request
			try:
				result = timelapse.execute({input_task: {'bbox': bbox, 'time_interval': (timestamp, timestamp)}})
			except:
				continue

			l = result.eopatch()
			img = np.clip(l.data['L1C_data'][0][..., [0,1,2]] * 2.5, 0, 1)

			tmp = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
			image[block[0]*img_height:(block[0]+1)*img_height,block[1]*img_width:(block[1]+1)*img_width] = tmp*255

		cv2.imwrite(os.path.join(output_folder, aoi, timestamp.replace('T','-').replace(':','-')+'_image.png'), image)

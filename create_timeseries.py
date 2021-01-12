# ---------------------------------------------------------------------------------------------------------- #
# Author: maups                                                                                              #
# ---------------------------------------------------------------------------------------------------------- #
import numpy as np
import json
import os
import datetime

# ---------------------------------------------------------------------------------------------------------- #
# Configuration                                                                                              #
# ---------------------------------------------------------------------------------------------------------- #
path_logs = './log/'
output_folder = './timeseries/'

day_window = 30

# ---------------------------------------------------------------------------------------------------------- #
# MAIN                                                                                                       #
# ---------------------------------------------------------------------------------------------------------- #
airports = [a.split('.')[0] for a in sorted(os.listdir(path_logs))]

for airport in airports:
	print('Processing', airport)
	timestamps = json.load(open(os.path.join(path_logs, airport+'.log')))

	# group counts per day/month
	last_day = '2000-01-01'
	day_count = {}
	month_series = {}
	for timestamp, count in timestamps.items():
		day = timestamp[:10]
		if day > last_day:
			last_day = day
		if day not in day_count:
			day_count[day] = [[] for i in range(49)]

		month = timestamp[:7]
		if month not in month_series:
			month_series[month] = [[] for i in range(49)]

		for i in range(49):
			if count[i] is not None:
				day_count[day][i].append(count[i])
				month_series[month][i].append(count[i])
	last_month = last_day[:7]

	# consolidate day counts
	for day, count in day_count.items():
		for i in range(49):
			if count[i]:
				day_count[day][i] = np.mean(count[i])
			else:
				day_count[day][i] = None

	# temporal smoothing for daily series
	day_series = {}
	for day, count in day_count.items():
		date = datetime.datetime(int(day[:4]), int(day[5:7]), int(day[8:10]))
		for i in range(day_window):
			iday = str(date).split()[0]
			if iday > last_day:
				break

			if iday not in day_series:
				day_series[iday] = [[] for j in range(49)]

			for j in range(49):
				if count[j] is not None:
					day_series[iday][j].append(count[j])

			date += datetime.timedelta(days=1)

	# consolidate series
	for day, count in day_series.items():
		signal = 0
		avg_blocks = 0
		for i in range(49):
			if count[i]:
				signal += np.mean(count[i])
				avg_blocks += len(count[i])
		avg_blocks /= 49
		day_series[day] = (signal, avg_blocks)

	for month, count in month_series.items():
		signal = 0
		avg_blocks = 0
		for i in range(49):
			if count[i]:
				signal += np.mean(count[i])
				avg_blocks += len(count[i])
		avg_blocks /= 49
		month_series[month] = (signal, avg_blocks)

	# dump series
	with open(os.path.join(output_folder, airport+'_day.log'), 'w') as fp:
		last=(-1.0,-1.0)
		for day, (signal, avg_blocks) in sorted(list(day_series.items())):
			if (signal, avg_blocks) != last:
				fp.write('%s %0.3f %0.3f\n' % (day, signal, avg_blocks))
				last = (signal, avg_blocks)

	with open(os.path.join(output_folder, airport+'_month.log'), 'w') as fp:
		last=(-1.0,-1.0)
		for month, (signal, avg_blocks) in sorted(list(month_series.items())):
			if (signal, avg_blocks) != last:
				fp.write('%s %0.3f %0.3f\n' % (month, signal, avg_blocks))
				last = (signal, avg_blocks)


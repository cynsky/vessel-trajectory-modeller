# Author: Xing Yifan A0105591J
import numpy as np
import math
import scipy.io as sio
import scipy.sparse as sp
from scipy.sparse import csc_matrix
import csv
import matplotlib.pyplot as plt
import datetime
import time
import os
from scipy import interpolate
import scipy.spatial.distance as distance
from collections import OrderedDict
import writeToCSV
import copy
import sys
import scipy.spatial.distance as DIST
import scipy.cluster.hierarchy as HAC
import random
from sklearn import metrics
import operator
import utils
import plotter

def getSetOfMMSI(data):
	return set(data[:, utils.dataDict["mmsi"]].astype(long))

def sortDataBasedOnTS(data):
	return np.asarray(sorted(data, key = lambda data_point: data_point[utils.dataDict["ts"]]))

def computeVesselMinDistanceMatrix (data, TIME_WINDOW = 3600):
	"""
	data: trajectory points with mmsi ids attached and sorted using ts
	"""
	mmsi_set = getSetOfMMSI(data)
	mmsi_list_dict = {item: list(mmsi_set).index(item) for item in mmsi_set}
	print mmsi_list_dict
	distance_matrix = np.zeros(shape = (len(mmsi_set), len(mmsi_set)))
	for i in range (0, distance_matrix.shape[0]):
		for j in range(i+1, distance_matrix.shape[1]):
			distance_matrix[i][j] = sys.maxint
			distance_matrix[j][i] = distance_matrix[i][j]

	i = 0
	while (i + 1 < len(data)):
		print "checking start index:", i, " total data length:", len(data)
		start_ts = data[i][utils.dataDict["ts"]]
		this_set_data_end_index = i + 1
		while(this_set_data_end_index < len(data) and data[this_set_data_end_index][utils.dataDict["ts"]] < start_ts + TIME_WINDOW):
			this_set_data_end_index += 1

		# check this set
		if (this_set_data_end_index - i > 1):
			# pairwise check
			for p in range(i, this_set_data_end_index):
				for q in range (i + 1, this_set_data_end_index):
					if(data[p][utils.dataDict["mmsi"]] != data[q][utils.dataDict["mmsi"]]): # two vessels
						this_distance = np.linalg.norm([utils.LatLonToXY ( \
							lat1 = data[p][utils.dataDict["latitude"]], \
							lon1 = data[p][utils.dataDict["longitude"]], \
							lat2 = data[q][utils.dataDict["latitude"]], \
							lon2 = data[q][utils.dataDict["longitude"]])], 2)
						p_index = mmsi_list_dict[data[p][utils.dataDict["mmsi"]]]
						q_index = mmsi_list_dict[data[q][utils.dataDict["mmsi"]]]
						if (this_distance < distance_matrix[p_index][q_index]):
							distance_matrix[p_index][q_index] = this_distance
							distance_matrix[q_index][p_index] = distance_matrix[p_index][q_index] # make sure symmetric

		# update indexes
		i = this_set_data_end_index

	print "final min_distance_matrix:\n", distance_matrix
	return mmsi_list_dict, distance_matrix
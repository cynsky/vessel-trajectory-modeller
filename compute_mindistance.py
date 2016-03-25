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
	mmsi_list_dict = {item: list(mmsi_set).index(item) for item in mmsi_set} # dictionary of mmsi to its index in the list
	print mmsi_list_dict
	distance_matrix = np.zeros(shape = (len(mmsi_set), len(mmsi_set)))
	vessel_distance_speed_dict = {} # [vesselid1_vesselid2: [SpeedDistanceTuple([distanace, relative_speed])]]
	for i in range (0, distance_matrix.shape[0]):
		for j in range(i+1, distance_matrix.shape[1]):
			distance_matrix[i][j] = sys.maxint
			distance_matrix[j][i] = distance_matrix[i][j]

			vessel_distance_speed_dict["{id1}_{id2}".format( \
				id1 = min(list(mmsi_set)[i],list(mmsi_set)[j]) , \
				id2 = max(list(mmsi_set)[i],list(mmsi_set)[j]))] = []

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
					if(data[p][utils.dataDict["mmsi"]] != data[q][utils.dataDict["mmsi"]] and \
						data[p][utils.dataDict["course_over_ground"]] != utils.UNKNOWN_COURSE_OVER_GROUND and \
						data[q][utils.dataDict["course_over_ground"]] != utils.UNKNOWN_COURSE_OVER_GROUND): # two vessels, both has direction info
						
						this_distance = np.linalg.norm([utils.LatLonToXY ( \
							lat1 = data[p][utils.dataDict["latitude"]], \
							lon1 = data[p][utils.dataDict["longitude"]], \
							lat2 = data[q][utils.dataDict["latitude"]], \
							lon2 = data[q][utils.dataDict["longitude"]])], 2)
						
						v_p_magnitude = data[p][utils.dataDict["speed_over_ground"]]
						v_q_magnitude = data[p][utils.dataDict["speed_over_ground"]]

						theta_p = data[p][utils.dataDict["course_over_ground"]]/10.0 / (180/math.pi) # to the true north, let y axis to be towards north
						theta_q = data[q][utils.dataDict["course_over_ground"]]/10.0 / (180/math.pi)

						v_p = np.array([v_p_magnitude * math.sin(theta_p), v_p_magnitude * math.cos(theta_p)])
						v_q = np.array([v_q_magnitude * math.sin(theta_q), v_q_magnitude * math.cos(theta_q)])

						p_index = mmsi_list_dict[data[p][utils.dataDict["mmsi"]]]
						q_index = mmsi_list_dict[data[q][utils.dataDict["mmsi"]]]
						# record the distance in the dict if close say, < 1km
						if (this_distance < 1):
							vessel_distance_speed_dict["{id1}_{id2}".format(\
								id1 = long(min(data[p][utils.dataDict["mmsi"]], data[q][utils.dataDict["mmsi"]])) , \
								id2 = long(max(data[p][utils.dataDict["mmsi"]], data[q][utils.dataDict["mmsi"]]))) ].append( \
								utils.SpeedDistanceTuple(distance = this_distance, speed = np.linalg.norm([v_q - v_p], 2)))

						if (this_distance < distance_matrix[p_index][q_index]):
							distance_matrix[p_index][q_index] = this_distance
							distance_matrix[q_index][p_index] = distance_matrix[p_index][q_index] # make sure symmetric
							if (this_distance == 0): # logging the collision case
								print "vessel ", data[p][utils.dataDict["mmsi"]], " and vessel ", data[q][utils.dataDict["mmsi"]], \
								"clashes at ", data[p][utils.dataDict["ts"]], "/", data[q][utils.dataDict["ts"]]

		# update indexes
		i = this_set_data_end_index

	print "final min_distance_matrix:\n", distance_matrix
	return mmsi_list_dict, distance_matrix, vessel_distance_speed_dict




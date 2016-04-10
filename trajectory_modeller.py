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
import interpolator
import clustering_worker
import compute_mindistance

class Point(object):
	def __init__(self,_x,_y):
		self.x = _x
		self.y = _y

def boxMeanTrajectoryPoints(trajectory_points, reference_lat, reference_lon):
	"""
	trajectory_points: in lat, lon;
	returns: geographical mean of these points in lat, lon
	"""
	trajectory_points_XY = convertListOfTrajectoriesToXY(reference_lat, reference_lon, [trajectory_points])[0]
	trajectory_points_XY_mean = np.mean(trajectory_points_XY, axis = 0)
	lat, lon = utils.XYToLatLonGivenOrigin(reference_lat, reference_lon, \
		trajectory_points_XY_mean[utils.data_dict_x_y_coordinate["x"]], trajectory_points_XY_mean[utils.data_dict_x_y_coordinate["y"]])
	trajectory_points_XY_mean[utils.dataDict["latitude"]] = lat
	trajectory_points_XY_mean[utils.dataDict["longitude"]] = lon
	return trajectory_points_XY_mean


def extractTrajectoriesUntilOD(data, originTS, originLatitude, originLongtitude, endTS, endLatitude, endLongtitude, show = True, save = False, clean = False, fname = ""):
	"""
	returns: OD_trajectories: in x,y coordinate;
			 OD_trajectories_lat_lon: in lat, lon coordinate;
	"""
	
	maxSpeed = 0
	for i in range(0, data.shape[0]):
		speed_over_ground = data[i][utils.dataDict["speed_over_ground"]]
		if(speed_over_ground > maxSpeed and speed_over_ground != 102.3): #1023 indicates speed not available
			maxSpeed = speed_over_ground
	print "This tanker maxSpeed:", maxSpeed, " knot"
	
	OD_trajectories = [] # origin destination endpoints trajectory
	i = 0
	while(i< data.shape[0]):
		cur_pos = data[i]
		if(utils.nearOrigin( \
			originLatitude, \
			originLongtitude, \
			cur_pos[utils.dataDict["latitude"]], \
			cur_pos[utils.dataDict["longitude"]], \
			thresh = 0.0) and \
			cur_pos[utils.dataDict["ts"]] == originTS): # must be exact point

			this_OD_trajectory = []
			this_OD_trajectory.append(cur_pos)
			i += 1
			while(i < data.shape[0] and \
				(not utils.nearOrigin( \
					endLatitude, \
					endLongtitude, \
					data[i][utils.dataDict["latitude"]], \
					data[i][utils.dataDict["longitude"]], \
					thresh = 0.0))):
				this_OD_trajectory.append(data[i])
				i += 1
			if(i < data.shape[0]):
				this_OD_trajectory.append(data[i]) # append the destination endpoint
			this_OD_trajectory = np.asarray(this_OD_trajectory) # make it to be an np 2D array

			""" box/radius approach in cleaning of points around origin"""
			j = 1
			print "checking points around origin:", j
			while(j < this_OD_trajectory.shape[0] and \
				utils.nearOrigin( \
					originLatitude, \
					originLongtitude, \
					this_OD_trajectory[j][utils.dataDict["latitude"]], \
					this_OD_trajectory[j][utils.dataDict["longitude"]], \
					thresh = utils.NEIGHBOURHOOD_ORIGIN)):
				j += 1
			print "last point around origin:", j
			this_OD_trajectory_around_origin = this_OD_trajectory[0:j]

			"""Take the box mean, treat timestamp as averaged as well"""
			this_OD_trajectory_mean_origin = boxMeanTrajectoryPoints(this_OD_trajectory_around_origin, originLatitude, originLongtitude)
			print "mean start point x,y : ", utils.LatLonToXY( \
				originLatitude, \
				originLongtitude, \
				this_OD_trajectory_mean_origin[utils.dataDict["latitude"]], \
				this_OD_trajectory_mean_origin[utils.dataDict["longitude"]])
			OD_trajectories.append(np.insert(this_OD_trajectory[j:],0,this_OD_trajectory_mean_origin, axis = 0))
			break  # only one trajectory per pair OD, since OD might be duplicated
		i += 1

	OD_trajectories = np.array(OD_trajectories)
	OD_trajectories_lat_lon = copy.deepcopy(OD_trajectories)
	for i in range(0, len(OD_trajectories)):
		for j in range(0, len(OD_trajectories[i])):
			x, y = utils.LatLonToXY(originLatitude, originLongtitude, OD_trajectories[i][j][utils.dataDict["latitude"]], OD_trajectories[i][j][utils.dataDict["longitude"]])
			OD_trajectories[i][j][utils.data_dict_x_y_coordinate["y"]] = y
			OD_trajectories[i][j][utils.data_dict_x_y_coordinate["x"]] = x
		# plotting purpose
		plt.scatter(OD_trajectories[i][0:len(OD_trajectories[i]),utils.data_dict_x_y_coordinate["x"]], \
			OD_trajectories[i][0:len(OD_trajectories[i]),utils.data_dict_x_y_coordinate["y"]])
	if(not plt.gca().yaxis_inverted()):
		plt.gca().invert_yaxis()
	if(save):
		plt.savefig("./{path}/{fname}.png".format(path = "plots", fname = fname))
	if(show):
		plt.show()
	if(clean):
		plt.clf()

	return OD_trajectories, OD_trajectories_lat_lon

def getDistance(point1, point2):
	dx, dy = utils.LatLonToXY(point1[utils.dataDict["latitude"]], point1[utils.dataDict["longitude"]], point2[utils.dataDict["latitude"]], point2[utils.dataDict["longitude"]])
	return (np.linalg.norm([dx,dy],2))

def alreadyInEndpoints(endpoints, target):
	for i in range(0, len(endpoints)):
		if(getDistance(endpoints[i], target) < utils.NEIGHBOURHOOD_ENDPOINT):
			return True
	return False

def extractEndPoints(data):
	"""
	Note: if the trajectory is discontinued because out of detection range, add that last point before out of range, and the new point in range as end point as well
	TODO: further cleaning of data is needed to extract better end points, eg. 8514019.csv end point 1,2 are actually of the same place but 3 is added due to error point
	"""
	endpoints = []
	print "data.shape:",data.shape
	i = 0
	while(i< data.shape[0]):
		start_point = data[i]
		start_index = i
		
		"""Find the next_point that marks the departure from endpoint"""
		while(i+1<data.shape[0]):
			next_point = data[i+1]
			"""
			If inter point distance > thresh and is not error signal (speed is indeed> 0)
			Or
			inter point time difference > thesh
			"""
			if((getDistance(start_point, next_point) > utils.NEIGHBOURHOOD_ENDPOINT \
				and next_point[utils.dataDict["speed_over_ground"]] > 0) or \
				(next_point[utils.dataDict["ts"]] - start_point[utils.dataDict["ts"]] > utils.BOUNDARY_TIME_DIFFERENCE) \
				and i == start_index # immediate point after start point
				):
				# print "found a point that is out of utils.NEIGHBOURHOOD_ENDPOINT:", datetime.datetime.fromtimestamp(start_point[utils.dataDict["ts"]]).strftime('%Y-%m-%dT%H:%M:%SZ'), \
				# datetime.datetime.fromtimestamp(next_point[utils.dataDict["ts"]]).strftime('%Y-%m-%dT%H:%M:%SZ')
				break;
			i += 1

		next_point = data[i] # back track to get the last data point that is still near start_point
		if(i - start_index > 0 and next_point[utils.dataDict["ts"]] - start_point[utils.dataDict["ts"]] > utils.STAYTIME_THRESH):
			if(len(endpoints) == 0 or (not (endpoints[len(endpoints) - 1] == start_point).all())): # if not just appended
				endpoints.append(start_point)
		elif((i+1) != data.shape[0]): # check boundary case
			"""TODO: is there a boundary informaiton on the area that AIS can detect?"""
			next_point_outside_neighbour = data[i+1]
			if(next_point_outside_neighbour[utils.dataDict["ts"]] - start_point[utils.dataDict["ts"]] > utils.BOUNDARY_TIME_DIFFERENCE and \
				(next_point_outside_neighbour[utils.dataDict["speed_over_ground"]] != 0 or \
				start_point[utils.dataDict["speed_over_ground"]] != 0)): # if start of new trajectory at a new position after some time, boundary case, (one of the speed should not be zero)
			# if(next_point_outside_neighbour[utils.dataDict["ts"]] - start_point[utils.dataDict["ts"]] > \
			# 	getDistance(start_point, next_point_outside_neighbour)/ \
			# 	(1*utils.KNOTTOKMPERHOUR) * 3600): #maximum knot
				print "append both, since start of new trajectory:", datetime.datetime.fromtimestamp(next_point[utils.dataDict["ts"]]).strftime('%Y-%m-%dT%H:%M:%SZ'), datetime.datetime.fromtimestamp(next_point_outside_neighbour[utils.dataDict["ts"]]).strftime('%Y-%m-%dT%H:%M:%SZ')
				if(len(endpoints) == 0 or (not (endpoints[len(endpoints) - 1] == next_point).all())): # if not just appended
					endpoints.append(next_point)
				if(len(endpoints) == 0 or (not (endpoints[len(endpoints) - 1] == next_point_outside_neighbour).all())): # if not just appended
					endpoints.append(next_point_outside_neighbour)

		elif((i+1) == data.shape[0]):
			if(len(endpoints) == 0 or (not (endpoints[len(endpoints) - 1] == next_point).all())): # if not just appended
				endpoints.append(next_point) # last point in the .csv record, should be an end point

		i += 1
	
	return endpoints

def convertListOfTrajectoriesToLatLon(originLatitude, originLongtitude, listOfTrajectories):
	for i in range(0, len(listOfTrajectories)):
		for j in range(0, len(listOfTrajectories[i])):
			lat, lon = utils.XYToLatLonGivenOrigin(originLatitude, originLongtitude, listOfTrajectories[i][j][utils.data_dict_x_y_coordinate["x"]], listOfTrajectories[i][j][utils.data_dict_x_y_coordinate["y"]])
			listOfTrajectories[i][j][utils.dataDict["latitude"]] = lat
			listOfTrajectories[i][j][utils.dataDict["longitude"]] = lon
	return listOfTrajectories

def convertListOfTrajectoriesToXY(originLatitude, originLongtitude, listOfTrajectories):
	for i in range(0, len(listOfTrajectories)):
		for j in range(0, len(listOfTrajectories[i])):
			x, y = utils.LatLonToXY(originLatitude, originLongtitude, listOfTrajectories[i][j][utils.dataDict["latitude"]], listOfTrajectories[i][j][utils.dataDict["longitude"]])
			listOfTrajectories[i][j][utils.data_dict_x_y_coordinate["y"]] = y
			listOfTrajectories[i][j][utils.data_dict_x_y_coordinate["x"]] = x
	return listOfTrajectories


def endPointMatchTrajectoryCentroid(endpoint, centroid, reference_lat, reference_lon):
	assert (len(centroid) > 0), "cluster centroid must be non empty"
	x, y = utils.LatLonToXY(reference_lat,reference_lon,endpoint[utils.dataDict["latitude"]], endpoint[utils.dataDict["longitude"]])
	centroid_start_x = centroid[0][utils.data_dict_x_y_coordinate["x"]]
	centroid_start_y = centroid[0][utils.data_dict_x_y_coordinate["y"]]
	if (np.linalg.norm([x - centroid_start_x, y - centroid_start_y], 2) < 20 * utils.NEIGHBOURHOOD_ENDPOINT):
		return True
	else:
		return False


def endPointsToRepresentativeTrajectoryMapping(endpoints, trajectories, cluster_label, reference_lat, reference_lon):
	"""
	trajectories: in XY coordinate by reference_lat, reference_lon
	endpoints: in lat, lon
	cluster_label: array of cluster label w.r.t array of trajectories, starting with cluster index 1
	"""
	endpoints_cluster_dict = {}
	class_trajectories_dict = clustering_worker.formClassTrajectoriesDict(cluster_label = cluster_label, data = trajectories)
	cluster_centroids_dict = {} # [cluster:centroid] dictionary
	for class_label, trajectories in class_trajectories_dict.iteritems():
		cluster_centroids_dict[class_label] =clustering_worker.getMeanTrajecotoryWithinClass(trajectories)

	for endpoint in endpoints:
		if (not "{lat}_{lon}".format(lat = endpoint[utils.dataDict["latitude"]], \
				lon = endpoint[utils.dataDict["longitude"]]) in endpoints_cluster_dict):
			endpoints_cluster_dict["{lat}_{lon}".format(lat = endpoint[utils.dataDict["latitude"]], \
				lon = endpoint[utils.dataDict["longitude"]])] = []
		for cluster, centroid in cluster_centroids_dict.iteritems():
			if (endPointMatchTrajectoryCentroid(endpoint, centroid, reference_lat, reference_lon)):
				endpoints_cluster_dict["{lat}_{lon}".format(lat = endpoint[utils.dataDict["latitude"]], \
				lon = endpoint[utils.dataDict["longitude"]])].append(utils.ClusterCentroidTuple(cluster = cluster - 1, centroid = centroid)) # offset by 1

	return endpoints_cluster_dict


def lookForEndPoints(endpoints, endpoint_str):
	for endpoint in endpoints:
		if ("{lat}_{lon}".format(lat = endpoint[utils.dataDict["latitude"]], \
				lon = endpoint[utils.dataDict["longitude"]]) == endpoint_str):
			return endpoint
	return None

def executeClustering(root_folder, all_OD_trajectories_XY, reference_lat, reference_lon):
	# fname = "10_tankers_dissimilarity_center_mass_cophenetic_distance_refined_endpoints"
	# fname = "10_tankers_dissimilarity_l2_inconsistent_refined_endpoints"
	fname = "10_tankers_dissimilarity_l2_cophenetic_distance_refined_endpoints"

	# fname = "10_tankers_dissimilarity_l2_inconsistent"
	# fname = "10_tankers_dissimilarity_l2_all_K"
	# fname = "10_tankers_dissimilarity_center_mass"
	# fname = "10_tankers_dissimilarity_center_mass_cophenetic_distance_cleaned"
	# fname = "10_tankers_dissimilarity_center_mass_inconsistent_cleaned"
	

	opt_cluster_label , cluster_labels, CH_indexes = clustering_worker.clusterTrajectories( \
		trajectories  = all_OD_trajectories_XY, \
		fname = fname, \
		path = utils.queryPath("tankers/cluster_result/{folder}".format(folder = fname)), \
		metric_func = clustering_worker.trajectoryDissimilarityL2, \
		# metric_func = clustering_worker.trajectoryDissimilarityCenterMass, \
		
		# user_distance_matrix = writeToCSV.loadData(root_folder + \
			# "/cluster_result/10_tankers_dissimilarity_center_mass/10_tankers_dissimilarity_center_mass_cleaned.npz"), \

		# user_distance_matrix = writeToCSV.loadData(root_folder + \
			# "/cluster_result/10_tankers_dissimilarity_l2_cophenetic_distance_cleaned/10_tankers_dissimilarity_l2_cophenetic_distance_cleaned.npz"), \
		
		user_distance_matrix = writeToCSV.loadData(root_folder + \
			"/cluster_result/10_tankers_dissimilarity_l2_cophenetic_distance_refined_endpoints" + \
			"/10_tankers_dissimilarity_l2_cophenetic_distance_refined_endpoints.npz"), \
		criterion = 'distance')

	print "opt_cluster_label:", opt_cluster_label
	print "opt_num_cluster:", len(set(opt_cluster_label))


	# print "distance between 1 and 4, should be quite small:", clustering_worker.trajectoryDissimilarityL2( \
	# 	all_OD_trajectories_XY[1], all_OD_trajectories_XY[4])
	# print "distance between 0 and 4, should be quite large:", clustering_worker.trajectoryDissimilarityL2( \
	# 	all_OD_trajectories_XY[0], all_OD_trajectories_XY[4])
	# print "center of mass measure distance between 1 and 4, should be quite small:", clustering_worker.trajectoryDissimilarityCenterMass( \
	# 	all_OD_trajectories_XY[1], all_OD_trajectories_XY[4])
	# print "center of mass measure distance between 0 and 4, should be quite large:", clustering_worker.trajectoryDissimilarityCenterMass( \
	# 	all_OD_trajectories_XY[0], all_OD_trajectories_XY[4])
	# print "matrix:\n", clustering_worker.getTrajectoryDistanceMatrix(\
	# 	all_OD_trajectories_XY, \
	# 	metric_func = clustering_worker.trajectoryDissimilarityL2)
	# plotter.plotListOfTrajectories(all_OD_trajectories_XY, show = True, clean = True, save = False, fname = "")
	
	"""Construct the endpoints to representative trajectory mapping"""
	filenames = ["8514019.csv", "9116943.csv", "9267118.csv", "9443140.csv", "9383986.csv", "9343340.csv", "9417464.csv", "9664225.csv", "9538440.csv", "9327138.csv"]
	endpoints = None
	for filename in filenames:
		this_vessel_endpoints = writeToCSV.readDataFromCSVWithMMSI( \
		root_folder + "/endpoints", \
		"{filename}_endpoints.csv".format(filename = filename[:filename.find(".")]))

		# Append to the total end points
		if(endpoints is None):
			endpoints = this_vessel_endpoints
		else:
			endpoints = np.concatenate((endpoints, this_vessel_endpoints), axis=0)

	cluster_centroids = clustering_worker.getClusterCentroids(opt_cluster_label, all_OD_trajectories_XY)
	cluster_centroids_lat_lon = {} # [cluster_label : centroid] dictionary
	for cluster_label, centroid in cluster_centroids.iteritems():
		cluster_centroids_lat_lon[cluster_label] = convertListOfTrajectoriesToLatLon(reference_lat, reference_lon, \
			[copy.deepcopy(centroid)])[0]
		# writeToCSV.writeDataToCSV(np.asarray(cluster_centroids_lat_lon[cluster_label]), root_folder + "/cleanedData/DEBUGGING", \
		# "refined_centroid_{i}".format(i = cluster_label))

	# flatten
	cluster_centroids_lat_lon_flattened = [point for cluster_label, centroid in cluster_centroids_lat_lon.iteritems() \
	for point in centroid]
	writeToCSV.writeDataToCSV(np.asarray(cluster_centroids_lat_lon_flattened), root_folder + "/cleanedData", \
		"centroids_" + fname)

	"""DEBUGGING,using unrefined data"""
	# point_to_examine = (1.2625833, 103.6827)
	# point_to_examine_XY = utils.LatLonToXY(reference_lat,reference_lon,point_to_examine[0], point_to_examine[1])
	# augmented_trajectories_from_point_to_examine_index = []
	# augmented_trajectories_from_point_to_examine = []
	# for i in range(0, len(all_OD_trajectories_XY)):
	# 	trajectory = all_OD_trajectories_XY[i]
	# 	if (np.linalg.norm([ \
	# 		point_to_examine_XY[0] - trajectory[0][utils.data_dict_x_y_coordinate["x"]], \
	# 		point_to_examine_XY[1] - trajectory[0][utils.data_dict_x_y_coordinate["y"]]], 2) < utils.NEIGHBOURHOOD_ENDPOINT):
	# 		augmented_trajectories_from_point_to_examine_index.append(i)
	# 		augmented_trajectories_from_point_to_examine.append(trajectory)
	# 		print "augmented_trajectories_from_point_to_examine_index:", augmented_trajectories_from_point_to_examine_index, \
	# 		"starting pos:", trajectory[0][utils.data_dict_x_y_coordinate["x"]], trajectory[0][utils.data_dict_x_y_coordinate["y"]] 
	# print "augmented_trajectories_from_point_to_examine_index:", augmented_trajectories_from_point_to_examine_index



	# augmented_trajectories_from_point_to_examine = convertListOfTrajectoriesToLatLon(reference_lat, reference_lon, copy.deepcopy(augmented_trajectories_from_point_to_examine))
	# for t in range(0, len(augmented_trajectories_from_point_to_examine)):
	# 	writeToCSV.writeDataToCSV(np.asarray(augmented_trajectories_from_point_to_examine[t]), root_folder + "/cleanedData/DEBUGGING", \
	# 	"DEBUGGING_augmented_{t}".format(t = augmented_trajectories_from_point_to_examine_index[t]))


	# augmented_trajectories_from_point_to_examine_clusters = []
	# for i in augmented_trajectories_from_point_to_examine_index:
	# 	augmented_trajectories_from_point_to_examine_clusters.append(opt_cluster_label[i])
	# augmented_trajectories_from_point_to_examine_clusters_unique = list(set(augmented_trajectories_from_point_to_examine_clusters))


	# class_trajectories_dict = clustering_worker.formClassTrajectoriesDict(opt_cluster_label, all_OD_trajectories_XY)

	# for i in augmented_trajectories_from_point_to_examine_clusters_unique:
	# 	writeToCSV.writeDataToCSV(np.asarray(cluster_centroids_lat_lon[i]), root_folder + "/cleanedData/DEBUGGING", \
	# 	"DEBUGGING_centroid_{i}".format(i = i))
	# 	print "cluster_centroids[{i}], starting point:".format(i = i), cluster_centroids[i][0]

	# 	"""save all trajectories under this cluster i """
	# 	class_trajectories = class_trajectories_dict[i]
	# 	class_trajectories_lat_lon = convertListOfTrajectoriesToLatLon(reference_lat, reference_lon, copy.deepcopy(class_trajectories))
	# 	for j in range(0, len(class_trajectories_lat_lon)):
	# 		print "class_trajectories[{i}], starting point:".format(i = i), class_trajectories[j][0]
	# 		writeToCSV.writeDataToCSV(np.asarray(class_trajectories_lat_lon[j]), \
	# 			utils.queryPath(root_folder + "/cleanedData/DEBUGGING/CLASS{i}".format(i = i)) , \
	# 			"DEBUGGING_class_{i}_trajectory_{j}".format(i = i , j = j))

	"""END DEBUGGING"""


	endpoints_cluster_dict = endPointsToRepresentativeTrajectoryMapping(\
		endpoints, \
		all_OD_trajectories_XY , \
		opt_cluster_label, \
		reference_lat, \
		reference_lon)

	empty_endpoints = []
	augmented_index_to_extra_label_mapping = {} # mapping from normal index to appended index in all_protocol_trajectories
	cluster_label_to_cluster_size = {} # 'cluster size' of the appended augmented trajectory in all_protocol_trajectories
	
	all_protocol_trajectories = [] # indexed by cluster label (offset by 1, cluster 1 -> all_protocol_trajectories[0])
	for label in range(np.min(opt_cluster_label), np.max(opt_cluster_label) + 1):
		assert (label in cluster_centroids_lat_lon), "{label} is supposed to be in the cluster_centroids_lat_lon dict".format(label = label)
		all_protocol_trajectories.append(cluster_centroids_lat_lon[label])
		cluster_label_to_cluster_size[label - 1] = len(np.where(opt_cluster_label == label)[0])
	assert(np.sum([size for label, size in cluster_label_to_cluster_size.iteritems()]) == len(opt_cluster_label)), "sum of individual label size should == total count"

	DEBUG_APPEND_INDEXS = []
	for endpoint_str, endpoint_tuple_list in endpoints_cluster_dict.iteritems():
		endpoint_starting_clusters = [item.cluster for item in endpoint_tuple_list] # get the list of cluster_labels of centroids to a certain endpoint

		if (len(endpoint_starting_clusters) == 0):
			"""If no centroid assigned, then assign the original augmented trajectory"""
			this_empty_endpoint = lookForEndPoints(endpoints, endpoint_str) # endpoints is in lat, lon
			if (this_empty_endpoint is None):
				raise ValueError("Error! should always be able to map back endpoints, but {p} is not found".format(p = endpoint_str))
			empty_endpoints.append(this_empty_endpoint)

			point_to_examine_XY = utils.LatLonToXY(reference_lat,reference_lon, \
				this_empty_endpoint[utils.dataDict["latitude"]], this_empty_endpoint[utils.dataDict["longitude"]])
			augmented_trajectories_from_point_to_examine_index = []
			augmented_trajectories_from_point_to_examine = []
			for i in range(0, len(all_OD_trajectories_XY)):
				trajectory = all_OD_trajectories_XY[i]
				if (np.linalg.norm([ \
					point_to_examine_XY[0] - trajectory[0][utils.data_dict_x_y_coordinate["x"]], \
					point_to_examine_XY[1] - trajectory[0][utils.data_dict_x_y_coordinate["y"]]], 2) < utils.NEIGHBOURHOOD_ENDPOINT):
					augmented_trajectories_from_point_to_examine_index.append(i)
					augmented_trajectories_from_point_to_examine.append(trajectory)
					# print "this found augmented_trajectories_from_point_to_examine_index:", \
					# augmented_trajectories_from_point_to_examine_index, \
					# "starting pos:", \
					# trajectory[0][utils.data_dict_x_y_coordinate["x"]], \
					# trajectory[0][utils.data_dict_x_y_coordinate["y"]] 
			print "all indexes (w.r.t all_OD_trajectories_XY) for this_empty_endpoint:", augmented_trajectories_from_point_to_examine_index

			DEBUG_APPEND_INDEXS.append(augmented_trajectories_from_point_to_examine_index)

			"""Append augmented_trajectories_from_point_to_examine to end of array of centroids and give extra label"""
			for augmented_index in augmented_trajectories_from_point_to_examine_index:
				if (not augmented_index in augmented_index_to_extra_label_mapping): 
					# if this normal trajectory is not appened, append it and mark in the augmented_index_to_extra_label_mapping
					augmented_index_to_extra_label_mapping[augmented_index] = len(all_protocol_trajectories)
					cluster_label_to_cluster_size[augmented_index_to_extra_label_mapping[augmented_index]] = 1
					all_protocol_trajectories.append(\
						convertListOfTrajectoriesToLatLon(reference_lat, reference_lon, \
							[copy.deepcopy(all_OD_trajectories_XY[augmented_index])])[0])
				else:
					cluster_label_to_cluster_size[augmented_index_to_extra_label_mapping[augmented_index]] += 1

				endpoints_cluster_dict[endpoint_str].append(utils.ClusterCentroidTuple(\
					cluster = augmented_index_to_extra_label_mapping[augmented_index], \
					centroid = all_protocol_trajectories[augmented_index_to_extra_label_mapping[augmented_index]]))

	"""Asserting and Saving of info for Agent Based Simulator"""
	assert (len(set([index for index_list in DEBUG_APPEND_INDEXS for index in index_list])) == \
		len(all_protocol_trajectories) - len(set(opt_cluster_label))), \
	"size of appended augmented trajectories should == len(DEBUG_APPEND_INDEXS)" 

	for index in range(0, len(all_protocol_trajectories)):
		assert(index in cluster_label_to_cluster_size), "all_protocol_trajectories's index mapping to cluster should be complete"
	
	for label, size in cluster_label_to_cluster_size.iteritems():
		print "label, size:", label, size

	print "number of endpoints that do not have clusters assigned to:", len(empty_endpoints)
	print "total number of endpoints:", len(endpoints)
	writeToCSV.writeDataToCSVWithMMSI(np.asarray(endpoints), root_folder + "/endpoints", "all_endpoints_with_MMSI")
	writeToCSV.writeDataToCSV(np.asarray(empty_endpoints), root_folder + "/cleanedData", \
		"non_starting_endpoints_10_tankers_dissimilarity_l2_cophenetic_distance_cleaned")
	writeToCSV.saveData([endpoints_cluster_dict], \
		filename = root_folder + "/cleanedData" + "/endpoints_cluster_dict" + fname)

	"""write all the all_protocol_trajectories for DEBUGGING purpose"""
	for i in range(0, len(all_protocol_trajectories)):
		protocol_trajectory = all_protocol_trajectories[i]
		writeToCSV.writeDataToCSV(\
			np.asarray(protocol_trajectory), \
			utils.queryPath(root_folder + "/cleanedData/DEBUGGING/ALL_PROTOCOLS"), \
			"all_protocol_{i}".format(i = i))

	"""Save related csv files for Agent Based Simulator"""
	writeToCSV.writeAllProtocolTrajectories(\
		path = utils.queryPath(root_folder+"ABMInput"), \
		file_name = "protocol_trajectories_with_cluster_size", \
		all_protocol_trajectories = all_protocol_trajectories, \
		cluster_label_to_cluster_size = cluster_label_to_cluster_size)

	writeToCSV.writeEndPointsToProtocolTrajectoriesIndexesWithMMSI(\
		path = utils.queryPath(root_folder+"ABMInput"), \
		file_name = "endpoints_to_protocol_trajectories", \
		endpoints = endpoints, \
		endpoints_cluster_dict = endpoints_cluster_dict)


def main():
	root_folder = "tankers"

	"""
	Get min distance between vessels
	"""
	"""sort the aggregateData with MMSI based on TS"""
	# data_with_mmsi = writeToCSV.readDataFromCSVWithMMSI(path = root_folder + "/cleanedData", filename = "aggregateData_with_mmsi.csv")
	# data_with_mmsi_sorted = compute_mindistance.sortDataBasedOnTS(data_with_mmsi)
	# writeToCSV.writeDataToCSVWithMMSI(data_with_mmsi_sorted, root_folder + "/cleanedData", "aggregateData_with_mmsi_sorted")

	"""Apply the computing of min distance using a timed window"""
	# data_with_mmsi_sorted = writeToCSV.readDataFromCSVWithMMSI(path = root_folder + "/cleanedData", filename = "aggregateData_with_mmsi_sorted.csv")
	# mmsi_set = compute_mindistance.getSetOfMMSI(data_with_mmsi_sorted)
	# print mmsi_set
	# print list(mmsi_set)

	# start_time = time.time()
	# mmsi_list_dict, min_distance_matrix, vessel_distance_speed_dict = \
	# compute_mindistance.computeVesselMinDistanceMatrix(data_with_mmsi_sorted, TIME_WINDOW = 1800)

	# writeToCSV.saveData([{ \
	# 	'mmsi_list_dict': mmsi_list_dict, \
	# 	'min_distance_matrix': min_distance_matrix, \
	# 	'vessel_distance_speed_dict': vessel_distance_speed_dict
	# 	}], filename = root_folder + "/cleanedData" + "/min_distance_matrix_with_mmsi_time_window_1800_sec")

	# print "time spent:", time.time() - start_time

	"""From already computed"""	
	# min_distance_matrix_result = writeToCSV.loadData(\
	# 	root_folder + "/cleanedData" + "/min_distance_matrix_with_mmsi_time_window_1800_sec.npz")
	# print "min_distance_matrix_result type:\n", type(min_distance_matrix_result)
	# mmsi_list_dict = min_distance_matrix_result[0]["mmsi_list_dict"]
	# min_distance_matrix = min_distance_matrix_result[0]["min_distance_matrix"]
	# vessel_distance_speed_dict = min_distance_matrix_result[0]["vessel_distance_speed_dict"]
	# print "min_distance_matrix loaded:\n", min_distance_matrix
	# min_of_min_distance = sys.maxint
	# for i in range(0, min_distance_matrix.shape[0]):
	# 	for j in range(i + 1, min_distance_matrix.shape[1]):
	# 		if (min_distance_matrix[i][j] < min_of_min_distance):
	# 			min_of_min_distance = min_distance_matrix[i][j]
	# print "min_distance_matrix min of 10 tankers:", min_of_min_distance

	# """write min distance records for Agent Based Simulator"""
	# writeToCSV.writeVesselSpeedToDistance(\
	# 	path = utils.queryPath(root_folder+"ABMInput"),\
	# 	file_name = "vessel_speed_to_distance", \
	# 	vessel_distance_speed_dict = vessel_distance_speed_dict)
	# writeToCSV.writeVesselMinDistanceMatrix(\
	# 	path = utils.queryPath(root_folder+"ABMInput"), \
	# 	file_name = "vessel_min_distance_matrix", \
	# 	mmsi_list_dict = mmsi_list_dict, \
	# 	min_distance_matrix = min_distance_matrix)
	# writeToCSV.writeMMSIs(\
	# 	path = utils.queryPath(root_folder+"ABMInput"), \
	# 	file_name = "mmsi_list", \
	# 	mmsi_list = [key for key, index in mmsi_list_dict.iteritems()])

	# raise ValueError("purpose stop for computing min distance between vessels")

	"""
	Test Clustering
	"""
	trajectories_to_cluster = writeToCSV.loadData(root_folder + "/" + "all_OD_trajectories_with_1D_data_refined.npz")
	# trajectories_to_cluster = writeToCSV.loadData(root_folder + "/" + "all_OD_trajectories_cleaned.npz")
	# trajectories_to_cluster = writeToCSV.loadData(root_folder + "/" + "all_OD_trajectories_9664225.npz")
	print "trajectories_to_cluster.shape: ", trajectories_to_cluster.shape
	print "type(trajectories_to_cluster): ", type(trajectories_to_cluster)
	print "len(trajectories_to_cluster): ", len(trajectories_to_cluster)
	
	# convert Lat, Lon to XY for clustering
	all_OD_trajectories_XY = convertListOfTrajectoriesToXY(utils.CENTER_LAT_SG, utils.CENTER_LON_SG, trajectories_to_cluster)
	executeClustering(root_folder = root_folder, \
		all_OD_trajectories_XY = all_OD_trajectories_XY, \
		reference_lat = utils.CENTER_LAT_SG, \
		reference_lon = utils.CENTER_LON_SG)
	raise ValueError("purpose stop for testing clustering")


	"""
	plot out the value space of the features, speed, accelerations, etc, for the aggregateData
	"""
	# filename = "aggregateData.npz"
	# path = "tankers/cleanedData"
	# data = writeToCSV.loadArray("{p}/{f}".format(p = path, f=filename))
	# for trajectory in trajectories_to_cluster:
		# plotter.plotFeatureSpace(trajectory)
	# raise ValueError("For plotting feature space only")

	"""
	Extract endpoints;
	TODO: Further cleaning of the data, sometimes the ship 'flys' around and out of a confined study window, need to tackle this situation
	"""
	filenames = ["8514019.csv", "9116943.csv", "9267118.csv", "9443140.csv", "9383986.csv", "9343340.csv", "9417464.csv", "9664225.csv", "9538440.csv", "9327138.csv"]
	# filenames = ["9664225.csv"]
	# filenames = ["8514019.csv"]
	endpoints = None
	all_OD_trajectories = []
	
	for i in range(0, len(filenames)):
		this_vessel_trajectory_points = writeToCSV.readDataFromCSV(root_folder + "/cleanedData", filenames[i])
		# Extract end points, along with MMSI
		this_vessel_endpoints = np.asarray(extractEndPoints(writeToCSV.readDataFromCSVWithMMSI(root_folder + "/cleanedData", filenames[i])))
		# Save end points, along with MMSI
		writeToCSV.writeDataToCSVWithMMSI( \
			this_vessel_endpoints, \
			root_folder + "/endpoints", \
			"{filename}_endpoints".format(filename = filenames[i][:filenames[i].find(".")]))
		print "this_vessel_endpoints.shape:", this_vessel_endpoints.shape

		# Append to the total end points
		if(endpoints is None):
			endpoints = this_vessel_endpoints
		else:
			endpoints = np.concatenate((endpoints, this_vessel_endpoints), axis=0)

		for s in range (0, len(this_vessel_endpoints) - 1):
			originLatitude = this_vessel_endpoints[s][utils.dataDict["latitude"]]
			originLongtitude = this_vessel_endpoints[s][utils.dataDict["longitude"]]
			origin_ts = this_vessel_endpoints[s][utils.dataDict["ts"]]

			endLatitude = this_vessel_endpoints[s + 1][utils.dataDict["latitude"]]
			endLongtitude = this_vessel_endpoints[s + 1][utils.dataDict["longitude"]]	
			end_ts = this_vessel_endpoints[s + 1][utils.dataDict["ts"]]
			
			"""Extracting trajectory between a pair of OD"""
			print "\n\nextracting endpoints between ", s, " and ", s + 1
			OD_trajectories, OD_trajectories_lat_lon = extractTrajectoriesUntilOD(\
				this_vessel_trajectory_points, \
				origin_ts, \
				originLatitude, \
				originLongtitude, \
				end_ts, \
				endLatitude, \
				endLongtitude, \
				show = False, save = True, clean = False, \
				fname = filenames[i][:filenames[i].find(".")] + "_trajectory_between_endpoint{s}_and{e}".format(s = s, e = s + 1))
				# there will be one trajectory between each OD		
			assert (len(OD_trajectories) > 0), "OD_trajectories extracted must have length > 0"
			print "number of trajectory points extracted : ", len(OD_trajectories[0])

			if(len(OD_trajectories[0]) > 2): # more than just the origin and destination endpoints along the trajectory
				writeToCSV.writeDataToCSV( \
					data = OD_trajectories_lat_lon[0],
					path = root_folder + "/trajectories", \
					file_name = "{filename}_trajectory_endpoint_{s}_to_{e}".format(filename = filenames[i][:filenames[i].find(".")], \
						s = s, \
						e = s + 1))
				"""
				Interpolation based on pure geographical trajectory, ignore temporal information
				"""
				interpolated_OD_trajectories = interpolator.geographicalTrajetoryInterpolation(OD_trajectories)
				plotter.plotListOfTrajectories( \
					interpolated_OD_trajectories, \
					show = False, \
					clean = True, \
					save = True, \
					fname = filenames[i][:filenames[i].find(".")] + "_interpolated_algo_3_between_endpoint{s}_and{e}".format(\
						s = s, \
						e = s + 1))
				
				"""
				Interpolation of 1D data: speed, rate_of_turn, etc; interpolated_OD_trajectories / OD_trajectories are both in X, Y coordinates
				"""
				if(len(interpolated_OD_trajectories) > 0):
					interpolated_OD_trajectories[0] = interpolator.interpolate1DFeatures( \
						interpolated_OD_trajectories[0], \
						OD_trajectories[0])

				# change X, Y coordinate to Lat, Lon
				interpolated_OD_trajectories_lat_lon = convertListOfTrajectoriesToLatLon( \
					originLatitude, originLongtitude, interpolated_OD_trajectories)
				if(len(interpolated_OD_trajectories_lat_lon) > 0):
					# since there should be only one trajectory between each pair of OD
					all_OD_trajectories.append(interpolated_OD_trajectories_lat_lon[0])
			else:
				print "no trajectories extracted between endpoints ", s , " and ", s + 1
				plt.clf()

	assert (not endpoints is None), "Error!: No endpoints extracted from the historial data of vessels" + "_".join(filenames)
	print "Final endpoints.shape:", endpoints.shape
	print "number of interpolated all_OD_trajectories:", len(all_OD_trajectories)

	"""
	save the augmented trajectories between endpoints as npz data file and the plot
	"""
	# remove error trajectories that are too far from Singapore
	all_OD_trajectories = removeErrorTrajectoryFromList(all_OD_trajectories)
	writeToCSV.saveData(all_OD_trajectories, root_folder + "/all_OD_trajectories_with_1D_data")
	# convert Lat, Lon to XY for displaying
	all_OD_trajectories_XY = convertListOfTrajectoriesToXY(utils.CENTER_LAT_SG, utils.CENTER_LON_SG, all_OD_trajectories)
	plotter.plotListOfTrajectories(all_OD_trajectories_XY, show = False, clean = True, save = True, fname = "tanker_all_OD_trajectories")


	"""
	Execute Clustering
	"""
	executeClustering(root_folder = root_folder, \
		all_OD_trajectories_XY = all_OD_trajectories_XY, \
		reference_lat = utils.CENTER_LAT_SG, \
		reference_lon = utils.CENTER_LON_SG)


if __name__ == "__main__":
	main()

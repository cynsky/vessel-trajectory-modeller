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

class Point(object):
	def __init__(self,_x,_y):
		self.x = _x
		self.y = _y


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
	
	OD_trajectories = [] # origin destination endpoints trajectory
	i = 0
	while(i< data.shape[0]):
		cur_pos = data[i]
		if(utils.nearOrigin(originLatitude, originLongtitude, cur_pos[utils.dataDict["latitude"]], cur_pos[utils.dataDict["longtitude"]], thresh = 0.0) and cur_pos[utils.dataDict["ts"]] == originTS): # must be exact point
			this_OD_trajectory = []
			this_OD_trajectory.append(cur_pos)
			i += 1
			while(i < data.shape[0] and (not utils.nearOrigin(endLatitude, endLongtitude, data[i][utils.dataDict["latitude"]], data[i][utils.dataDict["longtitude"]], thresh = 0.0))):
				this_OD_trajectory.append(data[i])
				i += 1
			if(i < data.shape[0]):
				this_OD_trajectory.append(data[i])
			this_OD_trajectory = np.asarray(this_OD_trajectory) # make it to be an np 2D array
			# box/radius approach in cleaning of points around origin
			j = 1
			print "checking points around origin:", j
			while(j < this_OD_trajectory.shape[0] and utils.nearOrigin(originLatitude, originLongtitude, this_OD_trajectory[j][utils.dataDict["latitude"]], this_OD_trajectory[j][utils.dataDict["longtitude"]], thresh = utils.NEIGHBOURHOOD_ORIGIN)):
				j += 1
			print "last point around origin:", j
			this_OD_trajectory_around_origin = this_OD_trajectory[0:j]
			"""Take the box mean, treat timestamp as averaged as well"""
			this_OD_trajectory_mean_origin = np.mean(this_OD_trajectory_around_origin, axis = 0) 
			print "mean start point x,y : ", utils.LatLonToXY(originLatitude, originLongtitude, this_OD_trajectory_mean_origin[utils.dataDict["latitude"]], this_OD_trajectory_mean_origin[utils.dataDict["longtitude"]])
			OD_trajectories.append(np.insert(this_OD_trajectory[j:],0,this_OD_trajectory_mean_origin, axis = 0))
			break  # only one trajectory per pair OD, since OD might be duplicated
		i += 1

	OD_trajectories = np.array(OD_trajectories)
	OD_trajectories_lat_lon = copy.deepcopy(OD_trajectories)
	for i in range(0, len(OD_trajectories)):
		for j in range(0, len(OD_trajectories[i])):
			x, y = utils.LatLonToXY(originLatitude, originLongtitude, OD_trajectories[i][j][utils.dataDict["latitude"]], OD_trajectories[i][j][utils.dataDict["longtitude"]])
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
	dx, dy = utils.LatLonToXY(point1[utils.dataDict["latitude"]], point1[utils.dataDict["longtitude"]], point2[utils.dataDict["latitude"]], point2[utils.dataDict["longtitude"]])
	return (np.linalg.norm([dx,dy],2))

def alreadyInEndpoints(endpoints, target):
	for i in range(0, len(endpoints)):
		if(getDistance(endpoints[i], target) < utils.NEIGHBOURHOOD_ENDPOINT):
			return True
	return False

def extractEndPoints(data):
	"""
	Note: if the trajectory is discontinued because out of detection range, add that last point before out of range, and the new point in range as end point as well
	TODO: further cleaning of data is needed to extract better end points, eg. 8514019.csv end point 2,3 are actually of the same place but 3 is added due to error point
	"""
	endpoints = []
	print "data.shape:",data.shape
	i = 0
	while(i< data.shape[0]):
		start_point = data[i]
		start_index = i
		
		while(i+1<data.shape[0]):
			# print "current start_point:", start_point
			next_point = data[i+1]
			if(getDistance(start_point, next_point) > utils.NEIGHBOURHOOD_ENDPOINT):
				# print "find a point that is out of utils.NEIGHBOURHOOD:", datetime.datetime.fromtimestamp(start_point[utils.dataDict["ts"]]).strftime('%Y-%m-%dT%H:%M:%SZ'), \
				datetime.datetime.fromtimestamp(next_point[utils.dataDict["ts"]]).strftime('%Y-%m-%dT%H:%M:%SZ')
				break;
			i += 1

		next_point = data[i] # back track to get the last data point that is still near start_point
		if(i - start_index > 0 and next_point[utils.dataDict["ts"]] - start_point[utils.dataDict["ts"]] > utils.STAYTIME_THRESH):
			# if(not alreadyInEndpoints(endpoints, start_point)): # But should not do the check if the returned endpoints are used to extract trajectories between them
			# print "append since stay more than half hour:", datetime.datetime.fromtimestamp(start_point[utils.dataDict["ts"]]).strftime('%Y-%m-%dT%H:%M:%SZ')
			if(len(endpoints) == 0 or (not (endpoints[len(endpoints) - 1] == start_point).all())): # if not just appended
				endpoints.append(start_point)

		# TODO: is there a boundary informaiton on the area that AIS can detect?
		elif((i+1) != data.shape[0]): # if still has a next postion which is outside neighbour
			next_point_outside_neighbour = data[i+1]
			if(next_point_outside_neighbour[utils.dataDict["ts"]] - start_point[utils.dataDict["ts"]] > 24*3600): # if start of new trajectory at a new position, or after one day
			# if(next_point_outside_neighbour[utils.dataDict["ts"]] - start_point[utils.dataDict["ts"]] > \
			# 	getDistance(start_point, next_point_outside_neighbour)/ \
			# 	(1*utils.KNOTTOKMPERHOUR) * 3600): #maximum knot
				# print "append both, since start of new trajectory:", datetime.datetime.fromtimestamp(next_point[utils.dataDict["ts"]]).strftime('%Y-%m-%dT%H:%M:%SZ'), datetime.datetime.fromtimestamp(next_point_outside_neighbour[utils.dataDict["ts"]]).strftime('%Y-%m-%dT%H:%M:%SZ')
				if(len(endpoints) == 0 or (not (endpoints[len(endpoints) - 1] == next_point).all())): # if not just appended
					endpoints.append(next_point)
				if(len(endpoints) == 0 or (not (endpoints[len(endpoints) - 1] == next_point_outside_neighbour).all())): # if not just appended
					endpoints.append(next_point_outside_neighbour)

		elif((i+1) == data.shape[0]):
			# print "append since last point", datetime.datetime.fromtimestamp(start_point[utils.dataDict["ts"]]).strftime('%Y-%m-%dT%H:%M:%SZ')
			if(len(endpoints) == 0 or (not (endpoints[len(endpoints) - 1] == next_point).all())): # if not just appended
				endpoints.append(next_point) # last point in the .csv record, should be an end point
		
		# DEBUGGING:
		# if(len(endpoints) >= 77):
			# break;
		i += 1
	
	return endpoints

def convertListOfTrajectoriesToLatLon(originLatitude, originLongtitude, listOfTrajectories):
	for i in range(0, len(listOfTrajectories)):
		for j in range(0, len(listOfTrajectories[i])):
			lat, lon = utils.XYToLatLonGivenOrigin(originLatitude, originLongtitude, listOfTrajectories[i][j][utils.data_dict_x_y_coordinate["x"]], listOfTrajectories[i][j][utils.data_dict_x_y_coordinate["y"]])
			listOfTrajectories[i][j][utils.dataDict["latitude"]] = lat
			listOfTrajectories[i][j][utils.dataDict["longtitude"]] = lon
	return listOfTrajectories

def convertListOfTrajectoriesToXY(originLatitude, originLongtitude, listOfTrajectories):
	for i in range(0, len(listOfTrajectories)):
		for j in range(0, len(listOfTrajectories[i])):
			x, y = utils.LatLonToXY(originLatitude, originLongtitude, listOfTrajectories[i][j][utils.dataDict["latitude"]], listOfTrajectories[i][j][utils.dataDict["longtitude"]])
			listOfTrajectories[i][j][utils.data_dict_x_y_coordinate["y"]] = y
			listOfTrajectories[i][j][utils.data_dict_x_y_coordinate["x"]] = x
	return listOfTrajectories

def isErrorTrajectory(trajectory, center_lat_sg, center_lon_sg):
	"""
	Checks if the give trajectory is too far from the Port Center, or only contains less than one trajectory point
	"""
	if(len(trajectory) <= 1):
		return True

	for i in range(0, len(trajectory)):
		dx, dy = utils.LatLonToXY (trajectory[i][utils.dataDict["latitude"]],trajectory[i][utils.dataDict["longtitude"]],center_lat_sg, center_lon_sg)
		if(np.linalg.norm([dx, dy], 2) > utils.MAX_DISTANCE_FROM_SG):
			return True
	return False

def removeErrorTrajectoryFromList(trajectories, center_lat_sg = 1.2, center_lon_sg = 103.8):
	"""
	trajectories: normal trajectories with lat and lon
	return: the list of trajectories with error ones removed
	"""
	i = 0
	while(i < len(trajectories)):
		if(isErrorTrajectory(trajectories[i], center_lat_sg, center_lon_sg)):
			if(isinstance(trajectories, list)): # if list, call the list's delete method
				trajectories.pop(i)
			elif(isinstance(trajectories, np.ndarray)): # if numpy.ndarray, call its delete method
				trajectories = np.delete(trajectories, i, 0)
		else:
			i += 1
	return trajectories

def main():
	root_folder = "tankers"

	"""
	Test Clustering
	"""
	# trajectories_to_cluster = writeToCSV.loadData(root_folder + "/" + "all_OD_trajectories.npz")
	# # trajectories_to_cluster = writeToCSV.loadData(root_folder + "/" + "all_OD_trajectories_9664225.npz")
	# print type(trajectories_to_cluster)
	# print len(trajectories_to_cluster)
	# trajectories_to_cluster = list(trajectories_to_cluster)
	# all_OD_trajectories_XY = convertListOfTrajectoriesToXY(utils.CENTER_LAT_SG, utils.CENTER_LON_SG, trajectories_to_cluster) # convert Lat, Lon to XY for displaying

	# fname = "10_tankers_dissimilarity_center_mass"
	# # fname = "tanker_9664225_dissimilarity_center_mass"
	# opt_cluster_label , cluster_labels, CH_indexes = clustering_worker.clusterTrajectories( \
	# 	all_OD_trajectories_XY, \
	# 	fname, \
	# 	utils.queryPath("tankers/cluster_result/{folder}".format(folder = fname)), \
	# 	metric_func = clustering_worker.trajectoryDissimilarityCenterMass)
	# print "opt_cluster_label:", opt_cluster_label
	# print "opt_num_cluster:", len(set(opt_cluster_label))

	# print "distance between 1 and 4, should be quite small:", clustering_worker.trajectoryDissimilarityL2(all_OD_trajectories_XY[1], all_OD_trajectories_XY[4])
	# print "distance between 0 and 4, should be quite large:", clustering_worker.trajectoryDissimilarityL2(all_OD_trajectories_XY[0], all_OD_trajectories_XY[4])
	# print "center of mass measure distance between 1 and 4, should be quite small:", clustering_worker.trajectoryDissimilarityCenterMass(all_OD_trajectories_XY[1], all_OD_trajectories_XY[4])
	# print "center of mass measure distance between 0 and 4, should be quite large:", clustering_worker.trajectoryDissimilarityCenterMass(all_OD_trajectories_XY[0], all_OD_trajectories_XY[4])
	# print "matrix:\n", clustering_worker.getTrajectoryDistanceMatrix(all_OD_trajectories_XY, metric_func = clustering_worker.trajectoryDissimilarityL2)
	# plotter.plotListOfTrajectories(all_OD_trajectories_XY, show = True, clean = True, save = False, fname = "")
	# raise ValueError("purpose stop of the testing clustering procedure")

	"""
	plot out the value space of the features, speed, accelerations, etc, for the aggregateData
	"""
	# filename = "aggregateData.npz"
	# path = "tankers/cleanedData"
	# data = writeToCSV.loadArray("{p}/{f}".format(p = path, f=filename))
	# plotter.plotFeatureSpace(data, utils.dataDict)
	# raise ValueError("For plotting feature space only")

	"""
	Extract endpoints;
	TODO: Further cleaning of the data, sometimes the ship 'flys' around and out of a confined study window, need to tackle this situation
	"""
	# filenames = ["8514019.csv", "9116943.csv", "9267118.csv", "9443140.csv", "9383986.csv", "9343340.csv", "9417464.csv", "9664225.csv", "9538440.csv", "9327138.csv"]
	filenames = ["9664225.csv"]
	endpoints = None
	all_OD_trajectories = []
	
	for i in range(0, len(filenames)):
		# Extract end points
		this_vessel_endpoints = np.asarray(extractEndPoints(writeToCSV.readDataFromCSV(root_folder + "/cleanedData", filenames[i])))
		# writeToCSV.writeDataToCSV(this_vessel_endpoints,root_folder + "/endpoints", "{filename}_endpoints".format(filename = filenames[i]))
		print "this_vessel_endpoints.shape:", this_vessel_endpoints.shape	
		# Append to the total end points
		if(endpoints is None):
			endpoints = this_vessel_endpoints
		else:
			endpoints = np.concatenate((endpoints, this_vessel_endpoints), axis=0)

		for s in range (0, len(this_vessel_endpoints) - 1):
		# for s in range (5, 6):
			originLatitude = this_vessel_endpoints[s][utils.dataDict["latitude"]]
			originLongtitude = this_vessel_endpoints[s][utils.dataDict["longtitude"]]
			origin_ts = this_vessel_endpoints[s][utils.dataDict["ts"]]

			endLatitude = this_vessel_endpoints[s + 1][utils.dataDict["latitude"]]
			endLongtitude = this_vessel_endpoints[s + 1][utils.dataDict["longtitude"]]	
			end_ts = this_vessel_endpoints[s + 1][utils.dataDict["ts"]]
			# if there could be possibly a trajectory between theses two this_vessel_endpoints; 
			# Could do a check here or just let the extractTrajectoriesUntilOD return empty array
			if(end_ts - origin_ts <= 3600 * 24):
				"""Extracting trajectory between a pair of OD"""
				print "\n\nextracting endpoints between ", s, " and ", s + 1
				OD_trajectories, OD_trajectories_lat_lon = extractTrajectoriesUntilOD(\
					writeToCSV.readDataFromCSV(root_folder + "/cleanedData", filenames[i]), \
					origin_ts, originLatitude, originLongtitude, end_ts, endLatitude, endLongtitude, \
					show = False, save = False, clean = True, \
					fname = filenames[i][:filenames[i].find(".")] + "_trajectory_between_endpoint{s}_and{e}".format(s = s, e = s + 1)) # there will be one trajectory between each OD		
				assert (len(OD_trajectories) > 0), "OD_trajectories extracted must have length > 0"
				print "number of trajectory points extracted : ", len(OD_trajectories[0])
				# writeToCSV.writeDataToCSV(OD_trajectories_lat_lon[0],root_folder + "/trajectories", "{filename}_trajectory_endpoint_{s}_to_{e}".format(filename = filenames[i][:filenames[i].find(".")], s = s, e = s + 1))

				"""
				Interpolation based on pure geographical trajectory, ignore temporal information
				"""
				# interpolator.geographicalTrajetoryInterpolation(trajectories_x_y_coordinate)
				interpolated_OD_trajectories = interpolator.geographicalTrajetoryInterpolation(OD_trajectories)
				# plotter.plotListOfTrajectories(interpolated_OD_trajectories, show = False, clean = True, save = True, fname = filenames[i][:filenames[i].find(".")] + "_interpolated_algo_3final_between_endpoint{s}_and{e}".format(s = s, e = s + 1))
				
				"""
				Interpolation of 1D data: speed, rate_of_turn, etc; interpolated_OD_trajectories / OD_trajectories are both in X, Y coordinates
				"""
				if(len(interpolated_OD_trajectories) > 0):
					interpolated_OD_trajectories[0] = interpolator.interpolate1DFeatures(interpolated_OD_trajectories[0], OD_trajectories[0])

				# change X, Y coordinate to Lat, Lon
				interpolated_OD_trajectories_lat_lon = convertListOfTrajectoriesToLatLon(originLatitude, originLongtitude, interpolated_OD_trajectories)
				if(len(interpolated_OD_trajectories_lat_lon) > 0):
					all_OD_trajectories.append(interpolated_OD_trajectories_lat_lon[0]) # since there should be only one trajectory between each pair of OD



	assert (not endpoints is None), "No endpoints extracted from the historial data"
	print "Final endpoints.shape:", endpoints.shape
	print "number of interpolated all_OD_trajectories:", len(all_OD_trajectories)
	writeToCSV.saveData(removeErrorTrajectoryFromList(all_OD_trajectories), root_folder + "/all_OD_trajectories_with_1D_data")

	all_OD_trajectories_XY = convertListOfTrajectoriesToXY(utils.CENTER_LAT_SG, utils.CENTER_LON_SG, all_OD_trajectories) # convert Lat, Lon to XY for displaying
	plotter.plotListOfTrajectories(all_OD_trajectories_XY, show = True, clean = True, save = True, fname = "tanker_all_OD_trajectories") # TODO: remove error trajectories that are too far from Singapore


if __name__ == "__main__":
	main()

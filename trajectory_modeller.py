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

def isErrorTrajectory(trajectory, center_lat_sg, center_lon_sg):
	"""
	Checks if the give trajectory is too far from the Port Center, or only contains less than one trajectory point
	"""
	if(len(trajectory) <= 1):
		return True

	for i in range(0, len(trajectory)):
		dx, dy = utils.LatLonToXY (trajectory[i][utils.dataDict["latitude"]],trajectory[i][utils.dataDict["longitude"]],center_lat_sg, center_lon_sg)
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
	# mmsi_list_dict, min_distance_matrix = compute_mindistance.computeVesselMinDistanceMatrix(data_with_mmsi_sorted, TIME_WINDOW = 1800)
	# print "time spent:", time.time() - start_time
	# writeToCSV.saveData([{ \
	# 	'mmsi_list_dict': mmsi_list_dict, \
	# 	'min_distance_matrix': min_distance_matrix \
	# 	}], filename = root_folder + "/cleanedData" + "/min_distance_matrix_with_mmsi_window_1800s")
	
	# # min_distance_matrix_result = writeToCSV.loadData(root_folder + "/cleanedData" + "/min_distance_matrix_with_mmsi.npz")
	# # print "min_distance_matrix_result:\n", min_distance_matrix_result, type(min_distance_matrix_result)
	# # min_distance_matrix = min_distance_matrix_result[0]["min_distance_matrix"]
	# # print "min_distance_matrix loaded:\n", min_distance_matrix
	# # print "min_distance_matrix min of 10 tankers:", np.min(min_distance_matrix)

	# raise ValueError("purpose stop for computing min distance between vessels")


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
	# filenames = ["8514019.csv"]
	endpoints = None
	all_OD_trajectories = []
	
	for i in range(0, len(filenames)):
		this_vessel_trajectory_points = writeToCSV.readDataFromCSV(root_folder + "/cleanedData", filenames[i])
		# Extract end points
		this_vessel_endpoints = np.asarray(extractEndPoints(writeToCSV.readDataFromCSV(root_folder + "/cleanedData", filenames[i])))
		writeToCSV.writeDataToCSV( \
			this_vessel_endpoints, \
			root_folder + "/endpoints", \
			"{filename}_endpoints".format(filename = filenames[i]))
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
				show = False, save = False, clean = False, \
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

	assert (not endpoints is None), "Error!: No endpoints extracted from the historial data of vessel" + filenames[i]
	print "Final endpoints.shape:", endpoints.shape
	print "number of interpolated all_OD_trajectories:", len(all_OD_trajectories)

	"""
	save the augmented trajectories between endpoints as npz data file
	"""
	# remove error trajectories that are too far from Singapore
	writeToCSV.saveData(removeErrorTrajectoryFromList(all_OD_trajectories), root_folder + "/all_OD_trajectories_with_1D_data")
	# convert Lat, Lon to XY for displaying
	all_OD_trajectories_XY = convertListOfTrajectoriesToXY(utils.CENTER_LAT_SG, utils.CENTER_LON_SG, all_OD_trajectories)
	plotter.plotListOfTrajectories(all_OD_trajectories_XY, show = True, clean = True, save = True, fname = "tanker_all_OD_trajectories")


	"""
	Test Clustering
	"""
	# # trajectories_to_cluster = writeToCSV.loadData(root_folder + "/" + "all_OD_trajectories.npz")
	# trajectories_to_cluster = writeToCSV.loadData(root_folder + "/" + "all_OD_trajectories_cleaned.npz")
	# print trajectories_to_cluster.shape
	# # trajectories_to_cluster = writeToCSV.loadData(root_folder + "/" + "all_OD_trajectories_9664225.npz")
	# print type(trajectories_to_cluster)
	# print len(trajectories_to_cluster)
	# trajectories_to_cluster = list(trajectories_to_cluster)
	# # convert Lat, Lon to XY for displaying
	# all_OD_trajectories_XY = convertListOfTrajectoriesToXY(utils.CENTER_LAT_SG, utils.CENTER_LON_SG, trajectories_to_cluster)

	# fname = "10_tankers_dissimilarity_l2_cophenetic_distance_cleaned"
	# # fname = "10_tankers_dissimilarity_l2_inconsistent"
	# # fname = "10_tankers_dissimilarity_l2_all_K"
	# # fname = "10_tankers_dissimilarity_center_mass"
	# # fname = "10_tankers_dissimilarity_center_mass_cophenetic_distance"
	# # fname = "tanker_9664225_dissimilarity_center_mass"

	# opt_cluster_label , cluster_labels, CH_indexes = clustering_worker.clusterTrajectories( \
	# 	trajectories  = all_OD_trajectories_XY, \
	# 	fname = fname, \
	# 	path = utils.queryPath("tankers/cluster_result/{folder}".format(folder = fname)), \
	# 	metric_func = clustering_worker.trajectoryDissimilarityL2, \
	# 	# user_distance_matrix = writeToCSV.loadData(root_folder + \
	# 	# 	"/cluster_result/10_tankers_dissimilarity_center_mass/10_tankers_dissimilarity_center_mass_cleaned.npz")
	# 	user_distance_matrix = writeToCSV.loadData(root_folder + \
	# 		"/cluster_result/10_tankers_dissimilarity_l2_cophenetic_distance_cleaned/10_tankers_dissimilarity_l2_cophenetic_distance_cleaned.npz")
	# 	)
	# print "opt_cluster_label:", opt_cluster_label
	# print "opt_num_cluster:", len(set(opt_cluster_label))

	# # print "distance between 1 and 4, should be quite small:", clustering_worker.trajectoryDissimilarityL2( \
	# # 	all_OD_trajectories_XY[1], all_OD_trajectories_XY[4])
	# # print "distance between 0 and 4, should be quite large:", clustering_worker.trajectoryDissimilarityL2( \
	# # 	all_OD_trajectories_XY[0], all_OD_trajectories_XY[4])
	# # print "center of mass measure distance between 1 and 4, should be quite small:", clustering_worker.trajectoryDissimilarityCenterMass( \
	# # 	all_OD_trajectories_XY[1], all_OD_trajectories_XY[4])
	# # print "center of mass measure distance between 0 and 4, should be quite large:", clustering_worker.trajectoryDissimilarityCenterMass( \
	# # 	all_OD_trajectories_XY[0], all_OD_trajectories_XY[4])
	# # print "matrix:\n", clustering_worker.getTrajectoryDistanceMatrix(\
	# # 	all_OD_trajectories_XY, \
	# # 	metric_func = clustering_worker.trajectoryDissimilarityL2)
	# # plotter.plotListOfTrajectories(all_OD_trajectories_XY, show = True, clean = True, save = False, fname = "")
	# raise ValueError("purpose stop of the testing clustering procedure")

if __name__ == "__main__":
	main()

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
import trajectory_modeller

def minDistanceAgainstCentroids(tr, centroids, metric_func):
	"""
	tr: one trajectory, in x, y coordinates
	centroids: array of cluster centroids, in x, y coordinates
	return: the distance of tr to the its closest centroid in centroids
	"""
	min_dist = sys.maxint
	min_index = -1
	for i in range(0, len(centroids)):
		this_dist = metric_func(tr, centroids[i])
		# print "centroids[{i}]".format( i = i), "'s length = ", len(centroids[i]), "distance = ", this_dist
		if (this_dist < min_dist):
			min_dist = this_dist
			min_index = i

	return min_dist, min_index


def main():
	metric_to_use = int(raw_input("use metric?\n" + "1. l2\n" + "2. center of mass\n"))
	root_folder = "tankers/out_sample_test"
	"""read centroids"""
	centroids = None
	if (metric_to_use == 1):
		centroids = writeToCSV.loadData("tankers/cleanedData/centroids_arr_l2.npz")
	elif(metric_to_use == 2):
		centroids = writeToCSV.loadData("tankers/cleanedData/centroids_arr_center_mass.npz")

	"""Extract endpoints, trajectories, augmentation"""
	filenames = ["9408475.csv", "9259769.csv"] # for out sample test
	# filenames = ["9408475.csv"]
	endpoints = None
	all_OD_trajectories = []
	"""Do the augmentation if not yet done"""
	if (not os.path.exists(root_folder + "/all_OD_trajectories_with_1D_data.npz")):
		for i in range(0, len(filenames)):
			this_vessel_trajectory_points = writeToCSV.readDataFromCSV(root_folder + "/cleanedData", filenames[i])
			# Extract end points, along with MMSI
			this_vessel_endpoints = np.asarray(trajectory_modeller.extractEndPoints(writeToCSV.readDataFromCSVWithMMSI(root_folder + "/cleanedData", filenames[i])))
			# Save end points, along with MMSI
			writeToCSV.writeDataToCSVWithMMSI( \
				this_vessel_endpoints, \
				utils.queryPath(root_folder + "/endpoints"), \
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
				OD_trajectories, OD_trajectories_lat_lon = trajectory_modeller.extractTrajectoriesUntilOD(\
					this_vessel_trajectory_points, \
					origin_ts, \
					originLatitude, \
					originLongtitude, \
					end_ts, \
					endLatitude, \
					endLongtitude, \
					show = False, save = True, clean = False, \
					fname = filenames[i][:filenames[i].find(".")] + "_trajectory_between_endpoint{s}_and{e}".format(s = s, e = s + 1), \
					path = utils.queryPath(root_folder + "/plots"))
					# there will be one trajectory between each OD		
				assert (len(OD_trajectories) > 0), "OD_trajectories extracted must have length > 0"
				print "number of trajectory points extracted : ", len(OD_trajectories[0])

				if(len(OD_trajectories[0]) > 2): # more than just the origin and destination endpoints along the trajectory
					writeToCSV.writeDataToCSV( \
						data = OD_trajectories_lat_lon[0],
						path = utils.queryPath(root_folder + "/trajectories"), \
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
							e = s + 1), \
						path = utils.queryPath(root_folder + "/plots"))
					
					"""
					Interpolation of 1D data: speed, rate_of_turn, etc; interpolated_OD_trajectories / OD_trajectories are both in X, Y coordinates
					"""
					if(len(interpolated_OD_trajectories) > 0):
						interpolated_OD_trajectories[0] = interpolator.interpolate1DFeatures( \
							interpolated_OD_trajectories[0], \
							OD_trajectories[0])

					# change X, Y coordinate to Lat, Lon
					interpolated_OD_trajectories_lat_lon = trajectory_modeller.convertListOfTrajectoriesToLatLon( \
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
		all_OD_trajectories = utils.removeErrorTrajectoryFromList(all_OD_trajectories)
		writeToCSV.saveData(all_OD_trajectories, root_folder + "/all_OD_trajectories_with_1D_data")
	else:
		all_OD_trajectories = writeToCSV.loadData(root_folder + "/all_OD_trajectories_with_1D_data.npz")

	"""convert Lat, Lon to XY for displaying"""
	all_OD_trajectories_XY = trajectory_modeller.convertListOfTrajectoriesToXY(utils.CENTER_LAT_SG, utils.CENTER_LON_SG, all_OD_trajectories)
	plotter.plotListOfTrajectories(\
		all_OD_trajectories_XY, \
		show = False, \
		clean = True, \
		save = True, \
		fname = "out_sample_tanker_all_OD_trajectories", path = utils.queryPath(root_folder + "/plots"))

	"""Test distance to cluster centroids"""
	centroids_XY = trajectory_modeller.convertListOfTrajectoriesToXY(\
		utils.CENTER_LAT_SG, utils.CENTER_LON_SG, centroids)
	
	for i in range(0, len(all_OD_trajectories_XY)):
		this_tr_XY = all_OD_trajectories_XY[i]
		if (metric_to_use == 1):
			this_tr_centroids_dist, according_pattern_index = minDistanceAgainstCentroids(\
				this_tr_XY, centroids_XY, clustering_worker.trajectoryDissimilarityL2)
			print "augmented trajectories[{i}]".format(i = i), \
			"'s best l2 distance is against cluster centroids[{i}], = ".format(i = according_pattern_index), \
			this_tr_centroids_dist, ", max allowed distance  = ", 1000
		elif(metric_to_use == 2):
			this_tr_centroids_dist, according_pattern_index = minDistanceAgainstCentroids(\
				this_tr_XY, centroids_XY, clustering_worker.trajectoryDissimilarityCenterMass)
			print "augmented trajectories[{i}]".format(i = i), \
			"'s best center of mass distance is against cluster centroids[{i}], = ".format(i = according_pattern_index), \
			this_tr_centroids_dist, ", max allowed distance  = ", 1.5



	return

if __name__ == "__main__":
	main()
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

### Start of Grid Interpolation Approach ### 
def findGridPointNearestTrajectoryPos(trajectory, grid_x_index, grid_y_index, grid_x, grid_y):
	x_coord = grid_x[grid_x_index][grid_y_index]
	y_coord = grid_y[grid_x_index][grid_y_index]
	grid_score = distanceScore(x_coord, y_coord, trajectory[0][utils.data_dict_x_y_coordinate["x"]], trajectory[0][utils.data_dict_x_y_coordinate["y"]])
	nearest_pos = 0
	for k in range(1, len(trajectory)):
		cur_distance_score = distanceScore(x_coord, y_coord, trajectory[k][utils.data_dict_x_y_coordinate["x"]], trajectory[k][utils.data_dict_x_y_coordinate["y"]])
		if(cur_distance_score < grid_score):
			grid_score = cur_distance_score
			nearest_pos = k
	return nearest_pos

def distanceScore(grid_x, grid_y, x, y):
	"""
	grid_x: Actual x coordinate on grid,
	grid_y: Actual y coordinate on grid
	"""
	return np.linalg.norm([grid_x - x, grid_y - y], 2)

def allGridVisited(visited):
	for i in range(0, visited.shape[0]):
		for j in range(0, visited.shape[1]):
			if(visited[i][j] == 0): # if there is one grid not yet visited, ie, marked as zero, then return False
				return False
	return True

def createTrajectoryPointRecordWithXY(grid_x_index, grid_y_index, grid_x, grid_y):
	"""
	grid_x: the 2D array records of actual grid_x coordinates,
	grid_y: the 2D array records of actual grid_y coordinates,
	grid_x_index: grid index to query grid_x, grid_y 2D array
	grid_y_index: grid index to query grid_x, grid_y 2D array
	"""
	return [0, 0, 0, grid_y[grid_x_index][grid_y_index], grid_x[grid_x_index][grid_y_index], 0, 0, 0]

def interpolateGeographicalGrid(trajectory):
	"""
	Input: trajectory in x, y coordinate
	return: grid interpolated trajectory (x,y) coordinates, in km (grid), of shape (n,8) with dummy 0 values for features other than (x,y)
	Note: Somtimes when len(trajectory) == 2, only returns one augmented point because all the points on that trajectory is within utils.NEIGHBOURHOOD_ORIGIN
	"""
	GRID_TO_NEAREST_POS = {}

	"""Get the min_x, min_y, max_x, max_y of the converted coordinates among the trajectory points"""
	origin = trajectory[0]
	end = trajectory[len(trajectory) -1]
	scale = 100
	min_x = 0
	max_x = 0
	min_y = 0	
	max_y = 0
	for i in range(0, len(trajectory)):
		x = trajectory[i][utils.data_dict_x_y_coordinate["x"]]
		y = trajectory[i][utils.data_dict_x_y_coordinate["y"]]
		if(x > max_x):
			max_x = x
		if(x < min_x):
			min_x = x
		if(y > max_y):
			max_y = y
		if(y < min_y):
			min_y = y
	print "check min_x, max_x, min_y, max_y:", min_x, max_x, min_y, max_y
	grid_x, grid_y = np.mgrid[min_x:max_x:complex(0,scale), min_y:max_y:complex(0, scale)]
	grid_score = np.zeros(shape = (scale, scale)) # the distance score, the smaller, the better

	
	"""Set up the scores for the grid"""
	print "set up complexity: ", scale*scale, " * ", len(trajectory)
	for i in range(0, scale):
		for j in range(0, scale):
			grid_score[i][j] = distanceScore(grid_x[i][j], grid_y[i][j], trajectory[0][utils.data_dict_x_y_coordinate["x"]], trajectory[0][utils.data_dict_x_y_coordinate["y"]])
			nearest_pos = 0
			for k in range(1, len(trajectory)):
				cur_distance_score = distanceScore(grid_x[i][j], grid_y[i][j], trajectory[k][utils.data_dict_x_y_coordinate["x"]], trajectory[k][utils.data_dict_x_y_coordinate["y"]])
				if(cur_distance_score < grid_score[i][j]):
					grid_score[i][j] = cur_distance_score
					nearest_pos = k
			GRID_TO_NEAREST_POS["{i}_{j}".format(i =i , j = j)] = nearest_pos
	print "grid_score set up done!"

	"""traverse the grid to get best approximate path of the trajectory"""
	# find start grid point
	start_grid_x = 0
	start_grid_y = 0
	for i in range(0, scale):
		for j in range(0, scale):
			if(distanceScore(grid_x[i][j], grid_y[i][j], origin[utils.data_dict_x_y_coordinate["x"]], origin[utils.data_dict_x_y_coordinate["y"]]) < \
				distanceScore(grid_x[start_grid_x][start_grid_y], grid_y[start_grid_x][start_grid_y], origin[utils.data_dict_x_y_coordinate["x"]], origin[utils.data_dict_x_y_coordinate["y"]])):
				start_grid_x = i
				start_grid_y = j
	# append the initial start grid point to result
	interpolated_trajectory = []
	interpolated_trajectory.append(createTrajectoryPointRecordWithXY(start_grid_x, start_grid_y, grid_x, grid_y))
	# start populating the grid points until meeting end
	pos = 0 # first data point
	cur_grid_x = start_grid_x # these are indexes, not actually coornidates
	cur_grid_y = start_grid_y
	visited = np.zeros(shape = (scale, scale))
	visited[cur_grid_x][cur_grid_y] = 1
	# print "start_grid_x, start_grid_y = ", grid_x[start_grid_x][start_grid_y], " , ", grid_y[start_grid_x][start_grid_y]
	# print "start_grid_x, start_grid_y = ", start_grid_x, " , ", start_grid_y

	### Algo1 ###
	# while( pos < len(trajectory) and (not allGridVisited(visited) ) and distanceScore(grid_x[cur_grid_x][cur_grid_y], grid_y[cur_grid_x][cur_grid_y], end[utils.data_dict_x_y_coordinate["x"]], end[utils.data_dict_x_y_coordinate["y"]]) > 0.1):
	# 	next_score = utils.MAX_FLOAT
	# 	next_grid_x = None
	# 	next_grid_y = None

	# 	for i in range(cur_grid_x - 1, cur_grid_x + 2):
	# 		for j in range(cur_grid_y -1, cur_grid_y + 2):
	# 			# only check up, down, left, right
	# 			if(not (i == cur_grid_x and j == cur_grid_y) and \
	# 				not (i == cur_grid_x - 1 and j == cur_grid_y - 1) and \
	# 				not (i == cur_grid_x - 1 and j == cur_grid_y + 1) and \
	# 				not (i == cur_grid_x + 1 and j == cur_grid_y + 1) and \
	# 				not (i == cur_grid_x + 1 and j == cur_grid_y - 1)):
	# 				if(i < 0 or i >= grid_score.shape[0] or j < 0 or  j >= grid_score.shape[1]): # if out of bound
	# 					this_option_score = utils.MAX_FLOAT
	# 				elif(visited[i][j] == 1): # if visited already
	# 					this_option_score = utils.MAX_FLOAT
	# 				elif(GRID_TO_NEAREST_POS["{i}_{j}".format(i = i, j = j)] < pos): # if corresponds to a previous data pos, don't wanna go back either
	# 					this_option_score = utils.MAX_FLOAT
	# 				else:
	# 					this_option_score = grid_score[i][j]
	# 				# update next_score, next_grid_x, next_grid_y if score smaller
	# 				if(this_option_score < next_score):
	# 					next_score = this_option_score
	# 					next_grid_x = i
	# 					next_grid_y = j
	# 	# after checking up, left, right, down
	# 	if(next_score == utils.MAX_FLOAT):
	# 		break; # still max float, can not populate anymore
	# 	else:
	# 		interpolated_trajectory.append(createTrajectoryPointRecordWithXY(next_grid_x, next_grid_y, grid_x, grid_y)) # append to result
	# 		visited[next_grid_x][next_grid_y] = 1 # mark as visited
	# 		cur_grid_x = next_grid_x
	# 		cur_grid_y = next_grid_y # update cur_grid_x, cur_grid_y
	# 		pos = GRID_TO_NEAREST_POS["{i}_{j}".format(i =next_grid_x , j = next_grid_y)]

	### Algo2 Go for the nearest next data pos : might work after initial cleaning of trajectory###
	# while( pos + 1 < len(trajectory) and (not allGridVisited(visited) ) and distanceScore(grid_x[cur_grid_x][cur_grid_y], grid_y[cur_grid_x][cur_grid_y], end[utils.data_dict_x_y_coordinate["x"]], end[utils.data_dict_x_y_coordinate["y"]]) > 0.05):
	# 	next_pos_point = trajectory[pos + 1]
	# 	next_score = utils.MAX_FLOAT
	# 	next_grid_x = None
	# 	next_grid_y = None

	# 	for i in range(cur_grid_x - 1, cur_grid_x + 2):
	# 		for j in range(cur_grid_y -1, cur_grid_y + 2):
	# 			# only check up, down, left, right
	# 			if(not (i == cur_grid_x and j == cur_grid_y) and \
	# 				not (i == cur_grid_x - 1 and j == cur_grid_y - 1) and \
	# 				not (i == cur_grid_x - 1 and j == cur_grid_y + 1) and \
	# 				not (i == cur_grid_x + 1 and j == cur_grid_y + 1) and \
	# 				not (i == cur_grid_x + 1 and j == cur_grid_y - 1)):
	# 				if( not (i < 0 or i >= grid_score.shape[0] or j < 0 or  j >= grid_score.shape[1]) and not (visited[i][j] == 1) ): # if not out of bound and not yet visited
	# 					this_option_score = distanceScore(grid_x[i][j], grid_y[i][j], next_pos_point[utils.data_dict_x_y_coordinate["x"]], next_pos_point[utils.data_dict_x_y_coordinate["y"]])
	# 					# visited[i][j] = 1 # mark as visited
	# 					if(this_option_score < next_score): # find the index with minimum distance to the next point
	# 						next_score = this_option_score
	# 						next_grid_x = i
	# 						next_grid_y = j

	# 	if(next_score == utils.MAX_FLOAT):
	# 		break;
	# 	else:
	# 		interpolated_trajectory.append(createTrajectoryPointRecordWithXY(next_grid_x, next_grid_y, grid_x, grid_y)) # append to result
	# 		visited[next_grid_x][next_grid_y] = 1 # mark as visited
	# 		pos = pos + 1 if (GRID_TO_NEAREST_POS["{i}_{j}".format(i = next_grid_x, j = next_grid_y)] == pos + 1) else pos # update data pos if needed
	# 		# pos = pos + 1 if (distanceScore(grid_x[next_grid_x][next_grid_y], grid_y[next_grid_x][next_grid_y], \
	# 		# next_pos_point[utils.data_dict_x_y_coordinate["x"]], next_pos_point[utils.data_dict_x_y_coordinate["y"]]) < 0.1) else pos # update data pos if needed
	# 		cur_grid_x = next_grid_x
	# 		cur_grid_y = next_grid_y # update cur_grid_x, cur_grid_y


	### Algo3 ###
	while( pos + 1 < len(trajectory) and (not allGridVisited(visited) ) and distanceScore(grid_x[cur_grid_x][cur_grid_y], grid_y[cur_grid_x][cur_grid_y], end[utils.data_dict_x_y_coordinate["x"]], end[utils.data_dict_x_y_coordinate["y"]]) > 0.1):
		next_pos_point = trajectory[pos + 1]
		next_score = utils.MAX_FLOAT
		next_grid_x = None
		next_grid_y = None

		flag_all_neighbours_closet_to_cur_pos = True
		for i in range(cur_grid_x - 1, cur_grid_x + 2):
			for j in range(cur_grid_y -1, cur_grid_y + 2):
				# only check up, down, left, right
				if(not (i == cur_grid_x and j == cur_grid_y) and \
					not (i == cur_grid_x - 1 and j == cur_grid_y - 1) and \
					not (i == cur_grid_x - 1 and j == cur_grid_y + 1) and \
					not (i == cur_grid_x + 1 and j == cur_grid_y + 1) and \
					not (i == cur_grid_x + 1 and j == cur_grid_y - 1)):
					if( not (i < 0 or i >= grid_score.shape[0] or j < 0 or  j >= grid_score.shape[1]) and not (visited[i][j] == 1) ): # if not out of bound and not yet visited
						if(GRID_TO_NEAREST_POS["{i}_{j}".format(i = i, j = j)] > pos):
							flag_all_neighbours_closet_to_cur_pos = False

		if(flag_all_neighbours_closet_to_cur_pos): # if all neighours are closest to current pos
			for i in range(cur_grid_x - 1, cur_grid_x + 2):
				for j in range(cur_grid_y -1, cur_grid_y + 2):
					# only check up, down, left, right
					if(not (i == cur_grid_x and j == cur_grid_y) and \
						not (i == cur_grid_x - 1 and j == cur_grid_y - 1) and \
						not (i == cur_grid_x - 1 and j == cur_grid_y + 1) and \
						not (i == cur_grid_x + 1 and j == cur_grid_y + 1) and \
						not (i == cur_grid_x + 1 and j == cur_grid_y - 1)):
						if( not (i < 0 or i >= grid_score.shape[0] or j < 0 or  j >= grid_score.shape[1]) and not (visited[i][j] == 1) ): # if not out of bound and not yet visited
							this_option_score = distanceScore(grid_x[i][j], grid_y[i][j], next_pos_point[utils.data_dict_x_y_coordinate["x"]], next_pos_point[utils.data_dict_x_y_coordinate["y"]])
							visited[i][j] = 1 # mark as visited
							if(this_option_score < next_score): # find the grid point with minimum distance to the next point
								next_score = this_option_score
								next_grid_x = i
								next_grid_y = j

			if(next_score == utils.MAX_FLOAT): # if all visited, break
				break;
			else:
				interpolated_trajectory.append(createTrajectoryPointRecordWithXY(next_grid_x, next_grid_y, grid_x, grid_y)) # append to result
				pos = pos + 1 if (GRID_TO_NEAREST_POS["{i}_{j}".format(i = next_grid_x, j = next_grid_y)] == pos + 1) else pos # update data pos if needed
				cur_grid_x = next_grid_x
				cur_grid_y = next_grid_y # update cur_grid_x, cur_grid_y
				print "Case1: cur_grid_x coord:", grid_x[cur_grid_x][cur_grid_y], ", cur_grid_y coord:", grid_y[cur_grid_x][cur_grid_y], ", pos: ", GRID_TO_NEAREST_POS["{i}_{j}".format(i =cur_grid_x , j = cur_grid_y)], \
				", pos data coordinate: (", trajectory[pos][utils.data_dict_x_y_coordinate["x"]], ",", trajectory[pos][utils.data_dict_x_y_coordinate["y"]], ")"

		else:
			for i in range(cur_grid_x - 1, cur_grid_x + 2):
				for j in range(cur_grid_y -1, cur_grid_y + 2):
					# only check up, down, left, right
					if(not (i == cur_grid_x and j == cur_grid_y) and \
						not (i == cur_grid_x - 1 and j == cur_grid_y - 1) and \
						not (i == cur_grid_x - 1 and j == cur_grid_y + 1) and \
						not (i == cur_grid_x + 1 and j == cur_grid_y + 1) and \
						not (i == cur_grid_x + 1 and j == cur_grid_y - 1)):
						if(i < 0 or i >= grid_score.shape[0] or j < 0 or  j >= grid_score.shape[1]): # if out of bound
							this_option_score = utils.MAX_FLOAT
						elif(visited[i][j] == 1): # if visited already
							this_option_score = utils.MAX_FLOAT
						elif(GRID_TO_NEAREST_POS["{i}_{j}".format(i = i, j = j)] < pos): # if corresponds to a previous data pos, don't wanna go back either
							this_option_score = utils.MAX_FLOAT
						else:
							this_option_score = grid_score[i][j]
						if(this_option_score < next_score): # update next_score, next_grid_x, next_grid_y if score smaller
							next_score = this_option_score
							next_grid_x = i
							next_grid_y = j

			# after checking up, left, right, down
			if(next_score == utils.MAX_FLOAT):
				break; # still max float, can not populate anymore
			else:
				visited[next_grid_x][next_grid_y] = 1
				interpolated_trajectory.append(createTrajectoryPointRecordWithXY(next_grid_x, next_grid_y, grid_x, grid_y)) # append to result
				cur_grid_x = next_grid_x
				cur_grid_y = next_grid_y # update cur_grid_x, cur_grid_y
				pos = GRID_TO_NEAREST_POS["{i}_{j}".format(i =next_grid_x , j = next_grid_y)] 
				print "Case2: cur_grid_x coord:", grid_x[cur_grid_x][cur_grid_y], ", cur_grid_y coord:", grid_y[cur_grid_x][cur_grid_y], ", pos: ", GRID_TO_NEAREST_POS["{i}_{j}".format(i =cur_grid_x , j = cur_grid_y)], \
				", pos data coordinate: (", trajectory[pos][utils.data_dict_x_y_coordinate["x"]], ",", trajectory[pos][utils.data_dict_x_y_coordinate["y"]], ")"

	print "len(interpolated_trajectory):", len(interpolated_trajectory)
	return np.asarray(interpolated_trajectory)

def geographicalTrajetoryInterpolation(trajectories_x_y_coordinate):
	# d = 0.1 # take a point every 100 metre

	interpolated_trajectories_x_y_coordinate = []
	for i in range(0, len(trajectories_x_y_coordinate)):
		interpolated_trajectories_x_y_coordinate.append(interpolateGeographicalGrid(trajectories_x_y_coordinate[i]))
	print "in geographicalTrajetoryInterpolation interpolated_trajectories_x_y_coordinate.shape:", np.asarray(interpolated_trajectories_x_y_coordinate).shape
	return interpolated_trajectories_x_y_coordinate

def interpolate1DFeatures(geo_augmented_trajectory, original_trajectory):
	"""
	geo_augmented_trajectory: augmented trajectory with feature other than x, y set as 0
	original_trajectory: original trajectory with full features
	Both needs to be in X, Y coordinates
	"""
	assert (len(original_trajectory) >= 1), "In interpolate1DFeatures: original_trajectory must not be empty"
	original_trajectory = np.asarray(original_trajectory)
	geo_augmented_trajectory = np.asarray(geo_augmented_trajectory)

	for feature_id in range(0, len(original_trajectory[0])):
		if (feature_id != utils.data_dict_x_y_coordinate["x"] \
			and feature_id != utils.data_dict_x_y_coordinate["y"] \
			and feature_id != utils.data_dict_x_y_coordinate["ts"]):
			feature_values_original_trajectory = original_trajectory[:, feature_id]
			feature_points_original_trajectory = original_trajectory[:, [utils.data_dict_x_y_coordinate["x"], utils.data_dict_x_y_coordinate["y"]]]
			assert(feature_points_original_trajectory.shape == (len(original_trajectory), 2)), "In interpolate1DFeatures: shape of original feature points incorrect"
			assert(len(feature_values_original_trajectory) == len(original_trajectory)), "In interpolate1DFeatures: length of original feature values incorrect"

			feature_points_augmented_trajectory = geo_augmented_trajectory[:, [utils.data_dict_x_y_coordinate["x"], utils.data_dict_x_y_coordinate["y"]]] # points where to interpolate data
			
			"""
			linear or cubic 2d interpolation will fill with nan values at requested points outside of the convex hull of the input points
			"""
			feature_values_augmented_trajectory = interpolate.griddata(feature_points_original_trajectory, \
				feature_values_original_trajectory, \
				feature_points_augmented_trajectory, \
				method='nearest')
			# nans    = np.array( np.where(  np.isnan(feature_values_augmented_trajectory) ) ).T
			# notnans = np.array( np.where( ~np.isnan(feature_values_augmented_trajectory) ) ).T
			
			# non_nan_values = np.array([item for item in feature_values_augmented_trajectory if ~np.isnan(item)])
			# print "min not nans:", np.min(non_nan_values), "max not nans:", np.max(non_nan_values)
			# sigma = ( np.max(non_nan_values) - np.min(non_nan_values) )/ 6.0
			# for p in nans:
			# 	feature_values_augmented_trajectory[p[0]] = sum( feature_values_augmented_trajectory[q[0]]*np.exp(-(sum((p-q)**2))/(2 * sigma**2 )) for q in notnans )

			"""
			Try rbf
			"""
			# rbfi = interpolate.Rbf( \
			# 	original_trajectory[:, utils.data_dict_x_y_coordinate["x"]], \
			# 	original_trajectory[:, utils.data_dict_x_y_coordinate["y"]], \
			# 	feature_values_original_trajectory)
			# feature_values_augmented_trajectory = rbfi( \
			# 	geo_augmented_trajectory[:, utils.data_dict_x_y_coordinate["x"]], \
			# 	geo_augmented_trajectory[:, utils.data_dict_x_y_coordinate["y"]])
			
			"""
			Try interp2d
			"""
			# f = interpolate.interp2d( \
			# 	original_trajectory[:, utils.data_dict_x_y_coordinate["x"]], \
			# 	original_trajectory[:, utils.data_dict_x_y_coordinate["y"]], \
			# 	feature_values_original_trajectory, \
			# 	kind = 'cubic', fill_value = None)
			# feature_values_augmented_trajectory = f( \
			# 	geo_augmented_trajectory[:, utils.data_dict_x_y_coordinate["x"]], \
			# 	geo_augmented_trajectory[:, utils.data_dict_x_y_coordinate["y"]])

			assert (len(feature_values_augmented_trajectory) == len(geo_augmented_trajectory)), \
			"In interpolate1DFeatures: interpolated feature values should have the same length as geo_augmented_trajectory"
			geo_augmented_trajectory[:, feature_id] = feature_values_augmented_trajectory
	
	return geo_augmented_trajectory
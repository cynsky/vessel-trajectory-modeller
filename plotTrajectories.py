# Author: Xing Yifan Yix14021
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

"""
CONSTANTS
"""
dataDict = {
"navigation_status":0,
"rate_of_turn":1,
"speed_over_ground":2,
"latitude":3,
"longtitude":4,
"course_over_ground":5,
"true_heading":6,
"ts":7
}

data_dict_x_y_coordinate = {
"navigation_status":0,
"rate_of_turn":1,
"speed_over_ground":2,
"y":3,
"x":4,
"course_over_ground":5,
"true_heading":6,
"ts":7
}


CENTER_LAT_SG = 1.2

CENTER_LON_SG = 103.8

geoScale = 600000.0

knotToKmPerhour = 1.85200

NEIGHBOURHOOD = 0.2

NEIGHBOURHOOD_ENDPOINT = 0.1

NEIGHBOURHOOD_ORIGIN = 0.1

STAYTIME_THRESH = 1800 # 1 hour

MAX_FLOAT = sys.float_info.max
MIN_FLOAT = sys.float_info.min

MAX_DISTANCE_FROM_SG = 100 # 100 km

class Point(object):
	def __init__(self,_x,_y):
		self.x = _x
		self.y = _y

def saveSparse (array, filename):
	np.savez(filename,data = array.data ,indices=array.indices,indptr =array.indptr, shape=array.shape )

def loadSparse(filename):
	loader = np.load(filename)
	return csc_matrix((  loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

def saveArray (array, filename):
	np.savez(filename,data = array)

def loadArray(filename):
	loader = np.load(filename)
	return np.array(loader['data'])

# from Lat Lon to X Y coordinates in Km
def LatLonToXY (lat1,lon1,lat2, lon2): # lat1 and lon1 are assumed to be origins, and all inputs are in proper lat lon
	# fix origin for display
	dx = (lon2-lon1)*40000*math.cos((lat1+lat2)*math.pi/360)/360
	dy = (lat1-lat2)*40000/360
	return dx, dy

# from X Y coordinates in Km to Lat Lon with a given origin Lat/Lon point as reference; Note: X-positive-axis is rightwards and Y-positive axis is downwards
def XYToLatLonGivenOrigin(lat1, lon1, x, y):
	lat2 = lat1 - y*360/40000
	lon2 = lon1 + x/(40000*math.cos((lat1+lat2)*math.pi/360)/360)
	return lat2, lon2

def nearOrigin(originLat, originLon, currentLat, currentLon, thresh = 0.5): # Default: if within 500 metre, then we regard as this start point close to the origin
	x, y = LatLonToXY(originLat, originLon, currentLat, currentLon)
	return (np.linalg.norm([x,y],2) <= thresh) 


def withinStudyWindow(originLatitude, originLongtitude, vesselLatitude,vesselLongtitude,currentLat, currentLon):
	max_X, max_Y = LatLonToXY(originLatitude, originLongtitude, vesselLatitude,vesselLongtitude)
	current_X, current_Y = LatLonToXY(originLatitude, originLongtitude, currentLat, currentLon)
	# if(current_X * max_X < 0): # if X direction is not consistent
	# 	return False
	# if(current_Y * max_Y < 0):
	# 	return False
	# if(abs(current_X) > abs(max_X)):
	# 	return False
	# if(abs(current_Y) > abs(max_Y)):
	# 	return False

	# Try a study window of a square surrounding the origin
	squareWindowLen = max(max_X, max_Y)
	if(abs(current_X) < squareWindowLen and abs(current_Y) < squareWindowLen):
		return True
	else:
		return False

def withinTimeConstriant(startTS, currentTS, timeWindowInHours):
	return ((currentTS - startTS) < timeWindowInHours * 3600)


def convertKnotToMeterPerSec (knot):
	knotToKmPerhour = 1.85200
	KmPerhourToMetrePerSec = 1/3.6
	return knot * knotToKmPerhour * KmPerhourToMetrePerSec

def getSpeedAccelerations(data, dataDict):
	accelerations = [] # acceleration will be in m/s^2
	timeAxis = [] # time axis in seconds
	startTime = data[0][dataDict["ts"]]
	distanceAxis = []
	startPoint = [data[0][dataDict["latitude"]], data[0][dataDict["longtitude"]]]
	for i in range (1, len(data)):
		curSpeed = data[i][dataDict["speed_over_ground"]]
		prevSpeed = data[i-1][dataDict["speed_over_ground"]]
		dt_secs = data[i][dataDict["ts"]] - data[i - 1][dataDict["ts"]]
		if(curSpeed == 102.3 or prevSpeed == 102.3):
			continue
		if(dt_secs == 0):
			continue
		accelerations.append((convertKnotToMeterPerSec(curSpeed) - convertKnotToMeterPerSec(prevSpeed))/float(dt_secs))
		timeAxis.append(data[i][dataDict["ts"]] - startTime)
		distanceAxis.append(np.linalg.norm([LatLonToXY(startPoint[0],startPoint[1], data[i][dataDict["latitude"]], data[i][dataDict["longtitude"]])],2))


	accelerations = np.asarray(accelerations)
	timeAxis = np.asarray(timeAxis)
	distanceAxis = np.asarray(distanceAxis)
	return accelerations, timeAxis, distanceAxis

def getSpeeds(data, dataDict):
	data = np.asarray(data)
	speeds = data[:,dataDict["speed_over_ground"]]
	timeAxis = []
	timeAxis.append(0)
	startTime = data[0][dataDict["ts"]]
	distanceAxis = []
	distanceAxis.append(0)
	startPoint = [data[0][dataDict["latitude"]], data[0][dataDict["longtitude"]]]
	for i in range(1, len(data)):
		dt_secs = data[i][dataDict["ts"]] - startTime
		timeAxis.append(dt_secs)
		distanceAxis.append(np.linalg.norm([LatLonToXY(startPoint[0],startPoint[1], data[i][dataDict["latitude"]], data[i][dataDict["longtitude"]])],2))
	
	i = 0
	while(i < len(speeds)):
		if(speeds[i] == 102.3): # remove the not available ones
			speeds = np.delete(speeds, i, 0)
		else:
			i += 1

	timeAxis = np.asarray(timeAxis)
	return speeds, timeAxis, distanceAxis

def getAngularSpeeds(data, dataDict):
	data = np.asarray(data)
	angularSpeeds = data[:,dataDict["rate_of_turn"]]
	timeAxis = []
	timeAxis.append(0)
	startTime = data[0][dataDict["ts"]]
	for i in range(1, len(data)):
		dt_secs = data[i][dataDict["ts"]] - startTime
		timeAxis.append(dt_secs)

	print angularSpeeds.shape
	i = 0
	while(i < len(angularSpeeds)):
		if(angularSpeeds[i] == 128 or angularSpeeds[i] == -128): # remove the not available ones
			angularSpeeds = np.delete(angularSpeeds, i)
		else:
			i += 1
	timeAxis = np.asarray(timeAxis)
	return angularSpeeds, timeAxis

def plotFeatureSpace(data, dataDict):
	num_bins = 50
	#1. plot out the acceleration profile
	accelerations, _ , _= getSpeedAccelerations(data,dataDict);
	print accelerations.shape
	print np.amax(accelerations)
	print np.amin(accelerations)
	# plt.plot(accelerations)
	plt.hist(accelerations, num_bins, normed= True, facecolor='green', alpha=0.5, stacked = True)
	plt.ylabel('Relative frequency')
	plt.xlabel('Acceleration Profile')
	plt.show()

	#2. plot out the speed profile
	speeds, _ , _= getSpeeds(data, dataDict)
	print np.amax(speeds)
	print np.amin(speeds)
	plt.hist(speeds, num_bins, normed= True, facecolor='green', alpha=0.5, stacked = True)
	plt.xlabel("speeds profile")
	plt.show()

	#3. plot out the turning rate profile
	angularSpeeds, _ = getAngularSpeeds(data, dataDict)
	print np.amax(angularSpeeds)
	print np.amin(angularSpeeds)
	plt.hist(angularSpeeds, num_bins, normed= True, facecolor='green', alpha=0.5, stacked = True)
	plt.xlabel("Angular speeds profile")
	plt.show()
	return

def plotTrajectoryProfiles(pathToSave, folderName, trajectories, dataDict, origin, end, savefig = True,showfig = True):
	if(not os.path.isdir("./{path}/{folderToSave}".format(path = pathToSave, folderToSave = folderName))):
		os.makedirs("./{path}/{folderToSave}".format(path = pathToSave, folderToSave = folderName))

	for i in range(0, len(trajectories)):
		last_point_in_trajectory = trajectories[i][len(trajectories[i]) -1]
		# end[0] is end latitude, end[1] is end longtitude
		if(nearOrigin(end[0],end[1],last_point_in_trajectory[dataDict["latitude"]], last_point_in_trajectory[dataDict["longtitude"]])): # only plot profile for those trajectories between origin and end		
			# 1. acceleration profile from start point to end point
			accelerations, timeAxis, distanceAxis = getSpeedAccelerations(trajectories[i], dataDict)
			plt.scatter(distanceAxis, accelerations, label = "trajectory_acceleration_profile_{i}".format(i = i))
			if(savefig):
				plt.savefig("./{path}/{folderToSave}/trajectory_acceleration_profile_{i}.png".format(path = pathToSave, folderToSave = folderName, i = i))
			if(showfig):
				plt.show()
			plt.clf()
			# 2. speed profile from start point to end point 
			speeds, timeAxis , distanceAxis= getSpeeds(trajectories[i], dataDict)
			plt.scatter(distanceAxis, speeds, label = "trajectory_speed_profile_{i}".format(i = i))
			if(savefig):
				plt.savefig("./{path}/{folderToSave}/trajectory_speed_profile_{i}.png".format(path = pathToSave, folderToSave = folderName, i = i))
			if(showfig):
				plt.show()
			plt.clf()

def plotOneTrajectory(trajectory, show = True, clean = True):
	"""
	Given one trajectory already converted into XY plane
	"""
	trajectory = np.asarray(trajectory)
	plt.plot(trajectory[0:len(trajectory),data_dict_x_y_coordinate['x']], trajectory[0:len(trajectory),data_dict_x_y_coordinate['y']])
	if(not plt.gca().yaxis_inverted()):
		plt.gca().invert_yaxis()
	if(show):
		plt.show()
	if(clean):
		plt.clf()

def plotListOfTrajectories(trajectories, show = True, clean = True, save = False, fname = "", path = "plots"):
	"""
	Give a list of trajectories that are already converted into XY plane
	"""
	for i in range(0, len(trajectories)):
		plotOneTrajectory(trajectories[i], False, False)
	if(not plt.gca().yaxis_inverted()):
		plt.gca().invert_yaxis()
	if(save):
		plt.savefig("./{path}/{fname}.png".format(path = path, fname = fname))
	if(show):
		plt.show()
	if(clean):
		plt.clf()

def getNumberOfPointsToTake(trajectories, data_dict_x_y_coordinate, interval = 60.0):
	"""
	Number of points to take = minimum change of time over the trajectory/60 seconds
	"""
	min_delta_time = None

	for i in range(0, len(trajectories)):
		delta_time = trajectories[i][len(trajectories[i]) -1][data_dict_x_y_coordinate['ts']] -  trajectories[i][0][data_dict_x_y_coordinate['ts']]
		if(min_delta_time == None):
			min_delta_time = delta_time
		elif(delta_time < min_delta_time):
			min_delta_time = delta_time
		print "number of poinst on trajectories[{i}]:".format(i =i), len(trajectories[i]), " ;delta_time: ", delta_time
	print "min_delta_time:", min_delta_time
	return int(min_delta_time/interval)

# TODO: if we need to consider temporal information, then, we need to further clean out the points on the trajectory where it rambles around the origin
def interpolateTrajectorypointsTemporal(trajectory_x_y_coordinate, points_time_axis, data_dict_x_y_coordinate):
	"""
	interpolate based on 't' temporal information as base coordinate
	"""
	print "in interpolateTrajectorypointsTemporal, len(points_time_axis)", len(points_time_axis)
	trajectory_x_y_coordinate = np.asarray(trajectory_x_y_coordinate)
	interpolated_trajectory_x_y_coordinate = np.zeros(shape = (len(points_time_axis), trajectory_x_y_coordinate.shape[1]))
	x = trajectory_x_y_coordinate[:,data_dict_x_y_coordinate['ts']] - trajectory_x_y_coordinate[0][data_dict_x_y_coordinate['ts']] 
	print "\nx:\n", x
	print "\npoints_time_axis:\n", points_time_axis

	# x should only be in the range of the points_time_axis, trim the unnecessary extra ones
	time_axis_end_index = len(x)
	if(len(x) > len(points_time_axis)):		
		for i in range(0, len(x)):
			if(x[i] > points_time_axis[len(points_time_axis) - 1]):
				break;
		if(i != len(x)):
			time_axis_end_index = i + 1

	
	print "in interpolateTrajectorypointsTemporal, original num points:", len(x)
	print "in interpolateTrajectorypointsTemporal, range of x needed:", time_axis_end_index
	x = x[0:time_axis_end_index]

	for j in range(0, trajectory_x_y_coordinate.shape[1]):
		if(j != data_dict_x_y_coordinate['ts']):
			y = trajectory_x_y_coordinate[0:time_axis_end_index,j]

			tmp = OrderedDict()
			for point in zip(x, y):
				tmp.setdefault(point, point)
			mypoints = tmp.values()
			x_trim = [point[0] for point in mypoints]
			y_trim = [point[1] for point in mypoints]
			
			"""TODO: debug numpy.linalg.linalg.LinAlgError: singular matrix"""			
			f = interpolate.interp1d(x_trim, y_trim, kind='cubic')
			interpolated_y = f(points_time_axis)
			interpolated_trajectory_x_y_coordinate[:,j] = interpolated_y
		else:
			interpolated_trajectory_x_y_coordinate[:,j] = points_time_axis + trajectory_x_y_coordinate[0][data_dict_x_y_coordinate['ts']]
	print "in interpolateTrajectorypointsTemporal, interpolated_trajectory_x_y_coordinate.shape:", interpolated_trajectory_x_y_coordinate.shape
	return interpolated_trajectory_x_y_coordinate


def temporalTrajectoryInterpolation(trajectories_x_y_coordinate):
	if(len(trajectories_x_y_coordinate) == 0):
		return
	interval = 120.0
	num_points = getNumberOfPointsToTake(trajectories_x_y_coordinate, data_dict_x_y_coordinate, interval)
	points_time_axis = np.arange(0, interval*(num_points + 1),interval)

	interpolated_trajectories_x_y_coordinate = []
	for i in range(0, len(trajectories_x_y_coordinate)):
		interpolated_trajectories_x_y_coordinate.append(interpolateTrajectorypointsTemporal(trajectories_x_y_coordinate[i], points_time_axis, data_dict_x_y_coordinate))
	print "interpolated_trajectories_x_y_coordinate.shape:", np.asarray(interpolated_trajectories_x_y_coordinate).shape	
	plotListOfTrajectories(interpolated_trajectories_x_y_coordinate)

	# plotOneTrajectory(trajectories_x_y_coordinate[39])
	# interpolated_test_39 = interpolateTrajectorypointsTemporal(trajectories_x_y_coordinate[39], points_time_axis, data_dict_x_y_coordinate)
	# plotOneTrajectory(interpolated_test_39)

	# plotOneTrajectory(trajectories_x_y_coordinate[43])
	# interpolated_test_43 = interpolateTrajectorypointsTemporal(trajectories_x_y_coordinate[43], points_time_axis, data_dict_x_y_coordinate)
	# plotOneTrajectory(interpolated_test_43)


def interpolateTrajectorypointsGeographical(trajectory,d):
	"""
	trajectory is already converted in XY coordinate
	"""
	trajectory = np.asarray(trajectory)
	x = trajectory[:,data_dict_x_y_coordinate['x']]
	y = trajectory[:,data_dict_x_y_coordinate['y']]
	plotOneTrajectory(trajectory, show = True, clean = True)

	# unique_x, indexes = np.unique(x,return_index = True)
	# indexes = sorted(indexes)
	# x = x[indexes]
	# y = y[indexes]
	

	print "len(x):", len(x)
	print "len(y):", len(y)
	print "len(set(x)):", len(set(x))
	print "len(set(y)):", len(set(y))

	tmp = OrderedDict()
	for point in zip(x, y):
		print point
		tmp.setdefault(point, point)
	mypoints = tmp.values()
	print "mypoints:\n", len(mypoints)
	x = [point[0] for point in mypoints]
	y = [point[1] for point in mypoints]

	print "x after trim:", len(x)
	print "y after trim:", len(y)

	f = interpolate.interp1d(x, y, kind='slinear')
	# tck,u = interpolate.splprep([x,y], s=0)

	# print max(x), x[len(x) -1]

	# Get equal distance points with d
	pre_x = x[0]
	pre_y = y[0]
	step = d/10.0
	interpolated_points = [Point(pre_x, pre_y)]
	"""TODO: fix the case if there are multiple y values for the same x coordinate"""
	for x_scan in np.arange(pre_x, max(x), step):
		y_scan = f(x_scan)
		# y_scan = interpolate.splev(x_scan, tck)
		if(distance.euclidean([pre_x,pre_y],[x_scan, y_scan]) >= d):
			interpolated_points.append(Point(x_scan, y_scan))
			pre_x = x_scan
			pre_y = y_scan # update pre point

	print "end of interpolation of X, Y for one trajectory:", len(interpolated_points)
	plt.plot([point.x for point in interpolated_points], [point.y for point in interpolated_points])
	if(not plt.gca().yaxis_inverted()):
		plt.gca().invert_yaxis()
	plt.show()

### Start of Grid Interpolation Approach ### 
def findGridPointNearestTrajectoryPos(trajectory, grid_x_index, grid_y_index, grid_x, grid_y):
	x_coord = grid_x[grid_x_index][grid_y_index]
	y_coord = grid_y[grid_x_index][grid_y_index]
	grid_score = distanceScore(x_coord, y_coord, trajectory[0][data_dict_x_y_coordinate["x"]], trajectory[0][data_dict_x_y_coordinate["y"]])
	nearest_pos = 0
	for k in range(1, len(trajectory)):
		cur_distance_score = distanceScore(x_coord, y_coord, trajectory[k][data_dict_x_y_coordinate["x"]], trajectory[k][data_dict_x_y_coordinate["y"]])
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
		x = trajectory[i][data_dict_x_y_coordinate["x"]]
		y = trajectory[i][data_dict_x_y_coordinate["y"]]
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
			grid_score[i][j] = distanceScore(grid_x[i][j], grid_y[i][j], trajectory[0][data_dict_x_y_coordinate["x"]], trajectory[0][data_dict_x_y_coordinate["y"]])
			nearest_pos = 0
			for k in range(1, len(trajectory)):
				cur_distance_score = distanceScore(grid_x[i][j], grid_y[i][j], trajectory[k][data_dict_x_y_coordinate["x"]], trajectory[k][data_dict_x_y_coordinate["y"]])
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
			if(distanceScore(grid_x[i][j], grid_y[i][j], origin[data_dict_x_y_coordinate["x"]], origin[data_dict_x_y_coordinate["y"]]) < \
				distanceScore(grid_x[start_grid_x][start_grid_y], grid_y[start_grid_x][start_grid_y], origin[data_dict_x_y_coordinate["x"]], origin[data_dict_x_y_coordinate["y"]])):
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
	# while( pos < len(trajectory) and (not allGridVisited(visited) ) and distanceScore(grid_x[cur_grid_x][cur_grid_y], grid_y[cur_grid_x][cur_grid_y], end[data_dict_x_y_coordinate["x"]], end[data_dict_x_y_coordinate["y"]]) > 0.1):
	# 	next_score = MAX_FLOAT
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
	# 					this_option_score = MAX_FLOAT
	# 				elif(visited[i][j] == 1): # if visited already
	# 					this_option_score = MAX_FLOAT
	# 				elif(GRID_TO_NEAREST_POS["{i}_{j}".format(i = i, j = j)] < pos): # if corresponds to a previous data pos, don't wanna go back either
	# 					this_option_score = MAX_FLOAT
	# 				else:
	# 					this_option_score = grid_score[i][j]
	# 				# update next_score, next_grid_x, next_grid_y if score smaller
	# 				if(this_option_score < next_score):
	# 					next_score = this_option_score
	# 					next_grid_x = i
	# 					next_grid_y = j
	# 	# after checking up, left, right, down
	# 	if(next_score == MAX_FLOAT):
	# 		break; # still max float, can not populate anymore
	# 	else:
	# 		interpolated_trajectory.append(createTrajectoryPointRecordWithXY(next_grid_x, next_grid_y, grid_x, grid_y)) # append to result
	# 		visited[next_grid_x][next_grid_y] = 1 # mark as visited
	# 		cur_grid_x = next_grid_x
	# 		cur_grid_y = next_grid_y # update cur_grid_x, cur_grid_y
	# 		pos = GRID_TO_NEAREST_POS["{i}_{j}".format(i =next_grid_x , j = next_grid_y)]

	### Algo2 Go for the nearest next data pos : might work after initial cleaning of trajectory###
	# while( pos + 1 < len(trajectory) and (not allGridVisited(visited) ) and distanceScore(grid_x[cur_grid_x][cur_grid_y], grid_y[cur_grid_x][cur_grid_y], end[data_dict_x_y_coordinate["x"]], end[data_dict_x_y_coordinate["y"]]) > 0.05):
	# 	next_pos_point = trajectory[pos + 1]
	# 	next_score = MAX_FLOAT
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
	# 					this_option_score = distanceScore(grid_x[i][j], grid_y[i][j], next_pos_point[data_dict_x_y_coordinate["x"]], next_pos_point[data_dict_x_y_coordinate["y"]])
	# 					# visited[i][j] = 1 # mark as visited
	# 					if(this_option_score < next_score): # find the index with minimum distance to the next point
	# 						next_score = this_option_score
	# 						next_grid_x = i
	# 						next_grid_y = j

	# 	if(next_score == MAX_FLOAT):
	# 		break;
	# 	else:
	# 		interpolated_trajectory.append(createTrajectoryPointRecordWithXY(next_grid_x, next_grid_y, grid_x, grid_y)) # append to result
	# 		visited[next_grid_x][next_grid_y] = 1 # mark as visited
	# 		pos = pos + 1 if (GRID_TO_NEAREST_POS["{i}_{j}".format(i = next_grid_x, j = next_grid_y)] == pos + 1) else pos # update data pos if needed
	# 		# pos = pos + 1 if (distanceScore(grid_x[next_grid_x][next_grid_y], grid_y[next_grid_x][next_grid_y], \
	# 		# next_pos_point[data_dict_x_y_coordinate["x"]], next_pos_point[data_dict_x_y_coordinate["y"]]) < 0.1) else pos # update data pos if needed
	# 		cur_grid_x = next_grid_x
	# 		cur_grid_y = next_grid_y # update cur_grid_x, cur_grid_y


	### Algo3 ###
	while( pos + 1 < len(trajectory) and (not allGridVisited(visited) ) and distanceScore(grid_x[cur_grid_x][cur_grid_y], grid_y[cur_grid_x][cur_grid_y], end[data_dict_x_y_coordinate["x"]], end[data_dict_x_y_coordinate["y"]]) > 0.1):
		next_pos_point = trajectory[pos + 1]
		next_score = MAX_FLOAT
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
							this_option_score = distanceScore(grid_x[i][j], grid_y[i][j], next_pos_point[data_dict_x_y_coordinate["x"]], next_pos_point[data_dict_x_y_coordinate["y"]])
							visited[i][j] = 1 # mark as visited
							if(this_option_score < next_score): # find the grid point with minimum distance to the next point
								next_score = this_option_score
								next_grid_x = i
								next_grid_y = j

			if(next_score == MAX_FLOAT): # if all visited, break
				break;
			else:
				interpolated_trajectory.append(createTrajectoryPointRecordWithXY(next_grid_x, next_grid_y, grid_x, grid_y)) # append to result
				pos = pos + 1 if (GRID_TO_NEAREST_POS["{i}_{j}".format(i = next_grid_x, j = next_grid_y)] == pos + 1) else pos # update data pos if needed
				cur_grid_x = next_grid_x
				cur_grid_y = next_grid_y # update cur_grid_x, cur_grid_y
				print "Case1: cur_grid_x coord:", grid_x[cur_grid_x][cur_grid_y], ", cur_grid_y coord:", grid_y[cur_grid_x][cur_grid_y], ", pos: ", GRID_TO_NEAREST_POS["{i}_{j}".format(i =cur_grid_x , j = cur_grid_y)], \
				", pos data coordinate: (", trajectory[pos][data_dict_x_y_coordinate["x"]], ",", trajectory[pos][data_dict_x_y_coordinate["y"]], ")"

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
							this_option_score = MAX_FLOAT
						elif(visited[i][j] == 1): # if visited already
							this_option_score = MAX_FLOAT
						elif(GRID_TO_NEAREST_POS["{i}_{j}".format(i = i, j = j)] < pos): # if corresponds to a previous data pos, don't wanna go back either
							this_option_score = MAX_FLOAT
						else:
							this_option_score = grid_score[i][j]
						if(this_option_score < next_score): # update next_score, next_grid_x, next_grid_y if score smaller
							next_score = this_option_score
							next_grid_x = i
							next_grid_y = j

			# after checking up, left, right, down
			if(next_score == MAX_FLOAT):
				break; # still max float, can not populate anymore
			else:
				visited[next_grid_x][next_grid_y] = 1
				interpolated_trajectory.append(createTrajectoryPointRecordWithXY(next_grid_x, next_grid_y, grid_x, grid_y)) # append to result
				cur_grid_x = next_grid_x
				cur_grid_y = next_grid_y # update cur_grid_x, cur_grid_y
				pos = GRID_TO_NEAREST_POS["{i}_{j}".format(i =next_grid_x , j = next_grid_y)] 
				print "Case2: cur_grid_x coord:", grid_x[cur_grid_x][cur_grid_y], ", cur_grid_y coord:", grid_y[cur_grid_x][cur_grid_y], ", pos: ", GRID_TO_NEAREST_POS["{i}_{j}".format(i =cur_grid_x , j = cur_grid_y)], \
				", pos data coordinate: (", trajectory[pos][data_dict_x_y_coordinate["x"]], ",", trajectory[pos][data_dict_x_y_coordinate["y"]], ")"

	print "len(interpolated_trajectory):", len(interpolated_trajectory)
	return np.asarray(interpolated_trajectory)

def geographicalTrajetoryInterpolation(trajectories_x_y_coordinate):
	# d = 0.1 # take a point every 100 metre

	interpolated_trajectories_x_y_coordinate = []
	for i in range(0, len(trajectories_x_y_coordinate)):
		# interpolated_trajectories_x_y_coordinate.append(interpolateTrajectorypointsGeographical(trajectories_x_y_coordinate[i],d))
		interpolated_trajectories_x_y_coordinate.append(interpolateGeographicalGrid(trajectories_x_y_coordinate[i]))
	print "in geographicalTrajetoryInterpolation interpolated_trajectories_x_y_coordinate.shape:", np.asarray(interpolated_trajectories_x_y_coordinate).shape
	return interpolated_trajectories_x_y_coordinate


def notNoise(prevPosition, nextPosition, MAX_SPEED):
	"""
	: MAX_SPEED is in knot
	"""
	dt = nextPosition[dataDict["ts"]] - prevPosition[dataDict["ts"]] # in secs
	dx, dy = LatLonToXY(prevPosition[dataDict["latitude"]], prevPosition[dataDict["longtitude"]], nextPosition[dataDict["latitude"]], nextPosition[dataDict["longtitude"]])
	return (np.linalg.norm([dx,dy],2) < (dt * convertKnotToMeterPerSec(MAX_SPEED))/1000.0) 

def extractAndPlotTrajectories(data, originLatitude, originLongtitude, endLatitude, endLongtitude, studyWindowLen = 8.0, timeWindow = 24, show = True):
	"""
	:studyWindowLen is in Km, timeWindow is in hours (usually AIS data is on daily base tracking)
	"""
	maxSpeed = 0
	for i in range(0, data.shape[0]):
		speed_over_ground = data[i][dataDict["speed_over_ground"]]
		if(speed_over_ground > maxSpeed and speed_over_ground != 102.3): #1023 indicates speed not available
			maxSpeed = speed_over_ground
	print "\n\nmaxSpeed:",maxSpeed, ", in km/h:",convertKnotToMeterPerSec(maxSpeed)


	print "origin lat:", originLatitude, ", origin Lon:",originLongtitude
	print "end lat:", endLatitude, ", end Lon:",endLongtitude
	max_x, max_y = LatLonToXY(originLatitude, originLongtitude, endLatitude, endLongtitude)
	print "max_x:", max_x, ",max_y:", max_y
	
	
	# timeWindow = studyWindowLen/(0.1*knotToKmPerhour) # based on a a slowest 0.1 knot speed estimation
	# assume a 50 km * 50 km study window, assume the vessel speed is around 8 knot, 10 km/h, then within half an hour time window, the study window should be 5* 5 km
	vesselLatitude, vesselLongtitude = XYToLatLonGivenOrigin(originLatitude, originLongtitude, studyWindowLen,-studyWindowLen)
	print "furthurst possible vessel Lat:",vesselLatitude, ", furthurst possible vessel Lon:",vesselLongtitude

	i = 0
	trajectories = []
	OD_trajectories = [] # origin destination endpoints trajectory
	while(i+1< data.shape[0]):
		currentPosition = data[i]
		if(nearOrigin(originLatitude, originLongtitude, currentPosition[dataDict["latitude"]], currentPosition[dataDict["longtitude"]], 0.005)):
			startTS = currentPosition[dataDict["ts"]] # the ts of the start point
			thisTrajectory = []
			
			prevPosition = data[i]
			nextPosition = data[i+1] # get next point inline
			# skip data if it is still around origin (noise)
			while(nearOrigin(currentPosition[dataDict["latitude"]], currentPosition[dataDict["longtitude"]], nextPosition[dataDict["latitude"]], nextPosition[dataDict["longtitude"]], NEIGHBOURHOOD)):
				i += 1
				nextPosition = data[i+1]
				prevPosition = data[i]
			# update start point and start time
			currentPosition = prevPosition
			startTS = currentPosition[dataDict["ts"]] # the ts of the updated start point
			thisTrajectory.append(currentPosition)			

			# if satisfy condition, append to this trajectory
			while(withinStudyWindow(originLatitude, originLongtitude, vesselLatitude,vesselLongtitude,nextPosition[dataDict["latitude"]], nextPosition[dataDict["longtitude"]]) and withinTimeConstriant(startTS, nextPosition[dataDict["ts"]], timeWindow)):
			# while(withinStudyWindow(originLatitude, originLongtitude, vesselLatitude,vesselLongtitude,nextPosition[dataDict["latitude"]], nextPosition[dataDict["longtitude"]])):
			# while(withinTimeConstriant(startTS, nextPosition[dataDict["ts"]], timeWindow)):
				"""
				Check if the next point is noise, somtimes, a noise point might be able to fulfill constraint put by maxSpeed
				"""	
				if(notNoise(prevPosition, nextPosition, maxSpeed)):
					thisTrajectory.append(nextPosition)
					i += 1
					if((i+1) >= data.shape[0]):
						break # reaching end of the file
					if(nearOrigin(endLatitude, endLongtitude, nextPosition[dataDict["latitude"]], nextPosition[dataDict["longtitude"]], NEIGHBOURHOOD)):
						print "break because reach end point"
						OD_trajectories.append(np.copy(np.array(thisTrajectory)))
						break
				prevPosition = nextPosition
				nextPosition = data[i+1]
			# out of the while loop, meaning the end of this trajectory with the constraint of current study window and time frame
			if(len(thisTrajectory) > 1): # impose a threshold, must contain at least certain number of points to be a trajectory
				trajectories.append(thisTrajectory)
		i += 1

	print "number of trajectories captured:",len(trajectories)
	if(len(trajectories) > 0):
		print "number of points in the trajectories[0]:", len(trajectories[0])
	# plot statistics of the captured trajectory: speed profile, acceleration profile from start point to end point 
	# plotTrajectoryProfiles(pathToSave, filename[0:filename.find(".")] + "trajectoryProfiles", trajectories, dataDict, [originLatitude, originLongtitude], [endLatitude,endLongtitude])

	# plot totalCoordinates of all points in all trajectories captured
	trajectories_x_y_coordinate = np.copy(trajectories)

	totalCoordinates = []
	for i in range(0, len(trajectories)):
		thisTrajectoryCoordinates = []

		for j in range(0, len(trajectories[i])):
			x, y = LatLonToXY(originLatitude, originLongtitude, trajectories[i][j][dataDict["latitude"]], trajectories[i][j][dataDict["longtitude"]])
			totalCoordinates.append([x,y])
			thisTrajectoryCoordinates.append([x,y])
			trajectories_x_y_coordinate[i][j][data_dict_x_y_coordinate['y']] = y # y corresponds to latitude value
			trajectories_x_y_coordinate[i][j][data_dict_x_y_coordinate['x']] = x # x corresponds to longtitude value

		thisTrajectoryCoordinates = np.asarray(thisTrajectoryCoordinates)
		
		plt.plot(thisTrajectoryCoordinates[0:len(thisTrajectoryCoordinates),0], thisTrajectoryCoordinates[0:len(thisTrajectoryCoordinates),1])

	totalCoordinates = np.asarray(totalCoordinates)
	print "totalCoordinates.shape:", totalCoordinates.shape
	plt.gca().invert_yaxis()
	# plt.savefig("{p}/{i}x{i}km_studywindow_{t}_timewindow_{f}_origin_{lat}_{lon}_trajectories.png".format(p = pathToSave, t = timeWindow, i = studyWindowLen, f = filename[0:filename.find(".")], lat = originLatitude, lon = originLongtitude))
	if(show):
		plt.show()

	OD_trajectories_lat_lon = copy.deepcopy(OD_trajectories)
	#Set X,Y coordinate for the OD_trajectories
	for i in range(0, len(OD_trajectories)):
		for j in range(0, len(OD_trajectories[i])):
			x, y = LatLonToXY(originLatitude, originLongtitude, OD_trajectories[i][j][dataDict["latitude"]], OD_trajectories[i][j][dataDict["longtitude"]])
			OD_trajectories[i][j][data_dict_x_y_coordinate["y"]] = y
			OD_trajectories[i][j][data_dict_x_y_coordinate["x"]] = x

	return trajectories_x_y_coordinate, OD_trajectories, OD_trajectories_lat_lon

def extractTrajectoriesUntilOD(data, originTS, originLatitude, originLongtitude, endTS, endLatitude, endLongtitude, show = True, save = False, clean = False, fname = ""):
	"""
	returns: OD_trajectories: in x,y coordinate;
			 OD_trajectories_lat_lon: in lat, lon coordinate;
	"""
	
	maxSpeed = 0
	for i in range(0, data.shape[0]):
		speed_over_ground = data[i][dataDict["speed_over_ground"]]
		if(speed_over_ground > maxSpeed and speed_over_ground != 102.3): #1023 indicates speed not available
			maxSpeed = speed_over_ground
	
	OD_trajectories = [] # origin destination endpoints trajectory
	i = 0
	while(i< data.shape[0]):
		cur_pos = data[i]
		if(nearOrigin(originLatitude, originLongtitude, cur_pos[dataDict["latitude"]], cur_pos[dataDict["longtitude"]], thresh = 0.0) and cur_pos[dataDict["ts"]] == originTS): # must be exact point
			this_OD_trajectory = []
			this_OD_trajectory.append(cur_pos)
			i += 1
			while(i < data.shape[0] and (not nearOrigin(endLatitude, endLongtitude, data[i][dataDict["latitude"]], data[i][dataDict["longtitude"]], thresh = 0.0))):
				this_OD_trajectory.append(data[i])
				i += 1
			if(i < data.shape[0]):
				this_OD_trajectory.append(data[i])
			this_OD_trajectory = np.asarray(this_OD_trajectory) # make it to be an np 2D array
			# box/radius approach in cleaning of points around origin
			j = 1
			print "checking points around origin:", j
			while(j < this_OD_trajectory.shape[0] and nearOrigin(originLatitude, originLongtitude, this_OD_trajectory[j][dataDict["latitude"]], this_OD_trajectory[j][dataDict["longtitude"]], thresh = NEIGHBOURHOOD_ORIGIN)):
				j += 1
			print "last point around origin:", j
			this_OD_trajectory_around_origin = this_OD_trajectory[0:j]
			this_OD_trajectory_mean_origin = np.mean(this_OD_trajectory_around_origin, axis = 0)
			print "mean start point x,y : ", LatLonToXY(originLatitude, originLongtitude, this_OD_trajectory_mean_origin[dataDict["latitude"]], this_OD_trajectory_mean_origin[dataDict["longtitude"]])
			OD_trajectories.append(np.insert(this_OD_trajectory[j:],0,this_OD_trajectory_mean_origin, axis = 0))
			break  # only one trajectory per pair OD, since OD might be duplicated
		i += 1

	OD_trajectories = np.array(OD_trajectories)
	OD_trajectories_lat_lon = copy.deepcopy(OD_trajectories)
	for i in range(0, len(OD_trajectories)):
		for j in range(0, len(OD_trajectories[i])):
			x, y = LatLonToXY(originLatitude, originLongtitude, OD_trajectories[i][j][dataDict["latitude"]], OD_trajectories[i][j][dataDict["longtitude"]])
			OD_trajectories[i][j][data_dict_x_y_coordinate["y"]] = y
			OD_trajectories[i][j][data_dict_x_y_coordinate["x"]] = x
		# plotting purpose
		plt.scatter(OD_trajectories[i][0:len(OD_trajectories[i]),data_dict_x_y_coordinate["x"]], \
			OD_trajectories[i][0:len(OD_trajectories[i]),data_dict_x_y_coordinate["y"]])
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
	dx, dy = LatLonToXY(point1[dataDict["latitude"]], point1[dataDict["longtitude"]], point2[dataDict["latitude"]], point2[dataDict["longtitude"]])
	return (np.linalg.norm([dx,dy],2))

def alreadyInEndpoints(endpoints, target):
	for i in range(0, len(endpoints)):
		if(getDistance(endpoints[i], target) < NEIGHBOURHOOD_ENDPOINT):
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
			if(getDistance(start_point, next_point) > NEIGHBOURHOOD_ENDPOINT):
				# print "find a point that is out of NEIGHBOURHOOD:", datetime.datetime.fromtimestamp(start_point[dataDict["ts"]]).strftime('%Y-%m-%dT%H:%M:%SZ'), \
				datetime.datetime.fromtimestamp(next_point[dataDict["ts"]]).strftime('%Y-%m-%dT%H:%M:%SZ')
				break;
			i += 1

		next_point = data[i] # back track to get the last data point that is still near start_point
		if(i - start_index > 0 and next_point[dataDict["ts"]] - start_point[dataDict["ts"]] > STAYTIME_THRESH):
			# if(not alreadyInEndpoints(endpoints, start_point)): # But should not do the check if the returned endpoints are used to extract trajectories between them
			# print "append since stay more than half hour:", datetime.datetime.fromtimestamp(start_point[dataDict["ts"]]).strftime('%Y-%m-%dT%H:%M:%SZ')
			if(len(endpoints) == 0 or (not (endpoints[len(endpoints) - 1] == start_point).all())): # if not just appended
				endpoints.append(start_point)

		# TODO: is there a boundary informaiton on the area that AIS can detect?
		elif((i+1) != data.shape[0]): # if still has a next postion which is outside neighbour
			next_point_outside_neighbour = data[i+1]
			if(next_point_outside_neighbour[dataDict["ts"]] - start_point[dataDict["ts"]] > 24*3600): # if start of new trajectory at a new position, or after one day
			# if(next_point_outside_neighbour[dataDict["ts"]] - start_point[dataDict["ts"]] > \
			# 	getDistance(start_point, next_point_outside_neighbour)/ \
			# 	(1*knotToKmPerhour) * 3600): #maximum knot
				# print "append both, since start of new trajectory:", datetime.datetime.fromtimestamp(next_point[dataDict["ts"]]).strftime('%Y-%m-%dT%H:%M:%SZ'), datetime.datetime.fromtimestamp(next_point_outside_neighbour[dataDict["ts"]]).strftime('%Y-%m-%dT%H:%M:%SZ')
				if(len(endpoints) == 0 or (not (endpoints[len(endpoints) - 1] == next_point).all())): # if not just appended
					endpoints.append(next_point)
				if(len(endpoints) == 0 or (not (endpoints[len(endpoints) - 1] == next_point_outside_neighbour).all())): # if not just appended
					endpoints.append(next_point_outside_neighbour)

		elif((i+1) == data.shape[0]):
			# print "append since last point", datetime.datetime.fromtimestamp(start_point[dataDict["ts"]]).strftime('%Y-%m-%dT%H:%M:%SZ')
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
			lat, lon = XYToLatLonGivenOrigin(originLatitude, originLongtitude, listOfTrajectories[i][j][data_dict_x_y_coordinate["x"]], listOfTrajectories[i][j][data_dict_x_y_coordinate["y"]])
			listOfTrajectories[i][j][dataDict["latitude"]] = lat
			listOfTrajectories[i][j][dataDict["longtitude"]] = lon
	return listOfTrajectories

def convertListOfTrajectoriesToXY(originLatitude, originLongtitude, listOfTrajectories):
	for i in range(0, len(listOfTrajectories)):
		for j in range(0, len(listOfTrajectories[i])):
			x, y = LatLonToXY(originLatitude, originLongtitude, listOfTrajectories[i][j][dataDict["latitude"]], listOfTrajectories[i][j][dataDict["longtitude"]])
			listOfTrajectories[i][j][data_dict_x_y_coordinate["y"]] = y
			listOfTrajectories[i][j][data_dict_x_y_coordinate["x"]] = x
	return listOfTrajectories

def isErrorTrajectory(trajectory, center_lat_sg, center_lon_sg):
	"""
	Checks if the give trajectory is too far from the Port Center
	"""
	for i in range(0, len(trajectory)):
		dx, dy = LatLonToXY (trajectory[i][dataDict["latitude"]],trajectory[i][dataDict["longtitude"]],center_lat_sg, center_lon_sg)
		if(np.linalg.norm([dx, dy], 2) > MAX_DISTANCE_FROM_SG):
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

def clusterPurity(groundTruth, clusterLabel, n_cluster):
	purity = np.zeros(n_cluster)
	startingLabel = min(clusterLabel)
	for index in range(startingLabel,startingLabel+n_cluster):
		true_label_distribution = groundTruth[clusterLabel == index]
		mode,modeArr= findMode(true_label_distribution)
		purity[index - startingLabel] = float(mode)/len(true_label_distribution)
	return purity

def trajectoryDissimilarityL2(t1, t2):
	i = 0
	j = 0
	dissimilarity = 0.0
	while(i < len(t1) and j < len(t2)):
		dissimilarity += DIST.euclidean(t1[i], t2[j])
		i += 1
		j += 1
	# only one of the following loops will be entered
	while(i < len(t1)):
		dissimilarity += DIST.euclidean(t1[i], t2[j - 1]) # j -1 to get the last point in t2
		i += 1

	while(j < len(t2)):
		dissimilarity += DIST.euclidean(t1[i - 1], t2[j])
		j += 1
	return dissimilarity

def withinClassVariation(class_trajectories, distance_matrix, metric_func):
	"""
	class_trajectories: the trajectories within this one class
	return: W for one class only
	"""
	mean = getMeanTrajecotoryWithinClass(class_trajectories)
	variation = 0
	for i in range(0,len(class_trajectories)):
		variation += metric_func(class_trajectories[i], mean)
	return variation

def betweenClassVariation(class_trajectories_dict, distance_matrix, metric_func):
	class_centroids = []
	for class_label, trajectories in class_trajectories_dict.iteritems():
		class_centroids.append(getMeanTrajecotoryWithinClass(trajectories))
	between_class_mean = getMeanTrajecotoryWithinClass(class_centroids) # get the mean of the class centroids of different clusters
	variation = 0
	class_weights = np.ones(len(class_centroids))
	for i in range(0, len(class_centroids)):
		variation += metric_func(class_centroids[i], between_class_mean) * class_weights[i] # here assume uniform weights among all classes
	return variation

def CHIndex(cluster_label, distance_matrix, data, metric_func = trajectoryDissimilarityL2):
	"""
	cluster_label: length n
	distance_matrix: shape (n, n)
	data: length n
	returns: CH(K) = ( B(K)/(k-1) )/( W(k)/(n-k) ), where K is number of clusters
	"""
	assert len(cluster_label) == len(data), "data and cluster_label length should be the same"
	class_trajectories_dict = {} # a dictionary of class lable to its cluster
	for i in range(0, len(cluster_label)):
		class_label = cluster_label[i]
		if(not class_label in class_trajectories_dict):
			class_trajectories_dict[class_label] = []
		class_trajectories_dict[class_label].append(data[i])
	W = 0 # within class variation, summed over classes
	for class_label, trajectories in class_trajectories_dict.iteritems():
		W += withinClassVariation(trajectories, distance_matrix, metric_func) # get the W for one class
	B = betweenClassVariation(class_trajectories_dict, distance_matrix, metric_func) # between class variation
	K = len(set(cluster_label))
	n = len(data)

	return  (B/ (K-1))/(W/(n - K))

def getMeanTrajecotoryPointAtIndex(trajectories, index):
	mean = np.zeros(len(data_dict_x_y_coordinate))
	count = 0
	for i in range(0, len(trajectories)):
		if(index < len(trajectories[i])):
			mean += np.asarray(trajectories[i][index])
			count += 1
	if(count != 0):
		mean = mean/count
	return mean

def getMeanTrajecotoryWithinClass(class_trajectories):
	max_length_index, max_length = max(enumerate([len(t) for t in class_trajectories]), key = operator.itemgetter(1))
	mean_trajectory = np.zeros(shape = (max_length, len(data_dict_x_y_coordinate)))
	for i in range(0, max_length):
		mean_trajectory[i] = getMeanTrajecotoryPointAtIndex(class_trajectories, i)
	return mean_trajectory

def getTrajectoryDistanceMatrix(trajectories, metric_func = trajectoryDissimilarityL2):
	n = len(trajectories)
	distance_matrix = np.zeros(shape = (n,n))
	for i in range(0, n):
		distance_matrix[i][i] = 0.0
		for j in range(i+1, n):
			distance_matrix[i][j] = metric_func(trajectories[i], trajectories[j])
			distance_matrix[j][i] = distance_matrix[i][j]
	return distance_matrix
			
def clusterTrajectories(trajectories, fname, path, metric_func = trajectoryDissimilarityL2):
	"""
	trajectories: the trajectories better to be in XY coordinates
	"""
	distance_matrix = getTrajectoryDistanceMatrix(trajectories, metric_func)
	print "distance_matrix:\n", distance_matrix
	v = DIST.squareform(distance_matrix)
	cluster_result = HAC.linkage(v, method = "average")
	dg = HAC.dendrogram(cluster_result)
	plt.xlabel("cluster_dengrogram_{fname}".format(fname = fname))
	plt.savefig("{path}/cluster_dengrogram_{fname}.png".format(fname = fname, path = path))
	plt.show()
	
	# Get the optimal number of clusters
	MIN_NUM_CLUSTER = 2
	MAX_NUM_CLUSTER = 300
	opt_num_cluster = MIN_NUM_CLUSTER
	opt_CH_index = MIN_FLOAT
	opt_cluster_label = None

	CH_indexes = []
	cluster_labels = []
	for i in range(MIN_NUM_CLUSTER, MAX_NUM_CLUSTER):
		this_cluster_label = HAC.fcluster(Z= cluster_result, t= i,criterion='maxclust')
		this_CH_index = CHIndex(this_cluster_label, distance_matrix, trajectories, metric_func)
		CH_indexes.append(this_CH_index)
		cluster_labels.append(this_cluster_label)
		if(this_CH_index > opt_CH_index):
			opt_CH_index = this_CH_index
			opt_num_cluster = i
			opt_cluster_label = this_cluster_label
		print "\nHAC Cluster label:\n", this_cluster_label
		print "number of labels by HAC:", len(set(this_cluster_label)), ";starting label is:", min(this_cluster_label)
		print "n_clusters				CH index"
		print i,"				", this_CH_index 

	plt.plot(range(MIN_NUM_CLUSTER, MAX_NUM_CLUSTER), CH_indexes)
	plt.xlabel("CH_indexes range plot")
	plt.show()
	return opt_cluster_label

def main():
	# originLongtitude = 62245670/geoScale
	# originLatitude = 718208/geoScale

	# filename = "1000019.npz"
	# filename = "1000019_extreme.npz"
	# filename = "9261126.npz"
	# filename = "3916119.npz"
	filename = "aggregateData.npz"
	# path = "cleanedData"
	# pathToSave = "cleanedData/graphs"
	path = "tankers/cleanedData"
	pathToSave = "tankersGraph"
	data = 	loadArray("{p}/{f}".format(p = path, f=filename))
	root_folder = "tankers"

	"""
	Test Clustering
	"""
	trajectories_to_cluster = writeToCSV.loadData(root_folder + "/" + "all_OD_trajectories.npz")
	print type(trajectories_to_cluster)
	print len(trajectories_to_cluster)
	trajectories_to_cluster = list(trajectories_to_cluster)
	all_OD_trajectories_XY = convertListOfTrajectoriesToXY(CENTER_LAT_SG, CENTER_LON_SG, trajectories_to_cluster) # convert Lat, Lon to XY for displaying
	opt_cluster_label = clusterTrajectories(all_OD_trajectories_XY, "10_tankers", pathToSave)
	print "opt_cluster_label:", opt_cluster_label
	print "opt_num_cluster:", len(set(opt_cluster_label))
	# print "distance between 1 and 4, should be quite small:", trajectoryDissimilarityL2(all_OD_trajectories_XY[1], all_OD_trajectories_XY[4])
	# print "distance between 0 and 4, should be quite large:", trajectoryDissimilarityL2(all_OD_trajectories_XY[0], all_OD_trajectories_XY[4])
	# print "matrix:\n", getTrajectoryDistanceMatrix(all_OD_trajectories_XY, metric_func = trajectoryDissimilarityL2)
	# plotListOfTrajectories(all_OD_trajectories_XY, show = True, clean = True, save = False, fname = "") # TODO: remove error trajectories that are too far from Singapore
	raise ValueError("purpose stop of the testing clustering procedure")


	"""
	plot out the value space of the features, speed, accelerations, etc
	"""
	# plotFeatureSpace(data, dataDict)
	# raise ValueError("For plotting feature space only")


	"""
	Extract endpoints;
	TODO: Further cleaning of the data, sometimes the ship 'flys' around and out of a confined study window, need to tackle this situation
	"""
	filenames = ["8514019.csv", "9116943.csv", "9267118.csv", "9443140.csv", "9383986.csv", "9343340.csv", "9417464.csv", "9664225.csv", "9538440.csv", "9327138.csv"]
	endpoints = None
	all_OD_trajectories = []
	# filenames = ["9664225.csv"]
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
			originLatitude = this_vessel_endpoints[s][dataDict["latitude"]]
			originLongtitude = this_vessel_endpoints[s][dataDict["longtitude"]]
			origin_ts = this_vessel_endpoints[s][dataDict["ts"]]

			endLatitude = this_vessel_endpoints[s + 1][dataDict["latitude"]]
			endLongtitude = this_vessel_endpoints[s + 1][dataDict["longtitude"]]	
			end_ts = this_vessel_endpoints[s + 1][dataDict["ts"]]
			# if there could be possibly a trajectory between theses two this_vessel_endpoints; 
			# Could do a check here or just let the extractAndPlotTrajectories return empty array
			if(end_ts - origin_ts <= 3600 * 24):
				"""Extracting trajectory between a pair of OD"""
				print "\n\nextracting endpoints between ", s, " and ", s + 1
				OD_trajectories, OD_trajectories_lat_lon = extractTrajectoriesUntilOD(\
					writeToCSV.readDataFromCSV(root_folder + "/cleanedData", filenames[i]), \
					origin_ts, originLatitude, originLongtitude, end_ts, endLatitude, endLongtitude, \
					show = False, save = False, clean = True, \
					fname = filenames[i][:filenames[i].find(".")] + "_trajectory_between_endpoint{s}_and{e}".format(s = s, e = s + 1)) # there will be one trajectory between each OD		
				print "number of trajectory points extracted : ", len(OD_trajectories[0])
				# writeToCSV.writeDataToCSV(OD_trajectories_lat_lon[0],root_folder + "/trajectories", "{filename}_trajectory_endpoint_{s}_to_{e}".format(filename = filenames[i][:filenames[i].find(".")], s = s, e = s + 1))

				"""
				Interpolation based on temporal information based trajectory, need to only consider trajectories between nicely found O-D
				"""
				# temporalTrajectoryInterpolation(OD_trajectories)

				"""
				Interpolation based on pure geographycal trajectory, ignore temporal information
				"""
				# geographicalTrajetoryInterpolation(trajectories_x_y_coordinate)
				interpolated_OD_trajectories = geographicalTrajetoryInterpolation(OD_trajectories)
				# plotListOfTrajectories(interpolated_OD_trajectories, show = False, clean = True, save = True, fname = filenames[i][:filenames[i].find(".")] + "_interpolated_algo_3final_between_endpoint{s}_and{e}".format(s = s, e = s + 1))
				# change X, Y coordinate to Lat, Lon
				interpolated_OD_trajectories_lat_lon = convertListOfTrajectoriesToLatLon(originLatitude, originLongtitude, interpolated_OD_trajectories)
				if(len(interpolated_OD_trajectories_lat_lon) > 0):
					all_OD_trajectories.append(interpolated_OD_trajectories_lat_lon[0]) # since there should be only one trajectory between each pair of OD

	assert (not endpoints is None), "No endpoints extracted from the historial data"
	print "Final endpoints.shape:", endpoints.shape
	print "number of interpolated all_OD_trajectories:", len(all_OD_trajectories)
	writeToCSV.saveData(removeErrorTrajectoryFromList(all_OD_trajectories), root_folder + "/all_OD_trajectories")

	all_OD_trajectories_XY = convertListOfTrajectoriesToXY(CENTER_LAT_SG, CENTER_LON_SG, all_OD_trajectories) # convert Lat, Lon to XY for displaying
	plotListOfTrajectories(all_OD_trajectories_XY, show = True, clean = True, save = True, fname = "tanker_all_OD_trajectories") # TODO: remove error trajectories that are too far from Singapore
	
	# endpoints = np.asarray(extractEndPoints(data))
	# writeToCSV.writeDataToCSV(endpoints,"tankers/endpoints", "aggregateData_endpoints")
	# endpoints = writeToCSV.readDataFromCSV("tankers/endpoints", "aggregateData_endpoints.csv")
	# print endpoints.shape
	
	# """
	# Find the trajectories that start near origin (within 500m radius) ;
	# within our observation studtWindowLen and within next timeWindow hours of the start
	# """
	# originLatitude = data[0][dataDict["latitude"]]
	# originLongtitude = data[0][dataDict["longtitude"]]
	# originLatitude = 718208/geoScale # for 1000019 only
	# originLongtitude = 62245670/geoScale # for 1000019 only
	
	# originLatitude = 1.2903432270023394 # test for aggregateData
	# originLongtitude = 103.72533768968383 # test for aggregateData

	# endLatitude = 1.237067419175662 # test for aggregateData
	# endLongtitude = 103.79633903503418 # test for aggregateData

	# trajectories_x_y_coordinate, OD_trajectories, OD_trajectories_lat_lon= extractAndPlotTrajectories(data, originLatitude, originLongtitude, endLatitude, endLongtitude)
	# # plotListOfTrajectories(OD_trajectories,True, True)
	# # writeToCSV.writeDataToCSV(OD_trajectories_lat_lon[2][185:],"tankers/trajectories", "aggregateData_OD_trajectory_{i}_afterFistHour3600".format(i = 2))
	# # for i in range(0, len(OD_trajectories_lat_lon)):
	# 	# writeToCSV.writeDataToCSV(OD_trajectories_lat_lon[i],"tankers/trajectories", "aggregateData_OD_trajectory_{i}".format(i = i))


if __name__ == "__main__":
	main()

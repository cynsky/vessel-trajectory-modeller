import os
import numpy as np
import math
import scipy.io as sio
import scipy.sparse as sp
from scipy.sparse import csc_matrix
import csv
import matplotlib.pyplot as plt
import datetime
import time
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
from collections import namedtuple

"""
CONSTANTS
"""

dataDict = {
"navigation_status":0,
"rate_of_turn":1,
"speed_over_ground":2,
"latitude":3,
"longitude":4,
"course_over_ground":5,
"true_heading":6,
"ts":7,
"mmsi":8
}

data_dict_x_y_coordinate = {
"navigation_status":0,
"rate_of_turn":1,
"speed_over_ground":2,
"y":3,
"x":4,
"course_over_ground":5,
"true_heading":6,
"ts":7,
"mmsi":8
}

CENTER_LAT_SG = 1.2

CENTER_LON_SG = 103.8

GEOSCALE = 600000.0

KNOTTOKMPERHOUR = 1.85200

NEIGHBOURHOOD = 0.2

NEIGHBOURHOOD_ENDPOINT = 0.1

NEIGHBOURHOOD_ORIGIN = 0.1

STAYTIME_THRESH = 1800 # 1 hour

MAX_FLOAT = sys.float_info.max

MAX_DISTANCE_FROM_SG = 100 # 100 km

BOUNDARY_TIME_DIFFERENCE = 7 * 3600

UNKNOWN_COURSE_OVER_GROUND = 3600

SpeedDistanceTuple = namedtuple('SpeedDistanceTuple', ['speed', 'distance'])

ClusterCentroidTuple = namedtuple('ClusterCentroidTuple', ['cluster', 'centroid'])

### For detection inland error data, adjust to specify the cleaning extent###
IN_LAND_UPPER_LAT = 1.391693577440254
IN_LAND_LOWER_LAT = 1.3354045918802477
IN_LAND_WEST_LON = 103.85238647460938
IN_LAND_EAST_LON = 103.89976501464844



def queryPath(path):
	"""
	checks if the given path exisits, if not existing, create and return it; else, just echo back it
	"""
	if(not os.path.isdir("./{path}".format(
		path = path))):
		os.makedirs("./{path}".format(
			path = path))
	return path


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
	KNOTTOKMPERHOUR = 1.85200
	KmPerhourToMetrePerSec = 1/3.6
	return knot * KNOTTOKMPERHOUR * KmPerhourToMetrePerSec

def notNoise(prevPosition, nextPosition, MAX_SPEED):
	"""
	MAX_SPEED: is in knot;
	returns: True if the distance between prevPosition and nextPosition can not be attained, i.e., noise data
	"""
	dt = nextPosition[dataDict["ts"]] - prevPosition[dataDict["ts"]] # in secs
	dx, dy = LatLonToXY(prevPosition[dataDict["latitude"]], prevPosition[dataDict["longitude"]], nextPosition[dataDict["latitude"]], nextPosition[dataDict["longitude"]])
	return (np.linalg.norm([dx,dy],2) < (dt * convertKnotToMeterPerSec(MAX_SPEED))/1000.0) 


def isInlandPoint(lat, lon):
	if(lat >= IN_LAND_LOWER_LAT and \
		lat <= IN_LAND_UPPER_LAT and \
		lon >= IN_LAND_WEST_LON and \
		lon <= IN_LAND_EAST_LON):
		return True
	else:
		return False

def isErrorTrajectory(trajectory, center_lat_sg, center_lon_sg):
	"""
	Checks if the give trajectory is too far from the Port Center, or only contains less than one trajectory point
	"""
	if(len(trajectory) <= 1):
		return True

	for i in range(0, len(trajectory)):
		lat = trajectory[i][dataDict["latitude"]]
		lon = trajectory[i][dataDict["longitude"]]
		dx, dy = LatLonToXY (lat, lon, center_lat_sg, center_lon_sg)
		if(np.linalg.norm([dx, dy], 2) > MAX_DISTANCE_FROM_SG):
			return True
		if (isInlandPoint(lat, lon)):
			return True
	return False

def removeErrorTrajectoryFromList(trajectories, center_lat_sg = 1.2, center_lon_sg = 103.8):
	"""
	note: list object passing to function will be by reference, while using np.delete(), returns a new copy thus need to return
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





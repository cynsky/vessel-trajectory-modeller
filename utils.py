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

GEOSCALE = 600000.0

KNOTTOKMPERHOUR = 1.85200

NEIGHBOURHOOD = 0.2

NEIGHBOURHOOD_ENDPOINT = 0.1

NEIGHBOURHOOD_ORIGIN = 0.1

STAYTIME_THRESH = 1800 # 1 hour

MAX_FLOAT = sys.float_info.max

MAX_DISTANCE_FROM_SG = 100 # 100 km


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
	max_X, max_Y = utils.LatLonToXY(originLatitude, originLongtitude, vesselLatitude,vesselLongtitude)
	current_X, current_Y = utils.LatLonToXY(originLatitude, originLongtitude, currentLat, currentLon)
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
	utils.KNOTTOKMPERHOUR = 1.85200
	KmPerhourToMetrePerSec = 1/3.6
	return knot * utils.KNOTTOKMPERHOUR * KmPerhourToMetrePerSec



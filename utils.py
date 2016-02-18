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
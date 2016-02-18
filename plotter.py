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


def getSpeedAccelerations(data):
	accelerations = [] # acceleration will be in m/s^2
	timeAxis = [] # time axis in seconds
	startTime = data[0][utils.dataDict["ts"]]
	distanceAxis = []
	startPoint = [data[0][utils.dataDict["latitude"]], data[0][utils.dataDict["longtitude"]]]
	for i in range (1, len(data)):
		curSpeed = data[i][utils.dataDict["speed_over_ground"]]
		prevSpeed = data[i-1][utils.dataDict["speed_over_ground"]]
		dt_secs = data[i][utils.dataDict["ts"]] - data[i - 1][utils.dataDict["ts"]]
		if(curSpeed == 102.3 or prevSpeed == 102.3):
			continue
		if(dt_secs == 0):
			continue
		accelerations.append((utils.convertKnotToMeterPerSec(curSpeed) - utils.convertKnotToMeterPerSec(prevSpeed))/float(dt_secs))
		timeAxis.append(data[i][utils.dataDict["ts"]] - startTime)
		distanceAxis.append(np.linalg.norm([utils.LatLonToXY(startPoint[0],startPoint[1], data[i][utils.dataDict["latitude"]], data[i][utils.dataDict["longtitude"]])],2))


	accelerations = np.asarray(accelerations)
	timeAxis = np.asarray(timeAxis)
	distanceAxis = np.asarray(distanceAxis)
	return accelerations, timeAxis, distanceAxis

def getSpeeds(data):
	data = np.asarray(data)
	speeds = data[:,utils.dataDict["speed_over_ground"]]
	timeAxis = []
	timeAxis.append(0)
	startTime = data[0][utils.dataDict["ts"]]
	distanceAxis = []
	distanceAxis.append(0)
	startPoint = [data[0][utils.dataDict["latitude"]], data[0][utils.dataDict["longtitude"]]]
	for i in range(1, len(data)):
		dt_secs = data[i][utils.dataDict["ts"]] - startTime
		timeAxis.append(dt_secs)
		distanceAxis.append(np.linalg.norm([utils.LatLonToXY(startPoint[0],startPoint[1], data[i][utils.dataDict["latitude"]], data[i][utils.dataDict["longtitude"]])],2))
	
	i = 0
	while(i < len(speeds)):
		if(speeds[i] == 102.3): # remove the not available ones
			speeds = np.delete(speeds, i, 0)
		else:
			i += 1

	timeAxis = np.asarray(timeAxis)
	return speeds, timeAxis, distanceAxis

def getAngularSpeeds(data):
	data = np.asarray(data)
	angularSpeeds = data[:,utils.dataDict["rate_of_turn"]]
	timeAxis = []
	timeAxis.append(0)
	startTime = data[0][utils.dataDict["ts"]]
	for i in range(1, len(data)):
		dt_secs = data[i][utils.dataDict["ts"]] - startTime
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

def plotFeatureSpace(data):
	num_bins = 50
	#1. plot out the acceleration profile
	accelerations, _ , _= getSpeedAccelerations(data);
	print accelerations.shape
	print np.amax(accelerations)
	print np.amin(accelerations)
	# plt.plot(accelerations)
	plt.hist(accelerations, num_bins, normed= True, facecolor='green', alpha=0.5, stacked = True)
	plt.ylabel('Relative frequency')
	plt.xlabel('Acceleration Profile')
	plt.show()

	#2. plot out the speed profile
	speeds, _ , _= getSpeeds(data)
	print np.amax(speeds)
	print np.amin(speeds)
	plt.hist(speeds, num_bins, normed= True, facecolor='green', alpha=0.5, stacked = True)
	plt.xlabel("speeds profile")
	plt.show()

	#3. plot out the turning rate profile
	angularSpeeds, _ = getAngularSpeeds(data)
	print np.amax(angularSpeeds)
	print np.amin(angularSpeeds)
	plt.hist(angularSpeeds, num_bins, normed= True, facecolor='green', alpha=0.5, stacked = True)
	plt.xlabel("Angular speeds profile")
	plt.show()
	return

def plotTrajectoryProfiles(pathToSave, folderName, trajectories, origin, end, savefig = True,showfig = True):
	if(not os.path.isdir("./{path}/{folderToSave}".format(path = pathToSave, folderToSave = folderName))):
		os.makedirs("./{path}/{folderToSave}".format(path = pathToSave, folderToSave = folderName))

	for i in range(0, len(trajectories)):
		last_point_in_trajectory = trajectories[i][len(trajectories[i]) -1]
		# end[0] is end latitude, end[1] is end longtitude
		if(utils.nearOrigin(end[0],end[1],last_point_in_trajectory[utils.dataDict["latitude"]], last_point_in_trajectory[utils.dataDict["longtitude"]])): # only plot profile for those trajectories between origin and end		
			# 1. acceleration profile from start point to end point
			accelerations, timeAxis, distanceAxis = getSpeedAccelerations(trajectories[i], utils.dataDict)
			plt.scatter(distanceAxis, accelerations, label = "trajectory_acceleration_profile_{i}".format(i = i))
			if(savefig):
				plt.savefig("./{path}/{folderToSave}/trajectory_acceleration_profile_{i}.png".format(path = pathToSave, folderToSave = folderName, i = i))
			if(showfig):
				plt.show()
			plt.clf()
			# 2. speed profile from start point to end point 
			speeds, timeAxis , distanceAxis= getSpeeds(trajectories[i], utils.dataDict)
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
	plt.plot(trajectory[0:len(trajectory),utils.data_dict_x_y_coordinate['x']], trajectory[0:len(trajectory),utils.data_dict_x_y_coordinate['y']])
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





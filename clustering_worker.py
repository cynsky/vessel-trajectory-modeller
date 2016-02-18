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

def clusterPurity(groundTruth, clusterLabel, n_cluster):
	purity = np.zeros(n_cluster)
	startingLabel = min(clusterLabel)
	for index in range(startingLabel,startingLabel+n_cluster):
		true_label_distribution = groundTruth[clusterLabel == index]
		mode,modeArr= findMode(true_label_distribution)
		purity[index - startingLabel] = float(mode)/len(true_label_distribution)
	return purity

def trajectoryDissimilarityL2(t1, t2):
	"""
	t1, t2 are two trajectories in X, Y coordinates;
	return the l2 distance between the x, y position of the two trajectory;
	"""
	i = 0
	j = 0
	dissimilarity = 0.0
	while(i < len(t1) and j < len(t2)):
		dissimilarity += DIST.euclidean([t1[i][utils.data_dict_x_y_coordinate["x"]] ,t1[i][utils.data_dict_x_y_coordinate["y"]]], \
			[t2[j][utils.data_dict_x_y_coordinate["x"]], t2[j][utils.data_dict_x_y_coordinate["y"]]])
		i += 1
		j += 1
	# only one of the following loops will be entered
	while(i < len(t1)):
		dissimilarity += DIST.euclidean([t1[i][utils.data_dict_x_y_coordinate["x"]], t1[i][utils.data_dict_x_y_coordinate["y"]]], \
		[t2[j - 1][utils.data_dict_x_y_coordinate["x"]], t2[j - 1][utils.data_dict_x_y_coordinate["y"]]]) # j -1 to get the last point in t2
		i += 1

	while(j < len(t2)):
		dissimilarity += DIST.euclidean([t1[i - 1][utils.data_dict_x_y_coordinate["x"]], t1[i - 1][utils.data_dict_x_y_coordinate["y"]]], \
			[t2[j][utils.data_dict_x_y_coordinate["x"]], t2[j][utils.data_dict_x_y_coordinate["y"]]])
		j += 1
	return dissimilarity

def getTrajectoryCenterofMass(t):
	"""
	t: trajectory in X, Y coordinates
	return the center of mass of trajectory t
	"""
	t = np.asarray(t) # make it a np array
	center_x = np.mean(t[:, utils.data_dict_x_y_coordinate["x"]])
	center_y = np.mean(t[:, utils.data_dict_x_y_coordinate["y"]])
	return np.asarray([center_x, center_y])

def distanceBetweenTwoTrajectoryPoint(p1, p2):
	"""
	p1, p2 are in X, Y coordinates
	return the l2 distance between theses two points' x, y geographical coordinates
	"""

	return DIST.euclidean( \
		[p1[utils.data_dict_x_y_coordinate["x"]], p1[utils.data_dict_x_y_coordinate["y"]]], \
		[p2[utils.data_dict_x_y_coordinate["x"]], p2[utils.data_dict_x_y_coordinate["y"]]])

def getTrajectoryLength(t):
	"""
	t: trajectory in X, Y coordinates
	return the sum of length of the trajectory
	"""
	distance = 0.0
	cur_pos = t[0]
	for i in range(1, len(t)):
		next_pos = t[i]
		distance += distanceBetweenTwoTrajectoryPoint(cur_pos, next_pos)
		cur_pos = next_pos # forward current position

	return distance

def getTrajectoryDisplacement(t):
	"""
	t: trajectory in X,Y coordinates
	return the (dx,dy) as displacement of the trajectory t
	"""
	return np.asarray([ \
		t[len(t) - 1][utils.data_dict_x_y_coordinate["x"]] - t[0][utils.data_dict_x_y_coordinate["x"]], \
		t[len(t) - 1][utils.data_dict_x_y_coordinate["y"]] - t[0][utils.data_dict_x_y_coordinate["y"]]
		])

def trajectoryDissimilarityCenterMass(t1, t2):
	"""
	t1 and t2: trajectory in X, Y coordinates
	return: the dissimilarity between these two trajectories using center of mass, trajectory length and trajectory displacement
	"""
	center_mass_t1 = getTrajectoryCenterofMass(t1)
	center_mass_t2 = getTrajectoryCenterofMass(t2)

	s1 = getTrajectoryDisplacement(t1)
	s2 = getTrajectoryDisplacement(t2)
	s1_norm = np.linalg.norm(s1, 2)
	s2_norm = np.linalg.norm(s2, 2)

	if(s1_norm == 0 and s2_norm == 0):
		cosine_dist = 0.0 # min of DIST.cosine
	elif(s1_norm == 0 or s2_norm == 0):
		cosine_dist = 2.0 # max of DIST.cosine
	else:
		cosine_dist = DIST.cosine(s1,s2)

	len_t1 = getTrajectoryLength(t1)
	len_t2 = getTrajectoryLength(t2)

	ctr_dist = DIST.euclidean(center_mass_t1, center_mass_t2)

	return  ctr_dist + ctr_dist * (abs(len_t1 - len_t2) / max(len_t1, len_t2)) if (max(len_t1, len_t2) > 0) else 0 + np.average([s1_norm, s2_norm]) * cosine_dist

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
	class_trajectories_dict = formClassTrajectoriesDict(cluster_label, data)
	W = 0 # within class variation, summed over classes
	for class_label, trajectories in class_trajectories_dict.iteritems():
		W += withinClassVariation(trajectories, distance_matrix, metric_func) # get the W for one class
	B = betweenClassVariation(class_trajectories_dict, distance_matrix, metric_func) # between class variation
	K = len(set(cluster_label))
	n = len(data)

	return  (B/ (K-1))/(W/(n - K)), B, W

def formClassTrajectoriesDict(cluster_label, data):
	"""
	cluster_label: length n
	data: length n
	returns: a dictionary of [cluster_label: [trajectories]]
	"""
	assert len(cluster_label) == len(data), "data and cluster_label length should be the same"
	class_trajectories_dict = {} # a dictionary of class label to its cluster
	for i in range(0, len(cluster_label)):
		class_label = cluster_label[i]
		if(not class_label in class_trajectories_dict):
			class_trajectories_dict[class_label] = []
		class_trajectories_dict[class_label].append(data[i])
	return class_trajectories_dict

def plotRepresentativeTrajectory(cluster_label, data, fname = "", path = ""):
	"""
	cluster_label: length n
	data: length n, needs to be in X, Y coordinate
	plots the cluster centroids of the current clustering
	"""
	assert len(cluster_label) == len(data), "data and cluster_label length should be the same"
	class_trajectories_dict = formClassTrajectoriesDict(cluster_label, data)
	centroids = []
	for class_label, trajectories in class_trajectories_dict.iteritems():
		centroids.append(getMeanTrajecotoryWithinClass(trajectories))
	plotter.plotListOfTrajectories(centroids, show = False, clean = True, save = (fname != "" and path != ""), fname = fname, path = path)

def getMeanTrajecotoryPointAtIndex(trajectories, index):
	"""
	trajectories: a list of trajectories
	index: a position on one trajectory
	returns: the mean trajectory point at the given position across all trajectories in the given list
	"""
	mean = np.zeros(len(utils.data_dict_x_y_coordinate))
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
	mean_trajectory = np.zeros(shape = (max_length, len(utils.data_dict_x_y_coordinate)))
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
	trajectories: the trajectories need to be in XY coordinates
	"""
	plot_path = utils.queryPath(path + "/plots")
	distance_matrix = getTrajectoryDistanceMatrix(trajectories, metric_func)
	# distance_matrix = writeToCSV.loadData("tankers/all_OD_trajectories_distance_matrix_l2.npz".format(path = path)) # load from the previously runned result
	print "distance_matrix:\n", distance_matrix
	writeToCSV.saveData(distance_matrix, path + "/" + fname)
	v = DIST.squareform(distance_matrix)
	cluster_result = HAC.linkage(v, method = "average")
	dg = HAC.dendrogram(cluster_result)
	plt.xlabel("cluster_dengrogram_{fname}".format(fname = fname))
	plt.savefig("{path}/cluster_dengrogram_{fname}.png".format(fname = fname, path = plot_path))
	plt.show()
	
	# Get the optimal number of clusters
	MIN_NUM_CLUSTER = 2
	MAX_NUM_CLUSTER = min(300,len(trajectories))
	opt_num_cluster = MIN_NUM_CLUSTER
	opt_CH_index = -1
	opt_cluster_label = None

	CH_indexes = []
	cluster_labels = []
	between_cluster_scatterness = []
	within_cluster_scatterness = []

	for i in range(MIN_NUM_CLUSTER, MAX_NUM_CLUSTER):
		this_cluster_label = HAC.fcluster(Z= cluster_result, t= i,criterion='maxclust')
		this_CH_index, B, W = CHIndex(this_cluster_label, distance_matrix, trajectories, metric_func)
		
		CH_indexes.append(this_CH_index)
		between_cluster_scatterness.append(B)
		within_cluster_scatterness.append(W)
		cluster_labels.append(this_cluster_label)
		
		if(this_CH_index > opt_CH_index):
			opt_CH_index = this_CH_index
			opt_num_cluster = i
			opt_cluster_label = this_cluster_label
		print "\nHAC Cluster label:\n", this_cluster_label
		print "number of labels by HAC:", len(set(this_cluster_label)), ";starting label is:", min(this_cluster_label)
		print "n_clusters				CH index"
		print i,"				", this_CH_index

		"""Plot the representative trajectories"""
		plotRepresentativeTrajectory(this_cluster_label, trajectories, \
			fname = "cluster_centroids_{n}_classes".format(n = len(set(this_cluster_label))), \
			path = plot_path)

	"""Plot the CH indexes chart"""
	plt.plot(range(MIN_NUM_CLUSTER, MAX_NUM_CLUSTER), CH_indexes)
	plt.xlabel("CH_indexes range plot")
	plt.savefig("{path}/CH_indexes_{fname}.png".format(fname = fname, path = plot_path))
	plt.show()

	"""Plot the B and W chart"""
	plt.plot(range(MIN_NUM_CLUSTER, MAX_NUM_CLUSTER), between_cluster_scatterness, label = "B")
	plt.plot(range(MIN_NUM_CLUSTER, MAX_NUM_CLUSTER), within_cluster_scatterness, label = "W")
	plt.xlabel("B and W plot")
	plt.legend()
	plt.savefig("{path}/B_and_W_{fname}.png".format(fname = fname, path = plot_path))
	plt.show()
	
	return opt_cluster_label, cluster_labels, CH_indexes






	
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
import writeToCSV
import utils
import os
from shutil import copyfile
import sys

def searchInputsForVesselType(vessel_type):
	dict_IMO = {}
	with open('dataSource/static.csv', 'rU') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		for row in reader:
			if (row[8].lower() == vessel_type.lower()):
				dict_IMO[row[0]] = True
	return dict_IMO


def main():
	vessel_type = raw_input("Please input the vessel type to search for raw .csv files named with IMO:")
	n = raw_input("Please specify how many .csv files you wish to copy over(Press Enter for All):")
	capacity = sys.maxint
	if (n != ''):
		capacity = int(n)
	
	utils.queryPath("{vessel_type}/input".format(vessel_type = vessel_type))
	dict_IMO = searchInputsForVesselType(vessel_type)
	print "number of IMOs with the specified type of {vessel_type} found:".format(vessel_type = vessel_type), len(dict_IMO)
	data_source = 'dataSource/dynamic/'
	count = 0
	for file_name in os.listdir(data_source):
		if (file_name.find(".csv") != -1):
			if (file_name[:file_name.find(".csv")] in dict_IMO):
				"""copy over to the input folder under vessel_type folder"""
				copyfile('{data_source}/{file_name}'.format(data_source = data_source, file_name = file_name), \
					"{vessel_type}/input/{file_name}".format(vessel_type = vessel_type, file_name = file_name))
				count += 1
				if (count >= capacity):
					break

if __name__ == "__main__":
	main()
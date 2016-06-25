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

def LatLonToXY (lat1, lon1,lat2, lon2): # lat1 and lon1 are the origins and all inputs are assumed to be in the right format of the lat lon
	dx = (lon2-lon1)*40000*math.cos((lat1+lat2)*math.pi/360)/360
	dy = (lat1-lat2)*40000/360
	return dx, dy

def isErrorData(prevData, currentData, maxSpeed):
	knotToKmPerhour = 1.85200
	dataDict = {
	"navigation_status":0,
	"rate_of_turn":1,
	"speed_over_ground":2,
	"latitude":3,
	"longitude":4,
	"course_over_ground":5,
	"true_heading":6,
	"ts":7
	}
	prevLat = prevData[dataDict["latitude"]] 
	prevLon = prevData[dataDict["longitude"]]
	prevTS = prevData[dataDict["ts"]]

	currentLat = currentData[dataDict["latitude"]] 
	currentLon = currentData[dataDict["longitude"]]
	currentTS = currentData[dataDict["ts"]]

	x,y = LatLonToXY(prevLat,prevLon, currentLat, currentLon)
	# higherSpeed = max(prevData[dataDict["speed_over_ground"]], currentData[dataDict["speed_over_ground"]])
	# if(higherSpeed > 0):
	# 	speedToUse = higherSpeed
	# else:
	# 	speedToUse = maxSpeed
		
	# if(currentTS == 1373689299):
		# print "currentTS:", datetime.datetime.fromtimestamp(currentTS).strftime('%Y-%m-%dT%H:%M:%SZ')
		# print "prevTS:", datetime.datetime.fromtimestamp(prevTS).strftime('%Y-%m-%dT%H:%M:%SZ')
		# print "supposed to be greater:"
		# print np.linalg.norm([x,y],2), float(maxSpeed) * knotToKmPerhour* float(currentTS - prevTS)/3600.0
		# print currentTS - prevTS  >= 0 and currentTS - prevTS < 3600 and np.linalg.norm([x,y],2) > float(maxSpeed) * knotToKmPerhour* (currentTS - prevTS)/3600
	
	# TODO: check what to do with data points of same TS but different report, just get rid of all of these points but the first one?
	if(currentTS == prevTS):
		return True
	# check within one hour possible error data
	if( currentTS - prevTS  > 0 and currentTS - prevTS < 3600 and np.linalg.norm([x,y],2) > float(maxSpeed) * knotToKmPerhour* (currentTS - prevTS)/3600):
		return True
	return False

def main():
	# assume that the input data are sorted
	utils.GEOSCALE = 600000.0
	record_mmsi = True

	# filename = "3916119.csv"
	# filename = "1000019.csv"
	# filename = "9261126.csv" # ship type is 40, High Speed Craft
	# fileNames = ["8514019.csv", "9116943.csv", "9267118.csv", "9443140.csv", "9383986.csv", "9343340.csv", "9417464.csv", "9664225.csv", "9538440.csv", "9327138.csv"]
	root_folder = raw_input("Input name for root_folder:")
	out_sample_test = raw_input("Out Sample Test?(y/n)") == "y"
	input_path = "{folder}/input/".format(folder = root_folder + ("/out_sample_test" if (out_sample_test) else ""))
	fileNames = []

	for input_filename in os.listdir(input_path):
			if (input_filename.find(".csv") != -1):
				fileNames.append(input_filename)

	if (out_sample_test):
		foldername = "{root_folder}/out_sample_test/cleanedData".format(root_folder = root_folder)
	else:
		foldername = "{root_folder}/cleanedData".format(root_folder = root_folder)

	utils.queryPath(foldername)

	aggregateData = None
	for index in range(0, len(fileNames)):
		filename = fileNames[index]
			
		data = []
		countSpeedGreaterThan10 = 0
		maxSpeed = 0
		with open('{input_path}/{filename}'.format(input_path = input_path, filename = filename), 'rU') as csvfile:
			reader = csv.DictReader(csvfile, dialect=csv.excel_tab, delimiter = ',')

			# skip the first iteration
			# iterrows = iter(reader)
			# next(iterrows)

			for row in reader:
				if(int(row["message_type"]) == 1 or int(row["message_type"]) == 2 or int(row["message_type"]) == 3):
					# if the Lat Lon info is not available, skip
					if(float(row["latitude"])/utils.GEOSCALE == 91 or float(row["longitude"])/utils.GEOSCALE == 181):
						continue
					
					time_str = row['timeStamp']
					timestamp = time.strptime(time_str, "%Y-%m-%dT%H:%M:%SZ") # parse the time
					dt_seconds = datetime.datetime(*timestamp[:6]).strftime("%s")
					# print x, y
					#Speed over ground
					speedOverGround = float(row["speed_over_ground"])/10.0
					if(speedOverGround > 1):
						countSpeedGreaterThan10 += 1
					if(speedOverGround > maxSpeed and speedOverGround != 102.3): #1023 indicates speed not available
						maxSpeed = speedOverGround
					trajectory_point = [ \
					int(row["navigation_status"]), \
					float(row["rate_of_turn"]), \
					speedOverGround,float(row["latitude"])/utils.GEOSCALE, \
					float(row["longitude"])/utils.GEOSCALE, \
					float(row["course_over_ground"]), \
					float(row["true_heading"]), \
					int(dt_seconds)]
					if (record_mmsi):
						trajectory_point.append(long(row["mmsi"]))
					data.append(trajectory_point)


		

		data = np.asarray(data)
		print "before cleaning data.shape:", data.shape
		# Clean out data that has trasient movement that does not make sense: for example, shift 1 km in 1 minute or soemthing
		print "Count of Time instances that speed is > 10 knot:", countSpeedGreaterThan10
		print "maxSpeed:", maxSpeed

		i = 1
		while(i < data.shape[0]):
			print "checking:", i
			if(isErrorData(data[i-1], data[i], maxSpeed)):
				data = np.delete(data, i, 0)
			else:
				i += 1
		
		print "after cleaning error data.shape:", data.shape

		if(aggregateData is None):
			aggregateData = data
		else:
			aggregateData = np.concatenate((aggregateData, data), axis=0)
		# writeToCSV.saveArray(data, "{foldername}/{f}".format(foldername = foldername, f = filename[0:filename.find(".")]))
		writeToCSV.writeDataToCSVWithMMSI(data, path = foldername, file_name = filename[0:filename.find(".")])
	
	print "aggregateData.shape:", aggregateData.shape
	# writeToCSV.saveArray(aggregateData, "{foldername}/{f}".format(foldername = foldername, f = "aggregateData"))
	# writeToCSV.writeDataToCSV(aggregateData, foldername, "aggregateData")
	writeToCSV.writeDataToCSVWithMMSI(aggregateData, foldername, "aggregateData_with_mmsi")
	# xy_coordinate = [item[3:5] for item in data]
	# xy_coordinate = np.asarray(xy_coordinate)
	# print xy_coordinate.shape c
	# plt.scatter([item[3] for item in data], [item[4] for item in data])
	# plt.savefig("vessel_points.png")
	# plt.show()
	
	return

if __name__ == "__main__":
	main()

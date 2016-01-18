import readCSV
import numpy as np 
import matplotlib.pyplot as plt
import datetime
import csv

def writeNpzToCSV(path, npz_file_name):
	data = readCSV.loadArray(path +"/"+ npz_file_name)
	print data.shape
	if(npz_file_name.find(".") != -1):
		file_name =  npz_file_name[:npz_file_name.find(".")]
	else:
		file_name = npz_file_name
	writeDataToCSV(data, path, file_name)
	return

def readDataFromCSV(path, filename):
	data = []
	with open('{path}/{filename}'.format(path = path, filename = filename), 'rU') as csvfile:
			reader = csv.DictReader(csvfile, dialect=csv.excel_tab, delimiter = ',')

			for row in reader:
					# time_str = row['timeStamp']
					# timestamp = time.strptime(time_str, "%Y-%m-%dT%H:%M:%SZ") # parse the time
					# dt_seconds = datetime.datetime(*timestamp[:6]).strftime("%s")
					trajectory_point = [int(float(row["navigation_status"])),float(row["rate_of_turn"]),float(row["speed_over_ground"]),float(row["latitude"]), float(row["longitude"]),float(row["course_over_ground"]), float(row["true_heading"]), int(float(row["ts"]))]
					data.append(trajectory_point)

	return np.asarray(data)



def writeDataToCSV(data, path, file_name):
	"""
	file_name: string of name of file, without .csv suffix
	"""
	# print path + "/" +npz_file_name[:npz_file_name.find(".")]+ ".csv"
	# raise ValueError
	dataDict = {
	"navigation_status":0,
	"rate_of_turn":1,
	"speed_over_ground":2,
	"latitude":3,
	"longitude":4,
	"course_over_ground":5,
	"true_heading":6,
	"ts":7 # is in UNIX timestamp of secs
	}

	# datetime.datetime.fromtimestamp(currentTS).strftime('%Y-%m-%dT%H:%M:%SZ')

	with open(path +"/"+ file_name+ ".csv", 'w') as csvfile:
		fieldnames = ['navigation_status', 'rate_of_turn', 'speed_over_ground', 'latitude', 'longitude', 'course_over_ground', 'true_heading','ts', 'ts_string']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		
		writer.writeheader()
	
		for i in range (0, data.shape[0]):
			writer.writerow({'navigation_status': data[i][dataDict['navigation_status']], 
				'rate_of_turn':data[i][dataDict['rate_of_turn']], 
				'speed_over_ground':data[i][dataDict['speed_over_ground']], 
				'latitude':data[i][dataDict['latitude']], 
				'longitude':data[i][dataDict['longitude']], 
				'course_over_ground':data[i][dataDict['course_over_ground']], 
				'true_heading':data[i][dataDict['true_heading']], 
				'ts':data[i][dataDict['ts']],
				'ts_string':datetime.datetime.fromtimestamp(data[i][dataDict['ts']]).strftime('%Y-%m-%dT%H:%M:%SZ')
				})

	return
		

def main():
	# path = "tankers/cleanedData"
	# # filename = "aggregateData.npz"
	# fileNames = ["8514019.npz", "9116943.npz", "9267118.npz", "9443140.npz", "9383986.npz", "9343340.npz", "9417464.npz", "9664225.npz", "9538440.npz", "9327138.npz"]
	# for i in range(0 , len(fileNames)):
	# 	filename = fileNames[i]
	# 	writeNpzToCSV(path = path , npz_file_name = filename)

	path = "cleanedData"
	writeNpzToCSV(path = path, npz_file_name = "1000019.npz")
	writeNpzToCSV(path = path, npz_file_name = "3916119.npz")
	writeNpzToCSV(path = path, npz_file_name = "9261126.npz")
	return

if __name__ == "__main__":
	main()
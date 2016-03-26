import readCSV
import numpy as np 
import matplotlib.pyplot as plt
import datetime
import csv
import utils

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

def saveData(data, filename):
	np.savez(filename,data = data)

def loadData(filename):
	loader = np.load(filename)
	return loader['data']

def writeNpzToCSV(path, npz_file_name):
	data = loadArray(path +"/"+ npz_file_name)
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


def readDataFromCSVWithMMSI(path, filename):
	"""
	filename: with .csv suffix
	"""
	data = []
	with open('{path}/{filename}'.format(path = path, filename = filename), 'rU') as csvfile:
			reader = csv.DictReader(csvfile, dialect=csv.excel_tab, delimiter = ',')

			for row in reader:
					# time_str = row['timeStamp']
					# timestamp = time.strptime(time_str, "%Y-%m-%dT%H:%M:%SZ") # parse the time
					# dt_seconds = datetime.datetime(*timestamp[:6]).strftime("%s")
					trajectory_point = [int(float(row["navigation_status"])), \
					float(row["rate_of_turn"]), \
					float(row["speed_over_ground"]), \
					float(row["latitude"]), \
					float(row["longitude"]), \
					float(row["course_over_ground"]), \
					float(row["true_heading"]), \
					int(float(row["ts"])), \
					int(float(row["mmsi"]))]
					data.append(trajectory_point)

	return np.asarray(data)


def writeDataToCSVWithMMSI(data, path, file_name):
	"""
	file_name: string of name of file, without .csv suffix
	"""
	with open(path +"/"+ file_name+ ".csv", 'w') as csvfile:
		fieldnames = [\
		'navigation_status', \
		'rate_of_turn', \
		'speed_over_ground', \
		'latitude', \
		'longitude', \
		'course_over_ground', \
		'true_heading',\
		'ts', \
		'ts_string', \
		'mmsi']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		
		writer.writeheader()
	
		for i in range (0, data.shape[0]):
			writer.writerow({'navigation_status': data[i][utils.dataDict['navigation_status']], 
				'rate_of_turn':data[i][utils.dataDict['rate_of_turn']], 
				'speed_over_ground':data[i][utils.dataDict['speed_over_ground']], 
				'latitude':data[i][utils.dataDict['latitude']], 
				'longitude':data[i][utils.dataDict['longitude']], 
				'course_over_ground':data[i][utils.dataDict['course_over_ground']], 
				'true_heading':data[i][utils.dataDict['true_heading']], 
				'ts':data[i][utils.dataDict['ts']],
				'ts_string':datetime.datetime.fromtimestamp(data[i][utils.dataDict['ts']]).strftime('%Y-%m-%dT%H:%M:%SZ'),
				'mmsi': data[i][utils.dataDict['mmsi']]
				})

	return



def writeDataToCSV(data, path, file_name):
	"""
	path: without trailing '/'
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
		fieldnames = [\
		'navigation_status', \
		'rate_of_turn', \
		'speed_over_ground', \
		'latitude', \
		'longitude', \
		'course_over_ground', \
		'true_heading',\
		'ts', \
		'ts_string']
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
		
def writeAllProtocolTrajectories(path, file_name, all_protocol_trajectories, cluster_label_to_cluster_size):
	"""
	file_name: string of name of file, without .csv suffix
	all_protocol_trajectories: list of protocol trajectories
	cluster_label_to_cluster_size: [all_protocol_trajectories's index : cluster size]
	note: protocol trajectories does not contain timestamp info
	"""

	with open(path +"/"+ file_name+ ".csv", 'w') as csvfile:
		fieldnames = [\
		'navigation_status', \
		'rate_of_turn', \
		'speed_over_ground', \
		'latitude', \
		'longitude', \
		'course_over_ground', \
		'true_heading',\
		'cluster_size'
		]
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		
		writer.writeheader()
	
		for i in range(0, len(all_protocol_trajectories)):
			this_trajectory = all_protocol_trajectories[i]
			for point in this_trajectory:
				writer.writerow({'navigation_status': point[utils.dataDict['navigation_status']], 
					'rate_of_turn':point[utils.dataDict['rate_of_turn']], 
					'speed_over_ground':point[utils.dataDict['speed_over_ground']], 
					'latitude':point[utils.dataDict['latitude']], 
					'longitude':point[utils.dataDict['longitude']], 
					'course_over_ground':point[utils.dataDict['course_over_ground']], 
					'true_heading':point[utils.dataDict['true_heading']], 
					'cluster_size':cluster_label_to_cluster_size[i]
					})
			writer.writerow({}) # write empty line between trajecotries to indicate start of new trajectory
	return

def writeEndPointsToProtocolTrajectoriesIndexesWithMMSI(path, file_name, endpoints, endpoints_cluster_dict):
	"""
	file_name: string of name of file, without .csv suffix
	endpoints: list of endpoints with mmsi
	endpoints_cluster_dict: [endpoint string: [utils.ClusterCentroidTuple]]
	"""
	with open(path +"/"+ file_name+ ".csv", 'w') as csvfile:
		fieldnames = [\
		'navigation_status', \
		'rate_of_turn', \
		'speed_over_ground', \
		'latitude', \
		'longitude', \
		'course_over_ground', \
		'true_heading',\
		'ts', \
		'ts_string', \
		'mmsi',\
		'protocol_trajectories']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		
		writer.writeheader()
	
		for i in range (0, len(endpoints)):
			writer.writerow({'navigation_status': endpoints[i][utils.dataDict['navigation_status']], 
				'rate_of_turn':endpoints[i][utils.dataDict['rate_of_turn']], 
				'speed_over_ground':endpoints[i][utils.dataDict['speed_over_ground']], 
				'latitude':endpoints[i][utils.dataDict['latitude']], 
				'longitude':endpoints[i][utils.dataDict['longitude']], 
				'course_over_ground':endpoints[i][utils.dataDict['course_over_ground']], 
				'true_heading':endpoints[i][utils.dataDict['true_heading']], 
				'ts':endpoints[i][utils.dataDict['ts']],
				'ts_string':datetime.datetime.fromtimestamp(endpoints[i][utils.dataDict['ts']]).strftime('%Y-%m-%dT%H:%M:%SZ'),
				'mmsi': endpoints[i][utils.dataDict['mmsi']],
				'protocol_trajectories': [ item.cluster for item in endpoints_cluster_dict[\
				"{lat}_{lon}".format(\
					lat = endpoints[i][utils.dataDict["latitude"]], \
					lon = endpoints[i][utils.dataDict["longitude"]])] ]
				})
	return

def writeVesselSpeedToDistance(path, file_name, vessel_distance_speed_dict):
	"""
	vessel_distance_speed_dict: a dictionary of ["id1_id2" : [utils.SpeedDistanceTuple]]
	"""
	with open(path +"/"+ file_name+ ".csv", 'w') as csvfile:
		fieldnames = [\
		'id1', \
		'id2', \
		'relative_speed', \
		'distance' \
		]
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		
		writer.writeheader()
	
		for vessel_pair_string, tuple_list in vessel_distance_speed_dict.iteritems():
			if (len(tuple_list) > 0):
				for speed_distance_tuple in tuple_list:
					writer.writerow({
						'id1':vessel_pair_string[:vessel_pair_string.find("_")],
						'id2':vessel_pair_string[vessel_pair_string.find("_") + 1:],
						'relative_speed': speed_distance_tuple.speed,
						'distance':speed_distance_tuple.distance
					})
				writer.writerow({}) # empty row indicating the start of new pair
	return

def writeVesselMinDistanceMatrix(path, file_name, mmsi_list_dict, min_distance_matrix):
	"""
	mmsi_list_dict: a dictionary of [mmsi_id: row / col index in min_distance_matrix]
	min_distance_matrix: a 2D dense matrix of shape (len(mmsi_list), len(mmsi_list))
	"""
	with open(path +"/"+ file_name+ ".csv", 'w') as csvfile:
		fieldnames = [\
		'id1', \
		'id2', \
		'min_distance'
		]
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		
		writer.writeheader()
		mmsi_list = [key for key, index in mmsi_list_dict.iteritems()]
		for i in range(0, len(mmsi_list)):
			for j in range (i + 1, len(mmsi_list)):
				writer.writerow({
					'id1':long(mmsi_list[i]),
					'id2':long(mmsi_list[j]),
					'min_distance': min_distance_matrix[mmsi_list_dict[long(mmsi_list[i])]][mmsi_list_dict[long(mmsi_list[j])]]
				})
	return

def writeMMSIs(path, file_name, mmsi_list):
	"""mmsi_list: list of mmsi (long)"""
	with open (path + "/" + file_name + ".csv", 'w') as csvfile:
		fieldnames = [\
		'mmsi'
		]
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		
		writer.writeheader()
		for i in range(0, len(mmsi_list)):
			writer.writerow({
				'mmsi': mmsi_list[i]
			})
	return

def main():
	# path = "tankers/cleanedData"
	# # filename = "aggregateData.npz"
	# fileNames = ["8514019.npz", "9116943.npz", "9267118.npz", "9443140.npz", "9383986.npz", "9343340.npz", "9417464.npz", "9664225.npz", "9538440.npz", "9327138.npz"]
	# for i in range(0 , len(fileNames)):
	# 	filename = fileNames[i]
	# 	writeNpzToCSV(path = path , npz_file_name = filename)

	# path = "cleanedData"
	# writeNpzToCSV(path = path, npz_file_name = "1000019.npz")
	# writeNpzToCSV(path = path, npz_file_name = "3916119.npz")
	# writeNpzToCSV(path = path, npz_file_name = "9261126.npz")
	root_folder = "tankers"
	list_of_trajectories = loadData(root_folder + "/" + "all_OD_trajectories_cleaned.npz")
	flattened_trajectory_points = np.array([point for trajectory in list_of_trajectories for point in trajectory])
	writeDataToCSV(data = flattened_trajectory_points, path = root_folder , file_name = "all_OD_trajectories_cleaned")
	
	return

if __name__ == "__main__":
	main()
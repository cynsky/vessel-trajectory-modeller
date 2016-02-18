import os

def queryPath(path):
	"""
	checks if the given path exisits, if not existing, create and return it; else, just echo back it
	"""
	if(not os.path.isdir("./{path}".format(
		path = path))):
		os.makedirs("./{path}".format(
			path = path))
	return path
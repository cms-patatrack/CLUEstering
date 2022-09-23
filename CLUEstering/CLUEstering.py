import pandas as pd
import numpy as np
import random as rnd
import time
import matplotlib.pyplot as plt
import CLUEsteringCPP as Algo 
from sklearn.datasets import make_blobs

def sign():
	sign = rnd.random()
	if sign > 0.5:
		return 1
	else:
		return -1

def makeBlobs(nSamples, Ndim, nBlobs=4, mean=0, sigma=0.5):
	"""
	Returns a test dataframe containing randomly generated 2-dimensional or 3-dimensional blobs. 

	Parameters:
	nSamples (int): The number
	Ndim (int): The number of dimensions.
	nBlobs (int): The number of blobs that should be produced. By default it is set to 4.
	mean (float): The mean of the gaussian distribution of the z values.
	sigma (float): The standard deviation of the gaussian distribution of the z values.
	"""

	if Ndim == 2:
		data = {'x0': [], 'x1': [], 'weight': []}
		blob_data, blob_labels = make_blobs(n_samples=nSamples)
		for i in range(nSamples):
			data['x0'] += [blob_data[i][0]]
			data['x1'] += [blob_data[i][1]]
			data['weight'] += [1]

		return pd.DataFrame(data)
	if Ndim == 3:
		data = {'x0': [], 'x1': [], 'x2': [], 'weight': []}
		z = np.random.normal(mean,sigma,100)
		centers = []
		for i in range(nBlobs):
			centers.append([sign()*15*rnd.random(),sign()*15*rnd.random()])
		for value in z: # for every z value, a layer is generated.
			blob_data, blob_labels = make_blobs(n_samples=nSamples,centers=np.array(centers))
			for i in range(nSamples):
				data['x0'] += [blob_data[i][0]]
				data['x1'] += [blob_data[i][1]]
				data['x2'] += [value]
				data['weight'] += [1]

		return pd.DataFrame(data)

class clusterer:
	def __init__(self, dc, rhoc, outlier, pPBin=10): 
		try:
			if float(dc) != dc:
				raise ValueError('The dc parameter must be a float')
			self.dc = dc
			if float(rhoc) != rhoc:
				raise ValueError('The rhoc parameter must be a float')
			self.rhoc = rhoc
			if float(outlier) != outlier:
				raise ValueError('The outlier parameter must be a float')
			self.outlier = outlier
			if type(pPBin) != int:
				raise ValueError('The pPBin parameter must be a int')
			self.pPBin = pPBin
		except ValueError as ve:
			print(ve)
			exit()
	def readData(self, inputData):
		"""
		Reads the data in input and fills the class members containing the coordinates of the points, the energy weight, the number of dimensions and the number of points.

		Parameters:
		inputData (pandas dataframe): The dataframe should contain one column for every coordinate, each one called 'x*', and one column for the weight.
		inputData (string): The string should contain the full path to a csv file containing the data.
		inputData (list or numpy array): The list or numpy array should contain a list of lists for the coordinates and a list for the weight.
		"""

		print('Start reading points')
		
		# numpy array
		if type(inputData) == np.array:
			try:
				if len(inputData) < 2:
					raise ValueError('Error: Inadequate data. The data must contain at least one coordinate and the energy.')
				self.coords = [coord for coord in self.inputData[:-1]]
				self.weight = self.inputData[-1] 
				if len(self.inputData[:-1]) > 10:
					raise ValueError('Error: The maximum number of dimensions supported is 10')
				self.Ndim = len(self.inputData[:-1])
				self.Npoints = len(self.weight)
			except ValueError as ve:
				print(ve)
				exit()

		# lists
		if type(inputData) == list:
			try:
				if len(inputData) < 2:
					raise ValueError('Error: Inadequate data. The data must contain at least one coordinate and the energy.')
				self.coords = [coord for coord in self.inputData[:-1]]
				self.weight = self.inputData[-1]
				if len(self.inputData[:-1]) > 10:
					raise ValueError('Error: The maximum number of dimensions supported is 10')
				self.Ndim = len(self.inputData[:-1])
				self.Npoints = len(self.weight)
			except ValueError as ve:
				print(ve)
				exit()

		# path to .csv file or pandas dataframe
		if type(inputData) == str or type(inputData) == pd.DataFrame:
			if type(inputData) == str:
				try:
					if inputData[-3:] != 'csv':
						raise ValueError('Error: The file is not a csv file.')
					df = pd.read_csv(inputData)
				except ValueError as ve:
					print(ve)
					exit()
			if type(inputData) == pd.DataFrame:
				try:
					if len(inputData.columns) < 2:
						raise ValueError('Error: Inadequate data. The data must contain at least one coordinate and the energy.')
					df = inputData
				except ValueError as ve:
					print(ve)
					exit()

			try:
				if not 'weight' in df.columns:
					raise ValueError('Error: The input dataframe must contain a weight column.')
					
				coordinate_columns = [col for col in df.columns if col[0] == 'x']
				if len(coordinate_columns) > 10:	
					raise ValueError('Error: The maximum number of dimensions supported is 10')
				self.Ndim = len(coordinate_columns)
				self.coords = []
				for col in coordinate_columns:
					self.coords.append(list(df[col]))
				self.weight = list(df['weight'])
				self.Npoints = len(self.weight)
			except ValueError as ve:
				print(ve)
				exit()

		print('Finished reading points')
	def runCLUE(self):
		"""
		Executes the CLUE clustering algorithm.

		Output:
		self.clusterIds (list): Contains the clusterId corresponding to every point.
		self.isSeed (list): For every point the value is 1 if the point is a seed or an outlier and 0 if it isn't.
		self.NClusters (int): Number of clusters reconstructed.
		self.clusterPoints (list): Contains, for every cluster, the list of points associated to id.
		self.pointsPerCluster (list): Contains the number of points associated to every cluster
		"""

		start = time.time_ns()
		clusterIdIsSeed = Algo.mainRun(self.dc,self.rhoc,self.outlier,self.pPBin,self.coords,self.weight,self.Ndim)
		finish = time.time_ns()
		self.clusterIds = clusterIdIsSeed[0]
		self.isSeed = clusterIdIsSeed[1]
		self.NClusters = len(list(set(self.clusterIds))) 

		clusterPoints = [[] for i in range(self.NClusters)]
		for i in range(self.Npoints):
			clusterPoints[self.clusterIds[i]].append(i)

		self.clusterPoints = clusterPoints
		self.pointsPerCluster = [len(clust) for clust in clusterPoints]

		data = {'clusterIds': self.clusterIds, 'isSeed': self.isSeed}
		self.outputDF = pd.DataFrame(data) 

		self.elapsed_time = (finish - start)/(10**6)
		print('CLUE run in ' + str(self.elapsed_time) + ' ms')
		print('Number of clusters found: ', self.NClusters)
	def inputPlotter(self):
		"""
		Plots the the points in input.
		"""

		if self.Ndim == 2:
			plt.scatter(self.coords[0],self.coords[1], s=1)
			plt.show()
		if self.Ndim >= 3:
			fig = plt.figure()
			ax = fig.add_subplot(projection='3d')
			ax.scatter(self.coords[0],self.coords[1],self.coords[2])

			plt.show()
	def clusterPlotter(self):
		"""
		Plots the clusters with a different colour for every cluster. 

		The points assigned to a cluster are prints as points, the seeds as stars and the outliers as little grey crosses. 
		"""
		
		if self.Ndim == 2:
			data = {'x0':self.coords[0], 'x1':self.coords[1], 'clusterIds':self.clusterIds, 'isSeed':self.isSeed}
			df = pd.DataFrame(data)

			df_clindex = df["clusterIds"]
			M = max(df_clindex) 
			dfs = df["isSeed"]

			df_out = df[df.clusterIds == -1] # Outliers
			plt.scatter(df_out.x0, df_out.x1, s=5, marker='x', color='0.4')
			for i in range(0,M+1):
				dfi = df[df.clusterIds == i] # ith cluster
				plt.scatter(dfi.x0, dfi.x1, s=5, marker='.')
			df_seed = df[df.isSeed == 1] # Only Seeds
			plt.scatter(df_seed.x0, df_seed.x1, s=20, color='r', marker='*')
			plt.show()
		if self.Ndim == 3:
			data = {'x0':self.coords[0], 'x1':self.coords[1], 'x2':self.coords[2], 'clusterIds':self.clusterIds, 'isSeed':self.isSeed}
			df = pd.DataFrame(data)

			df_clindex = df["clusterIds"]
			M = max(df_clindex) 
			dfs = df["isSeed"]
			fig = plt.figure()
			ax = fig.add_subplot(projection='3d')

			df_out = df[df.clusterIds == -1]
			ax.scatter(df_out.x0, df_out.x1, df_out.x2, s=15, color = 'grey', marker = 'x')
			for i in range(0,M+1):
				dfi = df[df.clusterIds == i]
				ax.scatter(dfi.x0, dfi.x1, dfi.x2, s=5, marker = '.')

			df_seed = df[df.isSeed == 1] # Only Seeds
			ax.scatter(df_seed.x0, df_seed.x1, df_seed.x2, s=20, color = 'r', marker = '*')

			plt.show()
	def toCSV(self,outputFolder,fileName):
		"""
		Creates a file containing the coordinates of all the points, their clusterIds and isSeed.	

		Parameters: 
		outputFolder (string): Full path to the desired ouput folder.
		fileName(string): Name of the file, with the '.csv' suffix.
		"""

		outPath = outputFolder + fileName
		data = {}
		for i in range(self.Ndim):
			data['x' + str(i)] = self.coords[i]
		data['clusterIds'] = self.clusterIds
		data['isSeed'] = self.isSeed

		df = pd.DataFrame(data)
		df.to_csv(outPath,index=False)

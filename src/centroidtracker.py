from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTracker:
	def __init__(self, maxDisappeared=50, maxDistance=50):
		self.nextObjectID = 0
		self.objects = OrderedDict() # store object IDs
		self.disappeared = OrderedDict() # store number of frames each object has disappeared for

		# max number of frames an object can disappear for before it's deregistered
		self.maxDisappeared = maxDisappeared

		# max dist between centroids to associate an object
		self.maxDistance = maxDistance

	# register an object
	def register(self, centroid):
		self.objects[self.nextObjectID] = centroid # assign centroid to next available object ID
		self.disappeared[self.nextObjectID] = 0
		self.nextObjectID += 1 # increment next available object ID

	# deregister an object
	def deregister(self, objectID):
		del self.objects[objectID]
		del self.disappeared[objectID]

	# update with new centroid info
	def update(self, rects):
		if len(rects) == 0: # if there are no bounding boxes:
			for objectID in list(self.disappeared.keys()):
				 # increment number of frames object has disappeared for
				self.disappeared[objectID] += 1

				# deregister an object if disppeared for max number of frames
				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)

			return self.objects

		# create array of input centroids for current frame
		inputCentroids = np.zeros((len(rects), 2), dtype="int")

		# loop over bounding boxes
		for (i, (startX, startY, endX, endY)) in enumerate(rects):
			# use bounding box coordinates to calculate centroid
			cX = int((startX + endX) / 2.0)
			cY = int((startY + endY) / 2.0)
			inputCentroids[i] = (cX, cY)

		# register all centroids if none currently exist
		if len(self.objects) == 0:
			for i in range(0, len(inputCentroids)):
				self.register(inputCentroids[i])
		
		else:
			objectIDs = list(self.objects.keys())
			objectCentroids = list(self.objects.values())

			# calculate dist between prev and new centroids
			D = dist.cdist(np.array(objectCentroids), inputCentroids)

			# find minimum row and col distances
			rows = D.min(axis=1).argsort()
			cols = D.argmin(axis=1)[rows]

			usedRows = set()
			usedCols = set()

			# loop over rows and cols
			for (row, col) in zip(rows, cols):
				if row in usedRows or col in usedCols:
					continue

				# if dist is greater than max dist, don't associate
				if D[row, col] > self.maxDistance:
					continue

				# update centroid
				objectID = objectIDs[row]
				self.objects[objectID] = inputCentroids[col]

				# reset disppeared counter
				self.disappeared[objectID] = 0

				usedRows.add(row)
				usedCols.add(col)

			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)

			# if more object centroids than input centroids, an object must have disappeared
			if D.shape[0] >= D.shape[1]:
				for row in unusedRows:
					# increment disappeared counter
					objectID = objectIDs[row]
					self.disappeared[objectID] += 1

					# deregister if disappeared for max number of frames
					if self.disappeared[objectID] > self.maxDisappeared:
						self.deregister(objectID)
			
			# otherwise, regiester new centroid
			else:
				for col in unusedCols:
					self.register(inputCentroids[col])

		return self.objects
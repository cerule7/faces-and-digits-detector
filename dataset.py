class Dataset():
	def __init__(self, trainlist, testlist, vallist):
		self.trainData = trainlist + vallist
		self.testData = testlist
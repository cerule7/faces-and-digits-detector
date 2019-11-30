import random

# returns vector of weights
def perceptronFaceClassifierTrainer(numColumns, numRows, trainData, labelVector):
	totalNumFeatures = (numColumns + 1) * (numRows + 1)
	featureVectorList = [] #list of feature vectors for each image in training data set
	for i in trainData:
		featureVectorList.append(featureVector(numColumns, numRows, trainData[i]))
	weightVector = [random.random() for i in range(len(totalNumFeatures))]
	bias = random.random()
	perfectRun = False #boolean for when to stop training
	iterations = 0 #counter for number of loops through training set
	while (perfectRun is False or iterations < 9999):
		successCounter = 0
		for i in trainData:
			predFunction = 0
			for j in featureVectorList[i]: #iterating over feature values of image i
				predFunction += featureVectorList[i][j] * weightVector[j]
			predFunction += bias
			prediction = 0  #0 means not face, 1 means face
			if(predFunction > 0):
				prediction = 1
			if(prediction == labelVector[i]):
				successCounter += 1
				if(i == len(trainData) and successCounter == i): #full pass of all correct predictions
					perfectRun = True
					break
			#since prediction failed we must adjust weights accordingly
			elif(prediction == 0):
				for j in featureVectorList[i]:
					weightVector[j] += featureVectorList[i][j]
				bias += 1
			else:
				for j in featureVectorList[i]:
					weightVector[j] -= featureVectorList[i][j]
				bias -= 1
		iterations += 1
	return weightVector.append(bias)

#returns 0 if not face, 1 if face
def isFace(image, weightVector, c, r):
	imageFeatures = featureVector(c, r, image)
	predFunction = 0
	for i in imageFeatures:
		predFunction += imageFeatures[i] * weightVector[i]
	predFunction += weightVector[len(weightVector) - 1] #bias
	if (predFunction <= 0):
		return 0
	return 1

# to do
# def perceptronDigitClassifierTrainer(numColumns, numRows, trainData, labelVector):
# 	totalNumFeatures = (numColumns + 1) * (numRow + 1)
# 	featureVectorList = [] #list of feature vectors for each image in training data set
# 	for i in train:
# 		featureVectorList.append(featureVector(numColumns, numRows, trainData[i]))
# 	weightVectors = [][] #list of weight vectors for each digit
# 	for i in weightVectors:
# 		for j in weightVectors[i]:
# 			weightVectors[i][j] = random.random()
# 	biasVector = []
# 	for i in biasVector:
# 		biasVector[i] = random.random()

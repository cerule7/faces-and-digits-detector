import random
from utils import featureVector 
import pickle

# returns vector of weights 
def perceptronFaceClassifierTrainer(trainData):
	featureVectorList = list()  #list of feature vectors for each image in training data set

	for i in trainData:
		featureVectorList.append(featureVector(i.image, r=10, c=10, A=70, Y=60))

	weightVector = [random.random() for i in range(len(featureVectorList[1]))]

	bias = random.random()
	perfectRun = False #boolean for when to stop training
	iterations = 0 #counter for number of loops through training set
	while (perfectRun is False and iterations < 1000): 
		successCounter = 0
		for i in range(0, len(trainData)):
			predFunction = 0

			for j in range(0, len(featureVectorList[i])): #iterating over feature values of image i
				predFunction += featureVectorList[i][j] * weightVector[j]

			predFunction += bias
			prediction = 0  #0 means not face, 1 means face
			if(predFunction > 0):
				prediction = 1
			if((prediction > 0 and int(trainData[i].label) == 1) or (prediction <= 0 and int(trainData[i].label) == 0)):
				successCounter += 1
				if(i == len(trainData) - 1 and successCounter == i): #full pass of all correct predictions
					perfectRun = True
					break
			#since prediction failed we must adjust weights accordingly
			elif(prediction <= 0):
				for j in range(0, len(featureVectorList[i])):
					weightVector[j] += featureVectorList[i][j]
				bias += 1
				# print('in loop {} '.format(weightVector[2]))
			elif(prediction > 0):
				for j in range(0, len(featureVectorList[i])):
					weightVector[j] -= featureVectorList[i][j]
				bias -= 1
		if iterations % 100== 0:
			print('training iteration {} complete'.format(iterations))
		iterations += 1
	weightVector.append(bias)
	return weightVector

#returns 0 if not face, 1 if face
def isFace(image, weightVector):
	imageFeatures = featureVector(image.image, r=10, c=10, A=70, Y=60)
	predFunction = 0
	for i in range(0, len(imageFeatures)):
		predFunction += imageFeatures[i] * weightVector[i] 
	predFunction += weightVector[len(weightVector) - 1] #bias
	if (predFunction > 0):
		return 1
	return 0


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
	

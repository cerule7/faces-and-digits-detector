import random
from utils import featureVector 
import pickle

<<<<<<< HEAD
# returns vector of weights
def perceptronFaceClassifierTrainer(numColumns, numRows, trainData, labelVector):
	totalNumFeatures = (numColumns + 1) * (numRows + 1)
	featureVectorList = [] #list of feature vectors for each image in training data set
=======
# returns vector of weights 
def perceptronFaceClassifierTrainer(trainData):
	featureVectorList = list()  #list of feature vectors for each image in training data set

>>>>>>> 635ec2c653d928de6cd650d37d58419af2b512bd
	for i in trainData:
		featureVectorList.append(featureVector(i.image, r=10, c=10, A=70, Y=60))

	weightVector = [random.random() for i in range(len(featureVectorList[1]))]

	bias = random.random()
	perfectRun = False #boolean for when to stop training
	iterations = 0 #counter for number of loops through training set
<<<<<<< HEAD
	while (perfectRun is False or iterations < 9999):
=======
	while (perfectRun is False and iterations < 1500): 
>>>>>>> 635ec2c653d928de6cd650d37d58419af2b512bd
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
<<<<<<< HEAD
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
=======
	for i in range(0, len(imageFeatures)):
		predFunction += imageFeatures[i] * weightVector[i] 
	predFunction += weightVector[len(weightVector) - 1] #bias
	if (predFunction > 0):
		return 1
	return 0

def perceptronDigitClassifierTrainer(trainData):
	featureVectorList = list()  #list of feature vectors for each image in training data set
	for i in trainData:
		featureVectorList.append(featureVector(i.image, r=14, c=14, A=28, Y=28))
	weightVector = [[random.random() for i in range(len(featureVectorList[1]))] for j in range(0, 10)]
	biasVector = [random.random() for i in range(0, 10)]

	perfectRun = False #boolean for when to stop training
	iterations = 0 #counter for number of loops through training set

	while (perfectRun is False and iterations < 1500):
		successCounter = 0
		predictorVector = [0 for x in range(0, 10)] #holds values of prediction for each digit respectively 

		for i in range(0, len(trainData)):
			if (perfectRun == True):
				break
			for k in range(0, len(predictorVector)):
				if (perfectRun == True):
					break
				for j in range(0, len(featureVectorList[i])): #iterating over feature values of image 
					predictorVector[k] += featureVectorList[i][j] * weightVector[k][j]

				predictorVector[k] += biasVector[k]
				prediction = findHighestPrediction(predictorVector)
				if(prediction == int(trainData[i].label)): #got prediction correct
					successCounter += 1
				if(i == len(trainData) - 1 and successCounter == i): #full pass of all correct predictions
					perfectRun = True
					break
				else:
					for x in range(0, len(weightVector[prediction])): #punishment for wrong prediction
						weightVector[prediction][x] -= featureVectorList[i][x] 
						biasVector[prediction] -= 1
					for x in range(0, len(weightVector[int(trainData[i].label)])): #enforcing ground truth weights
						weightVector[int(trainData[i].label)][x] += featureVectorList[i][x]
						biasVector[int(trainData[i].label)] += 1
		if iterations % 100== 0:
			print('training iteration {} complete'.format(iterations))
		iterations += 1

	for i in range(0, len(weightVector)):
		weightVector[i].append(biasVector[i])
	return weightVector

def findHighestPrediction(predictions):
	maxIndex = 0
	maxValue = predictions[0]
	for i in range(0, len(predictions)):
		if predictions[i] > maxValue:
			maxIndex = i
			maxValue = predictions[i]
	return maxIndex

def whichDigit(image, weights):
	imageFeatures = featureVector(image.image, r=14, c=14, A=28, Y=28)
	predictionVector = [0 for x in range(0, 10)]
	for i in range(0, len(weights)):
		for j in range(0, len(weights[i])):
			if (j != len(weights[i]) - 1): #j is not at bias position)
				predictionVector[i] = weights[i][j] * imageFeatures[j]
			else: 
				predictionVector[i] += weights[i][j] #adding bias
	return findHighestPrediction(predictionVector)


>>>>>>> 635ec2c653d928de6cd650d37d58419af2b512bd

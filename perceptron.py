import random
from utils import featureVector 
import pickle

# returns vector of weights 
def perceptronFaceClassifierTrainer(trainData):
	#totalNumFeatures = (numColumns + 1) * (numRows + 1)
	totalNumFeatures = 74
	featureVectorList = list()  #list of feature vectors for each image in training data set

	for i in trainData:
		featureVectorList.append(featureVector(i, totalNumFeatures))

	weightVector = [random.random() for i in range(totalNumFeatures)]

	bias = random.random()
	perfectRun = False #boolean for when to stop training
	iterations = 0 #counter for number of loops through training set
	while (perfectRun is False and iterations < 9999): 
		successCounter = 0
		for i in range(0, len(trainData)):
			predFunction = 0

			for j in featureVectorList[i]: #iterating over feature values of image i
				predFunction += featureVectorList[i][j] * weightVector[j]

			predFunction += bias
			prediction = 0  #0 means not face, 1 means face
			if(predFunction > 0):
				prediction = 1
			if(prediction == trainData[i].label):
				successCounter += 1

				if(i == len(trainData) - 1 and successCounter == i): #full pass of all correct predictions
					perfectRun = True
					break
			#since prediction failed we must adjust weights accordingly
			elif(prediction <= 0):
				for j in featureVectorList[i]:
					weightVector[j] -= featureVectorList[i][j]
				bias -= 1
			else:
				for j in featureVectorList[i]:
					weightVector[j] += featureVectorList[i][j]
				bias += 1

		iterations += 1
		if iterations % 1000 == 0:
			print('training iteration {} complete'.format(iterations))
	weightVector.append(bias)
	return weightVector

#returns 0 if not face, 1 if face
def isFace(image, weightVector):
	imageFeatures = featureVector(image, 74)
	predFunction = 0
	for i in range(0, len(imageFeatures)):
		predFunction += imageFeatures[i] * weightVector[i] 
	predFunction += weightVector[len(weightVector) - 1] #bias
	if (predFunction <= 0):
		return 0
	return 1


faces = pickle.load(open( "faces_dataset", "rb" ) )
weights = perceptronFaceClassifierTrainer(faces.trainData)
numcorrect = 0
numtotal = len(faces.testData)
for i in range(0, numtotal):
	prediction = isFace(faces.testData[i], weights)
	reality = faces.testData[i].label
	print('prediction {}: '.format(prediction))
	print('reality: {}'.format(reality))
	if int(prediction) == int(reality):
		numcorrect += 1

print('total accuracy: {}% for {} images'.format((numcorrect / numtotal) * 100, numtotal))

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
	

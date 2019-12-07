from perceptron import isFace, perceptronFaceClassifierTrainer, whichDigit, perceptronDigitClassifierTrainer
from naiveBayes import isFaceBayes
from nearestNeighbors import NNClassifier
import pickle
import random
import math
import winsound
from utils import featureVector

faces = pickle.load(open("faces_dataset", "rb"))
digits = pickle.load(open("digits_dataset", "rb"))
frequency = 2500  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second

def testNNFaces():
	for imageSetLength in range(int(len(faces.trainData) * 0.1), len(faces.trainData), int(len(faces.trainData) * 0.1)):
		numcorrect = 0
		numtotal = len(faces.testData)
		trainFeatureVectors = []
		for i in faces.trainData[0:imageSetLength]:
			trainFeatureVectors.append(featureVector(i.image, 7, 6, 70, 60))
		for i in range(0, numtotal):
			prediction = NNClassifier(faces.testData[i],  faces.trainData[0:imageSetLength], trainFeatureVectors, 7, 6, 70, 60)
			reality = faces.testData[i].label
			if int(prediction) == int(reality):
				numcorrect += 1
		print('total accuracy on faces: {}% using {}% of training data'.format((numcorrect / numtotal) * 100, math.ceil((imageSetLength / len(faces.trainData) * 100))))
	numcorrect = 0
	numtotal = len(faces.testData)
	trainFeatureVectors = []
	for i in faces.trainData:
		trainFeatureVectors.append(featureVector(i.image, 7, 6, 70, 60))
	for i in range(0, numtotal):
		prediction = NNClassifier(faces.testData[i], faces.trainData, trainFeatureVectors, 7, 6, 70, 60)
		reality = faces.testData[i].label
		if int(prediction) == int(reality):
			numcorrect += 1
	print('total accuracy on faces: {}% using 100% of training data'.format((numcorrect / numtotal) * 100))
	winsound.Beep(frequency, duration)

def testNNDigits():
	for imageSetLength in range(int(len(digits.trainData) * 0.1), len(digits.trainData), int(len(digits.trainData) * 0.1)):
		numcorrect = 0
		numtotal = len(digits.testData)
		trainFeatureVectors = []
		for i in digits.trainData[0:imageSetLength]:
			trainFeatureVectors.append(featureVector(i.image, 14, 4, 28, 28))
		for i in range(0, numtotal):
			prediction = NNClassifier(digits.testData[i],  digits.trainData[0:imageSetLength], trainFeatureVectors, 14, 4, 28, 28)
			reality = digits.testData[i].label
			if int(prediction) == int(reality):
				numcorrect += 1
		print('total accuracy on digits: {}% using {}% of training data'.format((numcorrect / numtotal) * 100, math.ceil((imageSetLength / len(digits.trainData) * 100))))
	numcorrect = 0
	numtotal = len(digits.testData)
	trainFeatureVectors = []
	for i in digits.trainData:
		trainFeatureVectors.append(featureVector(i.image, 14, 4, 28, 28))
	for i in range(0, numtotal):
		prediction = NNClassifier(digits.testData[i], digits.trainData, trainFeatureVectors, 14, 4, 28, 28)
		reality = digits.testData[i].label
		if int(prediction) == int(reality):
			numcorrect += 1
	print('total accuracy on digits: {}% using 100% of training data'.format((numcorrect / numtotal) * 100))
	winsound.Beep(frequency, duration)

def testPerceptronFaces():
	for imageSetLength in range(int(len(faces.trainData) * 0.1), len(faces.trainData), int(len(faces.trainData) * 0.1)):
		weights = perceptronFaceClassifierTrainer(faces.trainData[0:imageSetLength])
		numcorrect = 0
		numtotal = len(faces.testData)
		for i in range(0, numtotal):
			prediction = isFace(faces.testData[i], weights)
			reality = faces.testData[i].label
			if int(prediction) == int(reality):
				numcorrect += 1
		print('total accuracy on faces: {}% using {}% of training data'.format((numcorrect / numtotal) * 100, math.ceil((imageSetLength / len(faces.trainData) * 100))))
	# to catch the 100%
	if len(faces.trainData) % int(len(faces.trainData) * 0.1) != 0:
		weights = perceptronFaceClassifierTrainer(faces.trainData)
		numcorrect = 0
		numtotal = len(faces.testData)
		for i in range(0, numtotal):
			prediction = isFace(faces.testData[i], weights)
			reality = faces.testData[i].label
			if int(prediction) == int(reality):
				numcorrect += 1
		print('total accuracy on faces: {}% using 100% of training data'.format((numcorrect / numtotal) * 100))
		winsound.Beep(frequency, duration)

def testPerceptronDigits():
	trainingData = digits.trainData
	random.shuffle(trainingData)
	for imageSetLength in range(int(len(trainingData) * 0.1), len(trainingData), int(len(trainingData) * 0.1)):
		weights = perceptronDigitClassifierTrainer(trainingData[0:imageSetLength])
		numcorrect = 0
		numtotal = int(len(digits.testData) * 0.10)
		for i in range(0, numtotal):
			prediction = whichDigit(digits.testData[i], weights)
			reality = digits.testData[i].label
			if int(prediction) == int(reality):
				numcorrect += 1
		winsound.Beep(frequency, duration)
		print('total accuracy on digits: {}% using {}% of training data'.format((numcorrect / numtotal) * 100, math.ceil((imageSetLength / len(trainingData) * 100))))
	# to catch the 100%
	if len(trainingData) % int(len(trainingData) * 0.1) != 0:
		weights = perceptronDigitClassifierTrainer(trainingData)
		numcorrect = 0
		numtotal = int(len(digits.testData) * 0.10)
		for i in range(0, numtotal):
			prediction = whichDigit(digits.testData[i], weights)
			reality = digits.testData[i].label
			if int(prediction) == int(reality):
				numcorrect += 1
		print('total accuracy on digits: {}% using 100% of training data'.format((numcorrect / numtotal) * 100))
		winsound.Beep(frequency, duration * 10)

def testNaiveBayesFace():
	for imageSetLength in range(int(len(faces.trainData) * 0.1), len(faces.trainData), int(len(faces.trainData) * 0.1)):
		numcorrect = 0
		numtotal = len(faces.testData)
		for i in range(0, numtotal):
			prediction = isFaceBayes(faces.testData[i], faces.trainData[0:imageSetLength], 14, 14)
			reality = faces.testData[i].label
			if int(prediction) == int(reality):
				numcorrect += 1
		print('total accuracy on faces: {}% using {}% of training data'.format((numcorrect / numtotal) * 100, math.ceil((imageSetLength / len(faces.trainData) * 100))))
	# to catch the 100%
	if len(faces.trainData) % int(len(faces.trainData) * 0.1) != 0:
		numcorrect = 0
		numtotal = len(faces.testData)
		for i in range(0, numtotal):
			prediction = isFaceBayes(faces.testData[i], faces.trainData, 14, 14)
			reality = faces.testData[i].label
			if int(prediction) == int(reality):
				numcorrect += 1
		print('total accuracy on faces: {}% using 100% of training data'.format((numcorrect / numtotal) * 100))
		winsound.Beep(frequency, duration)

testNNDigits()
from perceptron import isFace, perceptronFaceClassifierTrainer, whichDigit, perceptronDigitClassifierTrainer
import pickle
import random
import math

faces = pickle.load(open("faces_dataset", "rb"))
digits = pickle.load(open("digits_dataset", "rb"))

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


def testPerceptronDigits():
	for imageSetLength in range(int(len(digits.trainData) * 0.1), len(digits.trainData), int(len(digits.trainData) * 0.1)):
		weights = perceptronDigitClassifierTrainer(digits.trainData[0:imageSetLength])
		numcorrect = 0
		numtotal = len(digits.testData)
		for i in range(0, numtotal):
			prediction = whichDigit(digits.testData[i], weights)
			reality = digits.testData[i].label
			if int(prediction) == int(reality):
				numcorrect += 1
		print('total accuracy on digits: {}% using {}% of training data'.format((numcorrect / numtotal) * 100, math.ceil((imageSetLength / len(digits.trainData) * 100))))

testPerceptronDigits()
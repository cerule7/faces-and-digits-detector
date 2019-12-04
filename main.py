from perceptron import isFace, perceptronFaceClassifierTrainer
import pickle
import random
import math

faces = pickle.load(open("faces_dataset", "rb"))

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

testPerceptronFaces()
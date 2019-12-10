from perceptron import isFace, perceptronFaceClassifierTrainer, whichDigit, perceptronDigitClassifierTrainer
from naiveBayes import isFaceBayes, whichDigitBayes, featureProbMatrixGen
from nearestNeighbors import NNClassifier
import pickle
import random
from utils import featureVector

faces = pickle.load(open("faces_dataset", "rb"))
digits = pickle.load(open("digits_dataset", "rb"))

def perceptronFace100(index):
	weights = perceptronFaceClassifierTrainer(faces.trainData)
	prediction = isFace(faces.testData[index], weights)
	reality = faces.testData[index].label
	print('Perceptron (Face): prediction: {} reality: {}'.format(prediction, reality))

def perceptronDigit100(index):
	weights = perceptronDigitClassifierTrainer(digits.trainData)
	prediction = whichDigit(digits.testData[index], weights)
	reality = digits.testData[index].label
	print('Perceptron (Digit): prediction: {} reality: {}'.format(prediction, reality))

def bayesFace100(index):
	matrixVector = featureProbMatrixGen(10, 10, 2, A=70, Y=60, trainData=faces.trainData)
	prediction = isFaceBayes(faces.testData[index], faces.trainData, matrixVector, 10, 10)
	reality = faces.testData[index].label
	print('Naive Bayes (Face): prediction: {} reality: {}'.format(prediction, reality))

def bayesDigit100(index):
	matrixVector = featureProbMatrixGen(14, 7, 10, A=28, Y=28, trainData= digits.trainData)
	prediction = whichDigitBayes(digits.testData[index], digits.trainData, matrixVector, 14, 7)
	reality = digits.testData[index].label
	print('Naive Bayes (Digit): prediction: {} reality: {}'.format(prediction, reality))

def knnFace100(index):
	trainFeatureVectors = []
	for i in faces.trainData:
		trainFeatureVectors.append(featureVector(i.image, 7, 6, 70, 60))
	prediction = NNClassifier(faces.testData[index], faces.trainData, trainFeatureVectors, 7, 6, 70, 60)
	reality = faces.testData[index].label
	print('KNN (Face): prediction: {} reality: {}'.format(prediction, reality))

def knnDigit100(index):
	trainFeatureVectors = []
	for i in digits.trainData:
		trainFeatureVectors.append(featureVector(i.image, 4, 14, 28, 28))
	prediction = NNClassifier(digits.testData[index], digits.trainData, trainFeatureVectors, 4, 14, 28, 28)
	reality = digits.testData[index].label
	print('KNN (Digits): prediction: {} reality: {}'.format(prediction, reality))

faceIndex = random.randint(0, len(faces.testData) - 1)
digitIndex = random.randint(0, len(digits.testData) - 1)

bayesFace100(faceIndex)
bayesDigit100(digitIndex)
knnFace100(faceIndex)
knnDigit100(digitIndex)
print('beginning face perceptron training')
perceptronFace100(faceIndex)
print('beginning digit perceptron training')
perceptronDigit100(digitIndex)

from utils import featureVector, findHighestPrediction
from numpy import ones

def featureProbMatrixGen(numColumns, numRows, numLabels, A, Y, trainData):
    totalNumFeatures = len(featureVector(trainData[0].image, numRows, numColumns, A, Y))
    matrixVector = [ones([totalNumFeatures, len(trainData)]) for x in range(0, numLabels)]
    for i in range(0, len(trainData)):
        feature = featureVector(trainData[i].image, numRows, numColumns, A, Y) #feature vector for ith training image
        for j in range(0, len(matrixVector[int(trainData[i].label)])):
            matrixVector[int(trainData[i].label)][j][feature[j]] += 1

    for i in range(0, len(matrixVector)):
        for j in range(0, len(matrixVector[i])):
            total = 0  #total of  individual feature values
            for k in range(0, len(matrixVector[i][j])):
                total += int(matrixVector[i][j][k])
            for k in range(0, len(matrixVector[i][j])):
                matrixVector[i][j][k] /= total

    #The following is to avoid a potential multiply by zero later
    for i in range(0, len(matrixVector)):
        for j in range(0, len(matrixVector[i])):
            for k in range(0, len(matrixVector[i][j])):
                if(matrixVector[i][j][k] == 0):
                    matrixVector[i][j][k] = .001
    return matrixVector

def isFaceBayes(image, trainingData, matrixVector, r, c):
    imageFeatures = featureVector(image.image, r, c, A=70, Y=60)
    faceCount = 0
    for i in trainingData:
        if(int(i.label) == 1):
            faceCount += 1
    probFace = faceCount / len(trainingData)
    probNotFace = 1 - probFace
    probImageGivenFace = 1
    for i in range(0, len(imageFeatures)):
        probImageGivenFace *= matrixVector[0][i][imageFeatures[i]]
        # the 0 in the index is the face feature matrix
    probImageGivenNotFace = 1
    for i in range(0, len(imageFeatures)):
        probImageGivenNotFace *= matrixVector[1][i][imageFeatures[i]]
        # the 1 in the index is the not face feature matrix
    probFaceGivenImage = probImageGivenFace * probFace
    probNotFaceGivenImage = probImageGivenNotFace * probNotFace
    if(probFaceGivenImage > probNotFaceGivenImage):
        return 0
    return 1

def whichDigitBayes(image, trainingData, matrixVector, c, r):
    imageFeatures = featureVector(image.image, r, c, A=28, Y=28)
    digitProbabilities = [0 for x in range(0, 10)]

    for i in range(0, len(trainingData)):
        digitProbabilities[int(trainingData[i].label)] += 1

    for i in range(0, len(digitProbabilities)):
        digitProbabilities[i] /= len(trainingData)

    probImageGivenDigits = [1 for i in range(0, 10)]  #vector of necessary conditional probabilities

    for i in range(0, len(probImageGivenDigits)):
        for j in range(0, len(imageFeatures)):
            probImageGivenDigits[i] *= matrixVector[i][j][imageFeatures[j]]

    probDigitsGivenImage = [probImageGivenDigits[i] * digitProbabilities[i] for i in range(0, len(probImageGivenDigits))] 

    return findHighestPrediction(probDigitsGivenImage)

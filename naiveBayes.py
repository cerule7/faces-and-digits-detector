<<<<<<< HEAD
def featureProbMatrixGen(numColumns, numRows, trainData, labelVector):
    matrixVector = [][][]
    totalNumFeatures = (numColumns + 1) * (numRows + 1)
    for i in len(labelVector):
        matrixVector[i].append(None) #dummy values so we can index latter
    for i in matrixVector:
        for j in range(0, totalNumFeatures):
            matrixVector[i].append(None) #again just making the data structure with empty values to index later
        for k in range(0, trainData[i].size())
            matrixVector[i][k].append(0)
	for i in trainData:
		feature = featureVector(numColumns, numRows, trainData[i]) #feature vector for ith training image
        for j in matrixVector[labelVector[i]]:
            matrixVector[labelVector[i]][j][feature[j]] += 1
             #adding one to position in probability matrix in position for each given feature and its respective value.
             #Will divide by total after
    for i in matrixVector:
        for j in matrixVector[i]:
            sum = 0  #total of  individual feature values
            for k in matrixVector[i][j]
                sum += matrixVector[i][j][k]
            for k in matrixVector[i][j]:
                matrixVector[i][j][k] = matrixVector[i][j][k]/sum

    #The following is to avoid a potential multiply by zero later
    for i in matrixVector:
        for j in matrixVector[i]:
            for k in matrixVector[i][j]:
                if(matrixVector[i][j][k] == 0):
                    matrixVector[i][j][k] = .0000000001
    return matrixVector
=======
from utils import featureVector 

def featureProbMatrixGen(numColumns, numRows, trainData, labelVector):
    matrixVector = []
    totalNumFeatures = (numColumns + 1) * (numRows + 1)
    for i in len(labelVector):
        matrixVector[i].append(None) #dummy values so we can index latter
    for i in matrixVector:
        for j in range(0, totalNumFeatures):
            matrixVector[i].append(None) #again just making the data structure with empty values to index later
        for k in range(0, trainData[i].size()):
            matrixVector[i][k].append(0)
    for i in trainData:
        feature = featureVector(numColumns, numRows, trainData[i]) #feature vector for ith training image
        for j in matrixVector[labelVector[i]]:
            matrixVector[labelVector[i]][j][feature[j]] += 1
             #adding one to position in probability matrix in position for each given feature and its respective value.
             #Will divide by total after
    for i in matrixVector:
        for j in matrixVector[i]:
            sum = 0  #total of  individual feature values
            for k in matrixVector[i][j]:
                sum += matrixVector[i][j][k]
            for k in matrixVector[i][j]:
                matrixVector[i][j][k] = matrixVector[i][j][k]/sum

    #The following is to avoid a potential multiply by zero later
    for i in matrixVector:
        for j in matrixVector[i]:
            for k in matrixVector[i][j]:
                if(matrixVector[i][j][k] == 0):
                    matrixVector[i][j][k] = .0000000001
    return matrixVector

        def isFaceBayes(image, train, labelVector, c, r):
            matrixVector = featureProbMatrixGen(train, labelVector, c, r)
            imageFeatures = featureVector(c, r, image)
            probFace = 0
            faceCount = 0
            for i in labelVector:
                if(labelVector[i] == 0):
                    faceCount += 1
            probFace = faceCount/len(labelVector)
            probNotFace = 1 - probFace
            probImageGivenFace = 1
            for i in imageFeatures:
                probImageGivenFace = probImageGivenFace * matrixVector[0][i][imageFeatures[i]]
                #performing calculation as described in video at 13:00. Note, the 0 in the index is the face feature matrix
            probImageGivenNotFace = 1
            for i in imageFeatures:
                probImageGivenFace = probImageGivenFace * matrixVector[1][i][imageFeatures[i]]
                #Note, the 1 in the index is the not face feature matrix
            probFaceGivenImage = probImageGivenFace * probFace
            probNotFaceGivenImage = probImageGivenNotFace * probNotFace
            if(probFaceGivenImage > probNotFaceGivenImage):
                return 0
            return 1

        def whatDigitBayes(image, train, labelVector, c, r):
            matrixVector =  featureProbMatrixGen(train, labelVector, c, r)
	        imageFeatures = featureVectorGenerator(c, r, image)
            digitProbibilities = [] #vector of probabilities for each digit
            q = 0
            while(q < 10):
                digitProbibilities.append(0) #dummy values for indexing
            for i in labelVector:
                digitProbibilities[labelVector[i]] += 1
            for i in digitProbibilities:
                digitProbabilities[i] = digitProbabilities[i]/len(labelVector)

            probImageGivenDigits = []  #vector of necessary conditional probabilities
            k = 0
            while(k < 10):
                probImageGivenDigits[i] = 1
            for i in probImageGivenDigits
                for j in imageFeatures:
                    probImageGivenDigits[i] = probImageGivenDigits[i] * matrixVector[i][j][imageFeatures[j]]
            probDigitsGivenImage = [] #final list of probabilities to compare
            for i in probImageGivenDigits:
                probDigitsGivenImage.append(probImageGivenDigits[i]*digitProbabilities[i]

            return probDigitsGivenImage.index(max(probDigitsGivenImage))
>>>>>>> 635ec2c653d928de6cd650d37d58419af2b512bd

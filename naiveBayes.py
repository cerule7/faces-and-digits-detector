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
